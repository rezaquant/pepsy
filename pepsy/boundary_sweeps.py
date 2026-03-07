"""Boundary-MPS construction and fitting utilities for DMRG-like PEPS contractions."""

import logging
import re
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .boundary_states import BdyMPS
from .core import build_optimizer, fidel_mps
from .dmrg_fit import FIT

logger = logging.getLogger(__name__)

__all__ = ["BdyMPS", "CompBdy", "fidel_mps", "build_optimizer", "opt_"]

# Backward-compatible alias.
opt_ = build_optimizer


@dataclass(frozen=True)
class DirectionSpec:
    """Direction-dependent tags and sweep extents."""

    cut_tag_id: str
    site_tag_id: str
    n_steps: int
    left_steps: int
    right_steps: int
    left_index: int


def max_tag_number(tags, tag_format):
    """Return the maximum numeric suffix matching tag pattern ``tag_format``."""
    prefix = tag_format[:-2]
    pattern = re.compile(rf"^{prefix}(\d+)$")

    nums = []
    for tag in tags:
        match = pattern.match(tag)
        if match:
            nums.append(int(match.group(1)))

    return max(nums) if nums else None


class CompBdy:  # pylint: disable=too-many-instance-attributes
    """Boundary MPS fitting driver for x/y contraction sweeps.

    Notes
    -----
    - This class mutates ``self.mps_boundaries`` when ``re_update=True``.
    - Per-step fidelity values are exposed via ``self.fidel`` and reset at
      the start of each :meth:`run` call.
    """

    def __init__(
        self,
        norm,
        mps_boundaries,
        *,
        opt="auto-hq",
        dmrg_run="eff",
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        if not isinstance(mps_boundaries, dict):
            raise TypeError("mps_boundaries must be a dictionary of boundary states.")

        self.norm = norm
        self.mps_boundaries = mps_boundaries
        self.opt = opt
        self.dmrg_run = dmrg_run

        # Runtime sweep options are configured per call in run/move methods.
        self.eq_norms = False
        self.n_iter = 4
        self.flat = False
        self.re_update = True
        self.re_tag = False
        self.visual_ = False
        self.fidel_ = False
        self.fidel = []
        self.pbar = False
        self.max_separation = 0
        self.direction = "y"

        self.warning_enabled = True
        self._warned_flat_initial_slice = False
        self.y_left = 0
        self.y_right = 0
        self.x_left = 0
        self.x_right = 0

        # Extract lattice sizes from tags.
        max_y = max_tag_number(list(norm.tags), "Y{}")
        max_x = max_tag_number(list(norm.tags), "X{}")
        if max_y is None or max_x is None:
            raise ValueError(
                "norm must include X*/Y* tags so lattice shape can be inferred."
            )
        self.Ly = 1 + max_y  # pylint: disable=invalid-name
        self.Lx = 1 + max_x  # pylint: disable=invalid-name
        self._update_separation()

    @property
    def fidelity(self):
        """Alias for ``self.fidel``."""
        return self.fidel

    def _reset_fidelity_history(self):
        """Reset stored fidelity values for a fresh public call."""
        self.fidel = []

    @staticmethod
    def _direction_base(direction):
        """Return canonical base direction: ``y`` or ``x``."""
        if direction.startswith("y"):
            return "y"
        if direction.startswith("x"):
            return "x"
        raise ValueError(f"Unsupported direction: {direction}")

    def _axis_length_from_cut_tag(self, cut_tag_id):
        """Return number of cut positions addressed by ``cut_tag_id``."""
        if cut_tag_id == "Y{}":
            return self.Ly
        if cut_tag_id == "X{}":
            return self.Lx
        raise ValueError(f"Unsupported cut_tag_id: {cut_tag_id}")

    def _direction_tags(self, direction):
        """Return ``(cut_tag_id, site_tag_id, n_steps)`` for a direction."""
        base = self._direction_base(direction)
        if base == "y":
            return "Y{}", "X{}", self.Ly - 1
        if base == "x":
            return "X{}", "Y{}", self.Lx - 1
        raise ValueError(f"Unsupported direction: {direction}")

    def _run_direction_spec(self, direction):
        """Return run-time sweep tags and left/right extents."""
        base = self._direction_base(direction)
        if base == "y":
            return DirectionSpec(
                cut_tag_id="Y{}",
                site_tag_id="X{}",
                n_steps=self.y_left + self.y_right,
                left_steps=self.y_left,
                right_steps=self.y_right,
                left_index=self.y_left,
            )
        if base == "x":
            return DirectionSpec(
                cut_tag_id="X{}",
                site_tag_id="Y{}",
                n_steps=self.x_left + self.x_right,
                left_steps=self.x_left,
                right_steps=self.x_right,
                left_index=self.x_left,
            )
        raise ValueError(f"Unsupported direction: {direction}")

    def _apply_runtime_overrides(
        self,
        *,
        mps_boundaries=None,
        re_tag=False,
        visual_=False,
        flat=False,
        fidel_=False,
        pbar=False,
        n_iter=4,
        eq_norms=False,
        re_update=True,
    ):  # pylint: disable=too-many-arguments
        """Apply run-time options explicitly for a single public call."""
        if mps_boundaries is not None:
            if not isinstance(mps_boundaries, dict):
                raise TypeError("mps_boundaries must be a dictionary of boundary states.")
            self.mps_boundaries = mps_boundaries

        self.re_tag = re_tag
        self.visual_ = visual_
        self.flat = flat
        self.fidel_ = fidel_
        self.pbar = pbar
        self.n_iter = n_iter
        self.eq_norms = eq_norms
        self.re_update = re_update

    def _update_separation(self):
        """Update left/right sweep extents from ``max_separation``."""
        if self.max_separation == 0:
            self.y_left = self.Ly // 2
            self.y_right = self.Ly - (self.Ly // 2)

            self.x_left = self.Lx // 2
            self.x_right = self.Lx - (self.Lx // 2)
        elif self.max_separation == 1:
            # y dir
            self.y_left = (self.Ly // 2) - 1
            self.y_right = self.Ly - (self.Ly // 2)

            # x dir
            self.x_left = (self.Lx // 2) - 1
            self.x_right = self.Lx - (self.Lx // 2)
        else:
            raise ValueError("max_separation must be 0 or 1.")

    def _cut_idx_and_key(self, side, step_idx, cut_tag_id):
        """Return ``(cut_idx, boundary_key, axis_len)`` for sweep side/step."""
        axis_len = self._axis_length_from_cut_tag(cut_tag_id)
        if side == "right":
            cut_idx = axis_len - step_idx - 1
            boundary_key = f"{cut_tag_id.format(step_idx)}_r"
        else:
            cut_idx = step_idx
            boundary_key = f"{cut_tag_id.format(step_idx)}_l"
        return cut_idx, boundary_key, axis_len

    @staticmethod
    def _previous_boundary_key(side, step_idx, cut_tag_id):
        """Return previous boundary key for a given side and step."""
        suffix = "_r" if side == "right" else "_l"
        return f"{cut_tag_id.format(step_idx - 1)}{suffix}"

    def _run_fit_solver(self, fit, boundary_mps):
        """Run selected fitting backend with explicit validation."""
        # Fidelity is tracked externally by CompBdy via self.fidel.
        verbose = False
        if boundary_mps.L == 1:
            fit.run(n_iter=self.n_iter, verbose=verbose)
            return
        if self.dmrg_run == "eff":
            fit.run_eff(n_iter=self.n_iter, verbose=verbose)
            return
        if self.dmrg_run == "global":
            fit.run(n_iter=self.n_iter, verbose=verbose)
            return
        raise ValueError(f"Unsupported dmrg_run mode: {self.dmrg_run}")

    def _maybe_visualize_fit(
        self,
        tn,
        boundary_mps,
        fit,
        site_tag_id,
        axis_len,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Draw intermediate networks when ``visual_`` is enabled."""
        if not self.visual_:
            return

        draw_tags = [site_tag_id.format(i) for i in range(axis_len)]
        draw_kwargs = {
            "legend": False,
            "show_inds": False,
            "show_tags": True,
            "figsize": (8, 8),
            "node_outline_darkness": 0.1,
            "node_outline_size": None,
            "highlight_inds_color": "darkred",
            "edge_scale": 2.0,
            "layout": "neato",
            "refine_layout": "auto",
            "highlight_inds": tn.outer_inds(),
        }
        tn.draw(draw_tags, **draw_kwargs)
        (tn & boundary_mps).draw(draw_tags, **draw_kwargs)
        fit.visual(figsize=(8, 8), show_inds="bond-size", tags_=[])

    def _fit_one_side(
        self,
        side,
        steps,
        progress_bar,
        cut_tag_id,
        site_tag_id,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        """Sweep one side boundary for ``steps`` and fit at each position."""
        previous = None

        for step_idx in range(steps):
            cut_idx, boundary_key, axis_len = self._cut_idx_and_key(
                side,
                step_idx,
                cut_tag_id,
            )
            tn_slice = self.norm.select(cut_tag_id.format(cut_idx), "any")

            if step_idx == 0 and self.flat:
                if self.warning_enabled and not self._warned_flat_initial_slice:
                    logger.warning(
                        "flat=True skips fitting for initial boundary slice (%s).",
                        boundary_key,
                    )
                    self._warned_flat_initial_slice = True
                previous = tn_slice
                continue

            if step_idx == 0:
                tn = tn_slice
            else:
                if previous is None:
                    raise ValueError("Missing previous boundary MPS during fitting.")
                tn = tn_slice | previous

            boundary_mps = self.mps_boundaries[boundary_key]
            if previous is not None and step_idx > 0:
                boundary_mps.exponent = complex(previous.exponent).real

            fit = FIT(
                tn,
                p=boundary_mps,
                inplace=False,
                site_tag_id=site_tag_id,
                opt=self.opt,
                re_tag=self.re_tag,
            )
            self._maybe_visualize_fit(tn, boundary_mps, fit, site_tag_id, axis_len)
            self._run_fit_solver(fit, boundary_mps)

            if self.eq_norms:
                fit.p.equalize_norms_(value=self.eq_norms)
            if self.fidel_:
                fidelity = fidel_mps(tn, fit.p)
                self.fidel.append(fidelity)

            if progress_bar is not None:
                if self.fidel_:
                    prod_fidelity = np.prod(self.fidel)
                    progress_bar.set_postfix({"F": complex(prod_fidelity).real})
                    progress_bar.refresh()
                progress_bar.update(1)

            previous = fit.p
            if self.re_update:
                self.mps_boundaries[boundary_key] = fit.p.copy()

        return previous

    def _fit_one_step(
        self,
        side,
        step_,
        cut_tag_id,
        site_tag_id,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        """Fit a single boundary step for one side and return updated MPS."""
        previous = None

        cut_idx, boundary_key, axis_len = self._cut_idx_and_key(side, step_, cut_tag_id)
        tn_slice = self.norm.select(cut_tag_id.format(cut_idx), "any")

        if step_ > 0:
            previous = self.mps_boundaries.get(
                self._previous_boundary_key(side, step_, cut_tag_id)
            )

        if step_ == 0 and self.flat:
            if self.warning_enabled and not self._warned_flat_initial_slice:
                logger.warning(
                    "flat=True skips fitting for initial boundary slice (%s).",
                    boundary_key,
                )
                self._warned_flat_initial_slice = True
            return tn_slice

        if step_ == 0:
            tn = tn_slice
        else:
            if previous is None:
                raise ValueError("Missing previous boundary MPS during fitting.")
            tn = tn_slice | previous

        boundary_mps = self.mps_boundaries[boundary_key]
        if previous is not None and step_ > 0:
            boundary_mps.exponent = complex(previous.exponent).real

        fit = FIT(
            tn,
            p=boundary_mps,
            inplace=False,
            site_tag_id=site_tag_id,
            opt=self.opt,
            re_tag=self.re_tag,
        )
        self._maybe_visualize_fit(tn, boundary_mps, fit, site_tag_id, axis_len)
        self._run_fit_solver(fit, boundary_mps)

        if self.eq_norms:
            fit.p.equalize_norms_(value=self.eq_norms)
        if self.fidel_:
            fidelity = fidel_mps(tn, fit.p)
            self.fidel.append(fidelity)

        previous = fit.p
        if self.re_update:
            self.mps_boundaries[boundary_key] = fit.p.copy()

        return previous


    @staticmethod
    def _direction_sides(direction):
        """Return ``(move_left, move_right)`` from direction selector."""
        return "left" in direction, "right" in direction

    def _build_final_boundary_network(self, spec, p_previous_l, p_previous_r):
        """Build final TN by combining left/right fitted boundaries."""
        if p_previous_r is None:
            raise ValueError("Boundary contraction failed: missing right boundary MPS.")
        if self.max_separation == 0:
            return p_previous_r if p_previous_l is None else (p_previous_r | p_previous_l)

        center = self.norm.select(spec.cut_tag_id.format(spec.left_index), "any")
        if p_previous_l is None:
            return p_previous_r | center
        return p_previous_r | center | p_previous_l

    def run(
        self,
        *,
        re_update=True,
        max_separation=0,
        mps_boundaries=None,
        re_tag=False,
        visual_=False,
        flat=False,
        fidel_=False,
        pbar=False,
        n_iter=4,
        eq_norms=True,
        direction="y",
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Run two-sided boundary sweeps and contract the final network.

        Parameters
        ----------
        re_update : bool, default=True
            Write fitted boundary MPS values back to ``self.mps_boundaries``.
        max_separation : int, default=0
            Separation mode (currently ``0`` or ``1``).
        mps_boundaries : dict | None, default=None
            Optional replacement boundary dictionary for this call.
        re_tag : bool, default=False
            Forwarded to fit backend.
        visual_ : bool, default=False
            Enable intermediate tensor-network drawings.
        flat : bool, default=False
            Skip first-step fitting and use raw slice directly.
        fidel_ : bool, default=False
            If ``True``, compute and store per-step fidelity values in
            ``self.fidel``.
        pbar : bool, default=False
            Show progress bar.
        n_iter : int, default=4
            Number of local fit iterations for each step.
        eq_norms : bool, default=True
            Forwarded normalization option for fitted MPS tensors.
        direction : str, default="y"
            Sweep selector.

        Returns
        -------
        complex | float
            Final contracted scalar.
        """
        # Fidelity history is run-local and resets for each run() call.
        self._reset_fidelity_history()
        self.max_separation = max_separation
        self._update_separation()
        self._warned_flat_initial_slice = False
        self._apply_runtime_overrides(
            mps_boundaries=mps_boundaries,
            re_tag=re_tag,
            visual_=visual_,
            flat=flat,
            fidel_=fidel_,
            pbar=pbar,
            n_iter=n_iter,
            eq_norms=eq_norms,
            re_update=re_update,
        )

        self.direction = direction
        spec = self._run_direction_spec(direction)
        with tqdm(
            total=spec.n_steps,
            desc="bdy_dmrg:",
            leave=True,
            position=0,
            colour="CYAN",
            disable=not self.pbar,
        ) as progress_bar:
            p_previous_l = self._fit_one_side(
                "left",
                spec.left_steps,
                progress_bar,
                spec.cut_tag_id,
                spec.site_tag_id,
            )
            p_previous_r = self._fit_one_side(
                "right",
                spec.right_steps,
                progress_bar,
                spec.cut_tag_id,
                spec.site_tag_id,
            )

        tn_f = self._build_final_boundary_network(spec, p_previous_l, p_previous_r)
        main, exp = tn_f.contract(all, optimize=self.opt, strip_exponent=True)
        return main * 10**exp

    def move_bdy(
        self,
        *,
        mps_boundaries=None,
        re_tag=False,
        visual_=False,
        flat=False,
        fidel_=False,
        pbar=False,
        n_iter=4,
        eq_norms=False,
        direction="y_left",
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Sweep and update stored boundary MPSs for a selected side/direction."""
        self._reset_fidelity_history()
        self._warned_flat_initial_slice = False
        self._apply_runtime_overrides(
            mps_boundaries=mps_boundaries,
            re_tag=re_tag,
            visual_=visual_,
            flat=flat,
            fidel_=fidel_,
            pbar=pbar,
            n_iter=n_iter,
            eq_norms=eq_norms,
            re_update=True,
        )

        self.direction = direction
        cut_tag_id, site_tag_id, n_steps = self._direction_tags(direction)
        move_left, move_right = self._direction_sides(direction)
        if not (move_left or move_right):
            raise ValueError(f"direction must include 'left' or 'right', got: {direction}")
        total_updates = n_steps * (int(move_left) + int(move_right))

        with tqdm(
            total=total_updates,
            desc="move:",
            leave=True,
            position=0,
            colour="CYAN",
            disable=not self.pbar,
        ) as progress_bar:
            if move_left:
                self._fit_one_side(
                    "left",
                    n_steps,
                    progress_bar,
                    cut_tag_id,
                    site_tag_id,
                )
            if move_right:
                self._fit_one_side(
                    "right",
                    n_steps,
                    progress_bar,
                    cut_tag_id,
                    site_tag_id,
                )

    def move_step_bdy(
        self,
        *,
        pos=0,
        mps_boundaries=None,
        re_tag=False,
        visual_=False,
        flat=False,
        fidel_=False,
        pbar=False,
        n_iter=4,
        eq_norms=False,
        direction="y_left",
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Fit and update one boundary step at position ``pos``."""
        self._reset_fidelity_history()
        self._warned_flat_initial_slice = False
        self._apply_runtime_overrides(
            mps_boundaries=mps_boundaries,
            re_tag=re_tag,
            visual_=visual_,
            flat=flat,
            fidel_=fidel_,
            pbar=pbar,
            n_iter=n_iter,
            eq_norms=eq_norms,
            re_update=True,
        )

        self.direction = direction
        cut_tag_id, site_tag_id, n_steps = self._direction_tags(direction)
        if not isinstance(pos, int):
            raise TypeError("pos must be an integer")
        if pos < 0 or pos >= n_steps:
            raise ValueError(f"pos must be in [0, {n_steps - 1}] for direction={direction}")

        move_left, move_right = self._direction_sides(direction)
        if not (move_left or move_right):
            raise ValueError(f"direction must include 'left' or 'right', got: {direction}")

        if move_left:
            self._fit_one_step(
                "left",
                pos,
                cut_tag_id=cut_tag_id,
                site_tag_id=site_tag_id,
            )
        if move_right:
            self._fit_one_step(
                "right",
                pos,
                cut_tag_id=cut_tag_id,
                site_tag_id=site_tag_id,
            )
