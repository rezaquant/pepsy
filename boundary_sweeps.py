"""Boundary-MPS construction and fitting utilities for DMRG-like PEPS contractions."""


import logging
import re
from dataclasses import dataclass

import numpy as np

from tqdm import tqdm

import autoray as ar
import jax

from boundary_states import BdyMPO
from dmrg_fit import FIT
from dmrg_helpers import fidel_mps, opt_

ar.register_function("torch", "stop_gradient", lambda x: x.detach())  # type: ignore[attr-defined]
ar.register_function("jax", "stop_gradient", jax.lax.stop_gradient)  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

__all__ = ["BdyMPO", "CompBdy", "fidel_mps", "opt_", "stop_grad"]


@dataclass(frozen=True)
class DirectionSpec:
    """Direction-dependent tags and sweep extents."""

    cut_tag_id: str
    site_tag_id: str
    n_steps: int
    left_steps: int
    right_steps: int
    left_index: int


def stop_grad(x):
    """Stop gradients for the active autoray backend."""
    return ar.do("stop_gradient", x)


def backend_numpy(dtype=np.float64):
    """Return a NumPy array caster for backend conversion."""

    def to_backend(x, dtype=dtype):
        return np.array(x, dtype=dtype)

    return to_backend



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
    """Boundary MPS fitting driver used for y/x/diagonal contraction sweeps."""

    def __init__(
        self,
        norm,
        mpo_b,
        opt="auto-hq",
        eq_norms=False,
        n_iter=4,
        flat=False,
        re_update=True,
        dmrg_run="eff",
        max_seperation=0,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self.norm = norm
        self.mpo_b = mpo_b
        self.opt = opt
        self.eq_norms = eq_norms
        self.n_iter = n_iter
        self.flat = flat
        self.re_update = re_update
        self.dmrg_run = dmrg_run
        self.re_tag = False
        self.visual_ = False
        self.fidel_ = False
        self.stop_grad_ = False
        self.pbar = False
        self.max_seperation = max_seperation
        self.direction = "y"
        self.warning_enabled = True
        self._warned_flat_initial_slice = False
        self.y_left = 0
        self.y_right = 0
        self.x_left = 0
        self.x_right = 0
        self.d_left = 0
        self.d_right = 0
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

    @staticmethod
    def _direction_base(direction):
        """Return canonical base direction: ``y``, ``x``, or ``diag``."""
        if direction.startswith("diag"):
            return "diag"
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
        if cut_tag_id == "Du{}":
            return self.Ly + self.Lx - 1
        raise ValueError(f"Unsupported cut_tag_id: {cut_tag_id}")

    def _direction_tags(self, direction):
        """Return ``(cut_tag_id, site_tag_id, n_steps)`` for a direction."""
        base = self._direction_base(direction)
        if base == "y":
            return "Y{}", "X{}", self.Ly - 1
        if base == "x":
            return "X{}", "Y{}", self.Lx - 1
        return "Du{}", "du{}", self.Ly + self.Lx - 2

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
        return DirectionSpec(
            cut_tag_id="Du{}",
            site_tag_id="du{}",
            n_steps=self.d_left + self.d_right,
            left_steps=self.d_left,
            right_steps=self.d_right,
            left_index=self.d_left,
        )

    def _apply_runtime_overrides(
        self,
        *,
        mpo_b=None,
        re_tag=None,
        visual_=None,
        flat=None,
        fidel_=None,
        stop_grad_=None,
        pbar=None,
        n_iter=None,
        eq_norms=None,
    ):  # pylint: disable=too-many-arguments
        """Apply optional runtime overrides for run/move methods."""
        if mpo_b is not None:
            self.mpo_b = mpo_b
        if re_tag is not None:
            self.re_tag = re_tag
        if visual_ is not None:
            self.visual_ = visual_
        if flat is not None:
            self.flat = flat
        if fidel_ is not None:
            self.fidel_ = fidel_
        if stop_grad_ is not None:
            self.stop_grad_ = stop_grad_
        if pbar is not None:
            self.pbar = pbar
        if n_iter is not None:
            self.n_iter = n_iter
        if eq_norms is not None:
            self.eq_norms = eq_norms

    def _update_separation(self):
        """Update left/right sweep extents from ``max_seperation``."""
        if self.max_seperation == 0:
            self.y_left = self.Ly // 2
            self.y_right = self.Ly - (self.Ly // 2)

            self.x_left = self.Lx // 2
            self.x_right = self.Lx - (self.Lx // 2)

            # diagonal
            l_d = self.Lx + self.Ly - 1
            self.d_left = l_d // 2
            self.d_right = l_d - l_d // 2
        elif self.max_seperation == 1:
            # y dir
            self.y_left = (self.Ly // 2) - 1
            self.y_right = self.Ly - (self.Ly // 2)

            # x dir
            self.x_left = (self.Lx // 2) - 1
            self.x_right = self.Lx - (self.Lx // 2)

            # diagonal
            l_d = self.Lx + self.Ly - 1
            self.d_left = (l_d // 2) - 1
            self.d_right = l_d - (l_d // 2)
        else:
            raise ValueError("max_seperation must be 0 or 1.")

    def _cut_idx_and_key(self, side, step_idx, cut_tag_id):
        """Return ``(cut_idx, mpo_key, axis_len)`` for sweep side/step."""
        axis_len = self._axis_length_from_cut_tag(cut_tag_id)
        if side == "right":
            cut_idx = axis_len - step_idx - 1
            mpo_key = f"{cut_tag_id.format(step_idx)}_r"
        else:
            cut_idx = step_idx
            mpo_key = f"{cut_tag_id.format(step_idx)}_l"
        return cut_idx, mpo_key, axis_len

    @staticmethod
    def _previous_mpo_key(side, step_idx, cut_tag_id):
        """Return previous MPO boundary key for a given side and step."""
        suffix = "_r" if side == "right" else "_l"
        return f"{cut_tag_id.format(step_idx - 1)}{suffix}"

    def _run_fit_solver(self, fit, boundary_mps):
        """Run selected fitting backend with explicit validation."""
        if boundary_mps.L == 1:
            fit.run(n_iter=self.n_iter)
            return
        if self.dmrg_run == "eff":
            fit.run_eff(n_iter=self.n_iter)
            return
        if self.dmrg_run == "global":
            fit.run(n_iter=self.n_iter)
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
        fidelity = -1.0
        previous = None

        for step_idx in range(steps):
            cut_idx, mpo_key, axis_len = self._cut_idx_and_key(
                side,
                step_idx,
                cut_tag_id,
            )
            tn_slice = self.norm.select(cut_tag_id.format(cut_idx), "any")

            if step_idx == 0 and self.flat:
                if self.warning_enabled and not self._warned_flat_initial_slice:
                    logger.warning(
                        "flat=True skips fitting for initial boundary slice (%s).",
                        mpo_key,
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

            boundary_mps = self.mpo_b[mpo_key]
            if previous is not None and step_idx > 0:
                boundary_mps.exponent = complex(previous.exponent).real

            fit = FIT(
                tn,
                p=boundary_mps,
                inplace=False,
                site_tag_id=site_tag_id,
                opt=self.opt,
                re_tag=self.re_tag,
                stop_grad_=self.stop_grad_,
            )
            self._maybe_visualize_fit(tn, boundary_mps, fit, site_tag_id, axis_len)
            self._run_fit_solver(fit, boundary_mps)

            if self.eq_norms:
                fit.p.equalize_norms_(value=self.eq_norms)
            if self.pbar and self.fidel_:
                fidelity = fidel_mps(tn, fit.p)

            if progress_bar is not None:
                if self.fidel_:
                    progress_bar.set_postfix({"F": complex(fidelity).real})
                    progress_bar.refresh()
                progress_bar.update(1)

            previous = fit.p
            if self.re_update:
                self.mpo_b[mpo_key] = fit.p.copy()

        return previous

    def _fit_one_step(
        self,
        side,
        step_,
        cut_tag_id,
        site_tag_id,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        """Fit a single boundary step for one side and return updated MPS."""
        fidelity = -1.0
        previous = None

        cut_idx, mpo_key, axis_len = self._cut_idx_and_key(side, step_, cut_tag_id)
        tn_slice = self.norm.select(cut_tag_id.format(cut_idx), "any")

        if step_ > 0:
            previous = self.mpo_b.get(self._previous_mpo_key(side, step_, cut_tag_id))

        if step_ == 0 and self.flat:
            if self.warning_enabled and not self._warned_flat_initial_slice:
                logger.warning(
                    "flat=True skips fitting for initial boundary slice (%s).",
                    mpo_key,
                )
                self._warned_flat_initial_slice = True
            return tn_slice

        if step_ == 0:
            tn = tn_slice
        else:
            if previous is None:
                raise ValueError("Missing previous boundary MPS during fitting.")
            tn = tn_slice | previous

        boundary_mps = self.mpo_b[mpo_key]
        if previous is not None and step_ > 0:
            boundary_mps.exponent = complex(previous.exponent).real

        fit = FIT(
            tn,
            p=boundary_mps,
            inplace=False,
            site_tag_id=site_tag_id,
            opt=self.opt,
            re_tag=self.re_tag,
            stop_grad_=self.stop_grad_,
        )
        self._maybe_visualize_fit(tn, boundary_mps, fit, site_tag_id, axis_len)
        self._run_fit_solver(fit, boundary_mps)

        if self.eq_norms:
            fit.p.equalize_norms_(value=self.eq_norms)
        if self.fidel_:
            fidelity = fidel_mps(tn, fit.p)
        _ = fidelity

        previous = fit.p
        if self.re_update:
            self.mpo_b[mpo_key] = fit.p.copy()

        return previous


    @staticmethod
    def _direction_sides(direction):
        """Return ``(move_left, move_right)`` from direction selector."""
        return "left" in direction, "right" in direction

    def _build_final_boundary_network(self, spec, p_previous_l, p_previous_r):
        """Build final TN by combining left/right fitted boundaries."""
        if p_previous_r is None:
            raise ValueError("Boundary contraction failed: missing right boundary MPS.")
        if self.max_seperation == 0:
            return p_previous_r if p_previous_l is None else (p_previous_r | p_previous_l)

        center = self.norm.select(spec.cut_tag_id.format(spec.left_index), "any")
        if p_previous_l is None:
            return p_previous_r | center
        return p_previous_r | center | p_previous_l

    def run(
        self,
        *,
        re_update=True,
        max_seperation=0,
        mpo_b=None,
        re_tag=None,
        visual_=None,
        flat=None,
        fidel_=None,
        stop_grad_=None,
        pbar=None,
        n_iter=None,
        eq_norms=None,
        direction="y",
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Run two-sided boundary sweeps and contract the final network."""
        if re_update is not None:
            self.re_update = re_update
        if max_seperation is not None:
            self.max_seperation = max_seperation
            self._update_separation()
        self._apply_runtime_overrides(
            mpo_b=mpo_b,
            re_tag=re_tag,
            visual_=visual_,
            flat=flat,
            fidel_=fidel_,
            stop_grad_=stop_grad_,
            pbar=pbar,
            n_iter=n_iter,
            eq_norms=eq_norms,
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
        mpo_b=None,
        re_tag=None,
        visual_=None,
        flat=None,
        fidel_=None,
        stop_grad_=False,
        pbar=None,
        n_iter=None,
        eq_norms=None,
        direction="y_left",
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Sweep and update stored boundary MPSs for a selected side/direction."""
        self.re_update = True
        self._apply_runtime_overrides(
            mpo_b=mpo_b,
            re_tag=re_tag,
            visual_=visual_,
            flat=flat,
            fidel_=fidel_,
            stop_grad_=stop_grad_,
            pbar=pbar,
            n_iter=n_iter,
            eq_norms=eq_norms,
        )

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
        mpo_b=None,
        re_tag=None,
        visual_=None,
        flat=None,
        fidel_=None,
        stop_grad_=False,
        pbar=None,
        n_iter=None,
        eq_norms=None,
        direction="y_left",
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Fit and update one boundary step at position ``pos``."""
        self.re_update = True
        self._apply_runtime_overrides(
            mpo_b=mpo_b,
            re_tag=re_tag,
            visual_=visual_,
            flat=flat,
            fidel_=fidel_,
            stop_grad_=stop_grad_,
            pbar=pbar,
            n_iter=n_iter,
            eq_norms=eq_norms,
        )

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
