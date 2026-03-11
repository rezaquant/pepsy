"""Boundary-MPS initialization utilities for PEPS boundary contraction."""

import logging
import re
import sys
import warnings

import numpy as np
import quimb.tensor as qtn

from .core import get_default_array_backend

logger = logging.getLogger(__name__)


def make_numpy_array_caster(dtype=np.float64):
    """Return a callable that casts arrays to a target NumPy dtype."""

    def to_backend(x, dtype=dtype):
        return np.array(x, dtype=dtype)

    return to_backend


class BdyMPS:
    """Build and store boundary MPS environments for PEPS contractions.

    The class precomputes left/right boundaries for both sweep axes:

    - Y-cuts with site tags ``X{}``
    - X-cuts with site tags ``Y{}``

    Boundary entries are stored in ``mps_b`` with keys:

    - ``Y{i}_l`` and ``Y{i}_r``
    - ``X{i}_l`` and ``X{i}_r``

    where ``i`` is the sweep-step index.

    Parameters
    ----------
    tn_flat : qtn.TensorNetwork | None
        Flattened tensor network used when ``flat=True``.
    tn_double : qtn.TensorNetwork | None
        Double-layer tensor network used when ``flat=False``.
    chi : int
        Maximum bond dimension for generated boundary MPS tensors.
    flat : bool
        If ``True``, initialize the first boundary slice directly from a
        flattened tensor network.
    to_backend : callable | None
        Function applied to tensor arrays before storage. If ``None``, uses
        :func:`pepsy.core.get_default_array_backend` when set, otherwise identity.
    seed : int
        Random seed used during boundary randomization.
    single_layer : bool
        If ``True``, initialize boundaries with single-layer construction.

    Attributes
    ----------
    mps_b : dict[str, qtn.MatrixProductState]
        Boundary map keyed by cut tags such as ``Y0_l`` and ``X2_r``.
    """
    # (side, site_tag_id, cut_tag_id) sweep definitions used to prebuild boundaries.
    _SWEEP_SPECS = (
        ("left", "X{}", "Y{}"),
        ("right", "X{}", "Y{}"),
        ("left", "Y{}", "X{}"),
        ("right", "Y{}", "X{}"),
    )

    def __init__(
        self,
        tn_flat=None,
        tn_double=None,
        chi=8,
        flat=False,
        to_backend=None,
        seed=1,
        single_layer=False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        if chi < 1:
            raise ValueError("chi must be >= 1")
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be callable or None")

        self.seed = seed
        self._chi_target = int(chi)
        self.flat = flat
        backend = to_backend if to_backend is not None else get_default_array_backend()
        self.to_backend = (lambda x: x) if backend is None else backend
        self.numpy_backend = make_numpy_array_caster(dtype="complex128")

        tn_ref = None
        if flat:
            if tn_flat is None:
                raise ValueError("tn_flat is required when flat=True")
            self._tn_norm = tn_flat.copy()
            tn_ref = tn_flat
        else:
            if tn_double is None:
                raise ValueError("tn_double is required when flat=False")
            self._tn_norm = tn_double.copy()
            tn_ref = tn_flat if tn_flat is not None else tn_double

        # Validate that lattice dimensions can be resolved early.
        self._infer_lattice_shape(tn_ref, self._tn_norm)

        if self.flat and single_layer:
            warnings.warn(
                "flat=True is incompatible with single_layer=True; "
                "using single_layer=False.",
                UserWarning,
                stacklevel=2,
            )
            single_layer = False

        self.mps_b = self._initialize_all_boundaries(single_layer=single_layer)

    def _initialize_all_boundaries(self, single_layer):
        """Build all configured boundary environments for this instance."""
        initializer = (
            self._initialize_single_layer_boundaries
            if single_layer
            else self._initialize_multi_layer_boundaries
        )
        boundaries = {}
        for side, site_tag_id, cut_tag_id in self._SWEEP_SPECS:
            boundaries |= initializer(
                side=side,
                site_tag_id=site_tag_id,
                cut_tag_id=cut_tag_id,
            )
        return boundaries

    @property
    def ly(self):
        """Lattice size along the vertical axis."""
        _, ly = self._infer_lattice_shape(None, self._tn_norm)
        return ly

    @property
    def lx(self):
        """Lattice size along the horizontal axis."""
        lx, _ = self._infer_lattice_shape(None, self._tn_norm)
        return lx

    @property
    def tn_norm(self):
        """Return the internal tensor network used to build boundaries."""
        return self._tn_norm

    @property
    def chi(self):
        """Return the current largest bond dimension across all boundary MPS."""
        if not getattr(self, "mps_b", None):
            return int(self._chi_target)
        return max(int(mps.max_bond()) for mps in self.mps_b.values())

    @chi.setter
    def chi(self, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError("chi must be an integer")
        value = int(value)
        if value < 1:
            raise ValueError("chi must be >= 1")
        self._chi_target = value

    @property
    def norm(self):
        """Return the mean of ``mps.norm()`` across all stored boundaries."""
        if not self.mps_b:
            raise ValueError("No boundary MPS available to compute norm.")

        norms = [mps.norm() for mps in self.mps_b.values()]
        total = norms[0]
        for value in norms[1:]:
            total = total + value
        return total / len(norms)

    def expand_bnd(self, chi, rand_strength=0.0, inplace=True):
        """Retune all stored boundary MPS bond dimensions to ``chi``.

        Behavior per boundary MPS:

        - if ``chi`` is larger than current ``max_bond()``, expand bonds
          with ``expand_bond_dimension(..., rand_strength=rand_strength)`` and
          re-canonicalize using ``left_canonicalize_(normalize=False)``
          without truncation.
        - if ``chi`` is smaller than current ``max_bond()``, compress with
          ``compress("left", max_bond=chi)``.
        - if equal, leave the boundary unchanged.

        Parameters
        ----------
        chi : int
            Target bond dimension.
        rand_strength : float, default=0.0
            Randomization strength used during bond expansion.
        inplace : bool, default=True
            If ``True``, update existing boundary MPS objects in place.
            If ``False``, retune copies and replace ``self.mps_b``.

        Returns
        -------
        BdyMPS
            ``self`` for chaining.
        """
        if not isinstance(chi, (int, np.integer)):
            raise TypeError("chi must be an integer")
        chi = int(chi)
        if chi < 1:
            raise ValueError("chi must be >= 1")
        if not self.mps_b:
            raise ValueError("No boundary MPS available to retune.")

        target = self.mps_b if inplace else {key: mps.copy() for key, mps in self.mps_b.items()}
        for key, mps in target.items():
            current_bond = int(mps.max_bond())
            if chi > current_bond:
                maybe_new = mps.expand_bond_dimension(
                    chi,
                    rand_strength=rand_strength,
                    inplace=True,
                )
                tuned = mps if maybe_new is None else maybe_new
                maybe_can = tuned.left_canonicalize_(normalize=False)
                tuned = tuned if maybe_can is None else maybe_can
                target[key] = tuned
            elif chi < current_bond:
                maybe_comp = mps.compress("left", max_bond=chi)
                tuned = mps if maybe_comp is None else maybe_comp
                target[key] = tuned

        if not inplace:
            self.mps_b = target
        self._chi_target = chi
        return self

    def normalize(self):
        """Normalize all stored boundary MPS in place.

        Returns
        -------
        BdyMPS
            ``self`` for chaining.
        """
        if not self.mps_b:
            raise ValueError("No boundary MPS available to normalize.")

        for mps in self.mps_b.values():
            mps.normalize()

        return self

    @staticmethod
    def _normalize_boundary_direction(direction):
        """Normalize direction aliases to boundary key prefixes."""
        if not isinstance(direction, str):
            raise TypeError("direction must be a string")

        mapping = {
            "y": "Y",
            "row": "Y",
            "rows": "Y",
            "x": "X",
            "col": "X",
            "cols": "X",
            "column": "X",
            "columns": "X",
        }
        key = direction.lower()
        if key in mapping:
            return mapping[key]

        raise ValueError("direction must be one of: 'y', 'x'")

    @staticmethod
    def _normalize_boundary_side(side):
        """Normalize side aliases to ``l`` / ``r`` suffixes."""
        if not isinstance(side, str):
            raise TypeError("side must be a string")

        key = side.lower()
        if key in {"left", "l"}:
            return "l"
        if key in {"right", "r"}:
            return "r"

        raise ValueError("side must be one of: 'left', 'right', 'l', 'r'")

    def available_boundary_keys(self, direction=None, side=None):
        """Return sorted boundary keys, optionally filtered by direction/side."""
        keys = sorted(self.mps_b)
        if direction is not None:
            prefix = self._normalize_boundary_direction(direction)
            keys = [entry for entry in keys if entry.startswith(prefix)]
        if side is not None:
            suffix = self._normalize_boundary_side(side)
            keys = [entry for entry in keys if entry.endswith(f"_{suffix}")]
        return keys

    def boundary_key(self, direction="y", step=0, side="left"):
        """Build and validate a boundary key from direction, step, and side."""
        if not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        step = int(step)
        if step < 0:
            raise ValueError("step must be >= 0")

        prefix = self._normalize_boundary_direction(direction)
        suffix = self._normalize_boundary_side(side)
        key = f"{prefix}{step}_{suffix}"
        if key not in self.mps_b:
            candidates = self.available_boundary_keys(direction=direction, side=side)
            available = ", ".join(candidates) if candidates else ", ".join(sorted(self.mps_b))
            raise KeyError(
                f"Unknown boundary key '{key}'. Available keys: {available}"
            )
        return key

    @staticmethod
    def _boundary_key_metadata(key):
        """Parse and return metadata for a boundary key."""
        match = re.match(r"^(Y|X)(\d+)_(l|r)$", key)
        if not match:
            return {
                "key": key,
                "prefix": None,
                "suffix": None,
                "direction": "unknown",
                "step": None,
                "side": "unknown",
            }

        prefix, step_txt, suffix = match.groups()
        direction = {"Y": "y", "X": "x"}[prefix]
        side = "left" if suffix == "l" else "right"
        return {
            "key": key,
            "prefix": prefix,
            "suffix": suffix,
            "direction": direction,
            "step": int(step_txt),
            "side": side,
        }

    def _boundary_cut_coordinate(self, key):
        """Return the concrete cut-tag index selected by a boundary key."""
        meta = self._boundary_key_metadata(key)
        if meta["prefix"] is None:
            raise ValueError(f"Invalid boundary key format: {key}")

        axis_length = {
            "Y": self.ly,
            "X": self.lx,
        }[meta["prefix"]]
        cut_index = (
            meta["step"] if meta["suffix"] == "l" else (axis_length - 1 - meta["step"])
        )
        return meta, cut_index

    @staticmethod
    def _is_cut_node(prefix, cut_index, x_pos, y_pos):
        """Return whether a grid node belongs to the highlighted cut."""
        if prefix == "Y":
            return y_pos == cut_index
        if prefix == "X":
            return x_pos == cut_index
        return False

    @staticmethod
    def _row_edge_symbol(prefix, suffix, cut_index, y_pos):
        """Return row edge symbol for a given cut and y index."""
        if prefix == "Y" and y_pos == cut_index:
            return ">>" if suffix == "l" else "<<"
        return "--"

    @staticmethod
    def _column_connector_symbol(prefix, suffix, cut_index, x_pos):
        """Return vertical connector symbol for a given cut and x index."""
        if prefix == "X" and x_pos == cut_index:
            return "v" if suffix == "l" else "^"
        return "|"

    def _build_grid_row(self, meta, cut_index, lx, y_pos):
        """Build a single PEPS row with highlighted cut markers."""
        prefix = meta["prefix"]
        suffix = meta["suffix"]
        row_width = 3 * lx - 2
        row_chars = [" "] * row_width
        for x_pos in range(lx):
            col = 3 * x_pos
            node = "●" if self._is_cut_node(prefix, cut_index, x_pos, y_pos) else "o"
            row_chars[col] = node

        edge = self._row_edge_symbol(prefix, suffix, cut_index, y_pos)
        for x_pos in range(lx - 1):
            col = 3 * x_pos + 1
            row_chars[col : col + 2] = list(edge)
        return "".join(row_chars)

    def _build_grid_connectors(self, meta, cut_index, lx, y_pos):
        """Build connector row between adjacent PEPS rows."""
        prefix = meta["prefix"]
        suffix = meta["suffix"]
        row_width = 3 * lx - 2
        conn_chars = [" "] * row_width
        for x_pos in range(lx):
            col = 3 * x_pos
            conn_chars[col] = self._column_connector_symbol(
                prefix,
                suffix,
                cut_index,
                x_pos,
            )
        return "".join(conn_chars)

    def _render_peps_grid_lines(self, key):
        """Render a 2D PEPS grid and overlay the cut selected by ``key``."""
        meta, cut_index = self._boundary_cut_coordinate(key)
        lx, ly = self.lx, self.ly

        lines = [
            (
                f"grid cut={meta['prefix']}{cut_index} "
                f"(direction={meta['direction']}, side={meta['side']})"
            )
        ]

        for y in range(ly - 1, -1, -1):
            row_line = self._build_grid_row(
                meta,
                cut_index,
                lx,
                y,
            )
            lines.append(f"Y{y:<2} {row_line}")

            if y > 0:
                conn_line = self._build_grid_connectors(
                    meta,
                    cut_index,
                    lx,
                    y,
                )
                lines.append("    " + conn_line)

        lines.append("    " + " ".join(f"X{x}" for x in range(lx)))
        return lines

    @staticmethod
    def _render_mps_structure_lines(mps):
        """Render an MPS structure as three text lines."""
        line_top = ""
        line_mid = ""
        line_bot = ""

        num_can_l, num_can_r = mps.count_canonized()
        seg_len = 1

        for idx in range(mps.L - 1):
            bond_dim = int(mps.bond_size(idx, idx + 1))
            seg_len = len(str(bond_dim))
            line_top += f"│{bond_dim}"
            line_mid += (
                ">"
                if idx < num_can_l
                else "<"
                if idx >= mps.L - num_can_r
                else "●"
            ) + ("─" if bond_dim < 100 else "━") * seg_len
            line_bot += "│" + " " * seg_len

        line_top += "│"
        line_mid += "<" if num_can_r > 0 else "●"
        line_bot += "│"

        if mps.cyclic and mps.L > 1:
            edge_dim = int(mps.bond_size(0, mps.L - 1))
            edge_len = len(str(edge_dim))
            edge_str = ("─" if edge_dim < 100 else "━") * edge_len
            line_top = f" {edge_dim}{line_top}{edge_dim} "
            line_mid = f"+{edge_str}{line_mid}{edge_str}+"
            line_bot = f" {' ' * edge_len}{line_bot}{' ' * edge_len} "

        return line_top, line_mid, line_bot

    @staticmethod
    def _normalize_boundary_key_alias(key):
        """Normalize user key aliases to canonical keys used in ``self.mps_b``."""
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        match = re.match(r"^(x|y)(\d+)_([lr])$", key.strip(), flags=re.IGNORECASE)
        if not match:
            return key
        prefix, step, side = match.groups()
        canon_prefix = "X" if prefix.lower() == "x" else "Y"
        return f"{canon_prefix}{step}_{side.lower()}"

    @staticmethod
    def _format_structure_lines(lines, max_width=None):
        """Return structure lines optionally chunked to ``max_width``."""
        if max_width is None:
            return list(lines)

        if not isinstance(max_width, int) or max_width <= 0:
            raise ValueError("max_width must be a positive integer or None")

        width = max(len(line) for line in lines)
        padded = [line.ljust(width) for line in lines]
        wrapped = []

        for start in range(0, width, max_width):
            end = start + max_width
            for line in padded:
                wrapped.append(line[start:end].rstrip())
            if end < width:
                wrapped.append("")
        return wrapped

    @staticmethod
    def _resolve_show_color(color):
        """Resolve color mode for ``show`` output."""
        if color == "auto":
            return bool(getattr(sys.stdout, "isatty", lambda: False)())
        if isinstance(color, bool):
            return color
        raise TypeError("color must be bool or 'auto'")

    @staticmethod
    def _ansi_wrap(text, code, enabled):
        """Wrap text with ANSI style code when enabled."""
        if not enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    @classmethod
    def _colorize_symbols(cls, line, enabled, *, grid_arrows=False):
        """Apply ANSI colors to key structural symbols."""
        if not enabled:
            return line

        replacements = [
            ("◆", "31"),  # highlighted cut node
            ("●", "31"),  # highlighted cut node (plain style)
            ("▶", "31"),
            ("◀", "31"),
            ("▼", "1;35"),
            ("▲", "1;35"),
            ("○", "36"),
            ("o", "36"),
        ]
        if grid_arrows:
            replacements.extend(((">", "31"), ("<", "31")))
        else:
            replacements.extend(((">", "1;32"), ("<", "1;34")))

        out = line
        for symbol, code in replacements:
            out = out.replace(symbol, cls._ansi_wrap(symbol, code, enabled))
        return out

    @classmethod
    def _style_show_line(cls, line, *, color_enabled, fancy):
        """Render one show line with optional fancy symbols and ANSI colors."""
        is_grid_title = line.startswith("grid cut=")
        is_grid_row = re.match(r"^Y\d+\s+", line) is not None
        is_x_axis = re.match(r"^\s+X\d+", line) is not None
        stripped = line.strip()
        is_conn_row = line.startswith("    ") and bool(stripped) and set(stripped) <= {
            "|",
            "v",
            "^",
            " ",
        }

        out = line
        if fancy and (is_grid_row or is_conn_row):
            out = out.replace("--", "──").replace(">>", "> ").replace("<<", " <")
            out = out.replace("|", "│").replace("v", "▼").replace("^", "▲")
            out = out.replace("o", "○")

        if color_enabled:
            if is_grid_title:
                return cls._ansi_wrap(out, "1;36", True)
            if is_grid_row:
                match = re.match(r"^(Y\d+\s+)(.*)$", out)
                if match:
                    prefix, body = match.groups()
                    return cls._ansi_wrap(prefix, "1;33", True) + cls._colorize_symbols(
                        body,
                        True,
                        grid_arrows=True,
                    )
            if is_x_axis:
                return cls._ansi_wrap(out, "1;33", True)
            return cls._colorize_symbols(out, True)

        return out

    @classmethod
    def _style_show_lines(cls, lines, *, color=True, fancy=True):
        """Apply styling to ``show`` output lines."""
        color_enabled = cls._resolve_show_color(color)
        return [
            cls._style_show_line(line, color_enabled=color_enabled, fancy=fancy) for line in lines
        ]

    def _resolve_selected_key(self, key, direction, side, step):
        """Resolve and validate selected boundary key from key or direction."""
        if key is not None and direction is not None:
            raise ValueError("Provide either 'key' or 'direction', not both.")

        if key is None:
            selected_key = self.boundary_key(
                direction="y" if direction is None else direction,
                step=step,
                side=side,
            )
        else:
            selected_key = self._normalize_boundary_key_alias(key)

        if selected_key not in self.mps_b:
            available = ", ".join(sorted(self.mps_b))
            raise KeyError(
                f"Unknown boundary key '{selected_key}'. Available keys: {available}"
            )
        return selected_key

    def _compose_show_lines(self, selected_key, max_width, show_key, show_grid):
        """Compose output lines for boundary visualization."""
        output_lines = []
        if show_key:
            meta = self._boundary_key_metadata(selected_key)
            _, cut_index = self._boundary_cut_coordinate(selected_key)
            output_lines.append(
                f"direction={meta['direction']} "
                f"side={meta['side']} "
                f"step={meta['step']} "
                f"cut={meta['prefix']}{cut_index}"
            )
        if show_grid:
            output_lines.extend(
                self._format_structure_lines(
                    self._render_peps_grid_lines(selected_key),
                    max_width=max_width,
                )
            )
            output_lines.append("")

        mps = self.mps_b[selected_key]
        mps_lines = self._render_mps_structure_lines(mps)
        output_lines.extend(self._format_structure_lines(mps_lines, max_width=max_width))
        return output_lines

    def show(
        self,
        key=None,
        *,
        direction=None,
        side="left",
        step=0,
        max_width=None,
        show_key=False,
        show_grid=True,
        color=True,
        fancy=True,
    ):  # pylint: disable=too-many-arguments
        """Display boundary view text (grid and MPS structure).

        Parameters
        ----------
        key : str | None, default=None
            Explicit boundary key such as ``Y0_l`` or ``X2_r``.
        direction : str | None, default=None
            Boundary direction alias (``y``, ``x``). Used when
            ``key`` is not provided.
        side : str, default="left"
            Side alias (``left``/``l`` or ``right``/``r``).
        step : int, default=0
            Sweep step index used with ``direction`` and ``side``.
        max_width : int | None, default=None
            Maximum text width per printed chunk.
        show_key : bool, default=False
            If ``True``, include selected key metadata line.
        show_grid : bool, default=True
            If ``True``, include 2D PEPS grid lines with the selected cut.
        color : bool | {"auto"}, default=True
            If enabled, apply ANSI colors to highlighted symbols and labels.
        fancy : bool, default=True
            If enabled, use a fancier unicode symbol set for grid display.

        Returns
        -------
        None
            This method prints a text view and does not return an MPS.
        """
        selected_key = self._resolve_selected_key(key, direction, side, step)

        output_lines = self._compose_show_lines(
            selected_key,
            max_width,
            show_key,
            show_grid,
        )
        styled_lines = self._style_show_lines(output_lines, color=color, fancy=fancy)
        for line in styled_lines:
            print(line)
        return None

    def show_all(self, direction=None, side=None, max_width=None, color=True, fancy=True):
        """Display all matching boundary keys."""
        keys = self.available_boundary_keys(direction=direction, side=side)
        for idx, key_name in enumerate(keys):
            self.show(
                key=key_name,
                max_width=max_width,
                show_key=False,
                show_grid=True,
                color=color,
                fancy=fancy,
            )
            if idx < len(keys) - 1:
                print()
        return keys


    @staticmethod
    def _max_tag_number(tags, prefix):
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        nums = [int(match.group(1)) for tag in tags if (match := pattern.match(tag))]
        return max(nums) if nums else None

    def _infer_lattice_shape(self, tn_ref, tn_fallback):
        """Infer ``(lx, ly)`` from explicit attributes or X/Y tags."""
        for tn in (tn_ref, tn_fallback):
            if tn is None:
                continue
            lx = getattr(tn, "Lx", None)
            ly = getattr(tn, "Ly", None)
            if lx is not None and ly is not None:
                return int(lx), int(ly)

        if tn_ref is not None and (
            getattr(tn_ref, "Lx", None) is None
            or getattr(tn_ref, "Ly", None) is None
        ):
            logger.warning(
                "Lx/Ly not found on reference TN; inferring lattice shape "
                "from X*/Y* tags."
            )

        tags = getattr(tn_fallback, "tags", ())
        max_x = self._max_tag_number(tags, "X")
        max_y = self._max_tag_number(tags, "Y")
        if max_x is not None and max_y is not None:
            return max_x + 1, max_y + 1

        raise ValueError(
            "Could not infer lattice shape. Provide a network with Lx/Ly "
            "or X*/Y* tags."
        )

    def _get_axis_length_for_site_tag(self, site_tag_id):
        if site_tag_id == "X{}":
            return self.ly
        if site_tag_id == "Y{}":
            return self.lx
        raise ValueError(f"Unsupported site_tag_id: {site_tag_id}")

    def _get_default_site_count(self, site_tag_id):
        if site_tag_id == "X{}":
            return self.lx
        if site_tag_id == "Y{}":
            return self.ly
        raise ValueError(f"Unsupported site_tag_id: {site_tag_id}")

    @staticmethod
    def _sort_site_tags_by_index(tags, site_tag_id):
        pattern = re.compile(
            "^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$"
        )
        matches = [tag for tag in tags if pattern.match(tag)]
        return sorted(matches, key=lambda tag: int(pattern.match(tag).group(1)))

    def _prepare_boundary_mps(self, mps):
        mps.apply_to_arrays(self.numpy_backend)
        mps.randomize(seed=self.seed, inplace=True)
        mps.apply_to_arrays(self.to_backend)
        mps.compress("left", max_bond=self._chi_target)
        mps.normalize()
        return mps

    @staticmethod
    def _view_as_mps_from_outer_inds(network, site_tag_id):
        """View a tensor network as an MPS using current outer indices."""
        network.view_as_(
            qtn.MatrixProductState,
            L=len(network.outer_inds()),
            site_tag_id=site_tag_id,
            site_ind_id=None,
            cyclic=False,
        )

    def _flat_first_step_boundary(self, network, site_tag_id):
        """Build first-step boundary directly from a flat slice."""
        self._view_as_mps_from_outer_inds(network, site_tag_id)
        network.apply_to_arrays(self.to_backend)
        return network

    @staticmethod
    def _previous_boundary_key(cut_tag_id, step, suffix):
        """Return previous-step boundary key."""
        return f"{cut_tag_id.format(step - 1)}_{suffix}"

    def _build_single_layer_boundary_mps(
        self,
        network,
        site_tag_id,
    ):  # pylint: disable=too-many-locals
        outer_inds = list(network.outer_inds())
        if not outer_inds:
            raise ValueError("Cannot build single-layer boundary MPS with no outer indices")

        site_tags = self._sort_site_tags_by_index(network.tags, site_tag_id)
        ordered_inds = []
        for tag in site_tags:
            selected = network.select(tag)
            local_outer = [idx for idx in selected.outer_inds() if idx in outer_inds]
            ordered_inds.extend(local_outer)

        # Keep grouped ordering when available, then include any remaining open indices.
        seen = set()
        ordered_unique = []
        for idx in ordered_inds:
            if idx not in seen:
                ordered_unique.append(idx)
                seen.add(idx)
        for idx in outer_inds:
            if idx not in seen:
                ordered_unique.append(idx)
                seen.add(idx)

        mps_length = len(ordered_unique)
        ind_sizes = {idx: network.ind_size(idx) for idx in ordered_unique}
        reindex_map = {f"k{pos}": idx for pos, idx in enumerate(ordered_unique)}

        tensors = [qtn.Tensor() for _ in range(mps_length)]
        for pos in range(mps_length):
            tensors[pos].add_tag(site_tag_id.format(pos))
            tensors[pos].new_ind(
                f"k{pos}",
                size=ind_sizes[reindex_map[f"k{pos}"]],
            )
            if pos < (mps_length - 1):
                tensors[pos].new_bond(tensors[pos + 1], size=self._chi_target)

        mps = qtn.TensorNetwork(tensors)
        mps.reindex_(reindex_map)
        mps.view_as_(
            qtn.MatrixProductState,
            L=mps_length,
            site_tag_id=site_tag_id,
            site_ind_id=None,
            cyclic=False,
        )
        return self._prepare_boundary_mps(mps)

    def _build_multi_layer_boundary_mps(self, network, site_tag_id, mps_length):
        grouped_inds = {pos: [] for pos in range(mps_length)}
        for pos in range(mps_length):
            tag = site_tag_id.format(pos)
            selected = network.select(tag, "any")
            grouped_inds[pos].extend(
                [
                    (idx, network.ind_size(idx))
                    for idx in selected.outer_inds()
                    if idx in network.outer_inds()
                ]
            )

        tensors = [qtn.Tensor() for _ in range(mps_length)]
        for pos in range(mps_length):
            for idx, size in grouped_inds[pos]:
                tensors[pos].new_ind(idx, size=size)
            tensors[pos].add_tag(site_tag_id.format(pos))
            if pos < (mps_length - 1):
                tensors[pos].new_bond(tensors[pos + 1], size=self._chi_target)

        mps = qtn.TensorNetwork(tensors)
        mps.view_as_(
            qtn.MatrixProductState,
            L=mps_length,
            site_tag_id=site_tag_id,
            site_ind_id=None,
            cyclic=False,
        )
        return self._prepare_boundary_mps(mps)

    def _initialize_single_layer_boundaries(
        self,
        side,
        site_tag_id="X{}",
        cut_tag_id="Y{}",
    ):
        length = self._get_axis_length_for_site_tag(site_tag_id)
        suffix = "l" if side == "left" else "r"
        boundaries = {}

        for step in range(length - 1):
            cut_pos = step if side == "left" else (length - 1 - step)
            key = f"{cut_tag_id.format(step)}_{suffix}"
            tn = self._tn_norm.select(cut_tag_id.format(cut_pos), "any").copy()

            if self.flat and step == 0:
                boundaries[key] = self._flat_first_step_boundary(tn, site_tag_id)
                continue

            if step == 0:
                mps_net = tn
            else:
                prev = boundaries[self._previous_boundary_key(cut_tag_id, step, suffix)]
                mps_net = tn | prev

            boundaries[key] = self._build_single_layer_boundary_mps(
                mps_net,
                site_tag_id,
            )

        return boundaries

    def _initialize_multi_layer_boundaries(
        self,
        side,
        site_tag_id="X{}",
        cut_tag_id="Y{}",
    ):
        length = self._get_axis_length_for_site_tag(site_tag_id)
        suffix = "l" if side == "left" else "r"
        default_count = self._get_default_site_count(site_tag_id)
        boundaries = {}

        for step in range(length - 1):
            cut_pos = step if side == "left" else (length - 1 - step)
            key = f"{cut_tag_id.format(step)}_{suffix}"
            tn = self._tn_norm.select(cut_tag_id.format(cut_pos), "any").copy()

            if self.flat and step == 0:
                boundaries[key] = self._flat_first_step_boundary(tn, site_tag_id)
                continue

            if step == 0:
                mps_net = tn
            else:
                prev = boundaries[self._previous_boundary_key(cut_tag_id, step, suffix)]
                mps_net = tn | prev

            boundaries[key] = self._build_multi_layer_boundary_mps(
                mps_net,
                site_tag_id,
                default_count,
            )

        return boundaries
