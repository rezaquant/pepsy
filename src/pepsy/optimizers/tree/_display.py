"""Small, dependency-free ASCII renderers for rooted tree operators."""

from __future__ import annotations

from numbers import Integral


def ascii_tree(
    plan,
    edge_dim,
    *,
    bond_dims=True,
    node_ids=False,
    color=False,
    marker="●",
    leaf_marker="◆",
    label_site=None,
):
    """Render a rooted plan with Quimb-style edge dimension annotations."""
    if label_site is None:
        label_site = lambda site: f"q{site}"

    gap = 3
    children_map = plan.children

    def render(node):
        is_leaf = not children_map.get(node, ())
        if hasattr(plan, "qubit_of_node"):
            site = plan.qubit_of_node.get(node)
        else:
            site = node
        if is_leaf:
            label = leaf_marker
            if site is not None:
                label += f" {label_site(site)}"
            width = max(1, len(label))
            return [label.center(width)], (width - 1) // 2, width

        label = f"{marker}{node}" if node_ids else marker
        if site is not None:
            label += f" {label_site(site)}"
        blocks = []
        for child in children_map[node]:
            lines, column, width = render(child)
            if bond_dims:
                dimension = str(edge_dim(node, child))
                if len(dimension) > width:
                    left = (len(dimension) - width) // 2
                    right = len(dimension) - width - left
                    lines = [" " * left + line + " " * right for line in lines]
                    column += left
                    width = len(dimension)
                lines = [dimension.center(width)] + lines
            blocks.append((lines, column, width))

        offsets = []
        cursor = 0
        for _lines, _column, width in blocks:
            offsets.append(cursor)
            cursor += width + gap
        total_width = cursor - gap
        child_columns = [
            offset + column
            for offset, (_lines, column, _width) in zip(offsets, blocks)
        ]
        parent_column = (child_columns[0] + child_columns[-1]) // 2

        connector = [" "] * total_width
        for index in range(child_columns[0], child_columns[-1] + 1):
            connector[index] = "─"
        for index, column in enumerate(child_columns):
            connector[column] = (
                "┌" if index == 0
                else "┐" if index == len(child_columns) - 1
                else "┬"
            )
        connector[parent_column] = "┴"

        height = max(len(lines) for lines, _column, _width in blocks)
        body = []
        for row in range(height):
            parts = []
            for index, (lines, _column, width) in enumerate(blocks):
                parts.append(lines[row] if row < len(lines) else " " * width)
                if index != len(blocks) - 1:
                    parts.append(" " * gap)
            body.append("".join(parts))

        return [label.center(total_width, " "), "".join(connector)] + body, parent_column, total_width

    drawing = "\n".join(line.rstrip() for line in render(plan.root)[0])
    if not color:
        return drawing

    # Keep the layout calculation independent of terminal escape sequences.
    # This mirrors the useful part of Quimb's coloured ``show`` output while
    # preserving copy/paste-friendly plain text by default.
    marker_escape = "\033[1;36m{}\033[0m"
    leaf_escape = "\033[1;33m{}\033[0m"
    return drawing.replace(marker, marker_escape.format(marker)).replace(
        leaf_marker, leaf_escape.format(leaf_marker),
    )


def ascii_lattice(
    plan,
    shape,
    site_coords,
    *,
    terms=None,
    node_ids=False,
):
    """Render a physical 2D/3D site layout and its operator supports.

    A :class:`TreePlan` contains structural internal nodes, so its native
    bonds cannot all be drawn as edges between physical lattice sites. This
    view therefore shows the physical coordinate array and lists the term
    supports, while :func:`ascii_tree` below remains the authoritative view
    of the virtual TreeMPO topology.
    """
    shape = tuple(int(size) for size in shape)
    coordinates = {int(site): tuple(point) for site, point in site_coords.items()}
    expected = set(plan.node_of_qubit)
    if set(coordinates) != expected:
        raise ValueError(
            "layout finder coordinates must cover every TreeMPO physical site."
        )

    normalized_terms = []
    if terms is not None:
        for support in terms:
            if isinstance(support, Integral):
                support = (int(support),)
            else:
                support = tuple(int(site) for site in support)
            normalized_terms.append(support)

    term_ids = {site: [] for site in expected}
    for index, support in enumerate(normalized_terms):
        for site in support:
            if site in term_ids:
                term_ids[site].append(index)

    def site_label(site):
        label = f"q{site}"
        if node_ids:
            label += f"/N{plan.node_of_qubit[site]}"
        if term_ids[site]:
            label += "[" + ",".join(f"t{index}" for index in term_ids[site]) + "]"
        return label

    lines = [
        f"physical lattice {shape}"
        + (f", map_mode={plan.map_mode!r}" if plan.map_mode else "")
    ]
    if len(shape) == 2:
        x_values = tuple(range(shape[0]))
        y_values = tuple(range(shape[1]))
        label_width = max(
            len(site_label(site))
            for site in expected
        )
        for x in reversed(x_values):
            row = []
            for y in y_values:
                site = next(
                    site for site, point in coordinates.items()
                    if tuple(point) == (x, y)
                )
                row.append(site_label(site).center(label_width))
            lines.append("──".join(row))
            if x:
                lines.append("  ".join("│".center(label_width) for _ in y_values))
    else:
        lines.append("coordinates:")
        for site in sorted(expected, key=lambda site: coordinates[site]):
            lines.append(f"  {site_label(site)} @ {coordinates[site]}")

    if normalized_terms:
        lines.append("terms:")
        for index, support in enumerate(normalized_terms):
            lines.append(
                f"  t{index}: "
                + " - ".join(f"q{site}" for site in support)
            )
    else:
        lines.append("terms: none recorded")
    return "\n".join(lines)
