"""Compression-hook compatibility must survive cached signature inspection."""

import inspect

from pepsy.optimizers.tree import TreeOptimizer


def test_compression_hook_cache_refreshes_for_replaced_overrides(monkeypatch):
    optimizer = TreeOptimizer(None, n=3, run=False)
    inspections = []
    signature = inspect.signature

    def counted(method):
        inspections.append(method)
        return signature(method)

    monkeypatch.setattr(inspect, "signature", counted)
    calls = []

    def legacy(u, v, *, max_bond, cutoff, reduced):
        calls.append((u, v, max_bond, cutoff, reduced))

    monkeypatch.setattr(optimizer, "_compress_edge_with_diagnostics", legacy)
    optimizer._compress_edge_compat(0, 1, max_bond=3, cutoff=0., reduction_proven=True)
    optimizer._compress_edge_compat(1, 2, max_bond=3, cutoff=0., reduction_proven=True)
    assert len(inspections) == 1
    assert calls == [(0, 1, 3, 0., True), (1, 2, 3, 0., True)]

    def modern(u, v, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(optimizer, "_compress_edge_with_diagnostics", modern)
    optimizer._compress_edge_compat(0, 1, reduction_proven=True)
    assert len(inspections) == 2
    assert calls[-1]["reduction_proven"] is True
    assert calls[-1]["compression_mode"] == optimizer.compression_mode
    assert calls[-1]["compression_seed"] == optimizer.compression_seed
