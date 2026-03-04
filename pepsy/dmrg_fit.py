"""DMRG local fitting utilities for MPS/MPO tensor networks.

This module provides local and environment-based sweep routines used by
boundary contraction code. The focus is to keep tensor-index handling explicit
and fail early when input structure is inconsistent.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import autoray as ar
import quimb.tensor as qtn

from .dmrg_helpers import fidel_mps, opt_

logger = logging.getLogger(__name__)

__all__ = [
    "FIT",
    "opt_",
    "fidel_mps",
    "energy_global",
    "gate_1d",
    "internal_inds",
    "canonize_mps",
]


def energy_global(MPO_origin, mps_a, opt="auto-hq"):  # pylint: disable=invalid-name
    """Compute global energy ``<mps_a|MPO_origin|mps_a>`` with normalization."""

    mps_a_ = mps_a.copy()
    mps_a_.normalize()
    p_h = mps_a_.H
    p_h.reindex_({f"k{i}": f"b{i}" for i in range(mps_a.L)})
    mpo_t = MPO_origin * 1.0

    energy_dmrg = (p_h | mpo_t | mps_a_).contract(all, optimize=opt)
    return energy_dmrg


def gate_1d(
    tn,
    where,
    G,  # pylint: disable=invalid-name
    ind_id="k{}",
    site_tags="I{}",
    cutoff=1.0e-12,
    contract="split-gate",
    inplace=False,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments

    """
    Apply a 1D gate to a tensor network at one or two sites.

    Args:
        tn:      Tensor network (quimb/qtn TensorNetwork).
        where:   Iterable of site indices; length 1 (single-qubit) or 2 (two-qubit).
        G:       Gate tensor (or matrix).
        ind_id: Format string for site indices (e.g., "k{}" -> "k3").
        site_tags: Format string for site tags   (e.g., "I{}" -> "I3").
        cutoff:  SVD cutoff (used for split contraction paths).
        contract: Contraction mode (e.g., "split-gate") or bool for single-qubit.
        inplace: Modify tn in place if True; otherwise return a new TN.

    Returns:
        TensorNetwork with the gate applied and site tags added.
    """

    if len(where) == 2:
        x, y = where
        tn = qtn.tensor_network_gate_inds(
            tn,
            G,
            [ind_id.format(x), ind_id.format(y)],
            contract=contract,
            inplace=inplace,
            **{"cutoff": cutoff},
        )
        # Add site tags on updated tensors.
        t = [tn.tensor_map[i] for i in tn.ind_map[ind_id.format(x)]][0]
        t.add_tag(site_tags.format(x))
        t = [tn.tensor_map[i] for i in tn.ind_map[ind_id.format(y)]][0]
        t.add_tag(site_tags.format(y))
    elif len(where) == 1:
        x, = where
        tn = qtn.tensor_network_gate_inds(
            tn,
            G,
            [ind_id.format(x)],
            contract=True,
            inplace=inplace,
        )
    else:
        raise ValueError("where must contain one or two site indices")

    return tn


def internal_inds(psi):
    """Return all internal (non-open) indices of ``psi``."""
    open_inds = psi.outer_inds()
    inner_inds = []
    for t in psi:
        for ind in t.inds:
            if ind not in open_inds:
                inner_inds.append(ind)
    return inner_inds


def canonize_mps(p, where, cur_orthog):
    """Canonize MPS on interval ``where`` and update ``cur_orthog`` in-place."""
    xmin, xmax = sorted(where)
    p.canonize([xmin, xmax], cur_orthog=cur_orthog)
    # update cur_orthog in place (preserving reference)
    cur_orthog[:] = [xmin, xmax]


class FIT:  # pylint: disable=too-many-instance-attributes
    """Local tensor fitting of an MPS/MPO against a target tensor network.

    Parameters
    ----------
    tn : TensorNetwork
        Target tensor network to fit.
    p : TensorNetwork
        Initial MPS/MPO state. Must support ``copy`` and canonization methods.
    cutoffs : float, optional
        Numerical cutoff for truncation (default: 1e-9).
    backend : str or None, optional
        Backend specification for tensor operations.
    re_tag : bool, default=False
        If True, (re)tag the target TN for environment construction.
    """

    def __init__(
        self,
        tn: qtn.TensorNetwork,
        p: Optional[qtn.TensorNetwork] = None,
        cutoffs: float = 1.e-12,
        backend: Optional[str] = None,
        site_tag_id: str = "I{}",
        opt: str = "auto-hq",
        range_int: Optional[Sequence[int]] = None,
        re_tag: bool = False,
        info: Optional[Dict[str, Any]] = None,
        warning: bool = False,
        inplace=False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        if p is None:
            raise ValueError("Initial MPS `p` must be provided for FIT.")
        if not isinstance(p, (qtn.MatrixProductState, qtn.MatrixProductOperator)):
            raise TypeError("Initial MPS `p` must be MatrixProductState or MatrixProductOperator.")
        if not isinstance(site_tag_id, str) or "{}" not in site_tag_id:
            raise ValueError("site_tag_id must be a format string containing '{}'.")

        self.L = int(p.L)  # pylint: disable=invalid-name

        self.p = p if inplace else p.copy()

        self.tn = tn.copy()


        if site_tag_id:
            self.p.view_as_(
                qtn.MatrixProductState,
                L=self.L,
                site_tag_id=site_tag_id,
                site_ind_id=None,
                cyclic=False,
            )

        self.site_tag_id = site_tag_id

        # cotengra path finder
        self.opt = opt

        # cutoffs and underlying backend
        self.cutoffs = cutoffs
        self.backend = backend

        # warnings being printed or not
        self.warning = warning

        # store cost function results
        self.loss: List[float] = []
        self.loss_: List[float] = []
        self.info: Dict[str, Any] = info or {}
        self.range_int: List[int] = list(range_int) if range_int is not None else []
        if self.range_int:
            if len(self.range_int) != 2:
                raise ValueError("range_int must be a sequence of two integers: (start, stop).")
            start, stop = self.range_int
            if start >= stop:
                raise ValueError("range_int must satisfy start < stop.")


        # Is there a better solution?
        # Reindex tensor network with random UUIDs for internal indices
        self.tn.reindex_({idx: qtn.rand_uuid() for idx in self.tn.inner_inds()})

        if set(self.tn.outer_inds()) != set(self.p.outer_inds()):
            raise ValueError("tn and p have different outer indices.")

        # Re-tag TN for effective environments when requested.
        if re_tag:
            self._re_tag()


    def visual(
        self,
        figsize=(14, 14),
        layout="neato",
        show_tags=False,
        tags_: Optional[Sequence[str]] = None,
        show_inds=False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Visualize the combined target network and current fitted state."""
        tag_list = tags_ if tags_ is not None else []
        tags = [self.site_tag_id.format(i) for i in range(self.L)] + tag_list
        return (self.tn & self.p).draw(
            tags,
            legend=False,
            show_inds=show_inds,
            show_tags=show_tags,
            figsize=figsize,
            node_outline_darkness=0.1,
            node_outline_size=None,
            highlight_inds_color="darkred",
            edge_scale=2.0,
            layout=layout,
            refine_layout="auto",
            highlight_inds=self.p.outer_inds(),
        )

    # -------------------------
    # Tagging methods
    # -------------------------
    def _deep_tag(self):
        """
        Propagates tags through the tensor network to ensure every tensor
        receives at least one site tag. Useful for layered TNs.
        """
        tn = self.tn
        count = 1

        while count >= 1:
            tags = tn.tags
            count = 0
            for tag in tags:
                tids = tn.tag_map[tag]
                neighbors = qtn.oset()
                for tid in tids:
                    t = tn.tensor_map[tid]
                    for ix in t.inds:
                        neighbors |= tn.ind_map[ix]
                for tid in neighbors:
                    t = tn.tensor_map[tid]
                    if not t.tags:
                        t.add_tag(tag)
                        count += 1

    def _re_tag(self):
        """Assign site tags on target TN tensors based on current boundary state."""
        # Drop all existing tags first.
        tn = self.tn
        tn.drop_tags()

        # Get outer indices and all site tags from current state.
        p = self.p
        site_tags = [self.site_tag_id.format(i) for i in range(p.L)]
        inds = list(p.outer_inds())

        # First-layer tagging: pick tensor directly connected to each boundary index.
        for site_tag in site_tags:
            site_outer = [idx for idx in p[site_tag].inds if idx in inds]
            if not site_outer:
                continue
            idx = site_outer[0]

            tids = list(tn.ind_map.get(idx, ()))
            if not tids:
                continue
            t = tn.tensor_map[tids[0]]

            if not t.tags:
                t.add_tag(site_tag)

        untagged_tensors = [tensor for tensor in tn if not tensor.tags]
        if untagged_tensors:
            if self.warning:
                logger.warning(
                    "%d tensors are still untagged after initial retagging; "
                    "propagating tags through neighbors.",
                    len(untagged_tensors),
                )
            self._deep_tag()


    def run(self, n_iter=6, verbose=True):
        """Run basic left-to-right local fitting sweeps.

        Parameters
        ----------
        n_iter : int
            Number of complete sweeps.
        verbose : bool
            If ``True``, append per-sweep fidelity values to ``self.loss``.
        """
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        psi = self.p
        L = self.L  # pylint: disable=invalid-name
        opt = self.opt
        site_tag_id = self.site_tag_id

        for _ in range(n_iter):
            for site in range(L):
                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1

                # Canonicalize psi at the current site
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)

                psi_h = psi.H.select([site_tag_id.format(site)], "!any")
                tn_ = psi_h | self.tn

                # Contract and normalize
                f = tn_.contract(all, optimize=opt)
                f = f.transpose(*psi[site].inds)

                norm_f = (f.H & f).contract(all) ** 0.5
                self.loss_.append(complex(norm_f).real)

                # Update tensor data
                psi[site].modify(data=f.data)

            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))

    def _build_env_right(self, psi, env_right):
        """
        Build right environments env_right["I{i}"] for i in 0..L-1.
        env_right[i] corresponds to contraction of site i and everything to the right (inclusive).
        """
        L = self.L  # pylint: disable=invalid-name
        opt = self.opt
        site_tag_id = self.site_tag_id

        # iterate from rightmost to leftmost
        for i in reversed(range(L)):
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block

            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
            else:
                # tie to previously computed right environment
                t |= env_right[site_tag_id.format(i + 1)]
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)




    def _right_range(self, psi, env_right, start, stop):
        """
        Build right environments env_right["I{i}"] for i in 0..L-1.
        env_right[i] corresponds to contraction of site i and everything to the right (inclusive).
        """
        L = self.L  # pylint: disable=invalid-name
        opt = self.opt
        site_tag_id = self.site_tag_id

        indx = None
        indx_ = None
        # iterate from rightmost to leftmost
        for count, i in enumerate(range(stop, start, -1)):
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            # Is there any tensor in tn to be included in env
            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block

            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
            else:
                if count == 0:
                    indx = psi.bond(stop + 1, stop)
                    indx_ = self.tn.bond(stop + 1, stop)

                # tie to previously computed right environment
                if env_right[site_tag_id.format(i + 1)] is not None:
                    t |= env_right[site_tag_id.format(i + 1)]
                    env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
                else:
                    if indx is None or indx_ is None:
                        raise ValueError("Right-range boundary indices are not initialized.")
                    t = t.reindex({indx: indx_})
                    env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)

    def _left_range(self, psi, site, count, env_left):
        """Update left environment incrementally for current site."""

        # get tensor at site from p
        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        opt = self.opt
        site_tag_id = self.site_tag_id

        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block

        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
        else:
            if count == 1:
                indx = psi.bond(site - 1, site)
                indx_ = self.tn.bond(site - 1, site)
                t = t.copy()
                t = t.reindex({indx: indx_})
                env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
            else:
                t |= env_left[site_tag_id.format(site - 1)]
                env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)



    def _update_env_left(self, psi, site: int, env_left):
        """Update left environment incrementally for current site."""

        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        opt = self.opt
        site_tag_id = self.site_tag_id

        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block

        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
        else:
            t |= env_left[site_tag_id.format(site - 1)]
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)


    def run_eff(self, n_iter=6, verbose=True):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Run environment-based fitting sweeps with cached left/right blocks.

        This method avoids rebuilding full contractions at each site by
        incrementally reusing left and right environments.
        """
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L  # pylint: disable=invalid-name
        opt = self.opt

        if L == 1:
            if self.warning:
                logger.warning("run_eff called for L=1; falling back to run().")
            self.run(n_iter=n_iter, verbose=verbose)
            return

        env_left = {site_tag_id.format(i): None for i in range(psi.L)}
        env_right = {site_tag_id.format(i): None for i in range(psi.L)}

        for _ in range(n_iter):
            for site in range(L):
                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1
                # Canonicalize psi at the current site
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)

                if site == 0:
                    self._build_env_right(psi, env_right)
                else:
                    self._update_env_left(psi, site - 1, env_left)

                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None

                tn = None
                if site == 0:
                    if tn_site is not None:
                        tn = tn_site | env_right[site_tag_id.format(site + 1)]
                    else:
                        tn = env_right[site_tag_id.format(site + 1)]

                if 0 < site < L - 1:
                    if tn_site is not None:
                        tn = (
                            tn_site
                            | env_right[site_tag_id.format(site + 1)]
                            | env_left[site_tag_id.format(site - 1)]
                        )
                    else:
                        tn = (
                            env_right[site_tag_id.format(site + 1)]
                            | env_left[site_tag_id.format(site - 1)]
                        )

                if site == L - 1:
                    if tn_site is not None:
                        tn = tn_site | env_left[site_tag_id.format(site - 1)]
                    else:
                        tn = env_left[site_tag_id.format(site - 1)]

                if tn is None:
                    raise ValueError("Failed to build effective tensor for current site.")

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=opt).transpose(
                        *psi[site_tag_id.format(site)].inds
                    )
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)
                else:
                    raise TypeError("Unexpected effective tensor type during run_eff.")

                norm_f = (f.H & f).contract(all) ** 0.5
                self.loss_.append(complex(norm_f).real)

                # Contract and normalize
                # Update tensor data
                psi[site].modify(data=f.data)

            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))



    def run_gate(
        self, n_iter=6, verbose=True
    ):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Run fitting restricted to ``range_int`` with gate-style sweeps.

        The algorithm canonicalizes the active interval and updates tensors
        using effective environments built from neighboring boundaries.
        """
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L  # pylint: disable=invalid-name
        opt = self.opt

        if L == 1:
            if self.warning:
                logger.warning("run_gate called for L=1; falling back to run().")
            self.run(n_iter=n_iter, verbose=verbose)
            return

        if len(self.range_int) != 2:
            raise ValueError("range_int must be set to (start, stop) before calling run_gate.")
        start, stop = self.range_int
        if start < 0 or stop >= L or start > stop:
            raise ValueError(f"range_int={self.range_int} is out of bounds for L={L}.")
        if stop == start:
            raise ValueError("run_gate requires range_int spanning at least two sites.")

        env_left = {site_tag_id.format(i): None for i in range(psi.L)}
        env_right = {site_tag_id.format(i): None for i in range(psi.L)}

        for _ in range(n_iter):
            for i in range(stop, start, -1):
                psi.right_canonize_site(i, bra=None)

            for count_, site in enumerate(range(start, stop + 1)):
                if count_ == 0:
                    self._right_range(psi, env_right, start, stop)
                else:
                    self._left_range(psi, site - 1, count_, env_left)

                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None

                tn = None
                if site == 0:
                    if tn_site is not None:
                        tn = tn_site | env_right[site_tag_id.format(site + 1)]
                    else:
                        tn = env_right[site_tag_id.format(site + 1)]

                if 0 < site < L - 1:

                    # Boundary consistency: the left and right indices must match between tn and p
                    if count_ == 0:
                        indx = psi.bond(start - 1, start)
                        indx_ = self.tn.bond(start - 1, start)
                        if tn_site is not None:
                            tn_site = tn_site.reindex({indx_: indx})
                    if count_ == stop - start:
                        indx = psi.bond(stop + 1, stop)
                        indx_ = self.tn.bond(stop + 1, stop)
                        if tn_site is not None:
                            tn_site = tn_site.reindex({indx_: indx})

                    if tn_site is not None:
                        if (
                            env_right[site_tag_id.format(site + 1)] is not None
                            and env_left[site_tag_id.format(site - 1)] is not None
                        ):
                            tn = (
                                tn_site
                                | env_right[site_tag_id.format(site + 1)]
                                | env_left[site_tag_id.format(site - 1)]
                            )
                        elif env_left[site_tag_id.format(site - 1)] is not None:
                            tn = tn_site | env_left[site_tag_id.format(site - 1)]
                        elif env_right[site_tag_id.format(site + 1)] is not None:
                            tn = tn_site | env_right[site_tag_id.format(site + 1)]
                        else:
                            tn = tn_site
                    else:
                        tn = (
                            env_right[site_tag_id.format(site + 1)]
                            | env_left[site_tag_id.format(site - 1)]
                        )

                if site == L - 1:
                    if tn_site is not None:
                        tn = tn_site | env_left[site_tag_id.format(site - 1)]
                    else:
                        tn = env_left[site_tag_id.format(site - 1)]

                if tn is None:
                    raise ValueError("Failed to build effective tensor for gate sweep.")

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=opt).transpose(
                        *psi[site_tag_id.format(site)].inds
                    )
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)
                else:
                    raise TypeError("Unexpected effective tensor type during run_gate.")

                norm_f = (f.H & f).contract(all) ** 0.5

                # norm_f = ar.do("norm", f.data)

                self.loss_.append(complex(norm_f).real)

                # Contract and normalize
                # Update tensor data
                psi[site].modify(data=f.data)

                if site < stop:
                    psi.left_canonize_site(site, bra=None)


            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))
