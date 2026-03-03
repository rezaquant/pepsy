"""Boundary-based tensor-network norm evaluation.

This module provides a class-based API to build ``norm = p.conj() | p`` and
contract it using boundary-MPS sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from uuid import uuid4
import warnings

import numpy as np

from boundary_states import BdyMPO, add_diagonalu_tags
from boundary_sweeps import CompBdy
from dmrg_helpers import backend_numpy


_TAG_X = re.compile(r"^X\d+$")
_TAG_Y = re.compile(r"^Y\d+$")
_TAG_DU = re.compile(r"^Du\d+$")


@dataclass
class BoundaryNormRunner:  # pylint: disable=too-many-instance-attributes
    """Compute tensor-network norm by boundary sweeps.

    Parameters
    ----------
    to_backend : callable | None
        Array converter applied to ``norm`` tensors. If ``None``, defaults to
        ``backend_numpy(np.complex128)``.
    opt : str or optimizer
        Contraction optimizer passed to boundary routines.
    chi : int
        Boundary MPS bond dimension.
    seed : int
        Random seed used by boundary-state initialization.
    single_layer : bool
        Forwarded to ``BdyMPO``.
    flat : bool
        Forwarded to ``BdyMPO`` / ``CompBdy``.
    dmrg_run : str
        Boundary fitting mode (e.g. ``"eff"``, ``"global"``).
    n_iter : int
        Number of fitting iterations.
    re_tag : bool
        Forwarded to ``CompBdy.run``.
    pbar : bool
        Show progress bars during boundary sweeps.
    stop_grad_ : bool
        Forwarded to ``CompBdy.run``.
    fidel_ : bool
        Forwarded to ``CompBdy.run``.
    visual_ : bool
        Forwarded to ``CompBdy.run``.
    re_update : bool
        Forwarded to ``CompBdy.run``.
    max_seperation : int
        Forwarded to ``CompBdy.run``.
    direction : str
        Sweep direction (e.g. ``"y"``, ``"x"``, ``"diag"`` variants).
    eq_norms : bool | float
        Forwarded to ``CompBdy`` and ``CompBdy.run``.
    add_diag_to_p : bool
        Add ``Du*``/``du*`` diagonal tags to input ``p`` before norm build.
    """

    to_backend: object = None
    opt: object = "auto-hq"
    chi: int = 20
    seed: int = 1
    single_layer: bool = False
    flat: bool = False
    dmrg_run: str = "eff"
    n_iter: int = 2
    re_tag: bool = False
    pbar: bool = True
    stop_grad_: bool = False
    fidel_: bool = False
    visual_: bool = False
    re_update: bool = True
    max_seperation: int = 0
    direction: str = "y"
    eq_norms: object = False
    add_diag_to_p: bool = True

    def _validate_and_prepare_input(self, p):
        tags = set(getattr(p, "tags", ()))

        if not any(_TAG_X.match(tag) for tag in tags) or not any(_TAG_Y.match(tag) for tag in tags):
            raise ValueError("Input network must contain X* and Y* lattice tags.")

        if not self.add_diag_to_p and not any(_TAG_DU.match(tag) for tag in tags):
            warnings.warn(
                "Input network has no Du* tags. Enabling diagonal tagging automatically.",
                RuntimeWarning,
            )
            self.add_diag_to_p = True

    def run(self, p):
        """Run boundary norm evaluation for input tensor network ``p``.

        Parameters
        ----------
        p : qtn.TensorNetwork
            Input PEPS-like tensor network.

        Returns
        -------
        tuple
            ``(cost, norm, bdy_mpo, comp_bdy)``
        """
        self._validate_and_prepare_input(p)

        to_backend = self.to_backend
        if to_backend is None:
            warnings.warn(
                "No backend caster provided. Using backend_numpy(np.complex128).",
                RuntimeWarning,
            )
            to_backend = backend_numpy(dtype=np.complex128)

        ket = p.copy()
        bra = p.copy().conj()
        if self.add_diag_to_p:
            ket = add_diagonalu_tags(ket)
            bra = add_diagonalu_tags(bra)

        bra.reindex_({idx: f"r{uuid4().hex}" for idx in ket.inner_inds()})
        ket.add_tag("KET")
        bra.add_tag("BRA")

        norm = bra | ket
        norm.apply_to_arrays(to_backend)

        bdy_mpo = BdyMPO(
            tn_flat=ket,
            tn_double=norm,
            opt=self.opt,
            chi=self.chi,
            flat=self.flat,
            to_backend=to_backend,
            seed=self.seed,
            single_layer=self.single_layer,
        )
        comp_bdy = CompBdy(
            norm,
            bdy_mpo.mps_b,
            opt=self.opt,
            eq_norms=self.eq_norms,
            n_iter=self.n_iter,
            flat=self.flat,
            re_update=self.re_update,
            dmrg_run=self.dmrg_run,
            max_seperation=self.max_seperation,
        )

        cost = comp_bdy.run(
            n_iter=self.n_iter,
            re_tag=self.re_tag,
            pbar=self.pbar,
            stop_grad_=self.stop_grad_,
            fidel_=self.fidel_,
            visual_=self.visual_,
            re_update=self.re_update,
            max_seperation=self.max_seperation,
            direction=self.direction,
            eq_norms=self.eq_norms,
        )
        return cost, norm, bdy_mpo, comp_bdy
