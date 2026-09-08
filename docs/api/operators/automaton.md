# `pepsy.operators.MPOAutomaton`

`MPOAutomaton` is the explicit structural layer for open-boundary MPOs. A
channel is a virtual state on one bond cut; a transition is a local operator
edge between two channel states. Every path from `start_state` to
`done_state` contributes one product operator.

```python
import numpy as np
from pepsy.operators import MPOAutomaton

x = np.array([[0.0, 1.0], [1.0, 0.0]])
z = np.diag([1.0, -1.0])

automaton = MPOAutomaton(6)
automaton.add_local_term(2, x, coefficient=0.3)
automaton.add_factorized_term((0, 5), (z, z), coefficient=-1.2)

# Exact tensor assembly. No SVD, QR, canonicalization, or compression occurs.
mpo = automaton.to_mpo()
```

For a nontrivial operator string between two endpoints, pass the intermediate
local operators explicitly:

```python
automaton.add_factorized_term(
    (1, 4),
    (x, x),
    string_operators=(z, z),
)
```

Use `add_product_term` for a product with more than two supported sites. The
support is given in chain order, and `string_operators` contains all local
operators on the omitted sites:

```python
automaton.add_product_term(
    (0, 2, 5),
    (x, z, x),
    string_operators=(z, z),
)
```

Automata can also be combined exactly at the channel level. `a.compose(b)`
forms the operator product `a @ b`, while `a.power(n)` forms an exact
non-negative integer power. `add_automaton` adds a direct-sum path with state
remapping, which is useful for explicit polynomial or time-step expansions:

```python
identity = MPOAutomaton.identity(6, 2)
correction = automaton.power(2)
identity.add_automaton(correction, coefficient=0.5)
mpo = identity.to_mpo()
```

These operations only assemble structural paths. They do not compress the
resulting channel space. `trim()` is the one exact structural cleanup: it
removes channels that are unreachable from either boundary without changing
any accepted operator path. Numerical bond compression remains an explicit
follow-up on the returned Quimb MPO.

The model-facing `ham_tn.to_mpo(...)` builder adds one dense-only structural
pass after its automaton or term assembly. It removes exact proportional and
roundoff-safe linearly dependent boundary channels before the optional Quimb
SVD, then keeps the existing public `compress`, `max_bond`, and backend
behavior. This is the deparallelization/delinearization stage; the explicit
`MPOAutomaton.to_mpo()` method above remains a raw, inspection-friendly
materialization. Non-NumPy tensors, including Torch and native Symmray data,
continue through their existing backend-aware paths.

The existing channel/transition tuple representation can be wrapped with
`MPOAutomaton.from_legacy(...)` and emitted with `to_legacy()`. This allows the
current Symmray MPO assembly code to adopt the common structural model without
changing its backend-specific charge handling in the first migration step.

`MPOAutomaton.to_mpo(compress=True)` is intentionally rejected. If an
approximate bond truncation is wanted, call Quimb's `mpo.compress(...)` as a
separate, visible operation after inspecting or recording the raw bond
dimensions.
