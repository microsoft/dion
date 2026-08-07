"""Dion3: an additional public name for the NorDion2 optimizer.

NorDion2 is the third-generation Dion update -- Dion2's submatrix selection and
error feedback with NorMuon's per-neuron normalization -- so it is also exposed
as ``Dion3``. This is an alias, not a subclass: ``Dion3 is NorDion2``, so the
two names are interchangeable everywhere, including ``isinstance`` checks. The
implementation lives in ``dion/nordion2.py``.

Parameter groups keep using ``algorithm="nordion2"`` under either name. That
string keys the optimizer state and megabatch grouping (see
``DistributedOrthoBase.step``) and is written into
``state_dict()["param_groups"]``, so it stays the internal identifier and
existing checkpoints load unchanged.
"""

from .nordion2 import NorDion2 as Dion3
