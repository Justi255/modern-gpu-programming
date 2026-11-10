from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")


def test_workspace_rejects_overlapping_states_and_inputs():
    from fa5060._workspace import check_workspace_disjoint
    storage = torch.empty(64, dtype=torch.float32)
    scratch = SimpleNamespace(partial=storage[:16], maxima=storage[16:24], sums=storage[16:24])
    with pytest.raises(ValueError, match="overlap"):
        check_workspace_disjoint(scratch, ())
    scratch.sums = storage[24:32]
    with pytest.raises(ValueError, match="overlap"):
        check_workspace_disjoint(scratch, (storage[3:5],))


def test_workspace_accepts_disjoint_views_of_one_allocation():
    from fa5060._workspace import check_workspace_disjoint
    storage = torch.empty(64, dtype=torch.float32)
    scratch = SimpleNamespace(partial=storage[:16], maxima=storage[16:24], sums=storage[24:32])
    check_workspace_disjoint(scratch, (storage[32:],))
