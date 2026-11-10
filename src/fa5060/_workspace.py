"""Metadata-only checks for writable decode scratch; no GPU imports or reads."""


def _span(tensor):
    if tensor is None or tensor.numel() == 0:
        return (0, 0)
    begin = tensor.data_ptr()
    elements = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return begin, begin + elements * tensor.element_size()


def check_workspace_disjoint(workspace, inputs):
    """Reject overlapping scratch writes and overlaps with live inputs/output.

    Disjoint contiguous slices of one packed scratch allocation remain valid.
    For strided inputs the bounding address span is deliberately conservative.
    """
    states = (workspace.partial, workspace.maxima, workspace.sums)
    for index, state in enumerate(states):
        low, high = _span(state)
        for other in (*states[index + 1:], *inputs):
            other_low, other_high = _span(other)
            if max(low, other_low) < min(high, other_high):
                raise ValueError("workspace buffers must not overlap each other or inputs/output")
