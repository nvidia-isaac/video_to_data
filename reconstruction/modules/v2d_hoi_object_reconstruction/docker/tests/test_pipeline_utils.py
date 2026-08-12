from v2d_hoi_object_reconstruction.docker._pipeline_utils import _interleave_gpu_slots


def test_interleave_gpu_slots_prefers_distinct_gpus():
    assert _interleave_gpu_slots([(0, 2), (1, 2)]) == [0, 1, 0, 1]


def test_interleave_gpu_slots_handles_uneven_capacity():
    assert _interleave_gpu_slots([(0, 3), (1, 1)]) == [0, 1, 0, 0]


def test_interleave_gpu_slots_handles_single_gpu():
    assert _interleave_gpu_slots([(0, 2)]) == [0, 0]


def test_interleave_gpu_slots_handles_no_capacity():
    assert _interleave_gpu_slots([]) == []
