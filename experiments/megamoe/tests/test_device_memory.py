import threading

import pytest

from deep_ep_experimental.megamoe.device_memory import (
    DeviceMemorySampler,
    combine_device_memory_samples,
    summarize_device_memory_peaks,
)


class FakeNPU:
    def __init__(self, free, total):
        self.free = free
        self.total = total
        self.device = threading.local()
        self.watch = None
        self.observed = threading.Event()

    def set_device(self, device):
        self.device.index = device

    def mem_get_info(self, device):
        assert device == self.device.index == 7
        reading = self.free, self.total
        if reading == self.watch:
            self.observed.set()
        return reading


def test_samples_transient_peak_on_explicit_device():
    gib = 1 << 30
    npu = FakeNPU(7 * gib, 8 * gib)
    npu.watch = 2 * gib, 8 * gib
    with DeviceMemorySampler(npu, 7, sample_interval_ms=1) as sampler:
        npu.free = 2 * gib
        # Wait until this allocation has actually been observed before releasing it.
        assert npu.observed.wait(3)
        npu.free = 6 * gib
    sampler.stop()
    report = sampler.report()
    assert report["initial_used_bytes"] == gib
    assert report["final_used_bytes"] == 2 * gib
    assert report["peak_used_bytes"] == 6 * gib
    assert report["peak_used_gib"] == 6
    assert report["peak_used_pct"] == 75
    assert report["sample_interval_ms"] == 1
    assert report["periodic_sample_count"] == report["sample_count"] - 2 >= 1


def test_short_phase_captures_both_boundaries_and_merges_local_peaks():
    npu = FakeNPU(75, 100)

    def measure(final_free):
        with DeviceMemorySampler(npu, 7, sample_interval_ms=60000) as sampler:
            with pytest.raises(RuntimeError, match="completed"):
                sampler.report()
            npu.free = final_free
        return sampler.report()

    setup = measure(10)
    npu.free = 20
    case = measure(30)
    combined = combine_device_memory_samples([setup, case])
    assert combined["initial_used_bytes"] == 25
    assert combined["final_used_bytes"] == 70
    assert combined["peak_used_bytes"] == 90
    assert combined["sample_count"] == 4
    assert combined["periodic_sample_count"] == 0
    assert setup["sample_count"] == 2
    with pytest.raises(ValueError, match="device_total_bytes"):
        combine_device_memory_samples([setup, dict(case, device_total_bytes=101)])


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_bad_interval(interval):
    with pytest.raises(ValueError, match="finite and positive"):
        DeviceMemorySampler(FakeNPU(50, 100), 7, interval)


def test_sampling_failure_never_yields_report():
    npu = FakeNPU(50, 100)
    npu.watch = 50, 150
    sampler = DeviceMemorySampler(npu, 7, sample_interval_ms=1)
    with pytest.raises(RuntimeError, match="capacity changed"):
        with sampler:
            npu.total = 150
            assert npu.observed.wait(3)
    with pytest.raises(RuntimeError, match="capacity changed"):
        sampler.report()


def test_invalid_initial_or_final_snapshot_and_exception_cleanup():
    npu = FakeNPU(101, 100)
    with pytest.raises(RuntimeError, match="Invalid NPU memory"):
        DeviceMemorySampler(npu, 7).start()
    npu.free = 10
    sampler = DeviceMemorySampler(npu, 7, sample_interval_ms=60000)
    with pytest.raises(LookupError, match="case failure"):
        with sampler:
            npu.free = -1
            raise LookupError("case failure")
    with pytest.raises(RuntimeError, match="Invalid NPU memory"):
        sampler.report()


def test_rank_summary_tracks_distinct_extrema_and_first_rank_on_ties():
    reports = [
        dict(device_total_bytes=1000, peak_used_bytes=840, peak_used_pct=84),
        dict(device_total_bytes=100, peak_used_bytes=90, peak_used_pct=90),
        dict(device_total_bytes=10, peak_used_bytes=5, peak_used_pct=50),
        dict(device_total_bytes=1000, peak_used_bytes=840, peak_used_pct=84),
    ]
    assert summarize_device_memory_peaks(reports) == dict(
        rank_count=4,
        max_peak_used_bytes=840,
        max_peak_used_rank=0,
        max_peak_used_pct=90,
        max_peak_used_pct_rank=1,
        min_headroom_bytes=5,
        min_headroom_rank=2,
    )
