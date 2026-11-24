"""Sample device-wide HBM usage. Peaks are lower bounds and include every device user."""

import math
import operator
import threading


class DeviceMemorySampler:
    """Sample total minus free bytes at entry, periodically during work, and at exit.

    The caller controls the work boundaries, including completion of asynchronous
    work. The sampler never synchronizes the device. Each measurement has one
    owner thread and uses one explicit device for its entire lifetime.
    """

    def __init__(self, npu, device, sample_interval_ms=10):
        if device is None:
            raise ValueError("HBM sampling requires an explicit NPU device")
        if not math.isfinite(sample_interval_ms) or sample_interval_ms <= 0:
            raise ValueError("sample_interval_ms must be finite and positive")
        self._npu = npu
        self._device = device
        self._interval_ms = sample_interval_ms
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._count = 0
        self._total = 0
        self._initial = 0
        self._last = 0
        self._peak = 0

    def _sample(self):
        free, total = map(operator.index, self._npu.mem_get_info(self._device))
        if total <= 0 or not 0 <= free <= total:
            raise ValueError(f"Invalid NPU memory reading: free={free}, total={total}")
        used = total - free
        if self._count == 0:
            self._total = total
            self._initial = used
        elif total != self._total:
            raise ValueError(f"NPU memory capacity changed from {self._total} to {total} bytes")
        self._last = used
        self._peak = max(self._peak, used)
        self._count += 1

    def _poll(self):
        try:
            # ACL's current context is thread-local, including on nonzero local ranks.
            self._npu.set_device(self._device)
            self._sample()
            self._ready.set()
            while not self._stop.wait(self._interval_ms / 1000):
                self._sample()
            self._sample()
        except BaseException as error:
            self._error = error
        finally:
            self._ready.set()

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError(f"HBM sampling failed on NPU {self._device}: {self._error}") from self._error

    def start(self):
        if self._thread is not None:
            raise RuntimeError("HBM sampler has already been started")
        self._thread = threading.Thread(target=self._poll, name="megamoe-hbm", daemon=True)
        self._thread.start()
        try:
            self._ready.wait()
            self._check_error()
        except BaseException:
            self._stop.set()
            self._thread.join()
            raise
        return self

    def stop(self):
        if self._thread is None:
            raise RuntimeError("HBM sampler has not been started")
        self._stop.set()
        self._thread.join()
        self._check_error()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.stop()
        except Exception as sampling_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(str(sampling_error))

    def report(self):
        """Return one completed local measurement, including boundary samples."""
        self._check_error()
        if not self._stop.is_set() or self._thread.is_alive() or self._count < 2:
            raise RuntimeError("HBM report requires a completed measurement")
        return dict(
            device=str(self._device),
            device_total_bytes=self._total,
            sample_interval_ms=self._interval_ms,
            sample_count=self._count,
            periodic_sample_count=self._count - 2,
            initial_used_bytes=self._initial,
            final_used_bytes=self._last,
            peak_used_bytes=self._peak,
            peak_used_gib=self._peak / (1 << 30),
            peak_used_pct=100 * self._peak / self._total,
        )


def combine_device_memory_samples(reports):
    """Combine chronologically ordered phases on one device, counting each sample once."""
    if not reports:
        raise ValueError("Need at least one HBM phase")
    combined = dict(reports[0])
    for report in reports[1:]:
        for key in ("device", "device_total_bytes", "sample_interval_ms"):
            if report[key] != combined[key]:
                raise ValueError(f"HBM phases disagree on {key}")
        combined["sample_count"] += report["sample_count"]
        combined["periodic_sample_count"] += report["periodic_sample_count"]
        combined["final_used_bytes"] = report["final_used_bytes"]
        if report["peak_used_bytes"] > combined["peak_used_bytes"]:
            for key in ("peak_used_bytes", "peak_used_gib", "peak_used_pct"):
                combined[key] = report[key]
    return combined


def summarize_device_memory_peaks(reports):
    """Summarize rank-ordered reports. Ties resolve to the lowest global rank."""
    if not reports:
        raise ValueError("Need at least one rank's HBM report")
    peak_bytes = [report["peak_used_bytes"] for report in reports]
    peak_pct = [report["peak_used_pct"] for report in reports]
    headroom = [report["device_total_bytes"] - used for report, used in zip(reports, peak_bytes)]
    max_bytes = max(peak_bytes)
    max_pct = max(peak_pct)
    min_free = min(headroom)
    return dict(
        rank_count=len(reports),
        max_peak_used_bytes=max_bytes,
        max_peak_used_rank=peak_bytes.index(max_bytes),
        max_peak_used_pct=max_pct,
        max_peak_used_pct_rank=peak_pct.index(max_pct),
        min_headroom_bytes=min_free,
        min_headroom_rank=headroom.index(min_free),
    )
