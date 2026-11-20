"""A physical EP32 communication worker can report two logical peers."""

import pytest

from deep_ep_experimental.megamoe import urma_trace


def test_ep32_trace_preserves_peer_ids_on_one_worker():
    np = pytest.importorskip("numpy")
    capacity = 4
    trace = np.zeros((urma_trace.ENGINE_COUNT, (capacity + 1) * urma_trace.RECORD_WORDS), dtype=np.uint64)
    trace[:, 0] = urma_trace.TRACE_SCHEMA
    trace[:, 1] = capacity
    trace[:, 4] = np.arange(urma_trace.ENGINE_COUNT)
    trace[2, 2] = capacity
    trace[2, 16:20] = (80, 105, 53, urma_trace.UNKNOWN)
    trace[2, 32:36] = (90, 110, 49, urma_trace.UNKNOWN)
    for offset, peer in enumerate((14, 30), 3):
        row = trace[2, offset * urma_trace.RECORD_WORDS : (offset + 1) * urma_trace.RECORD_WORDS]
        row[:4] = (100 + offset, 100 + offset, 65, peer)
        row[4:12] = urma_trace.UNKNOWN
    result = urma_trace.decode_trace(trace, rank=31, clock_mhz=1000)
    events = [event for event in result["traceEvents"] if event.get("args", {}).get("event_id") == 65]
    assert [event["args"]["peer"] for event in events] == [14, 30]


def test_ep32_worker_aggregate_does_not_attribute_traffic_to_one_peer():
    np = pytest.importorskip("numpy")
    timers = np.zeros((28, 3, 8, 16), dtype=np.uint64)
    dispatch = timers[:16, 2, 1]
    dispatch[:, 5] = urma_trace.DISPATCH_SEND_TO_RECEIVE
    dispatch[:, 6] = urma_trace.TIMER_SCHEMA
    dispatch[0, :5] = (100, 120, 20, 240, 2)
    dispatch[0, 7:10] = (150, 360, 3)
    result = urma_trace.communication_bandwidth(timers, 31, world=32, group_size=16)
    record = result["classes"][0]
    assert record["communication_class"] == "ALL_PEERS"
    assert record["workers"] == [0] and record["peers"] == []
    assert record["received_payload_bytes"] == 360
    assert record["payload_bytes"] == 240 and record["envelope_ticks"] == 50
