from scripts.live_latency import acknowledged_latencies


def test_coalesced_commands_are_not_counted_as_observed():
    sent = {8: 1.00, 9: 1.01, 10: 1.02, 11: 1.03}
    frames = [
        {"seq": 10, "received": 1.10},
        {"seq": 10, "received": 1.20},
        {"seq": 11, "received": 1.30},
        {"seq": 12, "received": 1.40},
    ]
    assert acknowledged_latencies(sent, frames) == [1.10 - 1.02, 1.30 - 1.03]
