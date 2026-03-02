from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from copex import metrics as metrics_mod


def test_get_collector_thread_safe_singleton() -> None:
    old = metrics_mod._global_collector
    metrics_mod._global_collector = None
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            instances = list(executor.map(lambda _: metrics_mod.get_collector(), range(24)))
        first = instances[0]
        assert all(instance is first for instance in instances)
    finally:
        metrics_mod._global_collector = old
