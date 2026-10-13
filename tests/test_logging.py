import logging

from bagcounter.logging import DedupFilter


def test_error_loop_collapses_to_one_line_per_window():
    f = DedupFilter(window_s=60)
    rec = logging.LogRecord("x", logging.ERROR, __file__, 10, "boom", None, None)
    passed = sum(f.filter(rec) for _ in range(10_000))
    assert passed == 1


def test_dedup_keeps_distinct_errors_and_info():
    f = DedupFilter(window_s=60)
    a = logging.LogRecord("x", logging.ERROR, __file__, 10, "a", None, None)
    b = logging.LogRecord("x", logging.ERROR, __file__, 11, "b", None, None)
    info = logging.LogRecord("x", logging.INFO, __file__, 12, "i", None, None)
    assert f.filter(a) and f.filter(b) and f.filter(info) and f.filter(info)
