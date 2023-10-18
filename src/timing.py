import time as _time

__all__ = [
    "time_block",
    "time_function",
    "disable_timing",
]


_recursion_depth = 0
_timing_disabled = False


class time_block:
    def __init__(self, name, round=True, enable=True):
        self.name = str(name)
        self.round = round
        self.enable = enable

    def __enter__(self):
        if self.enable:
            global _recursion_depth
            _recursion_depth += 1
            self.start = _time.perf_counter()

    def __exit__(self, ex_type, ex, ex_traceback):
        if not _timing_disabled and self.enable and ex is None:
            elapsed = _time.perf_counter() - self.start
            global _recursion_depth
            _recursion_depth -= 1
            if self.round:
                elapsed_str = f"{elapsed*1000:.0f} ms"
            else:
                elapsed_str = f"{elapsed*1000} ms"
            print(("  " * _recursion_depth) + f"[timing]  {self.name!r} took {elapsed_str}")
        return False


def time_function(func):
    def new_func(*args, **kwargs):
        with time_block(func.__name__):
            return func(*args, **kwargs)

    return new_func


def disable_timing():
    global time_block, time_function, _timing_disabled
    from contextlib import nullcontext

    time_block = nullcontext
    time_function = lambda func: func
    _timing_disabled = True
