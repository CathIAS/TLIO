import contextlib
import cProfile


@contextlib.contextmanager
def profile(filename, enabled=True):
    if enabled:
        profile = cProfile.Profile()
        profile.enable()
    try:
        yield
    finally:
        if enabled:
            profile.disable()
            profile.dump_stats(filename)
