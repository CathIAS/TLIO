import contextlib

with contextlib.redirect_stdout(None):
    # numba is too noisy
    from numba import jit
