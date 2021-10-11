from contextlib import contextmanager
import sys

@contextmanager
def perf_timed(name=None, f=sys.stdout):
    from time import perf_counter
    start = perf_counter()
    yield start
    stop = perf_counter()
    if name:
        f.write(name + " - ")
    f.write("Took %s\n" % (stop-start))
