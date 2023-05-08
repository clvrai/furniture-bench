import time
from functools import wraps


def set_frequency(freq):
    """Set frequency of the function."""

    def dec(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            while (time.time() - start) < (1.0 / float(freq)):
                time.sleep(0.001)
                pass
            return result

        return wrap

    return dec
