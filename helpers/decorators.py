import time
import typing as t


def timer(func: t.Callable) -> t.Callable:
    def wrapper(*args, **kwargs) -> t.Any:
        s = time.perf_counter()
        res = func(*args, **kwargs)
        print(f"Function {func.__name__} took: {time.perf_counter() - s: .4f}")
        return res

    return wrapper
