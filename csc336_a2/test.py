import numpy as np
import time

if __name__ == '__main__':
    A = np.random.rand(10000, 10000)
    x = np.random.rand(10000)
    start_time = time.perf_counter()
    _ = (2 * A) @ x
    time_taken = time.perf_counter() - start_time
    print(time_taken)
    start_time = time.perf_counter()
    _ = (A @ x) + (A @ x)
    time_taken = time.perf_counter() - start_time
    print(time_taken)