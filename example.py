import mymodule
import numpy as np
import timeit

a = np.random.randint(0, 1000, size=10000)


def sum_of_sines(a):
    return mymodule.sum_of_sines(a)


assert np.isclose(np.sum(np.sin(a)), sum_of_sines(a))

t = timeit.Timer(lambda: sum_of_sines(a))
print(t.timeit(100))

t = timeit.Timer(lambda: np.sum(np.sin(a)))
print(t.timeit(100))
