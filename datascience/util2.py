import numpy as np
import math

def make_array(*elements):
    """Returns an array containing all the arguments passed to this function.
    A simple way to make an array with a few elements.

    As with any array, all arguments should have the same type.

    >>> make_array(0)
    array([0])
    >>> make_array(2, 3, 4)
    array([2, 3, 4])
    >>> make_array("foo", "bar")
    array(['foo', 'bar'],
          dtype='<U3')
    >>> make_array()
    array([], dtype=float64)
    """
    return np.array(elements)
    

def percentile(p, arr=None):
    """Returns the pth percentile of the input array (the value that is at
    least as great as p% of the values in the array).

    If arr is not provided, percentile returns itself curried with p

    >>> percentile(74.9, [1, 3, 5, 9])
    5
    >>> percentile(75, [1, 3, 5, 9])
    5
    >>> percentile(75.1, [1, 3, 5, 9])
    9
    >>> f = percentile(75)
    >>> f([1, 3, 5, 9])
    5
    """
    if arr is None:
        return lambda arr: percentile(p, arr)
    if hasattr(p, '__iter__'):
        return np.array([percentile(x, arr) for x in p])
    if p == 0:
        return min(arr)
    assert 0 < p <= 100, 'Percentile requires a percent'
    i = (p/100) * len(arr)
    return sorted(arr)[math.ceil(i) - 1]
