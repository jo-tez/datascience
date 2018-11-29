import numpy as np
import math


__all__ = [
    'make_array', 
    'percentile', 
    'proportions_from_distribution', 
]


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



def sample_proportions(sample_size, probabilities):
    """Return the proportion of random draws for each outcome in a distribution.

    This function is similar to np.random.multinomial, but returns proportions
    instead of counts.

    Args:
        ``sample_size``: The size of the sample to draw from the distribution.

        ``probabilities``: An array of probabilities that forms a distribution.

    Returns:
        An array with the same length as ``probability`` that sums to 1.
    """
    return np.random.multinomial(sample_size, probabilities) / sample_size


def proportions_from_distribution(table, label, sample_size,
                                  column_name='Random Sample'):
    """
    Adds a column named ``column_name`` containing the proportions of a random
    draw using the distribution in ``label``.

    This method uses ``np.random.multinomial`` to draw ``sample_size`` samples
    from the distribution in ``table.column(label)``, then divides by
    ``sample_size`` to create the resulting column of proportions.

    Args:
        ``table``: An instance of ``Table``.

        ``label``: Label of column in ``table``. This column must contain a
            distribution (the values must sum to 1).

        ``sample_size``: The size of the sample to draw from the distribution.

        ``column_name``: The name of the new column that contains the sampled
            proportions. Defaults to ``'Random Sample'``.

    Returns:
        A copy of ``table`` with a column ``column_name`` containing the
        sampled proportions. The proportions will sum to 1.

    Throws:
        ``ValueError``: If the ``label`` is not in the table, or if
            ``table.column(label)`` does not sum to 1.
    """
    proportions = sample_proportions(sample_size, table.column(label))
    return table.with_column('Random Sample', proportions)
