"""
Input coercion shared by the return-oriented modules.

Every module here takes "one number per day", and every one of them meets the
same three ways of being handed something else: a column vector instead of a
flat array, an empty slice, or a NaN that would quietly change an answer
rather than raise. Keeping the check in one place is what stops the error
messages -- and, more importantly, the strictness -- from drifting apart.
"""

from __future__ import annotations

import numpy as np


def as_1d(name: str, values, allow_empty: bool = False) -> np.ndarray:
    """Coerce one input to a finite 1-D float array.

    Args:
        name: The argument's name, quoted back in any error message.
        values: Anything array-like -- a list, an ndarray, a pandas Series,
            or a single-column 2-D array.
        allow_empty: Whether an empty array is acceptable. Callers that want
            to report emptiness across a *pair* of series pass True and do
            that check themselves.

    Returns:
        A 1-D float ndarray.

    Raises:
        ValueError: If the input is not 1-D, is empty while ``allow_empty``
            is False, or holds a non-finite value.
    """
    array = np.asarray(values, dtype=float)

    # A column vector is the common shape accident (an sklearn output, a
    # DataFrame slice) and means the same thing; any other 2-D shape is a
    # real mistake.
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.ravel()
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (or a single-column 2-D array), got shape "
            f"{np.shape(values)}."
        )
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    # NaNs must not reach the arithmetic: a NaN prediction reads as a "down"
    # call or a flat day instead of an error, and a NaN return poisons every
    # figure after it. A non-finite value is a bug upstream, so say so here.
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(f"{name} contains {bad} non-finite value(s) (NaN or inf).")

    return array
