import numpy as np
import pytensor.tensor as pt

from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    LONG_MATRIX_NAMES,
    MATRIX_NAMES,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
    VECTOR_VALUED,
)


def make_default_coords(ss_mod):
    coords = {
        ALL_STATE_DIM: ss_mod.state_names,
        ALL_STATE_AUX_DIM: ss_mod.state_names,
        OBS_STATE_DIM: ss_mod.observed_states,
        OBS_STATE_AUX_DIM: ss_mod.observed_states,
        SHOCK_DIM: ss_mod.shock_names,
        SHOCK_AUX_DIM: ss_mod.shock_names,
    }

    return coords


def cleanup_states(states: list[str]) -> list[str]:
    """
    Remove meaningless symbols from state names

    Parameters
    ----------
    states, list of str
        State names generated by make_harvey_state_names

    Returns
    -------
    states, list of str
        State names for the Harvey statespace representation, with meaningless terms removed

    The state names generated by make_harvey_state_names includes some "meaningless" terms. For example, lags are
    indicated with L{i}.state. This includes L0.state, which is correctly just "state".

    In addition, sequential applications of the difference operator are denoted Dk^i, where k is the length of the
    difference, and i is the number of repeated applications. Dk^1 is thus just Dk.
    """

    out = []
    for state in states:
        state = state.replace("^1", "")
        state = state.replace("^0", "")
        state = state.replace("L0", "")
        state = state.replace("D0", "")
        out.append(state)
    return out


def make_harvey_state_names(p: int, d: int, q: int, P: int, D: int, Q: int, S: int) -> list[str]:
    """
    Generate informative names for the SARIMA states in the Harvey representation

    Parameters
    ----------
    p: int
        AR order
    d: int
        Number of ARIMA differences
    q: int
        MA order
    P: int
        Seasonal AR order
    D: int
        Number of seasonal differences
    Q: int
        Seasonal MA order
    S: int
        Seasonal length

    Returns
    -------
    state_names, list of str
        List of state names

    The Harvey state is not particularly interpretable, but it's also not totally opaque. This helper function makes
    a list of state names that can help users understand what they are getting back from the statespace. In particular,
    it is helpful to know how differences and seasonal differences are incorporated into the model
    """
    k_lags = max(p + P * S, q + Q * S + 1)
    has_diff = (d + D) > 0

    # First state is always data
    states = ["data"]

    # Differencing operations
    # The goal here is to get down to "data_star", the state that actually has the SARIMA dynamics applied to it.
    # To get there, first the data needs to be differenced d-1 times
    d_size = d + int(D > 0)
    states.extend([f"D1^{(i + 1)}.data" for i in range(d_size)[:-1]])

    # Next, if there are seasonal differences, we need to lag the ARIMA differenced state S times, then seasonal
    # difference it. This procedure is done D-1 times.

    arma_diff = [int(d_size > 1), d_size - 1]
    season_diff = [S, 0]
    curr_state = f"D{arma_diff[0]}^{arma_diff[1]}"
    for i in range(D):
        states.extend([f"L{j + 1}{curr_state}.data" for j in range(S - 1)])
        season_diff[1] += 1
        curr_state = f"D{arma_diff[0]}^{arma_diff[1]}D{season_diff[0]}^{season_diff[1]}"
        if i != (D - 1):
            states.append(f"{curr_state}.data")

    # Now we are at data_star. If we did any differencing, add it in.
    if has_diff:
        states.append("data_star")

    # Next, we add the time series dynamics states. These don't have a immediately obvious interpretation, so just call
    # them "state_1" .., "state_n".
    suffix = "_star" if "star" in states[-1] else ""
    states.extend([f"state{suffix}_{i + 1}" for i in range(k_lags - 1)])

    states = cleanup_states(states)

    return states


def make_SARIMA_transition_matrix(
    p: int, d: int, q: int, P: int, D: int, Q: int, S: int
) -> np.ndarray:
    r"""
    Make the transition matrix for a SARIMA model

    Parameters
    ----------
    p: int
        AR order
    d: int
        Number of ARIMA differences
    q: int
        MA order
    P: int
        Seasonal AR order
    D: int
        Number of seasonal differences
    Q: int
        Seasonal MA order
    S: int
        Seasonal length

    Returns
    -------
    T, ndarray
        The transition matrix associated with a SARIMA model of order (p,d,q)x(P,D,Q,S)

    Notes
    -----
    The transition matrix for the SARIMA model has a bunch of structure in it, especially when differences are included
    in the statespace model. This function will always assume the state space matrix is in the Harvey representation.

    Given this representation, the matrix can be divided into a bottom part and a top part. The top part has (S * D) + d
    rows, and is associated with the differencing operations. The bottom part has max(P*S+p, Q*S+q+1) rows, and is
    responsible for the actual time series dynamics.

    The bottom part of the matrix is quite simple, it is just a shifted identity matrix (called a "companion matrix"),
    responsible for "rolling" the states, so that at each transition, the value for :math:`x_{t-3}` becomes the value
    for :math:`x_{t-2}`, and so on.

    The top part is quite complex. The goal of this part of the matrix is to transform the raw data state, :math:`x_t`,
    into a stationary state, :math:`x_t^\star`, via the application of differencing operations,
    :math:`\Delta x_t = x_t - x_{t-1}`. For ARIMA differences (the little ``d``), this is quite simple. Sequential
    differences are representated as an upper-triangular matrix of ones. To see this, consider an example where ``d=3``,
    so that:
     .. math::

        \begin{align}
            x_t^\star &= \Delta^3 x_t \\
             &= \Delta^2 (x_t - x_{t-1})
             &= \Delta (x_t - 2x_{t-1} + x_{t-2})
             &= x_t - x_{t-1} - 2x_{t-1} + 2x_{t-3} + x_{t-2} - x_{t-3}
             &= x_t - 3x_{t-1} + 3x_{t-3} - x_{t-3}
        \end{align}

    If you choose a state vector :math:`\begin{bmatrix}x_t & \Delta x_t & \Delta^2 x_t & x_t^\star \end{bmatrix}^T`,
    you will find that:

    .. math::
        \begin{bmatrix}x_t \\ \Delta x_t \\ \Delta^2 x_t \\ x_t^\star \end{bmatrix} =
            \begin{bmatrix} 1 & 1 & 1 & 1 \\
                            0 & 1 & 1 & 1 \\
                            0 & 0 & 1 & 1 \\
                            0 & 0 & 0 & 1
            \end{bmatrix}
            \begin{bmatrix} x_{t-1} \\ \Delta x_{t-1} \\ \Delta^2 x_{t-1} \\ x_{t-1}^\star \end{bmatrix}

    Next are the seasonal differences. The highest seasonal difference stored in the states is one less than the
    seasonal difference order, ``D``. That is, if ``D = 1, S = 4``, there will be states :math:``x_{t-1}, x_{t-2},
    x_{t-3}, x_{t-4}, x_t^\star`, with :math:`x_t^\star = \Delta_4 x_t = x_t - x_{t-4}`. The level state can be
    recovered by adding :math:`x_t^\star + x_{t-4}`. To accomplish all of this, two things need to be inserted into the
    transition matrix:

        1. A shifted identity matrix to "roll" the lagged states forward each transition, and
        2. A pair of 1's to recover the level state by adding the last 2 states (:math:`x_t^\star + x_{t-4}`)

    Keeping the example of ``D = 1, S = 4``, the block that handles the seasonal difference will look this this:
    .. math::
        \begin{bmatrix} 0 & 0 & 0 & 1 & 1 \\
                        1 & 0 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 & 0 \\
                        0 & 0 & 1 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 \end{bmatrix}

    In the presence of higher order seasonal differences, there needs to be one block per difference. And the level
    state is recovered by adding together the last state from each block. For example, if ``D = 2, S = 4``, the states
    will be :math:`x_{t-1}, x_{t-2}, x_{t-3}, x_{t-4}, \Delta_4 x_{t-1}, \Delta_4 x_{t-2}, \Delta_4 x_{t-3},
    \Delta_4 x_{t-4} x_t^\star`, with :math:`x_t^\star = \Delta_4^2 = \Delta_4(x_t - x_{t-4}) = x_t - 2 x_{t-4} +
    x_{t-8}`. To recover the level state, we need :math:`x_t = x_t^\star + \Delta_4 x_{t-4} + x_{t-4}`. In addition,
    to recover :math:`\Delta_4 x_t`, we have to compute :math:`\Delta_4 x_t = x_t^\star + \Delta_4 x_{t-4} =
    \Delta_4(x_t - x_{t-4}) + \Delta_4 x_{t-4} = \Delta_4 x_t`. The block of the transition matrix associated with all
    this is thus:

    .. math::
        \begin{bmatrix} 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 1 \\
                        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 \\
                        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}

    When ARIMA differences and seasonal differences are mixed, the seasonal differences will be written in terms of the
    highest ARIMA difference order, and recovery of the level state will require the use of all the ARIMA differences,
    as well as the seasonal differences. In addition, the seasonal differences are needed to back out the ARIMA
    differences from :math:`x_t^\star`. Here is the differencing block for a SARIMA(0,2,0)x(0,2,0,4) -- the identites
    of the states is left an exercise for the motivated reader:

    .. math::
        \begin{bmatrix}
            1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 1 \\
            0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 1 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 1 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}
    """
    n_diffs = S * D + d
    k_lags = max(p + P * S, q + Q * S + 1)
    k_states = k_lags + n_diffs

    # Top Part
    # ARIMA differences
    T = np.zeros((k_states, k_states))
    diff_idx = np.triu_indices(d)
    T[diff_idx] = 1

    # Adjustment factors for difference states All of the difference states are computed relative to x_t_star using
    # combinations of states, so there's a lot of "backing out" that needs to happen here. The columns are the more
    # straightforward part. After the (d,d) upper triangle of 1s for the ARIMA lags, there will be (S - 1) zeros,
    # and then a 1. In addition, there is an extra column of 1s at position n_diffs + 1, corresponding to x_star itself.

    # This will slowly taper down, but first we build the "full" set of column indices with values
    base_col_idx = d + S + np.arange(D) * S - 1
    if len(base_col_idx) > 0:
        base_col_idx = np.r_[base_col_idx, base_col_idx[-1] + 1]

    # The first d rows -- associated with the ARIMA differences -- will have 1s in all columns.
    col_idx = np.tile(base_col_idx, d)
    row_idx = np.arange(d).repeat(D + 1)

    # Next, if there are seasonal differences, there will be more rows, with the columns slowly dropping off.
    # Starting from the d+1-th row, there will be 1 in the column positions every S rows, for a total of (D-1) rows.
    # Every row will drop 2 columns from the left of base_col_idx.
    for i in range(D):
        n = len(base_col_idx[i:])
        col_idx = np.r_[col_idx, base_col_idx[i:]]
        row_idx = np.r_[row_idx, np.full(n, d + S * i)]

    if D == 0 and d > 0:
        # Special case: If there are *only* ARIMA lags, there still needs to be a single column of 1s at position
        # [:d, d]
        row_idx = np.arange(d)
        col_idx = np.full(d, d)
    T[row_idx, col_idx] = 1

    if S > 0:
        # "Rolling" indices for seasonal differences
        (row_roll_idx, col_roll_idx) = np.diag_indices(S * D)
        row_roll_idx = row_roll_idx + d + 1
        col_roll_idx = col_roll_idx + d

        # Rolling indices have a zero after every diagonal of length S-1
        T[row_roll_idx, col_roll_idx] = 1
        zero_idx = row_roll_idx[S - 1 :: S], col_roll_idx[S - 1 :: S]
        T[zero_idx] = 0

    # Bottom part
    # Rolling indices for the "compute" states, x_star
    star_roll_row, star_roll_col = np.diag_indices(k_lags - 1)
    star_roll_row = star_roll_row + n_diffs
    star_roll_col = star_roll_col + n_diffs + 1

    T[star_roll_row, star_roll_col] = 1

    return T


def conform_time_varying_and_time_invariant_matrices(A, B):
    """
    Adjust either A or B to conform to the other in the time dimension

    In the context of building a structural model from components, it might be the case that one component has
    time-varying statespace matrices, while the other does not. In this case, it is not possible to concatenate
    or block diagonalize the pair of matrices A and B without first expanding the time-invariant matrix to have a
    time dimension. This function checks if exactly one of the two time varies, and adjusts the other accordingly if
    need be.

    Parameters
    ----------
    A: pt.TensorVariable
        An anonymous statespace matrix
    B: pt.TensorVariable
        An anonymous statespace matrix

    Returns
    -------
    (A, B): Tuple of pt.TensorVariable
        A and B, with one or neither expanded to have a time dimension.
    """

    if A.name == B.name:
        name = A.name
    else:
        if all([X.name not in MATRIX_NAMES + LONG_MATRIX_NAMES for X in [A, B]]):
            raise ValueError(
                "At least one matrix passed to conform_time_varying_and_time_invariant_matrices should be a "
                "statespace matrix"
            )
        name = A.name if A.name in MATRIX_NAMES + LONG_MATRIX_NAMES else B.name

    time_varying_ndim = 3 - int(name in VECTOR_VALUED)

    if not all([x.ndim == time_varying_ndim for x in [A, B]]):
        return A, B

    T_A, *A_dims = A.type.shape
    T_B, *B_dims = B.type.shape

    if T_A == T_B:
        return A, B

    if T_A == 1:
        A_out = pt.repeat(A, B.shape[0], axis=0)
        A_out = pt.specify_shape(A_out, (T_B, *tuple(A_dims)))
        A_out.name = A.name

        return A_out, B

    if T_B == 1:
        B_out = pt.repeat(B, A.shape[0], axis=0)
        B_out = pt.specify_shape(B_out, (T_A, *tuple(B_dims)))
        B_out.name = B.name

        return A, B_out

    return A, B


def get_exog_dims_from_idata(exog_name, idata):
    if exog_name in idata.posterior.data_vars:
        exog_dims = idata.posterior[exog_name].dims[2:]
    elif exog_name in getattr(idata, "constant_data", []):
        exog_dims = idata.constant_data[exog_name].dims
    elif exog_name in getattr(idata, "mutable_data", []):
        exog_dims = idata.mutable_data[exog_name].dims
    else:
        exog_dims = None

    return exog_dims
