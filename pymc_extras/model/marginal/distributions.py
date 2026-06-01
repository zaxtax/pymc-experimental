import warnings

from collections.abc import Sequence

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc.distributions import Bernoulli, Categorical, DiscreteUniform
from pymc.distributions.distribution import _support_point, support_point
from pymc.logprob.abstract import MeasurableOp, _logprob
from pymc.logprob.basic import conditional_logp, logp
from pymc.pytensorf import constant_fold
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import Mode
from pytensor.graph import FunctionGraph, Op, vectorize_graph
from pytensor.graph.basic import Variable, equal_computations
from pytensor.graph.replace import clone_replace, graph_replace
from pytensor.scan import map as scan_map
from pytensor.scan import scan
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.type import RandomType

from pymc_extras.distributions import DiscreteMarkovChain


class MarginalRV(OpFromGraph, MeasurableOp):
    """Base class for Marginalized RVs"""

    def __init__(
        self,
        *args,
        dims_connections: tuple[tuple[int | None], ...],
        dims: tuple[Variable, ...],
        **kwargs,
    ) -> None:
        self.dims_connections = dims_connections
        self.dims = dims
        super().__init__(*args, **kwargs)

    @property
    def support_axes(self) -> tuple[tuple[int]]:
        """Dimensions of dependent RVs that belong to the core (non-batched) marginalized variable."""
        marginalized_ndim_supp = self.inner_outputs[0].owner.op.ndim_supp
        support_axes_vars = []
        for dims_connection in self.dims_connections:
            ndim = len(dims_connection)
            marginalized_supp_axes = ndim - marginalized_ndim_supp
            support_axes_vars.append(
                tuple(
                    -i
                    for i, dim in enumerate(reversed(dims_connection), start=1)
                    if (dim is None or dim > marginalized_supp_axes)
                )
            )
        return tuple(support_axes_vars)

    def __eq__(self, other):
        # Just to allow easy testing of equivalent models,
        # This can be removed once https://github.com/pymc-devs/pytensor/issues/1114 is fixed
        if type(self) is not type(other):
            return False

        return equal_computations(
            self.inner_outputs,
            other.inner_outputs,
            self.inner_inputs,
            other.inner_inputs,
        )

    def __hash__(self):
        # Just to allow easy testing of equivalent models,
        # This can be removed once https://github.com/pymc-devs/pytensor/issues/1114 is fixed
        return hash((type(self), len(self.inner_inputs), len(self.inner_outputs)))


@_support_point.register
def support_point_marginal_rv(op: MarginalRV, rv, *inputs):
    """Support point for a marginalized RV.

    The support point of a marginalized RV is the support point of the inner RV,
    conditioned on the marginalized RV taking its support point.
    """
    outputs = rv.owner.outputs

    fgraph = op.fgraph.clone()
    inner_inputs = fgraph.inputs
    inner_outputs = fgraph.outputs
    del op

    inner_rv = inner_outputs[outputs.index(rv)]
    marginalized_inner_rv, *other_dependent_inner_rvs = (
        out for out in inner_outputs if out is not inner_rv and not isinstance(out.type, RandomType)
    )

    # Replace references to inner rvs by the dummy variables (including the marginalized RV)
    # This is necessary because the inner RVs may depend on each other
    marginalized_inner_rv_dummy = marginalized_inner_rv.clone()
    # Map inner rvs to dummies, saving what outer output each corresponds to.
    # We need dummies because inner RVs may depend on each other.
    inner_to_dummy_replacements = []
    dummy_to_outer_replacements = []
    for other_inner_rv in other_dependent_inner_rvs:
        dummy = other_inner_rv.clone()
        inner_to_dummy_replacements.append((other_inner_rv, dummy))
        dummy_to_outer_replacements.append((dummy, outputs[inner_outputs.index(other_inner_rv)]))

    fgraph.replace(marginalized_inner_rv, marginalized_inner_rv_dummy, import_missing=True)
    fgraph.replace_all(tuple(inner_to_dummy_replacements), import_missing=True)

    # Get support point of inner RV and marginalized RV
    inner_rv_support_point = support_point(inner_rv)
    marginalized_inner_rv_support_point = support_point(marginalized_inner_rv)

    fgraph = FunctionGraph(outputs=[inner_rv_support_point], clone=False)
    # Replace the marginalized RV dummy by its support point
    fgraph.replace(
        marginalized_inner_rv_dummy, marginalized_inner_rv_support_point, import_missing=True
    )
    # Replace the inner inputs by the outer inputs
    fgraph.replace_all(tuple(zip(inner_inputs, inputs)), import_missing=True)
    # Replace other dependent RVs dummies by the respective outer outputs.
    # PyMC will replace them by their support points later
    fgraph.replace_all(tuple(dummy_to_outer_replacements), import_missing=True)

    [rv_support_point] = fgraph.outputs
    return rv_support_point


class MarginalFiniteDiscreteRV(MarginalRV):
    """Base class for Marginalized Finite Discrete RVs"""


class MarginalDiscreteMarkovChainRV(MarginalRV):
    """Base class for Marginalized Discrete Markov Chain RVs"""


def get_domain_of_finite_discrete_rv(rv: TensorVariable) -> tuple[int, ...]:
    op = rv.owner.op
    dist_params = rv.owner.op.dist_params(rv.owner)
    if isinstance(op, Bernoulli):
        return (0, 1)
    elif isinstance(op, Categorical):
        [p_param] = dist_params
        [p_param_length] = constant_fold([p_param.shape[-1]])
        return tuple(range(p_param_length))
    elif isinstance(op, DiscreteUniform):
        lower, upper = constant_fold(dist_params)
        return tuple(np.arange(lower, upper + 1))
    elif isinstance(op, DiscreteMarkovChain):
        P, *_ = dist_params
        return tuple(range(pt.get_vector_length(P[-1])))

    raise NotImplementedError(f"Cannot compute domain for op {op}")


def reduce_batch_dependent_logps(
    dependent_dims_connections: Sequence[tuple[int | None, ...]],
    dependent_ops: Sequence[Op],
    dependent_logps: Sequence[TensorVariable],
) -> TensorVariable:
    """Combine the logps of dependent RVs and align them with the marginalized logp.

    This requires reducing extra batch dims and transposing when they are not aligned.

       idx = pm.Bernoulli(idx, shape=(3, 2))  # 0, 1
       pm.Normal("dep1", mu=idx.T[..., None] * 2, shape=(3, 2, 5))
       pm.Normal("dep2", mu=idx * 2, shape=(7, 2, 3))

       marginalize(idx)

       The marginalized op will have dims_connections = [(1, 0, None), (None, 0, 1)]
       which tells us we need to reduce the last axis of dep1 logp and the first of dep2 logp,
       as well as transpose the remaining axis of dep1 logp before adding the two element-wise.

    """
    from pymc_extras.model.marginal.graph_analysis import get_support_axes

    reduced_logps = []
    for dependent_op, dependent_logp, dependent_dims_connection in zip(
        dependent_ops, dependent_logps, dependent_dims_connections
    ):
        if dependent_logp.type.ndim > 0:
            # Find which support axis implied by the MarginalRV need to be reduced
            # Some may have already been reduced by the logp expression of the dependent RV (e.g., multivariate RVs)
            dep_supp_axes = get_support_axes(dependent_op)[0]

            # Dependent RV support axes are already collapsed in the logp, so we ignore them
            supp_axes = [
                -i
                for i, dim in enumerate(reversed(dependent_dims_connection), start=1)
                if (dim is None and -i not in dep_supp_axes)
            ]
            dependent_logp = dependent_logp.sum(supp_axes)

            # Finally, we need to align the dependent logp batch dimensions with the marginalized logp
            dims_alignment = [dim for dim in dependent_dims_connection if dim is not None]
            dependent_logp = dependent_logp.transpose(*dims_alignment)

        reduced_logps.append(dependent_logp)

    reduced_logp = pt.add(*reduced_logps)
    return reduced_logp


def align_logp_dims(dims: tuple[tuple[int, None]], logp: TensorVariable) -> TensorVariable:
    """Align the logp with the order specified in dims."""
    dims_alignment = [dim for dim in dims if dim is not None]
    return logp.transpose(*dims_alignment)


def inline_ofg_outputs(op: OpFromGraph, inputs: Sequence[Variable]) -> tuple[Variable]:
    """Inline the inner graph (outputs) of an OpFromGraph Op.

    Whereas `OpFromGraph` "wraps" a graph inside a single Op, this function "unwraps"
    the inner graph.
    """
    return graph_replace(
        op.inner_outputs,
        replace=tuple(zip(op.inner_inputs, inputs)),
        strict=False,
    )


class NonSeparableLogpWarning(UserWarning):
    pass


def warn_non_separable_logp(values):
    if len(values) > 1:
        warnings.warn(
            "There are multiple dependent variables in a FiniteDiscreteMarginalRV. "
            f"Their joint logp terms will be assigned to the first value: {values[0]}.",
            NonSeparableLogpWarning,
            stacklevel=2,
        )


DUMMY_ZERO = pt.constant(0, name="dummy_zero")


@_logprob.register(MarginalFiniteDiscreteRV)
def finite_discrete_marginal_rv_logp(op: MarginalFiniteDiscreteRV, values, *inputs, **kwargs):
    # Clone the inner RV graph of the Marginalized RV
    marginalized_rv, *inner_rvs = inline_ofg_outputs(op, inputs)

    # Obtain the joint_logp graph of the inner RV graph
    inner_rv_values = dict(zip(inner_rvs, values))
    marginalized_vv = marginalized_rv.clone()
    rv_values = inner_rv_values | {marginalized_rv: marginalized_vv}
    logps_dict = conditional_logp(rv_values=rv_values, **kwargs)

    # Reduce logp dimensions corresponding to broadcasted variables
    marginalized_logp = logps_dict.pop(marginalized_vv)
    joint_logp = marginalized_logp + reduce_batch_dependent_logps(
        dependent_dims_connections=op.dims_connections,
        dependent_ops=[inner_rv.owner.op for inner_rv in inner_rvs],
        dependent_logps=[logps_dict[value] for value in values],
    )

    # Compute the joint_logp for all possible n values of the marginalized RV. We assume
    # each original dimension is independent so that it suffices to evaluate the graph
    # n times, once with each possible value of the marginalized RV replicated across
    # batched dimensions of the marginalized RV

    # PyMC does not allow RVs in the logp graph, even if we are just using the shape
    marginalized_rv_shape = constant_fold(tuple(marginalized_rv.shape), raise_not_constant=False)
    marginalized_rv_domain = get_domain_of_finite_discrete_rv(marginalized_rv)
    marginalized_rv_domain_tensor = pt.moveaxis(
        pt.full(
            (*marginalized_rv_shape, len(marginalized_rv_domain)),
            marginalized_rv_domain,
            dtype=marginalized_rv.dtype,
        ),
        -1,
        0,
    )

    try:
        joint_logps = vectorize_graph(
            joint_logp, replace={marginalized_vv: marginalized_rv_domain_tensor}
        )
    except Exception:
        # Fallback to Scan
        def logp_fn(marginalized_rv_const, *non_sequences):
            return graph_replace(joint_logp, replace={marginalized_vv: marginalized_rv_const})

        joint_logps = scan_map(
            fn=logp_fn,
            sequences=marginalized_rv_domain_tensor,
            non_sequences=[*values, *inputs],
            mode=Mode().including("local_remove_check_parameter"),
            return_updates=False,
        )

    joint_logp = pt.logsumexp(joint_logps, axis=0)

    # Align logp with non-collapsed batch dimensions of first RV
    joint_logp = align_logp_dims(dims=op.dims_connections[0], logp=joint_logp)

    warn_non_separable_logp(values)
    # We have to add dummy logps for the remaining value variables, otherwise PyMC will raise
    dummy_logps = (DUMMY_ZERO,) * (len(values) - 1)
    return joint_logp, *dummy_logps


def _compute_hmm_emission_logp(chain_rv, dependent_rvs, dims_connections, values):
    """Compute emission log-probabilities for every possible state at each time step.

    Parameters
    ----------
    chain_rv : TensorVariable
        The DiscreteMarkovChain random variable.
    dependent_rvs : list of TensorVariable
        Dependent (emission) random variables.
    dims_connections : list of tuple
        Batch dimension mappings between chain and each dependent RV.
    values : list of TensorVariable
        Value variables for the dependent RVs.

    Returns
    -------
    batch_logp_emissions : TensorVariable
        Shape (n_states, *batch_dims, n_steps). Logp of emissions for each state at each step.
    batch_chain_value : TensorVariable
        Shape (n_states, *chain_shape). Domain values broadcast to chain shape, used for
        vectorizing init-distribution logp.
    """
    P = chain_rv.owner.inputs[0]
    domain = pt.arange(P.shape[-1], dtype="int32")

    # Break dependency between chain and dependent RVs
    chain_value = chain_rv.clone()
    dependent_rvs_clone = clone_replace(dependent_rvs, {chain_rv: chain_value})
    logp_emissions_dict = conditional_logp(dict(zip(dependent_rvs_clone, values)))

    # Reduce and add the batch dims beyond the chain dimension
    reduced_logp_emissions = reduce_batch_dependent_logps(
        dependent_dims_connections=dims_connections,
        dependent_ops=[dependent_rv.owner.op for dependent_rv in dependent_rvs_clone],
        dependent_logps=[logp_emissions_dict[value] for value in values],
    )

    # Vectorize over the domain of the chain
    chain_shape = constant_fold(tuple(chain_rv.shape))
    batch_chain_value = pt.moveaxis(pt.full((*chain_shape, domain.size), domain), -1, 0)
    batch_logp_emissions = vectorize_graph(reduced_logp_emissions, {chain_value: batch_chain_value})

    return batch_logp_emissions, batch_chain_value


def _hmm_forward_backward(batch_logp_emissions, batch_logp_init_dist, log_P, return_samples=False):
    """Run forward-backward on precomputed emission log-probabilities.

    Parameters
    ----------
    batch_logp_emissions : TensorVariable
        Shape (n_states, *batch, n_steps). Emission log-probs for each state and step.
    batch_logp_init_dist : TensorVariable
        Shape (n_states, *batch). Initial state log-probabilities.
    log_P : TensorVariable
        Shape (n_states, n_states, *batch). Log transition matrix.
    return_samples : bool
        If True, also return FFBS samples.

    Returns
    -------
    log_gamma : TensorVariable
        Posterior marginal log-probabilities: (*batch, n_steps, n_states).
    samples : TensorVariable (if return_samples=True)
        Sampled state sequences: (*batch, n_steps).
    """
    from pytensor.tensor.special import log_softmax

    # ----- Forward pass (alpha) -----
    # alpha_0(j) = log pi_j + log b_j(o_0)
    log_alpha_0 = batch_logp_init_dist + batch_logp_emissions[..., 0]

    def step_alpha(logp_emission, log_alpha, log_P):
        step_log_prob = pt.logsumexp(log_alpha[:, None, ...] + log_P, axis=0)
        return logp_emission + step_log_prob

    log_alpha_seq = scan(
        step_alpha,
        non_sequences=[log_P],
        outputs_info=[log_alpha_0],
        sequences=pt.moveaxis(batch_logp_emissions[..., 1:], -1, 0),
        return_updates=False,
    )
    log_alphas = pt.concatenate([log_alpha_0[None, ...], log_alpha_seq], axis=0)

    # ----- Backward pass (beta) -----
    log_beta_T = pt.zeros_like(log_alpha_0)

    def step_beta(logp_emission, log_beta, log_P):
        return pt.logsumexp(log_P + logp_emission[None, ...] + log_beta[None, ...], axis=1)

    rev_emissions = batch_logp_emissions[..., :0:-1]

    log_beta_seq = scan(
        step_beta,
        non_sequences=[log_P],
        outputs_info=[log_beta_T],
        sequences=pt.moveaxis(rev_emissions, -1, 0),
        return_updates=False,
    )
    log_betas = pt.concatenate([log_beta_seq[::-1], log_beta_T[None, ...]], axis=0)

    # ----- Posterior marginals (gamma) -----
    log_gamma = log_alphas + log_betas
    log_gamma = log_softmax(log_gamma, axis=1)
    log_gamma = pt.moveaxis(pt.moveaxis(log_gamma, 1, -1), 0, -2)

    if not return_samples:
        return log_gamma

    # ----- FFBS -----
    ffbs_rng = pytensor.shared(np.random.default_rng())

    p_last = log_softmax(log_alphas[-1], axis=0)
    _, last_state = Categorical.dist(logit_p=p_last.T, rng=ffbs_rng, return_next_rng=True)

    rev_alphas = log_alphas[:-1][::-1]

    def step_ffbs(log_alpha_t, rng, prev_state, log_P):
        log_trans_to_prev = log_P[:, prev_state]
        log_prob = log_softmax(log_alpha_t + log_trans_to_prev, axis=0)
        next_rng, state = Categorical.dist(logit_p=log_prob.T, rng=rng, return_next_rng=True)
        return next_rng, state

    _, rev_states = scan(
        step_ffbs,
        non_sequences=[log_P],
        outputs_info=[ffbs_rng, last_state],
        sequences=[rev_alphas],
        return_updates=False,
    )
    states = pt.concatenate([rev_states[::-1], last_state[None, ...]], axis=0)
    samples = pt.moveaxis(states, 0, -1)

    return log_gamma, samples


def _prepare_hmm_quantities(chain_rv, batch_chain_value):
    """Compute log initial distribution and log transition matrix."""
    P, n_steps_, init_dist_, rng = chain_rv.owner.inputs
    chain_shape = constant_fold(tuple(chain_rv.shape))

    # Initial distribution logp
    init_dist_value = init_dist_.type()
    logp_init_dist = logp(init_dist_, init_dist_value)
    batch_logp_init_dist = vectorize_graph(
        logp_init_dist, {init_dist_value: batch_chain_value[..., :1]}
    ).squeeze(-1)

    # Log transition matrix: move core dims to front
    P = pt.atleast_Nd(P, n=len(chain_shape) + 1)
    P = pt.moveaxis(P, (-2, -1), (0, 1))
    log_P = pt.log(P)

    return batch_logp_init_dist, log_P


def hmm_posterior_marginals(
    chain_rv, dependent_rvs, dims_connections, dep_values, *, return_samples=False
):
    """Compute posterior state marginals and optionally samples for a DiscreteMarkovChain.

    Parameters
    ----------
    chain_rv : TensorVariable
        The DiscreteMarkovChain random variable.
    dependent_rvs : list of TensorVariable
        Dependent (emission) random variables.
    dims_connections : list of tuple
        Batch dimension mappings from chain to each dependent RV.
    dep_values : list of TensorVariable
        Value variables for the dependent RVs.
    return_samples : bool
        If True, also return sampled state sequences via FFBS.

    Returns
    -------
    log_gamma : TensorVariable
        Posterior marginal log-probabilities: (*batch, n_steps, n_states).
    samples : TensorVariable (if return_samples=True)
        Sampled state sequences: (*batch, n_steps).
    """
    batch_logp_emissions, batch_chain_value = _compute_hmm_emission_logp(
        chain_rv, dependent_rvs, dims_connections, dep_values
    )
    batch_logp_init_dist, log_P = _prepare_hmm_quantities(chain_rv, batch_chain_value)
    return _hmm_forward_backward(
        batch_logp_emissions, batch_logp_init_dist, log_P, return_samples=return_samples
    )


@_logprob.register(MarginalDiscreteMarkovChainRV)
def marginal_hmm_logp(op, values, *inputs, **kwargs):
    chain_rv, *dependent_rvs = inline_ofg_outputs(op, inputs)

    P, n_steps_, init_dist_, rng = chain_rv.owner.inputs
    chain_shape = constant_fold(tuple(chain_rv.shape))

    # Step 1: Compute emission log-probabilities
    batch_logp_emissions, batch_chain_value = _compute_hmm_emission_logp(
        chain_rv, dependent_rvs, op.dims_connections, values
    )

    # Step 2: Compute the transition probabilities via the forward algorithm
    # alpha_t = p(y | s_t) * sum_{s_{t-1}}(p(s_t | s_{t-1}) * alpha_{t-1})

    # Compute initial logp
    init_dist_value = init_dist_.type()
    logp_init_dist = logp(init_dist_, init_dist_value)
    batch_logp_init_dist = vectorize_graph(
        logp_init_dist, {init_dist_value: batch_chain_value[..., :1]}
    ).squeeze(-1)
    log_alpha_init = batch_logp_init_dist + batch_logp_emissions[..., 0]

    def step_alpha(logp_emission, log_alpha, log_P):
        step_log_prob = pt.logsumexp(log_alpha[:, None, ...] + log_P, axis=0)
        return logp_emission + step_log_prob

    # Add implicit dimensions of P, and place core dimensions at the front
    P = pt.atleast_Nd(P, n=len(chain_shape) + 1)
    P = pt.moveaxis(P, (-2, -1), (0, 1))
    log_P = pt.log(P)

    log_alpha_seq = scan(
        step_alpha,
        non_sequences=[log_P],
        outputs_info=[log_alpha_init],
        sequences=pt.moveaxis(batch_logp_emissions[..., 1:], -1, 0),
        return_updates=False,
    )
    # Final logp is just the sum of the last scan state
    joint_logp = pt.logsumexp(log_alpha_seq[-1], axis=0)

    # Align logp with non-collapsed batch dimensions of first RV
    remaining_dims_first_emission = list(op.dims_connections[0])
    # The last dim of chain_rv was removed when computing the logp
    remaining_dims_first_emission.remove(chain_rv.type.ndim - 1)
    joint_logp = align_logp_dims(remaining_dims_first_emission, joint_logp)

    # If there are multiple emission streams, we have to add dummy logps for the remaining value variables. The first
    # return is the joint probability of everything together, but PyMC still expects one logp for each emission stream.
    warn_non_separable_logp(values)
    dummy_logps = (DUMMY_ZERO,) * (len(values) - 1)
    return joint_logp, *dummy_logps
