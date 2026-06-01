# How `recover_marginals` Works — Detailed Explanation

## Overview

`recover_marginals` is a post-hoc recovery utility. After you **sample** from a marginalized PyMC model (one where some discrete variables have been analytically integrated out), you no longer have posterior samples for those marginalized variables. `recover_marginals` reconstructs them: for each marginalized variable, it computes the **posterior log-probability** (given the sampled continuous parameters) and optionally draws **posterior samples** via the conditional distribution P(marginalized_var | everything_else).

It lives in `pymc_extras/model/marginal/marginal_model.py`.

---

## Step-by-Step Walkthrough

### Step 1: Argument handling and migration guard

```python
if isinstance(idata, Model):
    raise TypeError("The order of arguments … changed. The first input must be an idata")
```

This is a temporary migration guard. In older versions the model came first; now the inference data (`idata`) comes first.

### Step 2: Unmarginalize the model

```python
unmarginal_model = unmarginalize(model)
```

This is key. The sampling model only contains `MarginalRV` ops — black-box Ops that wrap the original graph and expose a marginal log-probability. `unmarginalize` inlines (unwraps) all those `MarginalRV` Ops back into explicit `FreeRV` nodes, recovering the **original full graph** with all discrete variables present. From this unmarginalized model, we can discover:

```python
model_var_names = set(rv.name for rv in model.free_RVs)
marginalized_rv_names = [
    rv.name for rv in unmarginal_model.free_RVs if rv.name not in model_var_names
]
```

### Step 3: Extract posterior points from the InferenceData

```python
posterior_pts, stacked_dims = dataset_to_point_list(
    idata["posterior"].dataset[[rv.name for rv in model.free_RVs]],
    sample_dims=("chain", "draw"),
)
transformed_posterior_pts = transform_posterior_pts(model, posterior_pts)
```

- `dataset_to_point_list` converts the xarray Dataset into a flat list of dictionaries — one per (chain, draw) combination — each dictionary mapping variable names to NumPy values.
- `transform_posterior_pts` applies the model's **transforms** (e.g., log for HalfNormal) to convert sampled values back to the **unconstrained/transformed** space. This is necessary because the log-probability computation operates in transformed space.

### Step 4: Iterate over each marginalized variable

For each marginalized variable, there are **two distinct code paths**: one for finite discrete distributions (`Bernoulli`, `Categorical`, `DiscreteUniform`) and one for `DiscreteMarkovChain` (HMMs).

#### Path A: Finite Discrete Variables (Bernoulli, Categorical, DiscreteUniform)

##### 4A.1: Find dependent RVs and re-marginalize other variables

```python
dependent_rvs = [rv for rv in find_conditional_dependent_rvs(...)
    if rv.name not in other_marginalized_rvs_names]
marginalized_model = marginalize(unmarginal_model, other_marginalized_rvs_names)
```

If there are multiple marginalized variables (e.g., nested indexing `idx → sub_idx`), we want the **conditional** P(var | other_marginalized_vars, params). So we re-marginalize everything *except* the target variable. This gives us a model where only one variable is marginalized, and we can compute its conditional log-probability via the standard logp machinery.

##### 4A.2: Compute the joint log-probability over the domain

```python
logps = marginalized_model.logp(vars=[marginalized_var_to_recover, *dependent_rvs], sum=False)
marginalized_logp, *dependent_logps = logps
joint_logp = marginalized_logp + reduce_batch_dependent_logps(...)
```

`model.logp(sum=False)` returns elementwise log-probabilities. The marginalized variable's logp and the dependent variables' logps are added together elementwise. `reduce_batch_dependent_logps` handles the alignment of batch dimensions — e.g., if the marginalized variable has shape `(3, 2)` and the dependent has shape `(7, 2, 3)`, it sums over the extra batch dimension and transposes to align.

##### 4A.3: Enumerate over the domain

```python
rv_domain = get_domain_of_finite_discrete_rv(var_to_recover)  # e.g. (0, 1) for Bernoulli
rv_domain_tensor = pt.moveaxis(pt.full((*rv_shape, len(rv_domain)), rv_domain, ...), -1, 0)
batched_joint_logp = vectorize_graph(joint_logp, replace={marginalized_value: rv_domain_tensor})
joint_logp_norm = log_softmax(batched_joint_logp, axis=-1)
```

`vectorize_graph` replaces the marginalized value variable with a tensor that enumerates all possible values along a new axis (e.g., for Bernoulli shape `(3,2)`: a `(2, 3, 2)` tensor of `[0, 1]` values repeated). The result is the **unnormalized log-joint** for each possible state. `log_softmax` normalizes across that new axis to produce proper conditional log-probabilities.

##### 4A.4: Optionally draw samples

```python
rv_draws = Categorical.dist(logit_p=batched_joint_logp)
```

Samples are drawn from a Categorical distribution parameterized by the unnormalized log-probabilities.

##### 4A.5: Compile and evaluate

```python
rv_loglike_fn = compile_pymc(inputs=other_values, outputs=outputs, ...)
logvs = [rv_loglike_fn(**vs) for vs in transformed_posterior_pts]
```

A compiled PyTensor function is evaluated on each posterior point, producing log-probabilities and optional samples.

#### Path B: DiscreteMarkovChain (HMMs)

##### 4B.1: Same setup — find dependents, re-marginalize others

##### 4B.2: Use the Forward-Backward algorithm

```python
outputs = hmm_posterior_marginals(chain_rv=chain_rv, dependent_rvs=...,
    dims_connections=..., dep_values=dep_values, return_samples=return_samples)
```

`hmm_posterior_marginals` in `distributions.py`:
1. Computes **emission log-probabilities** for each time step via `_compute_hmm_emission_logp`
2. Extracts the transition matrix `P` and initial distribution
3. Runs the **forward-backward algorithm** to compute posterior marginal probabilities `P(S_t = s | observations, params)` for every time step and every state
4. Optionally runs **forward-filtering backward-sampling (FFBS)** to draw state sequences

This is vastly more efficient than brute-force enumeration (which would be O(K^T) for T time steps and K states). Forward-backward is O(T·K²).

##### 4B.3: Compile and evaluate (same as Path A)

### Step 5: Reshape results into xarray dataset

```python
rv_dict[var_name] = samples.reshape(tuple(len(coord) for coord in stacked_dims.values()) + samples.shape[1:])
rv_dict["lp_" + var_name] = logps.reshape(...)
```

Results are reshaped from flat (chain, draw) lists back into proper multi-indexed arrays with appropriate dimensions. The log-probability variable gets the suffix `lp_` and has an extra dimension representing the domain of the discrete variable.

### Step 6: Merge into InferenceData

```python
rv_dataset = dict_to_dataset(rv_dict, ...)
idata["posterior"] = idata["posterior"].assign(rv_dataset)
```

The new variables (`idx`, `lp_idx`, etc.) are merged back into the posterior group of the original InferenceData tree.

---

## Key Design Insights

1. **Unmarginalize → re-marginalize pattern**: For multiple marginalized variables, each variable is conditioned on the others by re-marginalizing everything else. This turns a joint problem into a sequence of conditional ones.

2. **Transformed space**: All log-probability computation happens in the *transformed* (unconstrained) space. This is standard PyMC convention but is important to note — the returned log-probabilities are "transformed-space" log-probabilities.

3. **Batch dimension handling**: `subgraph_batch_dim_connection` and `reduce_batch_dependent_logps` handle the alignment of batch axes between marginalized variables and their dependent RVs, which is the trickiest part of the implementation.

4. **Efficiency**: For simple discrete variables, domain enumeration is O(K) per draw. For HMMs, the forward-backward algorithm avoids exponential blowup.

5. **Named dims**: When the marginalized variable has named dims (e.g., `dims="year"`), these are preserved in the output dataset, ensuring the recovered variables slot naturally into the existing coordinates.
