# Plan: Add DiscreteMarkovChain support to `recover_marginals`

## Status: Branch `recover_hmm` — implementation exists, needs tests and polish

The core implementation already exists on this branch. The changes span three areas:

1. **`marginal_model.py`**: New `is_markov_chain` branch in `recover_marginals` (lines 451–504) that uses forward-backward instead of brute-force enumeration.
2. **`distributions.py`**: Refactored HMM internals into reusable helpers (`_compute_hmm_emission_logp`, `_hmm_forward_backward`, `_prepare_hmm_quantities`) plus a new public `hmm_posterior_marginals` entry point that can return both log-marginals and FFBS samples.
3. **No tests yet** — the test file is untouched versus `main`.

Below is a phased plan to finish and land this feature.

---

## Phase 1: Refactoring (already done)

### 1.1 Extract reusable HMM building blocks

The old `marginal_hmm_logp` (logp handler for `MarginalDiscreteMarkovChainRV`) was monolithic: emission computation, initial-distribution processing, and the forward algorithm were all in one function. These were split into three helpers:

| Helper | Purpose |
|--------|---------|
| `_compute_hmm_emission_logp(chain_rv, dependent_rvs, dims_connections, values)` | Returns `(batch_logp_emissions, domain)` — a `(n_states, *batch, n_steps)` tensor of log-emission-probs for every possible state at every time step. |
| `_prepare_hmm_quantities(chain_rv, batch_logp_emissions)` | Returns `(batch_logp_init_dist, log_P)` — the vectorized log-initial-distribution and the log-transition-matrix with core dims moved to the front. |
| `_hmm_forward_backward(batch_logp_emissions, batch_logp_init_dist, log_P, return_samples)` | Runs the full forward-backward algorithm. If `return_samples=True`, also runs forward-filtering backward-sampling (FFBS). |

The refactored `marginal_hmm_logp` now simply calls `_compute_hmm_emission_logp` + `_prepare_hmm_quantities` + a scan-based forward pass, reusing exactly the same building blocks that `recover_marginals` uses.

### 1.2 New public entry point: `hmm_posterior_marginals`

```python
def hmm_posterior_marginals(chain_rv, dependent_rvs, dims_connections,
                            dep_values, *, return_samples=False):
```

This is the single call `recover_marginals` uses. It orchestrates all three helpers:

```
hmm_posterior_marginals
  ├── _compute_hmm_emission_logp   → batch_logp_emissions
  ├── _prepare_hmm_quantities       → batch_logp_init_dist, log_P
  └── _hmm_forward_backward        → log_gamma [, samples]
```

### 1.3 New code path in `recover_marginals`

The existing finite-discrete recovery path enumerates the domain (brute-force, `vectorize_graph` over all possible values). For chains, that would be `O(K^T)` — exponential in sequence length and completely infeasible.

The new path (lines 451–504):

1. **Same preamble** as the finite-discrete path: `unmarginalize`, find dependents, find other-marginalized, re-marginalize everything except the target RV.
2. **Get the chain and dependent RVs from that re-marginalized model.**
3. **Compute `dims_connections`** via `subgraph_batch_dim_connection` — tells us how batch axes map from chain to each dependent.
4. **Call `hmm_posterior_marginals`** — returns `log_gamma` (posterior log-marginals, shape `(*batch, n_steps, n_states)`) and optionally FFBS samples (shape `(*batch, n_steps)`).
5. **Compile** the resulting graph via `compile_pymc`.
6. **Evaluate** on each posterior point from the MCMC chains.
7. **Reshape** and merge into the `DataTree` — same logic as the finite-discrete path.

---

## Phase 2: Tests to write

No tests exist for `recover_marginals` on `DiscreteMarkovChain`. Need at minimum:

### 2.1 Basic HMM recovery (Normal emissions)

```python
def test_recover_marginals_hmm_normal():
    """Recover states for a 2-state HMM with Normal emissions."""
    with pm.Model() as m:
        P = pm.Dirichlet("P", a=np.ones((2, 2)))
        init_dist = pm.Dirichlet("init_dist", a=np.ones(2))
        chain = DiscreteMarkovChain("chain", P=P, init_dist=init_dist, steps=5)
        mu = pm.Normal("mu", mu=chain, sigma=1, shape=(5,))
        y = pm.Normal("y", mu=mu, sigma=0.1, observed=[0, 1, 0, 1, 0])

    marg_m = marginalize(m, [chain])
    with marg_m:
        idata = pm.sample(draws=50, tune=50, chains=1, random_seed=42)

    idata = recover_marginals(idata, return_samples=True)
    post = idata.posterior
    assert "chain" in post
    assert "lp_chain" in post
    assert post.chain.shape == (1, 50, 5)
    assert post.lp_chain.shape == (1, 50, 5, 2)  # 2 states
```

### 2.2 HMM with Categorical emissions

```python
def test_recover_marginals_hmm_categorical():
    """Recover states for an HMM with Categorical emissions."""
    with pm.Model() as m:
        P = pm.Dirichlet("P", a=np.ones((3, 3)))
        init_dist = pm.Dirichlet("init_dist", a=np.ones(3))
        chain = DiscreteMarkovChain("chain", P=P, init_dist=init_dist, steps=10)
        # 3x4 emission matrix
        emit = pm.Dirichlet("emit", a=np.ones((3, 4)))
        y = pm.Categorical("y", p=emit[chain], observed=np.random.randint(0, 4, 10))

    marg_m = marginalize(m, [chain])
    with marg_m:
        idata = pm.sample(draws=50, tune=50, chains=1, random_seed=42)

    idata = recover_marginals(idata, return_samples=True)
    post = idata.posterior
    assert post.lp_chain.shape == (1, 50, 10, 3)
```

### 2.3 HMM with multiple emission streams

```python
def test_recover_marginals_hmm_multiple_emissions():
    """Two independent emission streams depending on the same chain."""
    with pm.Model() as m:
        P = pm.Dirichlet("P", a=np.ones((2, 2)))
        init_dist = pm.Dirichlet("init_dist", a=np.ones(2))
        chain = DiscreteMarkovChain("chain", P=P, init_dist=init_dist, steps=5)
        mu1 = pm.Normal("mu1", mu=chain, sigma=1, shape=(5,))
        mu2 = pm.Normal("mu2", mu=2 * chain - 1, sigma=0.5, shape=(5,))

    marg_m = marginalize(m, [chain])
    with marg_m:
        idata = pm.sample(draws=50, tune=50, chains=1, random_seed=42)

    idata = recover_marginals(idata, return_samples=True)
    post = idata.posterior
    assert "chain" in post
    assert post.chain.shape == (1, 50, 5)
```

### 2.4 Batched HMM

```python
def test_recover_marginals_hmm_batched():
    """Recover from a batched HMM with shape=(batch, steps)."""
    with pm.Model() as m:
        P = pm.Dirichlet("P", a=np.ones((2, 2)))
        init_dist = pm.Dirichlet("init_dist", a=np.ones(2))
        chain = DiscreteMarkovChain("chain", P=P, init_dist=init_dist, shape=(3, 5))
        # chain has shape (3, 5), mu has shape (5, 3) — transposed batch dims
        mu = pm.Normal("mu", mu=chain.T, sigma=0.5, shape=(5, 3))

    marg_m = marginalize(m, [chain])
    with marg_m:
        idata = pm.sample(draws=50, tune=50, chains=1, random_seed=42)

    idata = recover_marginals(idata, return_samples=True)
    post = idata.posterior
    assert post.chain.shape == (1, 50, 3, 5)
    assert post.lp_chain.shape == (1, 50, 3, 5, 2)
```

### 2.5 dims preservation

```python
def test_recover_marginals_hmm_dims():
    """Named dims should be preserved."""
    with pm.Model(coords={"step": np.arange(7)}) as m:
        P = pm.Dirichlet("P", a=np.ones((2, 2)))
        init_dist = pm.Dirichlet("init_dist", a=np.ones(2))
        chain = DiscreteMarkovChain("chain", P=P, init_dist=init_dist, dims=["step"])
        mu = pm.Normal("mu", mu=chain, sigma=1, dims=["step"])

    marg_m = marginalize(m, [chain])
    with marg_m:
        idata = pm.sample(draws=10, tune=10, chains=1, random_seed=42)

    idata = recover_marginals(idata, return_samples=True)
    post = idata.posterior
    assert post.chain.dims == ("chain", "draw", "step")
    assert post.lp_chain.dims == ("chain", "draw", "step", "lp_chain_dim")
```

### 2.6 No-sample mode (`return_samples=False`)

Verify that only `lp_chain` is produced and `chain` is absent.

### 2.7 Analytical correctness (ground truth)

For a simple 2-state model where the truth can be computed analytically with a small number of draws, compare `post.lp_chain` against manually computed forward-backward results.

---

## Phase 3: Bugs and rough edges to fix

### 3.1 FFBS RNG seeding ✅ FIXED

`_hmm_forward_backward` previously hardcoded:

```python
ffbs_rng = pytensor.shared(np.random.default_rng(0))
ffbs_rng, last_state = Categorical.dist(logit_p=p_last.T, rng=ffbs_rng, return_next_rng=True)
```

This bypassed `compile_pymc(random_seed=seed)` — the hardcoded seed 0 made FFBS samples deterministic regardless of the `random_seed` argument.

**Fix** (applied): Follow the same pattern as the finite discrete path at `marginal_model.py:571`, which simply calls `Categorical.dist(logit_p=...)` without an explicit `rng=`, letting `compile_pymc` seed the default RNG stream:

```python
# No explicit rng= — uses default RNG, seeded by compile_pymc
init_rng, last_state = Categorical.dist(logit_p=p_last.T, return_next_rng=True)

# init_rng (the advanced RNG state) is fed into scan's outputs_info
# step_ffbs continues to thread it internally with rng=rng
```

The `step_ffbs` inner function still uses `rng=rng` — that's correct, it's threading the RNG state *from the previous iteration*, which originally came from the default stream.

### 3.2 `_hmm_forward_backward` backward pass: off-by-one check

The backward pass reverses emissions with `batch_logp_emissions[..., :0:-1]`. Double-check with a 3-step test case (T=3) that the beta sequence is correct and matches standard HMM literature. The slice `:0:-1` gives `[T-1, ..., 1]` (excludes index 0), which aligns with `log_beta_T` being zeros. But verify that `log_beta_seq[::-1]` concatenated with `log_beta_T` produces exactly T betas.

### 3.3 Batched FFBS shape correctness

In batched HMMs, the FFBS `step_ffbs` uses `log_P[:, prev_state]`. This selects the column for a given `prev_state`, which is a scalar per batch element. Confirm this broadcasts correctly with:

```
log_P shape:     (n_states, n_states, *batch)
prev_state shape: (*batch,)
```

The indexing `log_P[:, prev_state]` should give shape `(n_states, *batch)` — correct for the Categorical logits.

### 3.4 Mixed marginalized variables (chain + regular discrete)

Test that `recover_marginals` works when a `DiscreteMarkovChain` and a regular `Bernoulli` are both marginalized in the same model. The `other_marginalized_rvs_names` logic handles this, but no test exercises it.

---

## Phase 4: Documentation

### 4.1 Docstring for `recover_marginals`

Already adequate — mentions `DiscreteMarkovChain` in the error message. Consider adding an explicit HMM example to the docstring.

### 4.2 User-facing docs

Add a notebook or `.rst` page in `docs/` showing an end-to-end HMM example:

```
Model → marginalize → sample → recover_marginals → posterior state analysis
```

### 4.3 Notes on limitations

- Only `n_lags=1` is supported (already enforced in `marginalize`)
- Transition matrix must be 2D (no batched P with `shape=(n_states, n_states, batch)`) — verify if this is still true
- `return_samples=True` requires that the model's dependency graph is separable (no non-separable logp warnings)

---

---

## Dev notes

This project uses a pixi environment for development:

```
# Python
~/upstream/pixi-envs/pymc-dev/.pixi/envs/default/bin/python

# pytest
~/upstream/pixi-envs/pymc-dev/.pixi/envs/default/bin/pytest
```

Prefix both with the venv path or activate via `pixi shell` inside `~/upstream/pixi-envs/pymc-dev/`.

## Algorithm Summary

The HMM recovery path avoids the O(K^T) brute-force enumeration used for ordinary discrete variables. Instead:

```
For each posterior draw (params):

  1. Compute emission log-probs for each state at each time step
     log b_j(o_t) for j=1..K, t=1..T
     → tensor of shape (K, *batch, T)

  2. Vectorize initial distribution logp over batch
     log π_j for j=1..K
     → tensor of shape (K, *batch)

  3. Forward pass (alpha):
     α_1(j) = log π_j + log b_j(o_1)
     α_t(j) = log b_j(o_t) + logsumexp_i [log P(i→j) + α_{t-1}(i)]
     → O(T·K²) via pytensor scan

  4. Backward pass (beta):
     β_T(j) = 0
     β_t(j) = logsumexp_i [log P(j→i) + log b_i(o_{t+1}) + β_{t+1}(i)]
     → O(T·K²) via pytensor scan

  5. Posterior marginals:
     log γ_t(j) = α_t(j) + β_t(j)   (then log_softmax over j)

  6. (Optional) FFBS sampling:
     Draw s_T ~ softmax(α_T)
     for t = T-1 down to 1:
       draw s_t ~ softmax(α_t + log P(·→ s_{t+1}))
     → O(T·K) via pytensor scan

Reshape results into xarray dataset and merge into InferenceData.
```

---

## Design Note: Why independent forward and backward passes?

`_hmm_forward_backward` computes alpha (forward) and beta (backward) as two **independent** `pytensor.scan` passes, combining them only at the end: `log_gamma = log_alphas + log_betas`. Neither pass references the other's intermediate state.

Some formulations derive the posterior marginals (gamma) as a "derivative" of the backward algorithm — computing beta only and expressing gamma implicitly from it, without materializing alpha. This code deliberately avoids that approach for three reasons:

1. **FFBS needs alpha anyway.** When `return_samples=True`, the forward-filtering backward-sampling step draws `s_t` proportional to `α_t(i) · log P(i→s_{t+1})`. Even if gamma were derived from beta alone, the forward quantities would still have to be materialized for sampling. There's no way around storing alpha.

2. **No computational savings.** Both the forward and backward scans are O(T·K²). Deriving alpha from beta (or gamma from beta alone) doesn't reduce the asymptotic work — you'd still be doing the same logsumexp operations, just organized differently. The two-scan approach is already optimal.

3. **Clarity.** Independent passes map directly to the canonical HMM literature (Rabiner 1989). The alpha recursion (`α_t = emission × P^T × α_{t-1}`) and beta recursion (`β_t = P × emission × β_{t+1}`) are instantly recognizable and verifiable against any HMM reference.

The only coupling between the passes is at the gamma combination step, which is a trivial elementwise addition in log-space.
