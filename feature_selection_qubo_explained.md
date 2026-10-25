# Feature Selection as a QUBO Problem

## 1. What problem are we solving?

Feature selection chooses a useful subset of candidate explanatory variables
before fitting a predictive model. In a credit-risk application, the outcome is
whether a borrower defaults, and the candidate features represent information
that may help predict default.

Using every available feature is not necessarily desirable:

- Some features contain little information about the outcome.
- Highly correlated features can repeat essentially the same information.
- Large feature sets can increase data-collection costs, model complexity, and
  the risk of overfitting.

The goal here is to choose approximately **k informative, nonredundant features
from p candidates**. The implementation balances predictive relevance,
redundancy, and the desired number of selected features.

This is a filter-based selection method: it scores a subset using statistical
relationships in the data, rather than training and evaluating a classifier
inside the optimization loop.

## 2. Data and binary decision variables

Let:

- `X` be an `N x p` matrix: N observations and p candidate features.
- `y` be a binary default indicator: 1 for default and 0 otherwise.
- `k` be the target number of features.

For each candidate feature i, define:

$$
x_i =
\begin{cases}
1, & \text{if feature } i \text{ is selected},\\
0, & \text{otherwise}.
\end{cases}
$$

The selected subset is \(S = \{i : x_i = 1\}\), with cardinality
\(|S| = \sum_{i=1}^{p} x_i\).

There are p binary decision variables, not N. Increasing N changes the
statistical estimates used to build an instance; increasing p changes the
dimension of the optimization problem. There are \(2^p\) possible subsets.

## 3. Relevance, redundancy, and subset size

### Feature relevance

Each feature has a relevance score \(R_i\). In the saved instances, this is its
estimated mutual information with the default indicator, normalized by the
largest feature relevance within that instance:

$$
R_i =
\frac{\widehat{I}(X_i;y)}
     {\max_j \widehat{I}(X_j;y)}.
$$

If all estimated mutual information scores are zero, they remain zero instead
of being divided by zero. Otherwise, normalized relevance lies between 0 and 1.
The implementation uses `mutual_info_classif` with three neighbors and
`random_state=42`.

Higher relevance is rewarded. The generator also supports absolute Pearson
correlation with the target as an alternative, but the saved 36-instance set
uses mutual information.

### Feature redundancy

For each pair of features, define:

$$
C_{ij} = |\operatorname{corr}(X_i, X_j)|.
$$

Pairs with large absolute Pearson correlation are penalized when both features
are selected. Positive and negative correlations both count as redundancy.
Undefined correlations, such as those involving a constant feature, are
replaced with zero by the generator.

### Cardinality penalty

The number of selected features is encouraged to stay near k through:

$$
\gamma \left(\sum_{i=1}^{p}x_i-k\right)^2.
$$

This penalty is zero at exactly k selected features and positive otherwise.
It is a **soft penalty**, not a hard equality constraint: a finite gamma does
not guarantee that the returned subset contains exactly k features.

## 4. The feature-selection objective

The intended objective is:

$$
\min_{x \in \{0,1\}^{p}} F(x)
= -\alpha \sum_{i=1}^{p} R_i x_i
  + \beta \sum_{i<j} C_{ij}x_ix_j
  + \gamma \left(\sum_{i=1}^{p}x_i-k\right)^2.
$$

| Component | Role |
|---|---|
| Negative relevance term | Rewards selecting informative features |
| Positive redundancy term | Discourages selecting correlated pairs |
| Squared cardinality term | Discourages selecting too many or too few features |

The saved instances use:

| Parameter | Value |
|---|---:|
| Relevance weight, alpha | 1.0 |
| Redundancy weight, beta | 0.8 |
| Cardinality penalty, gamma | 2.5 |

These weights define the optimization trade-off; they are not fitted
classification-model coefficients.

## 5. Converting the objective into QUBO coefficients

QUBO means **Quadratic Unconstrained Binary Optimization**. The cardinality
requirement appears as a penalty in the objective, so the only explicit
restriction is that every decision variable is binary.

Because \(x_i^2=x_i\) for binary variables:

$$
\left(\sum_i x_i-k\right)^2
= (1-2k)\sum_i x_i + 2\sum_{i<j}x_ix_j + k^2.
$$

Substitution gives an upper-triangular QUBO:

$$
f_Q(x) = \sum_{i=1}^{p} Q_{ii}x_i
       + \sum_{i<j} Q_{ij}x_ix_j,
$$

with:

$$
Q_{ii} = -\alpha R_i + \gamma(1-2k),
$$

$$
Q_{ij} = \beta C_{ij} + 2\gamma \quad \text{for } i<j,
$$

and \(Q_{ij}=0\) for \(i>j\).

The generator omits the constant term \(\gamma k^2\), because adding a constant
cannot change which subset minimizes the objective. Consequently:

$$
F(x)=f_Q(x)+\gamma k^2.
$$

The solver reports \(f_Q(x)\), not F(x). Add \(\gamma k^2\) when the full
penalized objective is needed. Neither value is a prediction accuracy,
probability, or monetary loss.

**Coefficient convention matters:** each exported off-diagonal coefficient
already contains the complete contribution for its pair. It must be counted
once, not doubled. If constructing a symmetric matrix for \(x^\mathsf{T}Qx\),
place half the exported pair coefficient in each symmetric off-diagonal entry.

## 6. A small illustrative example

Suppose p = 3, k = 2, and:

$$
R=(1.0,\;0.8,\;0.2), \qquad
C_{12}=0.9,\quad C_{13}=0.1,\quad C_{23}=0.2.
$$

With the weights above:

$$
Q =
\begin{pmatrix}
-8.5 & 5.72 & 5.08\\
0 & -8.3 & 5.16\\
0 & 0 & -7.7
\end{pmatrix}.
$$

Comparing all subsets containing exactly two features:

| Selected features | QUBO objective, f_Q | Full objective, F = f_Q + 10 |
|---|---:|---:|
| 1 and 2 | -11.08 | -1.08 |
| 1 and 3 | -11.12 | -1.12 |
| 2 and 3 | -10.84 | -0.84 |

Features 1 and 3 are preferred, even though feature 2 is more relevant than
feature 3 individually. Selecting features 1 and 2 incurs a much larger
redundancy penalty. This illustrates why selecting only the top individual
relevance scores can differ from solving the joint selection problem.

This example is illustrative, not an entry in the saved benchmark set.

## 7. What the 36 saved instances represent

The instances were generated by
[variableselection_qubo2.py](variableselection_qubo2.py) from a synthetic master
dataset with 10,000 observations and 200 features, using simulation seed 42.
Features combine two latent risk factors with random weights and additive
noise. The binary outcome is sampled from a logistic default probability
depending on the same latent factors. These are **synthetic benchmark data,
not observed borrower records**.

The sample-size grid is:

`N = 100, 500, 1000, 3000, 5000, 10000`

Each sample size is paired with all six feature counts:

| Candidate features, p | Target selected features, k | Coefficient records per instance |
|---:|---:|---:|
| 10 | 2 | 55 |
| 30 | 3 | 465 |
| 50 | 5 | 1,275 |
| 100 | 10 | 5,050 |
| 150 | 15 | 11,325 |
| 200 | 20 | 20,100 |

The target is `k = max(2, p // 10)`. Thus, the 10-feature case targets two
features; the other configurations target 10 percent of the candidates.
All upper-triangular coefficients in these instances are nonzero, giving
`p * (p + 1) / 2` records per instance.

This produces 6 x 6 = **36 instances**, with **229,620 coefficient records** in
total. Instances use the first N observations and first p columns of the same
master dataset, so configurations are nested rather than independent datasets.
Repeated timing runs do not create additional instances.

The [instance index](qubo_instances_xqx_index.csv) maps each instance to:

- `instance_index`: its 1-based position in the text file.
- `n_samples`: observations used to estimate relevance and redundancy.
- `n_features`: candidate features and binary decision variables.
- `k_target`: desired number of selected features.

For example, instance 1 uses N = 100, p = 10, k = 2; instance 36 uses
N = 10,000, p = 200, k = 20. A coefficient index refers to a feature's column
position, not to an observation.

### Original datasets before Q-matrix conversion

The [raw dataset folder](qubo_raw_instances/) contains 36 text files, one for
each configuration. Its [index](qubo_raw_instances/index.csv) maps
`instance_index`, N, p, and k to the corresponding filename.

Each file has comment headers beginning with `#`, followed by tab-delimited
numeric rows in this order:

```text
feature_1  feature_2  ...  feature_p  target
```

Feature values are written with 17 significant digits and targets as binary
integers. The files preserve the original, unnormalized X/y slices, not
relevance scores, correlation matrices, or QUBO coefficients. They can be
loaded as follows:

```python
import numpy as np

data = np.loadtxt(r"qubo_raw_instances\instance_01_N100_p10.txt")
X = data[:, :-1]
y = data[:, -1].astype(int)
```

The generator exports raw datasets before constructing Q. Raw-data writing
does not count toward the Q-construction benchmark timings. All 36 saved raw
datasets have been checked to reproduce their corresponding saved Q matrices
exactly using mutual-information relevance and the documented parameters.

## 8. Text-file format and solving

The [solver input](qubo_instances_xqx.txt) begins with the number of instances.
Each instance then contains:

```text
n_variables n_nonzero_records
i j coefficient
i j coefficient
...
```

Indices are 1-based, and only nonzero upper-triangular entries are written.
Coefficients use 17 significant decimal digits to preserve float64 values
without integer rounding or rescaling. The file contains QUBO coefficients,
not the raw feature matrix or target observations.

The updated [D-FLIP C++ solver](<D:/MCSR_codes/rflip code/xqx.cpp>) must be run with
`--qubo` for these files:

```powershell
Set-Location "C:\Users\hwang\Documents\summitted\otherpapers"
& "D:\MCSR_codes\rflip code\xqx.exe" ".\qubo_instances_xqx.txt" ".\qubo_solutions_xqx.csv" --qubo --csv --max-count 1000000
```

This mode minimizes the original upper-triangular objective without doubling
off-diagonal terms. With the default maximization build, it negates
coefficients internally and restores the original objective sign in the
reported result. Omitting `--qubo` uses the legacy input convention and would
solve a different problem.

### Saved solution CSV

The [solution CSV](qubo_solutions_xqx.csv) has one row per instance, in input
order. Match `instance_index` to either instance index CSV.

| Column | Meaning |
|---|---|
| `instance_index` | 1-based position in the QUBO input |
| `input_file` | QUBO input filename |
| `n_variables` | Number of candidate features, p |
| `objective` | Original minimization objective f_Q, recomputed from the saved vector |
| `elapsed_seconds` | Measured search runtime |
| `tabu_tenure` | Tabu tenure used for this instance |
| `search_evaluations` | Actual counted search evaluations |
| `max_search_evaluations` | Configured evaluation limit |
| `selected_count` | Number of features in the saved solution |
| `selected_features` | Semicolon-separated, 1-based feature indices |
| `binary_solution` | Space-separated binary vector in feature-column order |

For example, `selected_features = 2;5` corresponds to a binary vector with
entries 2 and 5 equal to 1. The vector is separated by spaces so leading zeros
are not lost through numeric interpretation of a compact bit string.

All 36 saved solutions use a limit of **1,000,000 counted search evaluations
per instance**. The limit is enforced in local search, destruction, and
construction, including when it ends partway through a scan. The counter
tracks the solver's existing per-variable search scans; it is not a count of
every internal multi-flip calculation or a wall-clock time limit.

Without `--max-count`, the default remains 50,000,000 times p counted
evaluations per instance. Without `--csv`, the solver retains its legacy
tab-delimited summary format. Solver perturbations are seeded from the
current time, so repeated runs can return different solutions.

## 9. How to interpret and validate a solution

For a recovered decision vector x:

1. Identify the selected features from the entries equal to 1.
2. Report the actual selected count alongside `k_target`.
3. Compute relevance, redundancy, and the cardinality penalty separately.
4. Increase or otherwise calibrate gamma if deviations from k are unacceptable,
   or use an explicitly cardinality-constrained method.
5. Evaluate a downstream classifier on held-out data to assess predictive value.

For real predictive modeling, estimate relevance and redundancy using only
training data within each validation split. Computing these quantities using
held-out outcomes would leak information into feature selection.

The QUBO objective is a surrogate for useful feature selection: univariate
mutual information and pairwise linear correlation do not capture every
possible interaction or nonlinear redundancy. A lower QUBO objective does
not by itself establish better out-of-sample prediction.

Finally, the tabu-search solver is a heuristic. Its reported value is the
best found during that run, not a certificate of global optimality.

## 10. Comparing D-FLIP, Gurobi, D-Wave tabu, and OpenJij

The [comparison experiment](compare_qubo_solvers.py) solves the same saved Q
matrices using:

1. D-FLIP, with its native r-flip/tabu search.
2. Gurobi, modeling the upper-triangular objective as a binary quadratic
   minimization problem.
3. D-Wave Ocean's local, classical `dwave.samplers.TabuSampler`.
4. OpenJij's classical `SASampler`, using simulated annealing.

D-FLIP is the display name used for the existing r-flip/tabu implementation
in [the C++ source](<D:/MCSR_codes/rflip code/xqx.cpp>). The executable,
technical filenames, and existing command-line options retain their original
names for compatibility. The display-name change does not change the
algorithm or its numerical results.

The saved comparison CSVs and plot legends also use D-FLIP, without rerunning
the solvers. Their manifests record this presentation-only update while
preserving the original execution metadata, source hashes, and logs.

The D-Wave solver is not a quantum processor and does not require a cloud
account or access token. OpenJij simulated annealing also runs locally without
cloud credentials. Gurobi requires a valid local license.

### Running the experiment

For a complete Windows setup guide, an explicit all-method/all-instance
command, and a results-verification checklist, see the [README](README.md).

Dependencies are listed in
[requirements_solver_comparison.txt](requirements_solver_comparison.txt).
If they are missing, install them with:

```powershell
Set-Location "C:\Users\hwang\Documents\summitted\otherpapers"
python -m pip install -r ".\requirements_solver_comparison.txt"
```

Run the complete experiment:

```powershell
python ".\compare_qubo_solvers.py" --build-xqx
```

`--build-xqx` rebuilds the executable from the adjacent C++ source using `g++`.
It can be omitted when the executable is already current. Use `--xqx-exe` to
specify a different executable location.

The defaults are **36 instances x 4 solvers x 3 repeats = 432 runs**, with a
**five-second search limit per run**, seeds 42, 43, and 44, and one Gurobi
thread. D-FLIP and D-Wave tabu are single-threaded; OpenJij executes one annealing
read at a time. Solver execution order is shuffled reproducibly within each
instance/repeat. Maximum allocated search time is 36 minutes, plus setup and
I/O and possible time-limit overruns; Gurobi can terminate earlier after
proving optimality.

For a short integration check on **one instance only**, rather than a full
experiment:

```powershell
python ".\compare_qubo_solvers.py" --instances 1 --time-limit 0.1 --repeats 1 --output-dir ".\solver_comparison_openjij_smoke"
```

Use `--instances 1 6 36` to select specific instances, and `--output-dir` for
each additional experiment. A nonempty output directory is rejected to
protect previous results.

### Objective gaps

Every returned binary vector is checked, and its objective is independently
recomputed from the same Q matrix. Off-diagonal terms are counted once for
all four solvers.

For each instance, let \(f_{\mathrm{ref}}\) be the lowest feasible objective
observed across all solvers and all repeated runs. The script reports:

$$
\operatorname{gap}_{\mathrm{abs}}=f_{\mathrm{solver}}-f_{\mathrm{ref}},
$$

$$
\operatorname{gap}_{\%}
=100\,\frac{f_{\mathrm{solver}}-f_{\mathrm{ref}}}
               {|f_{\mathrm{ref}}|}.
$$

Using the absolute value in the denominator handles negative minimization
objectives correctly. When the reference is zero, zero absolute gaps are
reported as zero percent; positive percentage gaps are undefined, explicitly
logged, and left missing. Absolute gaps remain available.

`reference_kind` distinguishes:

- `gurobi_optimal`: an OPTIMAL Gurobi run supports the reference within
  numerical tolerances. Relative and absolute MIP stopping gaps are both set
  to zero; solver numerical tolerances still apply.
- `best_observed`: no Gurobi optimality certificate supports the reference.
  These are best-known-solution gaps, not proven optimality gaps.
- `unavailable`: no feasible solution was returned for the instance.

Gurobi's `gurobi_mip_gap_pct` is a separate bound-based quantity. It is not
substituted for the common objective gap used to compare the solvers.

### Computing-time measurements

- `solve_time_sec` is D-FLIP's native search timer, Gurobi's optimizer Runtime,
  or wall time around D-Wave/OpenJij sampling, including Python wrappers.
- `wall_time_sec` additionally includes per-run model construction,
  serialization, D-FLIP process startup, and result reading.
- One-time imports, input loading, and the shared Gurobi environment are
  outside per-run measurements.
- D-Wave uses exactly one read per trial, with a millisecond timeout equal
  to the requested seconds rounded up. The timeout is not multiplied by a
  separate number of reads.
- OpenJij has no native wall-clock timeout. The adapter builds its sparse
  binary model once, then executes serial one-read annealing batches until
  the time budget is exhausted, retaining the best validated vector across
  batches. Setup is outside the sampling timer. Matching sparse construction
  and sampling modes preserve isolated zero-bias variables.
- Each OpenJij batch uses `--openjij-sweeps` sweeps (default 1,000), the
  single-spin-flip updater, automatic geometric temperature scheduling, and
  reinitialized states. Batch seeds are drawn from
  `random.Random(run_seed).getrandbits(31)` to avoid merely shifting the same
  consecutive-seed sequence between repeated runs.
- Native limits are cooperative, and OpenJij can overrun by its final
  annealing batch. Actual times and `time_overrun_sec` are recorded rather
  than claiming a strict timeout. Very long sweep counts can cause larger
  overruns.

This is a final-incumbent comparison under equal allocated search limits,
not a time-to-target experiment. Heuristics typically consume their time
limit, whereas Gurobi can finish early after proving optimality. Timed
stochastic searches may return different vectors even with fixed seeds,
because stopping also depends on clock resolution and machine scheduling.

### Experiment outputs

The default output directory is `solver_comparison_results`.

| File | Contents |
|---|---|
| `runs.csv` | Each trial's status, objective, gaps, times, bounds, selected features, binary vector, seed, OpenJij batch count/sweeps, and error details |
| `instance_summary.csv` | Per-instance/per-solver best and mean objectives, gap statistics, runtime statistics, and coverage |
| `solver_summary.csv` | Aggregate runtime, objective gaps, feasible/error counts, and instances matching the reference |
| `comparison.png` | Time and objective-gap plots against feature count; means and standard deviations across sample sizes and repeats |
| `manifest.json` | Settings, package versions, platform information, input/script/executable hashes, and completion status |
| `experiment.log` | Progress, solver failures, missing incumbents, and reference warnings |

Partial trial results are saved after every run. Solver failures remain
visible as ERROR rows; they never become zero objectives or successful
fallback solutions. Runs without a feasible vector retain their termination
status and missing objective. Missing solutions cause a nonzero process exit
code, and failed runs are excluded from runtime means.

API references:
[D-Wave TabuSampler](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.TabuSampler.sample.html)
and [Gurobi parameters](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html),
plus the [OpenJij project](https://github.com/OpenJij/OpenJij).
