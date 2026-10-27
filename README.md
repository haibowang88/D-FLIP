# Derivative-Pruned r-Flip (D-FLIP) standalone solver for feature-selection QUBOs


This guide uses **Windows PowerShell**. Commands start in the project root,
one level above this README. Replace the example project path if necessary. For Sparse and Dense instances, see [Sparse vs Dense](./Sparse_vs_Dense.md)

## 1. What the executable does

The executable reads one or more precomputed QUBO matrices and searches for
a good binary vector for each instance:

```text
minimize E(z) = sum_i Q[i,i] z[i] + sum_{i<j} Q[i,j] z[i] z[j]
subject to z[i] in {0, 1}
```

For feature selection, `z[i] = 1` means retain encoded feature `i`.
The solver is a **classical heuristic**. It reports the best vector it finds,
not a proof of global optimality or a bound on the remaining optimality gap.

It does **not**:

- read a raw insurance, credit-card, or bankruptcy dataset;
- construct relevance/redundancy coefficients;
- preprocess data, train a predictor, or calculate RMSE/AUC;
- infer feature names or target cardinality from an index CSV;
- enforce a hard `sum(z) == k` constraint.

Prepare the matrix and preserve its feature-column mapping before running
the executable. In cross-validation, build each QUBO from training data only.

For the model used in this project, see
[the feature-selection formulation](../feature_selection_qubo_explained.md).
The standalone solver can also minimize other valid upper-triangular QUBOs.

## 2. Quick start: the 36 saved instances

No Python interpreter or solver license is required for this command.
The executable must be compatible with your Windows system; if Windows
reports missing runtime DLLs, see [Troubleshooting](#10-troubleshooting).

```powershell
Set-Location "your local director path"

$Exe = (Resolve-Path -LiteralPath ".\D-FLIP.exe").Path
$InputFile = (Resolve-Path -LiteralPath ".\qubo_instances.txt").Path
$OutputDir = Join-Path $PWD.Path ("results\dflip_standalone_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
New-Item -ItemType Directory -Path $OutputDir -ErrorAction Stop | Out-Null
$ResultCsv = Join-Path $OutputDir "solutions.csv"

& $Exe $InputFile $ResultCsv `
    --qubo `
    --csv `
    --time-limit 60 `
    --seed 42 `
    --max-count 9223372036854775807

if ($LASTEXITCODE -ne 0) {
    throw "D-FLIP failed. Do not treat a partial output file as a completed run."
}

$Rows = @(Import-Csv -LiteralPath $ResultCsv)
if ($Rows.Count -ne 36) {
    throw "Expected 36 solution rows; found $($Rows.Count)."
}
$Rows | Select-Object instance_index, n_variables, objective, selected_count, elapsed_seconds
Write-Output "Saved: $ResultCsv"
```

To use the other filename, change only the executable assignment:

```powershell
$Exe = (Resolve-Path -LiteralPath ".\D-FLIP.exe").Path
```

Important:

- **Always use `--qubo` for these saved instances.** Without it, the program
  uses its legacy maximization/doubled-off-diagonal convention.
- **Always use `--csv` to save selected features and binary vectors.**
  A `.csv` filename alone does not enable CSV output.
- The time limit is **per instance**, not per input file. A 60-second cap
  for 36 instances allocates approximately **36 minutes** of search, plus
  loading, initialization, output writing, and possible cooperative overruns.
  Change `60` to `5` for approximately three minutes of allocated search.
- The native program overwrites an existing output file. The commands use
  a new timestamped directory to preserve previous results.
- Input and output must be **different files**. Never use the input matrix
  filename as the output path.

## 3. Command-line reference

```text
D-FLIP.exe INPUT.txt OUTPUT.csv --qubo --csv
    [--time-limit SECONDS] [--seed INTEGER] [--max-count INTEGER]
```

`D-FLIP.exe` accepts exactly the same arguments.

| Argument | Meaning | Default / accepted values |
| --- | --- | --- |
| `INPUT.txt` | File containing the instance count and QUBO records | Required; must exist |
| `OUTPUT.csv` | Output filename; parent directory must already exist | Required; existing file is overwritten |
| `--qubo` | Minimize floating-point upper-triangular QUBO coefficients | Off by default; **required for this guide** |
| `--csv` | Write one solution row per instance, including its vector | Off by default; otherwise output is a brief tab-separated summary |
| `--time-limit SECONDS` | Cooperative search-time cap for each instance | Omitted: no time cap; supplied value must be finite and strictly positive |
| `--seed INTEGER` | Seed the process-wide C/C++ pseudo-random stream | Omitted: current time; supplied range `0` to `4294967295` |
| `--max-count INTEGER` | Limit counted search evaluations per instance | Default `50000000 * n_variables`; supplied range `1` to `9223372036854775807` |

The search stops when **either** its time budget or evaluation-count budget
is exhausted. The large `--max-count` in the examples prevents the count
cap from normally ending a time-budgeted trial early. Counted evaluations
are implementation-specific; they are not iterations or CPU instructions.

An explicit `--time-limit 0` is invalid. Omit the option to run with only
the evaluation-count cap. A finite positive time limit is recommended.

The native executable does **not** support `--instances`, `--repeats`,
`--threads`, `--k`, or runtime tabu-tenure/r-opt options. Those must not be
copied from other Python runners or older source variants.

To inspect usage, invoke the executable with no arguments:

```powershell
& ".\D-FLIP.exe"
```

Usage is printed to standard error and the program returns exit code `1`
because required arguments were omitted. This is expected for this probe;
it is not a successful solver run. There is no dedicated `--help` option.

## 4. Input format: upper-triangular QUBO records

The file is plain **ASCII text** with whitespace-separated tokens:

```text
number_of_instances
n_variables n_records
i j coefficient
i j coefficient
...
n_variables n_records
i j coefficient
...
```

Rules:

1. The first integer is the number of instances in the file.
2. Each instance begins with its variable count and coefficient-record count.
3. Exactly `n_records` coefficient records follow that instance's header.
4. Indices are **one-based** and satisfy `1 <= i <= j <= n_variables`.
5. Store each pair only once. Do not include both `(i,j)` and `(j,i)`.
6. Omitted entries are zero; explicit zero entries are allowed.
7. Coefficients must be finite. Decimal and scientific notation are
   accepted; use `.` as the decimal separator and no thousands separators.
8. Do not include comments, CSV headers, commas, feature names, or a
   prefixed `p`/`p qubo` header.
9. Declare the correct instance count and do not append extra records.

**Do not double off-diagonal terms.** A supplied record `1 2 5.08`
contributes exactly `5.08 * z[1] * z[2]` in QUBO mode.

If your existing matrix `A` is symmetric and its intended objective is
`z.T @ A @ z`, convert it to upper-triangular records using:

```text
Q[i,i] = A[i,i]
Q[i,j] = A[i,j] + A[j,i]    for i < j
```

Simply discarding the lower triangle of a symmetric matrix changes that
objective. Conversely, the project's existing upper-triangular matrices
already contain full pair contributions and need no such conversion.

The native reader is not a strict duplicate/trailing-record validator:
repeated pairs overwrite earlier values, and trailing data can remain
unread after the declared instances. Follow the rules above even when a
malformed file does not produce an error. The optional project helper
[qubo_io.py](../qubo_io.py) provides stricter input validation and
full-precision export; using it requires Python and NumPy, unlike running
the executable itself.

### Tiny complete example

This file contains **one instance, three variables, and six coefficients**:

```text
1
3 6
1 1 -8.5
1 2 5.08
1 3 5.72
2 2 -8.3
2 3 5.64
3 3 -7.7
```

Its unique global minimum is `z = [1, 1, 0]` with objective approximately
`-11.72`. This tiny problem can be checked by enumerating its eight binary
vectors. It is an installation/format example, not a solver-quality benchmark.

Create and solve it without Python, starting in the project root:

```powershell
$ToyDir = Join-Path $PWD.Path ("results\dflip_toy_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
New-Item -ItemType Directory -Path $ToyDir -ErrorAction Stop | Out-Null
$ToyInput = Join-Path $ToyDir "toy_qubo.txt"
$ToyResult = Join-Path $ToyDir "toy_solution.csv"

@'
1
3 6
1 1 -8.5
1 2 5.08
1 3 5.72
2 2 -8.3
2 3 5.64
3 3 -7.7
'@ | Set-Content -LiteralPath $ToyInput -Encoding Ascii

& ".\D-FLIP.exe" $ToyInput $ToyResult `
    --qubo --csv --time-limit 0.1 --seed 42 `
    --max-count 9223372036854775807
if ($LASTEXITCODE -ne 0) {
    throw "Toy QUBO run failed."
}
Import-Csv -LiteralPath $ToyResult | Format-List
```

The expected selected-feature list is `1;2`, and the binary vector is
`1 1 0`. Floating-point output can contain additional decimal digits.

## 5. Output CSV and feature interpretation

With `--csv`, the native header is:

```text
instance_index,input_file,n_variables,objective,elapsed_seconds,tabu_tenure,search_evaluations,max_search_evaluations,selected_count,selected_features,binary_solution
```

| Field | Interpretation |
| --- | --- |
| `instance_index` | Position in this input file, starting at 1 |
| `input_file` | Input filename passed to the executable |
| `n_variables` | Number of binary decision variables / candidate encoded features |
| `objective` | Recomputed minimized energy `E(z)`; **lower is better** |
| `elapsed_seconds` | Native elapsed search time, including search initialization but excluding matrix parsing and result writing |
| `tabu_tenure` | Tenure used by this build; not a user-supplied CLI setting |
| `search_evaluations` | Counted evaluations performed |
| `max_search_evaluations` | Evaluation-count cap for this instance |
| `selected_count` | Number of ones in the saved vector |
| `selected_features` | Semicolon-separated **one-based** selected indices, e.g. `1;2` |
| `binary_solution` | Space-separated 0/1 vector in original matrix-column order, e.g. `1 1 0` |

CSV rows do not contain the random seed or a repeat number. Record these
externally, as in the three-run example below.

**Index mapping:** selected index `1` means the first column used to build
the QUBO, or index `0` in Python. For a preprocessed dataset, these can be
one-hot-encoded columns rather than original raw fields. Save the exact
ordered feature-name list from the matrix-construction step.

For the provided 36-instance file,
[the index CSV](../qubo_instances_index.csv) maps `instance_index` to
sample size, feature count, and `k_target`. It is metadata, **not an input
to the executable**. The saved grid has six sample sizes and six feature
counts, covering all 36 combinations.

Console messages such as `Solved instance 1/36` report progress.
Output is buffered; do not assume every row has been durably flushed while
the process is still active. A nonzero exit or interruption can leave an
empty or partial file. Confirm exit code `0` and the expected row count.

## 6. Three runs per instance: keep the best solution

The following standalone PowerShell example runs the entire input file
three times, with seeds **42, 43, and 44**, at **60 seconds per instance**.
It saves:

- `repeat_1_seed_42.csv`, `repeat_2_seed_43.csv`, `repeat_3_seed_44.csv`;
- `all_runs.csv`, containing every completed trial plus repeat/seed/budget;
- `best_runs.csv`, retaining the lowest objective per instance, including
  its selected features, binary vector, repeat, seed, and timing.

For the 36-instance input, a successful run produces **108 trial rows**
and **36 best rows**. Allocated search time is approximately **108 minutes**
(1 hour 48 minutes), plus overhead and possible overruns.

Run this block from the project root; it sets its own paths:

```powershell
$Exe = (Resolve-Path -LiteralPath ".\D-FLIP.exe").Path
$InputFile = (Resolve-Path -LiteralPath ".\qubo_instances.txt").Path
$TimeLimit = 60
$Seeds = @(42, 43, 44)
$ExpectedInstances = [int]((Get-Content -LiteralPath $InputFile -TotalCount 1).Trim())
$RunDir = Join-Path $PWD.Path ("results\dflip_best_of_3_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
New-Item -ItemType Directory -Path $RunDir -ErrorAction Stop | Out-Null
$AllRuns = @()

for ($Repeat = 1; $Repeat -le $Seeds.Count; $Repeat++) {
    $Seed = $Seeds[$Repeat - 1]
    $TrialCsv = Join-Path $RunDir ("repeat_{0}_seed_{1}.csv" -f $Repeat, $Seed)

    & $Exe $InputFile $TrialCsv `
        --qubo --csv --time-limit $TimeLimit --seed $Seed `
        --max-count 9223372036854775807
    if ($LASTEXITCODE -ne 0) {
        throw "Repeat $Repeat failed. Earlier repeat files remain in $RunDir."
    }

    $TrialRows = @(Import-Csv -LiteralPath $TrialCsv)
    if ($TrialRows.Count -ne $ExpectedInstances) {
        throw "Repeat $Repeat returned $($TrialRows.Count) rows; expected $ExpectedInstances."
    }
    foreach ($Row in $TrialRows) {
        $Row | Add-Member -NotePropertyName repeat -NotePropertyValue $Repeat
        $Row | Add-Member -NotePropertyName seed -NotePropertyValue $Seed
        $Row | Add-Member -NotePropertyName time_limit_sec -NotePropertyValue $TimeLimit
    }
    $AllRuns += $TrialRows
    $AllRuns | Export-Csv -LiteralPath (Join-Path $RunDir "all_runs.csv") -NoTypeInformation -Encoding UTF8

    $BestRows = @(
        $AllRuns | Group-Object instance_index | ForEach-Object {
            $_.Group | Sort-Object `
                @{Expression = {[double]::Parse($_.objective, [System.Globalization.CultureInfo]::InvariantCulture)}}, `
                @{Expression = {[int]$_.repeat}} |
                Select-Object -First 1
        } | Sort-Object @{Expression = {[int]$_.instance_index}}
    )
    $BestRows | Export-Csv -LiteralPath (Join-Path $RunDir "best_runs.csv") -NoTypeInformation -Encoding UTF8
}

if ($AllRuns.Count -ne ($ExpectedInstances * $Seeds.Count) -or $BestRows.Count -ne $ExpectedInstances) {
    throw "Incomplete repeated-run coverage."
}
Write-Output "Completed $($AllRuns.Count) trials; saved $($BestRows.Count) best solutions in $RunDir"
```

The objective is sorted **numerically**, not lexicographically. Equal
objectives retain the earliest repeat. Best-so-far exports are refreshed
after each **complete input-file repeat**, not after every native instance.
The retained objective and vector always come from the same trial.

### Reproducibility of batch runs

The executable seeds its random stream once per process, not once per
instance. Later instances consume the stream left by earlier instances.
Consequently, solving a matrix alone can differ from solving it inside a
multi-instance file, even with the same process seed.

Wall-clock stopping also makes identical results across machines or system
loads impossible to guarantee. Record the executable/version or SHA256,
input-file hash, instance order, seed, and budgets.

The project's Python four-solver benchmark launches D-FLIP separately for
each instance/trial. Its seeding protocol therefore differs from this
native multi-instance batch example. Do not claim the two procedures
necessarily reproduce identical solution vectors.

## 7. Cardinality and the feature-selection objective

The project's feature-selection builder uses:

```text
F(z) = -alpha * sum_i R[i] z[i]
       + beta * sum_{i<j} C[i,j] z[i] z[j]
       + gamma * (sum_i z[i] - k)^2
```

Its upper-triangular coefficients are:

```text
Q[i,i] = -alpha * R[i] + gamma * (1 - 2*k)
Q[i,j] = beta * C[i,j] + 2*gamma       for i < j
```

For the saved feature-selection instances, `alpha=1`, `beta=0.8`, and
`gamma=2.5`. The native reported objective omits the constant:

```text
E(z) = F(z) - gamma * k^2
F(z) = reported_objective + gamma * k^2
```

The constant does not change the minimizing vector. It does change a
percentage gap's denominator, so do not compare relative gaps calculated
using different objective offsets.

The cardinality term is a **soft penalty**. Check `selected_count`
against the intended `k` or the index CSV's `k_target`; the executable does
not read or enforce that target separately. If the returned size differs,
review the penalty/model or apply an explicitly documented external repair
or constrained solver. Merely giving D-FLIP a larger time limit does not
turn a soft penalty into a hard equality.

The optimization objective is not predictive accuracy. Evaluate a selected
subset on held-out data before claiming improved RMSE, AUC, or operational
utility.

## 8. Limits and practical resource considerations

The compatible source accepts **1 to 30,000 variables** per instance.
This is a parser/source limit, not a practical scalability guarantee.
The solver allocates a dense `double` matrix even when the text input is
sparse, requiring approximately `8 * p^2` bytes for coefficients:

| Variables p | Matrix storage only |
| --- | --- |
| 200 | 0.32 MB |
| 2,000 | 32 MB |
| 10,000 | 800 MB |
| 30,000 | 7.2 GB |

These are decimal units and exclude all other allocations and overhead.
Use a machine with adequate RAM; do not assume a very large sparse file
will require little memory after loading.

Search parameters such as the maximum multi-move size and tabu tenure
are compiled into this build rather than exposed as CLI options.


## 10. Troubleshooting

| Symptom | Check / action |
| --- | --- |
| Executable is not found | Start in the project root, quote paths with spaces, and use PowerShell's `&` call operator. |
| Windows reports a missing DLL | Use the matching compiler/runtime distribution and make its runtime DLLs available through `PATH`; do not download arbitrary DLLs from untrusted sites. |
| `Invalid or incomplete option` | Use the supported named options; do not pass the old positional `time tabu r-opt` arguments or Python runner options. |
| `Invalid instance count` | The first token must be an integer instance count; save plain ASCII, not a CSV or a UTF-8 file with a byte-order mark. |
| `Invalid QUBO coefficient record` | Check one-based indices, upper-triangular ordering, finite values, and exact record counts. |
| Input/output `cannot be opened` | Check paths, permissions, and whether the output parent directory exists. |
| Unexpected objective/sign or selected set | Confirm `--qubo` is present, off-diagonal terms are counted once, and input records use the intended feature order. |
| Too few/many selected features | Cardinality is a soft penalty; inspect `selected_count` and the coefficient model. |
| Runtime is shorter than expected | The evaluation-count cap can stop search early; inspect `search_evaluations` and `max_search_evaluations`. |
| Runtime slightly exceeds the cap | Limits are cooperative and exclude input/output overhead; inspect `elapsed_seconds` separately from total process time. |
| Output has no selected-feature vector | Add `--csv`; the default text summary does not contain it. |
| Repeated run has different results | Specify a seed, preserve input ordering, and account for batch-stream consumption and time-limited search variability. |
| Only some instance rows are present | Check exit code and input headers. Interrupted/failed runs are not complete results; buffered output may be partial. |

