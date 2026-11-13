# OCA — Standalone Operating Guide for AI Agents

**Ordinal Correlation Algorithm (OCA).** Version 9.0.0.

This document is written for an autonomous agent — or a human in a hurry — who has
been handed this repository and must use it without reading the source. It is
self-contained: everything needed to configure a study, run it, read the results and
interpret the two dominance criteria is below.

Read sections 1 and 4 first. The rest is reference material you consult on demand.

---

## 0. Table of contents

1. What OCA does, and what it does not do
2. Environment
3. The data contract
4. Minimal run, in three steps
5. Configuration reference
6. The algorithm, stage by stage
7. **The two dominance criteria — do not confuse them**
8. Where each result lives
9. Running an autonomous exploration
10. Measured limitations and traps
11. Decision tables: request → action
12. Troubleshooting
13. Conventions to respect when modifying the code
14. Known limits (Limites connues)
15. Where to read more

---

## 1. What OCA does, and what it does not do

### 1.1. What it does

OCA answers a question of **ordering**, not of magnitude: among several candidate
quantities measured on the same population — typically stresses computed at several
*critical zones* of a mechanical component — **which one dominates the others, and
by what margin?**

It is designed for the situation where a downstream analysis must be carried out on
*one* representative quantity rather than on all of them, and where the choice of
that quantity must be justified rather than asserted. The deliverable is a ranked
hierarchy, a dominance graph, and a regularisation envelope that bounds the
**zones retained by the selection** (the zones the selection discarded are not part
of the envelope, see §5.4).

The pipeline, in order, for **each** declared sub-directory (each one is an
independent study, see §3.1):

1. **Read** the workbook of the sub-directory (or merge per-zone files, see §3.4),
   resolve the zone and variable lists, and compute the declared synthetic variables
   (ratios such as `De/E`).
2. **Initial ordinal correlation matrix** over *all* critical zones, before any
   selection. Ordinal, not linear: the coefficient in row *i*, column *j* is the
   **proportion of observations** for which zone *j* is majorised by zone *i*.
3. **Stage-1 ordering graph**, deduced from that matrix, in which every critical
   zone is represented — including those carrying no order relation above the
   threshold.
4. **Descriptive plots**: box plots, histograms, scatter plots.
5. **GPO selection** (One-vs-All): for each zone, count the observations where it
   majorises *all* the others simultaneously.
6. **Resolution of the regularisation target**: declared by name, resolved
   automatically, or added as a synthetic column.
7. **κ regularisation**: find the smallest multiplier κ such that the target zone,
   scaled by κ, envelopes every **retained** zone on a declared share of
   observations.
8. **Augmented ordinal matrix and final dominance graph**, envelope included.
9. **Parsimony index β**, sampling table, consolidated LaTeX report and
   `output.json`.

### 1.2. What it does not do

- It is **not** a regression tool. It produces no predictive model and no equation.
  (Its companion project, RMR, does that.)
- It does **not** test statistical significance. Every quantity it reports is a
  proportion or a count over the observations supplied, descriptive by construction.
- It does **not** impute missing values.
- It assumes the quantities compared are **commensurable** — measured in the same
  unit, on the same population. Comparing a stress with a displacement would produce
  a formally valid ranking that means nothing.

---

## 2. Environment

Python **3.12 or later**. Scientific stack: `pandas`, `numpy`, `scipy`,
`matplotlib`, `seaborn`, `networkx`, `openpyxl` (workbooks) and `Jinja2` (needed by
pandas to write the LaTeX tables). PDF assembly uses `pypdf` (with `PyPDF2`
accepted as a fallback). LaTeX reports need **`pdflatex`** (or `tectonic`), set by
`latex_output.compiler`.

Verifying the installation, from the repository root:

```bash
python -c "import pandas, numpy, scipy, matplotlib, seaborn, networkx, openpyxl, jinja2; print('ok')"
pdflatex --version
python main_OCA.py --config examples/CAS_TEST_00/input.json
```

| Component | Absent means |
|---|---|
| A Python package | One line naming the package and the command `pip install -r requirements.txt` (exit code 3). |
| LaTeX compiler | One warning, printed once. The figure PDFs (`02-`, `03-`, `05-`, `06-`, `08-`, `09-`, `10-`), the `.tex` sources, every `.txt` trace and `output.json` are still written, the exit code is 0, and the consolidated report `00-` is built from the figures only and declared **partial** (`output.json`: `report.partial`, `report.missing_sections`). The LaTeX PDFs `01-`, `04-`, `07-`, `11-` are missing. |
| `pypdf` / `PyPDF2` | No consolidated report `00-`. Individual PDFs still produced. |

---

## 3. The data contract

### 3.1. Required shape

One workbook per declared sub-directory of `merged_subdir` (`.xlsx`, first sheet
only; or `.csv` with `,` or `;` separator; `.xls` is refused, convert it). Inside
each:

- the first row is the header;
- one column per design variable (names as in `feature_cols_init`);
- one column per critical zone (names as in `critical_zones`);
- one row per observation;
- numeric values in every compared column (text columns that are not declared, such
  as a comment column, are ignored).

**A first column without a header (or named `Unnamed`, `index`, `id`, `config`,
`config_id`, `n`) is an index and is dropped (one log line says so); any other first
column is kept like the others.** Exception: a first column with one of those usual
index names that is **cited** in `critical_zones`, `feature_cols_init` or in a
synthetic-variable expression (`user_defined_vars_cols`, `target_zone`) is a quantity
and is kept. Only the first column is concerned: an `n` in second position is an
ordinary column. Both layouts, with and without an index column, give the same
results.

**Each sub-directory is an independent study.** Declaring three sub-directories with
300 rows each gives three studies of 300 rows, with three target choices and three
κ values; nothing is concatenated across sub-directories and no 900-row population is
ever analysed. To analyse a larger population, put all its rows in one workbook.

### 3.1.1. Empty zone or variable lists

`critical_zones` and `feature_cols_init` may be empty, `null`, `""` or absent. The
missing list is then deduced from the numeric columns of the workbook:

| `critical_zones` | `feature_cols_init` | Resolution |
|---|---|---|
| declared | declared | each name must exist, be numeric, and belong to one list only |
| declared | empty | variables = numeric columns that are not zones |
| empty | declared | zones = numeric columns that are not variables |
| empty | empty | **all** numeric columns are zones, no design variable, with a warning |

The resolved lists are written to the log, to `output.json` (`column_resolution`) and
to the report. Names absent from the workbook, duplicates, blanks and names present
in both lists are refused with the list of available columns.

### 3.2. What the algorithm needs to be meaningful

| Property | Why it matters |
|---|---|
| At least **two** observations and **two** critical zones | A dominance analysis of one quantity is vacuous; one observation carries no order. |
| Finite values | A missing or infinite value in a compared zone is refused with the column, the count and the observation numbers. |
| Commensurable zones | The comparison `zone_j ≤ zone_i` must have physical meaning. |
| Non-constant zones | An ordinal correlation is computed on comparisons. A constant zone compares equal to itself everywhere and carries no order information. |
| Enough observations | Every reported figure is a proportion. With 20 observations, one observation is five percentage points. |
| Non-zero denominators | A synthetic variable such as `De/E` divides. A zero in `E` is a division by zero. |

### 3.3. File layout

```
input/
  01-first-configuration/dataSet_ALL.xlsx
  02-second-configuration/dataSet_ALL.xlsx
  03-third-configuration/dataSet_ALL.xlsx
```

```json
"data_files": "dataSet_ALL.xlsx",
"merged_subdir": ["01-first-configuration", "02-second-configuration",
                  "03-third-configuration"]
```

`merged_subdir` also accepts a single string, for one sub-directory.

### 3.4. Per-zone files (a different mode, not a batch)

For **one** sub-directory, if `<input_dir>/dataSet_<zone>/dataSet_<zone>.xlsx` exists
for every **declared** zone, those files are merged on a configuration identifier
built from the design variables (`feature_cols_init`, else every column but the
last, which carries the target). When the variables are deduced, their order and set
are fixed by the **first** file; a file with a different set of variables is refused
with a message naming the files and columns, and the columns of the following files
are reordered on the first one before the keys are built, so permuting columns from
one zone file to another changes nothing. Duplicates removed, duplicates with
contradicting values (a warning; the first occurrence is kept, so the result depends
on the row order) and configurations outside the intersection are counted in the log.
The zones must be declared in this mode.

### 3.5. Safe paths

The output folder of each sub-directory is emptied at each run. Before any cleaning
the program refuses a `merged_subdir` entry containing a path separator or `..` or
that is absolute, a `data_files` that is not a plain file name, an `output_dir` equal
to or containing `input_dir`, an `output_json`, an `output_files.*` entry or a
consolidated-report name that designates a source file (merged workbooks and per-zone
files, paths resolved), and two outputs that designate the same file. Zone, variable,
synthetic-variable and target names that start with `$` or look like a label OCA
generates (`2.00$\times$A`) are refused too: they would overwrite a computed column.

---

## 4. Minimal run, in three steps

To try the program first, run the shipped example from the repository root:
`python main_OCA.py --config examples/CAS_TEST_00/input.json`. For a study of your
own, `python main_OCA.py --init` writes a template `input.json` and an `input/`
folder in the current directory.

### Step 1 — place the data

```
<workdir>/input/<sub-directory>/<data_files>
```

### Step 2 — write `input.json`

Copy the shipped `input.json` — it is the reference, commented inline. `//`
comments are accepted and stripped silently; accented characters are accepted; keys
suffixed `__options` document the admissible values and are ignored by the
computation.

Change only the data keys. This four-key file runs as it is:

```json
{
  "critical_zones": ["ZoneA", "ZoneB", "ZoneC"],
  "feature_cols_init": ["V1", "V2", "V3"],
  "data_files": "myData.xlsx",
  "merged_subdir": ["run-01"]
}
```

Every other key takes its default (no synthetic variable, automatic target,
threshold 0.99, κ from 1 to 3 by 0.01, LaTeX compiler `auto`). The two lists may be
left empty, see §3.1.1. There is no default study: the data files are always named.

### Step 3 — run

```bash
python main_OCA.py                  # uses ./input.json
python main_OCA.py --config x.json  # uses another file
```

A configuration or data error is reported in one or two lines naming the offending
key by its full path (for example `gpo_regularization.kappa_step`), never as a Python
trace (`--debug` shows the trace of an unexpected error). Run without data, the
program points to `--init` and to `examples/`.

### Reading the result

| Question | File |
|---|---|
| Which zone dominates, and how are they ranked? | `01-Step1_Dominance_Hierarchy_Result.pdf` |
| Everything in one document | `00-OCA_Report.pdf` |
| Machine-readable | `output.json` |
| What the run actually did | `OCA_run.log` |

---

## 5. Configuration reference

### 5.1. Data

| Key | Type | Meaning |
|---|---|---|
| `critical_zones` | list of str | The quantities to compare. May be empty, see §3.1.1. |
| `feature_cols_init` | list of str | Design variables. May be empty, see §3.1.1. |
| `data_files` | str | Workbook name (`.xlsx` or `.csv`), a plain file name, identical in every sub-directory. |
| `merged_subdir` | str or list | Sub-directories to process, **each an independent study** (§3.1). |
| `input_dir`, `output_dir`, `output_json` | str | Roots and trace path. |

A `null` or empty optional field takes its default (a note is logged); a wrong type
or an out-of-domain value is refused with the full key path. Unknown keys are
ignored with a warning.

### 5.2. Synthetic variables

```json
"user_defined_vars": false,
"user_defined_vars_cols": []
```

`user_defined_vars: true` adds one column per entry of `user_defined_vars_cols`
(which must then be listed). The entry `"De/E"` is read as the ratio of column `De`
to column `E`. A zero denominator makes the value infinite and the run is refused
with the expression named.

These columns become candidates like any other, and in particular a synthetic
variable may serve as the regularisation target.

### 5.3. GPO selection

```json
"gpo_selection": { "threshold_low": 0.5, "threshold_high": 10.0 }
```

Two thresholds, in percent of the One-vs-All count, with **different roles**:

- `threshold_low` is the only retention filter: a zone whose frequency `p` is
  **strictly below** it is dropped from step 2. Raising it drops more zones.
- `threshold_high` filters nothing: zones whose `p` is **strictly above** it are only
  labelled "dimensioning" in the table and in the notice. Zones above it are kept.

The decisions (both thresholds, the automatic target, the cumulative `P`) use the
percentages **rounded to 0.01 %**, the convention of the reference paper: a zone at
0.4975 % reads 0.50 % and is not dropped by a 0.5 % low threshold.

### 5.4. κ regularisation

```json
"gpo_regularization": {
  "target_zone": "none",
  "threshold": 0.99,
  "kappa_min": 1.0,
  "kappa_max": 3.0,
  "kappa_step": 0.01
}
```

| Key | Meaning |
|---|---|
| `target_zone` | `"none"` (default) resolves the target automatically; a column name imposes it; a name absent from the table is added as a synthetic column. When several zones tie at the maximum frequency, the first in the column order of the count table (count ascending, then declaration order) is chosen; ties and rule are logged and written to `output.json`. |
| `threshold` | Share of observations the envelope must cover. `0.99` means the scaled target must majorise **each retained zone** on 99 % of observations. |
| `kappa_min`, `kappa_max`, `kappa_step` | The sweep, with `0 < kappa_min <= kappa_max` and `kappa_step > 0`. κ is the smallest multiplier reaching the threshold. |

**What the envelope covers.** The search runs on the zones **kept by the selection**
(`updated_critical_zones`, target included). Zones dropped by `threshold_low` are not
part of κ nor of β: κ·target does not bound them. Include them by lowering
`threshold_low` if the envelope must cover them.

**Grid.** The grid is built by index, `kappa_min + i * kappa_step`; the upper bound
is included if it belongs to the grid, to within the rounding tolerance (1e-9
relative), so `[1, 2]` with a 0.3 step stops at 1.9. At most 100 000 points, all
distinct (a grid whose points cannot be told apart at float precision is refused). The
κ published in `output.json` is the float actually tested, never re-rounded; only
the labels and the log are rounded, to as many decimals as the step requires, at
least 2 (a 0.001 step gives 3).

**Overflow.** If the envelope `κ × target` or a ratio of β leaves the float domain
(values around 1e308), the run is refused with a message: an infinite product would
majorise every zone and is not a success.

**Success is `kappa_found`, not the value of κ.** A κ equal to `kappa_max` is a
success if `kappa_found` is true (the threshold is met exactly at the bound). When
`kappa_found` is false, the reported κ is the **last point tested**, and
`output.json` (`kappa_search`) gives the number of points tested, that last point and
the lowest coverage reached there.

### 5.5. Ordinal graph

```json
"ordinal_graph": { "threshold": 0.99 }
```

An edge is drawn from *i* to *j* when the proportion of observations with
`zone_j ≤ zone_i` reaches the threshold. Lowering it thickens the graph with weaker
relations; raising it keeps only near-total dominances. **This threshold governs the
graph only, never the selection.**

### 5.6. Reporting and style

| Key | Meaning |
|---|---|
| `latex_output.sample_size` | Rows of the data extract reproduced in the report. |
| `latex_output.max_columns_per_block` | Column count per table block, for wide tables. |
| `latex_output.table_font_size` | `tiny`, `scriptsize`, `footnotesize`, `small`, `normalsize`. |
| `latex_output.table_column_separation` | LaTeX column separation, e.g. `2pt`. |
| `latex_output.compiler` | `auto` (default: `pdflatex`, else `tectonic`), `pdflatex`, `tectonic`, `none` (`.tex` sources only). |
| `sample_random_state` | Seed of the reproduced extract. Affects presentation only. |
| `color_style.color_map_matrix` | Colormap of the **correlation matrices**. |
| `color_style.color_map_critical_zones` | Colormap of the **zone series** in plots. Deliberately dissociated from the previous one since version 7: a diverging map suits a matrix, a qualitative one suits series. |
| `color_style.saturation`, `alpha` | Intensity and opacity. |
| `color_style.apply_to` | `background` colours the cell, `text` colours the glyph in bold with saturation toward black. |

---

## 6. The algorithm, stage by stage

### 6.1. Ordinal correlation matrix

For each ordered pair of columns *(i, j)*, the coefficient is

```
p(i, j) = share of observations where  column_j ≤ column_i
```

This is a **proportion**, not a Pearson or Spearman coefficient. It is not
symmetric: `p(i, j)` and `p(j, i)` sum to one plus the share of ties. A coefficient
of 1 means total dominance of *i* over *j* on the sample.

The matrix is computed **twice**: once on all zones before any selection
(deliverable `02-`), and once on the retained zones augmented with the
regularisation envelope after it (`05-`).

**Reading `A < B`.** In the text traces of the graphs (`Step1_OrdinalGraph_Targets.txt`,
`OCA_OrdinalGraph_Targets.txt`), in the comparison chains and in `output.json`
(`chains`), `A < B` reads "A precedes B in the dominance ranking": A majorises B in
the proportion given (`p(A, B) >= threshold`). A sentence recalls it at the top of
the graph traces.

### 6.2. GPO selection, One-vs-All

For each zone, count the observations where it majorises **all** the others
simultaneously. Four figures are reported: `n` the count, `p` the percentage, `N`
the total, `P` the cumulative percentage.

The One-vs-All requirement is strict, and section 7 explains what it costs. The
percentages are rounded to 0.01 % **before** the selection decisions (§5.3).

### 6.3. Regularisation target and κ

The target zone alone does not bound the others everywhere — that is the point of
a dominance analysis rather than a maximum. The regularisation seeks the smallest
multiplier κ such that `κ × target` majorises every **retained** zone (the zones the
selection kept, target included) on at least `threshold` of the observations. The
result is an **envelope** of the retained zones; zones dropped by the low threshold
are outside its scope (§5.4).

### 6.4. Parsimony index β

β is, for each observation, the minimum over the retained zones of the ratio
`κ × target / zone`; the report gives its maximum and mean. A value close to 1 means
a tight envelope, a high value a conservative one. Read it alongside κ.

β reads as a margin only for **strictly positive** values. If a retained zone holds
negative or zero values, a warning is logged and `output.json` carries
`metrics.beta_defined: false` with the reason; the computation is unchanged, and a
zero denominator gives a ratio set to `+inf` (published `null`).

---

## 7. The two dominance criteria — do not confuse them

**This is the single most important section of this guide.** OCA computes two
distinct dominance measures, and mistaking one for the other invalidates any
conclusion drawn.

### 7.1. Dominance frequency `p` — One-vs-All

Computed by the stage-1 count. A zone scores the share of observations where it
majorises **all the others simultaneously**.

**Role: it selects.** It governs the ranking, the hierarchy and the filtering.

**Its weakness, and it is structural: it saturates.** As soon as one zone dominates
broadly, every other falls to zero — *whatever their mutual relations*. Three zones
that are perfectly ordered among themselves all score zero if a fourth dominates
them all. They are then no longer separated by this criterion, and no amount of data
will separate them.

### 7.2. Matrix dominance score — pairwise

Computed from the ordinal matrix. The score of a column is the mean, over all the
others, of the proportion of observations where it majorises them:

```
s_i = (1 / (m-1)) × Σ_{j ≠ i} p(i, j)
```

**Role: it disposes.** It orders the columns for presentation, and it separates
exactly what the frequency `p` cannot.

**It has no saturation.** A pairwise comparison remains informative even when a
third column dominates both.

### 7.3. The rule

> The **frequency selects**; the **matrix score disposes**. Never read one as the
> other.

Concretely: a zone with `p = 0 %` is not "without order relation". It may be
perfectly ranked among its peers — the One-vs-All criterion simply cannot see it.
Look at the matrix score, and at the ordinal graph, before concluding.

---

## 8. Where each result lives

Deliverables are numbered by decreasing importance. Each numbered folder of
`merged_subdir` receives its own set, plus the merged workbook.

| Prefix | Content |
|---|---|
| `00-` | **Consolidated report.** Everything, merged. |
| `01-` | **Dominance hierarchy.** The ranked result — the answer to the question. |
| `02-` | Initial correlation matrix, all zones, before selection |
| `03-` | Stage-1 ordering graph, all zones represented |
| `04-` | Critical-zone sample extract |
| `05-` | Ordinal correlation matrix, envelope included |
| `06-` | Final dominance graph |
| `07-` | Critical-zone table |
| `08-` | Box plots |
| `09-` | Histograms |
| `10-` | Scatter plots |
| `11-` | Analytical notice, report introduction |
| `OCA_Sort_Targets.txt` | Ranking, plain text |
| `Step1_OrdinalCorrMat_Targets.txt` | Stage-1 matrix, plain text |
| `Step1_OrdinalGraph_Targets.txt`, `OCA_OrdinalGraph_Targets.txt` | Graphs, plain text |
| `OCA_run.log` | Execution log |

`output.json` carries the machine-readable trace: retained zone, κ, β, dominance
frequencies, matrix scores, and the resolved configuration. `config_file` is the
configuration path **as provided** when relative; an absolute path is reduced to the file name. The matrices it carries are
rounded to 4 decimals while edges, cumulative sums and degrees are decided on the
**unrounded** proportions: a proportion published as equal to the threshold at 4
decimals may carry no edge, and `ordinal_relations` are published to 2 decimals.

**Attempt status.** At the start of each run the previous `output_json` is replaced by
`{"status": "running", "started_at": ...}`, so a consumer watching that file alone never
reads an earlier success. At the end it carries `"status": "complete"` (`"partial"` if
some sub-directories of a batch failed, see `failed_subdirs`), the rest of its content
being unchanged; on a global failure it becomes `{"status": "failed", "reason": ...}`.
The output folders of failed sub-directories are not touched and may hold the results of
an earlier run.

When the consolidated report is partial and the notice could not be compiled (no LaTeX,
or compilation failed), `00-` opens on a "Rapport partiel" page, produced without LaTeX,
that lists the missing sections. A PDF whose compilation failed is never merged, even if
it is still on disk.

**Read `output.json` rather than parsing the PDFs.**

---

## 9. Running an autonomous exploration

OCA has no computation branches to sweep, so an exploration is narrower than for a
regression tool. Three axes are worth varying, in this order.

1. **The low selection threshold** `gpo_selection.threshold_low`. This is the axis
   that changes *which zones are retained*, and therefore everything downstream.
   Raise it to drop more zones and force a decision, lower it to see how many zones
   are genuine candidates. `threshold_high` is not an axis: it only labels zones as
   "dimensioning" and filters nothing.
2. **The regularisation target.** Compare `"none"` against each plausible column
   named explicitly. A target resolved automatically is convenient; a target
   justified physically is publishable. The comparison tells you whether the two
   agree.
3. **The envelope threshold** `gpo_regularization.threshold`. Report κ as a function
   of it — 0.95, 0.99, 1.0. A κ that grows sharply near 1.0 signals a small number
   of stubborn observations, which are worth inspecting individually.

Do **not** sweep `ordinal_graph.threshold` expecting the selection to move: it
governs the graph only.

For each trial, run in an isolated copy of the code and data so no trial can see
another's outputs, and keep the trace of the choice — which thresholds were
compared, and by what margin the winner won — rather than every figure.

---

## 10. Measured limitations and traps

### 10.1. The One-vs-All criterion saturates

See section 7. This is not a defect to fix but a property to know. It is the reason
the matrix score exists.

### 10.2. Read `kappa_found`, not the value of κ

The success criterion is `kappa_found`. A κ equal to `kappa_max` with `kappa_found`
true is a success at the bound. With `kappa_found` false, the reported κ is only the
last point tested (not necessarily `kappa_max`): read it with the coverage given in
`output.json` (`kappa_search`), not as a tight bound.

### 10.2.1. The envelope covers the retained zones

κ and β are computed on the zones kept by the selection. A zone dropped by
`threshold_low` (for example one that dominates on 0.1 % of the rows only) is not
bounded by the envelope. Lower `threshold_low` to include it.

### 10.3. The ordinal matrix is not symmetric

`p(i, j) + p(j, i) = 1 + share of ties`. Do not read half the matrix and infer the
other half, and do not describe it as a correlation matrix in the Pearson sense.

### 10.4. Synthetic variables divide without protection

`"De/E"` divides column `De` by column `E`. A zero denominator gives an infinite
value, and the run is refused naming the expression; a near-zero gives an enormous
value that propagates into the comparisons. Check the denominator before declaring
the ratio.

### 10.5. Every figure is a proportion

There is no significance test anywhere in OCA. A dominance frequency of 100 % on
twenty observations and on nine hundred are not the same evidence, and the program
reports both identically. Quote the observation count alongside any proportion.

### 10.6. A constant zone carries no order information

It compares equal to itself on every observation. It will appear in the matrix with
coefficients that reflect ties rather than dominance. Exclude it, or expect to
interpret ties. The linear (Pearson) coefficient of the scatter plots is not defined
for a constant series: it is not computed and the figure says "variance nulle".

### 10.7. The three studies of a batch are not one population

See §3.1. Nothing is concatenated across the sub-directories of `merged_subdir`.

---

## 11. Decision tables: request → action

| Instruction | Action |
|---|---|
| "which zone should we keep?" | Run as shipped, read `01-`. Then check section 7: if several zones score `p = 0 %`, read their matrix scores before concluding they are unordered. |
| "justify the choice" | Report the dominance frequency **and** the matrix score **and** the ordinal graph. The three together are the argument; any one alone is not. |
| "bound the zones" | That is the κ envelope, on the retained zones (§5.4). Report κ, `kappa_found`, the threshold used, and β (with `beta_defined`). To bound a zone the selection dropped, lower `threshold_low` first. |
| "impose this zone as the target" | `gpo_regularization.target_zone: "<name>"`. If the name is absent from the table it is added as a synthetic column. |
| "the graph is unreadable" | Raise `ordinal_graph.threshold`. This changes the graph only, never the selection. |
| "the tables overflow the page" | Reduce `latex_output.max_columns_per_block`, or step `table_font_size` down. |
| "make it reproducible" | Everything is deterministic except the reproduced extract, governed by `sample_random_state`. |

---

## 12. Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| Error naming missing columns | `critical_zones` or `feature_cols_init` disagrees with the workbook header (the message lists the available columns). Compare exactly, case and spaces included. |
| A first variable or zone "disappears" | It was taken for an index column (blank header, or named `index`, `id`, `config`, `config_id`, `n`; the log says so). Cite it in `critical_zones`, `feature_cols_init` or an expression to keep it, or rename the column. |
| `status` is `running` in `output_json` | The run is in progress or was killed before the end. `failed` carries the reason. |
| Workbook not found | `data_files` must be the same name in every sub-directory of `merged_subdir`, and paths are relative to the working directory. |
| Message pointing to `--init` and `examples/` | No data at the declared path (fresh clone, empty `input/`). |
| LaTeX PDFs missing, `.tex` present | No LaTeX compiler, or its compilation failed: see the log (first error cited) and `report.missing_sections`. `00-` is then partial. |
| No `00-` consolidated report | `pypdf` absent, or no PDF at all. |
| κ equals `kappa_max` | Read `kappa_found` (§10.2): true is a success at the bound, false means no multiplier of the swept range met the threshold. |
| `beta_defined` false, `β` null | Negative or zero values among the retained zones (§6.4). |
| Infinite or enormous values in the matrix | A synthetic variable divided by zero. See §10.4. |
| Every zone scores `p = 0 %` except one | Expected saturation of the One-vs-All criterion. See §7. |

---

## 13. Conventions to respect when modifying the code

- Configuration reading is tolerant: `//` comments are stripped silently, keys
  suffixed `__options` are self-documentation and are removed before computation,
  accented characters are accepted. Do not add a syntax warning for a comment.
- Every deliverable name comes from the `output_files` block of the configuration.
  Do not hard-code one.
- The two dominance criteria must remain distinct, and their docstrings state why.
  Merging them would destroy the discrimination that the matrix score provides.
- The two colormaps are dissociated on purpose since version 7: a diverging map for
  matrices, a qualitative one for zone series. Reuniting them would make one of the
  two illegible.
- Extend the harness in `tests/` when you add a capability. A check that cannot fail
  proves nothing: verify that it fails on the unpatched code.
- Validation commands, all from the repository root: `python -m unittest
  tests.test_robustness_v9 -v`, `python tests/validate_pipeline.py [--no-run]`,
  `python tests/validate_against_reference.py [--no-run]`. Both harnesses default to
  `examples/CAS_TEST_00/input.json`. The reference harness compares against the
  **version 8.0** state of the example (the original outputs were not kept): it
  proves invariance since 8.0, not absolute correctness.
- Errors of input are `ConfigError`s of one or two lines naming the key by its full
  path; never let a raw `TypeError`, `ValueError` or `KeyError` reach the user.

---

## 14. Known limits (Limites connues)

These are accepted and documented, not defects to work around silently.

- **Rounded matrices.** See section 8: the matrices of `output.json` are rounded to 4
  decimals, edges are decided on the unrounded proportions, so an edge cannot always be
  rebuilt from the published matrix when a cell is within 5e-5 of the threshold.
- **Selection on rounded percentages.** Decisions use `p` rounded to 0.01 % (paper
  convention). The automatic target is the first maximum of the count table, so
  rounding ties can favour the zone with the smaller count.
- **Both lists empty.** With neither `critical_zones` nor `feature_cols_init`, every
  numeric column is a zone, including quantities of different nature. Declare at
  least one list when zones and variables are not commensurable.
- **Per-zone duplicates.** Duplicated configurations with contradicting values keep
  the first occurrence: the result depends on the row order (a warning says so).
- **`.txt` traces are cp1252.** A character that encoding cannot represent becomes
  `?`, so two names differing only by such characters become indistinguishable in the
  `.txt` files (JSON, workbook and tables keep the exact names).
- **Extreme magnitudes.** Values near the float limit (1e308) are refused when the
  envelope overflows; figures drawn on such magnitudes may fail earlier in matplotlib.
- **Labels of the envelope.** The label of the envelope column rounds κ to the
  displayed decimals (up to 15): two κ closer than that share a label, never a value
  (the JSON keeps the tested float).
- **Validation harnesses.** `validate_pipeline.py` checks the first sub-directory of a
  batch against the exported traceability workbook; `validate_against_reference.py`
  compares the first sub-directory against the version 8.0 state. Neither proves the
  correctness of every sub-directory nor re-reads the sources.
- **Old results of failed sub-directories** are not removed (see section 8).

---

## 15. Where to read more

| Document | Content |
|---|---|
| `input.json` | The reference configuration, commented inline |
| `README.md` | Installation, configuration, outputs, validation (in French) |
| `CHANGELOG.md` | Version-by-version record |
| `examples/CAS_TEST_00/` | A complete worked example: configuration, data and reference outputs |
| `examples/README.md` | How to replay the example and what it shows |
