# Alpaca 🦙

**A**daptive **L**inear **P**iecewise **A**pproximation with **C**ombinatorial **A**ugmentation

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/utnopt/alpaca/releases/tag/v0.2.0)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Alpaca is a Python-based optimization framework for solving nonlinear programming (NLP) and mixed-integer nonlinear programming (MINLP) problems using piecewise linear (PWL) relaxations.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Computational Studies](#computational-studies)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Supported Expressions](#supported-expressions)
- [Stair-Locatelli Cuts](#stair-locatelli-cuts)
- [MPIP](#mpip)
- [External Solvers](#external-solvers)
- [References](#references)
- [License](#license)
- [Authors](#authors)

---

## Overview

Alpaca transforms nonlinear optimization problems into mixed-integer linear programs (MILPs) by decomposing complex expressions into low-dimensional components and approximating them with piecewise linear functions. The framework supports reading models in the OSiL (Optimization Services instance Language) format and provides advanced features such as:

- Automatic expression tree decomposition
- Multiple PWL formulation methods
- Bound propagation and tightening
- Cutting plane generation via Multipartite Implication Polytopes (MPIP)
- Stair-Locatelli cuts for tighter bilinear relaxations
- Automated computational study pipeline with parallel execution
- LaTeX table and TikZ plot generation for result evaluation

---

## Features

| Feature | Description |
|---------|-------------|
| **OSiL Model Import** | Parse optimization models from `.osil` XML files |
| **Expression Decomposition** | Automatically decompose nonlinear expressions into bilinear, multilinear, and one-dimensional components |
| **PWL Methods** | Multiple Choice Method and Delta Method for domain discretization |
| **Bilinear Handling** | McCormick envelopes, sum-of-squares reformulation, piecewise constant relaxation, or nonlinear |
| **Bound Propagation** | Manual propagation and Optimization-Based Bound Tightening (OBBT) |
| **Breakpoint Generation** | Uniform, adaptive (error-based), and neural network-based strategies |
| **Stair-Locatelli Cuts** | Domain projection and cutting planes for tighter bilinear relaxations |
| **MPIP Separation** | Advanced cutting planes based on multipartite implication polytopes |
| **Computational Studies** | Parallel execution pipeline with subprocess-based job scheduling |
| **Result Evaluation** | CSV-based evaluation with filtering, statistics, and outlier detection |
| **LaTeX Output** | Automated generation of LaTeX tables and TikZ scatter/bar plots |
| **Solver Support** | Gurobi and SCIP backends |

---

## Installation

### Prerequisites

- Python 3.11 or 3.12
- One of the following MIP solvers:
  - [Gurobi](https://www.gurobi.com/) (commercial, free academic license)
  - [SCIP](https://www.scipopt.org/) (open source via `pyscipopt`)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/utnopt/alpaca.git
cd alpaca

# Checkout the release tag
git checkout v0.2.0

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Install Solver Backend

Alpaca requires at least one MIP solver backend. Install your preferred solver:

```bash
# For SCIP (open source)
pip install pyscipopt

# For Gurobi (requires license)
pip install gurobipy
```

### Additional Dependencies for Studies

The computational study module requires additional packages for evaluation:

```bash
pip install numpy pandas scikit-learn
```

---

## Quick Start

### Basic Usage

```python
import alpaca as alp

# Load model from OSiL file
alpaca = alp.read_model_from_osil("path/to/model.osil")

# Configure logging
alpaca.configure_logging("logs/optimization.log", level="INFO")

# Customize settings (optional)
alpaca.customize_settings({"number_of_breakpoints": 10, "external_solver": "gurobi"})

# Build the PWL relaxation model
alpaca.build_pwl_relaxation_solver()

# Solve
alpaca.solve()
```

### Using a Configuration File

```python
import alpaca as alp

alpaca = alp.read_model_from_osil("instances/alkyl.osil")
alpaca.customize_settings("config/settings.json")
alpaca.build_pwl_relaxation_solver()
alpaca.solve()
```

### Accessing Statistics

```python
import alpaca as alp

alpaca = alp.read_model_from_osil("instances/pooling_adhya1pq.osil")
alpaca.customize_settings({"external_solver": "gurobi"})
alpaca.configure_logging("logs/run.log")
alpaca.build_pwl_relaxation_solver()
alpaca.solve()

# Access collected statistics
print(f"Solving time: {alpaca.statistics.solving_time}")
print(f"Solution value: {alpaca.statistics.solution_value}")
print(f"MIP gap: {alpaca.statistics.mip_gap}")
print(f"Build time: {alpaca.statistics.build_time}")
print(f"Locatelli cuts: {alpaca.statistics.locatelli_nr_cuts}")
```

---

## Computational Studies

Alpaca includes a full-featured computational study module for benchmarking configurations across multiple instances.

### Command-Line Interface

```bash
# Run a study with default paths
python -m alpaca.study run

# Run with custom paths
python -m alpaca.study run \
    -i ./instances \
    -c ./configs \
    -r ./results \
    -l ./results/logs \
    -w 4 \
    -t 4

# Run in background (survives shell close)
nohup python -m alpaca.study run -i ./instances -c ./configs -r ./results > study.log 2>&1 &

# Evaluate existing results
python -m alpaca.study evaluate --csv results/raw/study_results_2024-01-01_12-00-00.csv

# Evaluate with custom options
python -m alpaca.study evaluate \
    --csv results/raw/study_results.csv \
    -r ./results \
    --timelimit 3600 \
    --meantrim 0.05 \
    --base b_a_s_e
```

### CLI Options

| Command | Option | Default | Description |
|---------|--------|---------|-------------|
| `run` | `-i, --instances` | `./instances` | Directory containing `.osil` files |
| `run` | `-c, --configs` | `./configs` | Directory containing `.json` config files |
| `run` | `-r, --results` | `./results` | Output directory for CSV, tables, plots |
| `run` | `-l, --logs` | `./results/logs` | Log directory (`none` to disable) |
| `run` | `-w, --workers` | auto | Max parallel workers |
| `run` | `-t, --threads-per-job` | 4 | Threads per solver job |
| `run` | `--no-evaluate` | — | Skip evaluation after run |
| `evaluate` | `--csv` | (required) | Path to CSV results file |
| `evaluate` | `--timelimit` | 3600 | Time limit for timeout detection |
| `evaluate` | `--meantrim` | 0.05 | Outlier trimming fraction |
| `evaluate` | `--base` | `b_a_s_e` | Base config name for comparisons |

### Study Pipeline Architecture

The study pipeline uses **subprocess-based parallelism**:

1. **Discovery**: Finds all `.osil` instances and `.json` configs
2. **Scheduling**: Spawns separate Python processes per instance (visible in `ps aux`)
3. **Execution**: Each subprocess runs all configs sequentially for one instance, reusing OBBT bounds and Locatelli vertices across configs
4. **Collection**: Results are written immediately to CSV as each job completes
5. **Evaluation**: Generates LaTeX tables and TikZ plots from the CSV

```bash
# Monitor running jobs
ps aux | grep alpaca.study.run_job

# Watch results in real-time
tail -f results/raw/study_results_*.csv
```

### Generated Outputs

After a study completes, the following outputs are produced:

```
results/
├── raw/
│   ├── study_results_2024-01-01_12-00-00.csv
│   └── errors_2024-01-01_12-00-00.log
├── tables/
│   ├── table_instance_solution_time.tex
│   ├── table_instance_nr_nodes.tex
│   ├── table_instance_root_gap_reduction.tex
│   ├── table_instance_mip_gap.tex
│   ├── table_instance_model_size.tex
│   ├── table_instance_domain_volume_polygon.tex
│   └── table_instance_domain_volume_polytope.tex
├── plots/
│   ├── plot_root_gap_reduction.tex
│   └── plot_root_gap_reduction_bar.tex
└── logs/
    ├── instance1_config1.log
    └── instance1_config2.log
```

### Instance Filters

The evaluator applies intelligent filters to ensure meaningful comparisons:

| Filter | Description |
|--------|-------------|
| `NONE` | All complete instances (every config has a result) |
| `ALL_REACHED_ROOT` | All configs reached the root node before time limit |
| `ALL_FOUND_SOLUTION` | All configs found a feasible solution |
| `NON_EMPTY_BILINEAR_DOMAIN` | All configs have non-empty bilinear domains |
| `ALL_TERMINATED` | No config hit the time limit |
| `ALL_TERMINATED_CONSISTENT` | All terminated with consistent solution values |
| `BRANCH_AND_BOUND` | At least one config used branch-and-bound (>1 node) |
| `ROOT_SUBOPTIMAL` | At least one config has a suboptimal root solution |

---

## Configuration

Settings can be provided via a JSON file or a Python dictionary.

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed for reproducibility |
| `solver_time_limit` | int | 7200 | Time limit in seconds |
| `solver_thread_limit` | int | 4 | Number of threads |
| `external_solver` | str | `"scip"` | `"scip"` or `"gurobi"` |
| `number_of_breakpoints` | int | 5 | Number of PWL breakpoints per variable |
| `relaxation_tolerance` | float | 1e-4 | Tolerance for adaptive breakpoint generation |
| `pwl_method` | str | `"multiple_choice"` | `"multiple_choice"`, `"delta"`, or `"none"` |
| `approximation` | int | 0 | Approximation mode |

### Bilinear Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bilinear_handling` | int | 0 | `0`: McCormick, `1`: Sum of squares, `2`: Piecewise constant, `3`: Nonlinear |
| `reformulate_multilinear_to_bilinear` | int | 1 | Decompose multilinear terms into bilinear chains |

### Bound Propagation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bound_propagation` | int | 0 | `0`: Manual, `1`: OBBT, `2`: OBBT on bilinear |
| `bound_propagation_rounds` | int | 3 | Number of propagation iterations |
| `bound_propagation_obbt_time_limit` | int | 300 | OBBT time budget in seconds |

### Breakpoint Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `breakpoint_generation` | int | 0 | `0`: Uniform, `1`: Neural network, `2`: Adaptive |
| `feature/nnbp/learning_rate` | float | 1e-7 | Learning rate for NN-based generation |
| `feature/nnbp/nr_of_samples` | int | 1000 | Number of training samples |
| `feature/nnbp/time_limit` | int | 100 | Training time limit |
| `feature/nnbp/queue_size` | int | 5 | Queue size for convergence |
| `feature/nnbp/convergence_tol` | float | 1e-2 | Convergence tolerance |

### Stair-Locatelli

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature/stair_locatelli` | int | 0 | `0`: Disabled, `1`: Locatelli, `2`: Stair-Locatelli, `3`: Indicator Locatelli |
| `feature/stair_locatelli/grid_size` | int | 10 | Grid size for domain projection |
| `feature/stair_locatelli/mu` | float | 1e-3 | Distance from lower bound |
| `feature/stair_locatelli/obbt_time_limit` | int | 1800 | OBBT time limit for projection |
| `feature/stair_locatelli/evaluation_grid_size` | int | 100 | Grid size for volume evaluation |

### MPIP Features

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature/mpip/separation` | int | 0 | Enable separation cuts |
| `feature/mpip/mccormick` | int | 0 | Add MPIP-based McCormick constraints |
| `feature/mpip/corner` | int | 0 | Add corner constraints |
| `feature/mpip/stair` | int | 0 | Add stair constraints |
| `feature/mpip/stripe` | int | 0 | Add stripe constraints |
| `feature/mpip/bar` | int | 0 | Add bar constraints |
| `feature/mpip/frequency` | int | 10 | Separation callback frequency |
| `feature/mpip/useless_threshold` | float | 0.1 | Threshold for useless cuts |
| `feature/mpip/reset_interval` | int | 300 | Interval for cut pool reset |

### Instance Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter/no_bilinear_expressions` | int | 0 | Skip instances with no bilinear expressions |
| `filter/no_mpip_instances` | int | 0 | Skip instances with no MPIP structures |
| `filter/unbounded_variables` | int | 1 | Filter unbounded variables |
| `filter/max_nr_variables` | int | 1e6 | Maximum number of variables |

### Example Configuration File

```json
{
    "seed": 42,
    "solver_time_limit": 3600,
    "solver_thread_limit": 4,
    "external_solver": "gurobi",
    "number_of_breakpoints": 10,
    "pwl_method": "multiple_choice",
    "bilinear_handling": 0,
    "bound_propagation": 1,
    "bound_propagation_rounds": 3,
    "feature/stair_locatelli": 2,
    "feature/stair_locatelli/grid_size": 10,
    "feature/mpip/separation": 1,
    "feature/mpip/mccormick": 1,
    "feature/mpip/frequency": 10
}
```

---

## Architecture

```
alpaca/
├── src/alpaca/
│   ├── __init__.py              # Package init, exports Alpaca and read_model_from_osil
│   ├── main.py                  # Main Alpaca class and entry point
│   ├── settings.py              # Static and user-configurable settings
│   ├── run.py                   # Example usage and script execution
│   ├── solver/
│   │   └── solver.py            # Solver orchestration
│   ├── model_data/
│   │   ├── model_data.py        # Central model container
│   │   ├── variable.py          # Variable representation
│   │   └── constraint.py        # Constraint representation
│   ├── model_buildup/
│   │   ├── osil_reader.py       # OSiL file parser
│   │   ├── expression_tree.py   # Expression tree decomposition
│   │   ├── multilinear_handler.py
│   │   ├── bound_propagator.py  # Bound propagation & OBBT
│   │   ├── breakpoint_generator.py
│   │   └── pwl_handler.py       # PWL relaxation application
│   ├── expressions/
│   │   ├── expression.py
│   │   ├── expression_container.py
│   │   ├── one_dim_expression.py
│   │   ├── bilinear_expression.py
│   │   ├── bilinear_binary_expression.py
│   │   ├── bilinear_mixed_binary_expression.py
│   │   ├── multilinear_expression.py
│   │   ├── linear_expression.py
│   │   └── nonlinear_expression.py
│   ├── pwl/
│   │   ├── pwl_method.py
│   │   ├── multiple_choice_method.py
│   │   └── delta_method.py
│   ├── breakpoints/
│   │   ├── breakpoint_adaptive.py
│   │   └── breakpoint_neural_network.py
│   ├── external_solvers/
│   │   ├── mip_model.py         # MIP model construction
│   │   └── solver_wrapper.py    # Gurobi/SCIP abstraction
│   ├── locatelli/
│   │   ├── stair_locatelli.py       # Stair-Locatelli orchestration
│   │   ├── domain_projector.py      # Feasible domain polygon computation
│   │   └── locatelli_cut_generator.py # Cutting plane generation
│   ├── mpip/
│   │   ├── mpip.py              # MPIP data structure
│   │   ├── mpip_handler.py      # MPIP extraction
│   │   └── separation/
│   │       ├── mpip_separationhandler.py
│   │       └── mpip_separator.py
│   ├── stats/
│   │   └── statistics.py        # Statistics collection and reporting
│   ├── study/
│   │   ├── __main__.py          # CLI entry point (python -m alpaca.study)
│   │   ├── pipeline.py          # Parallel study execution pipeline
│   │   ├── run_job.py           # Single-instance subprocess worker
│   │   ├── evaluator.py         # CSV-based result evaluation and filtering
│   │   ├── latex_generator.py   # LaTeX table generation
│   │   └── tikz_generator.py    # TikZ plot generation
│   └── utils/
│       ├── inout.py             # File I/O and logger configuration
│       ├── logger.py            # Logging utilities
│       ├── geometry.py          # Geometric utility functions
│       └── lsf/
│           └── localized_string_factory.py  # Centralized string constants
├── test/                        # Test suite
├── pyproject.toml
└── README.md
```

---

## Supported Expressions

### One-Dimensional Functions

| Expression | Mathematical Form |
|------------|-------------------|
| Square | $x^2$ |
| Exponential | $e^x$ |
| Natural Logarithm | $\ln(x)$ |
| Square Root | $\sqrt{x}$ |
| Sine | $\sin(x)$ |
| Cosine | $\cos(x)$ |
| Log Base 10 | $\log_{10}(x)$ |
| Hyperbolic Tangent | $\tanh(x)$ |
| Inverse | $x^{-1}$ |
| Absolute Value | $\|x\|$ |
| Power | $x^y$ |

### Multilinear Expressions

- **Bilinear**: $z = x \cdot y$
- **Bilinear Binary**: $z = x \cdot y$ where $x, y \in \{0, 1\}$
- **Bilinear Mixed Binary**: $z = b \cdot x$ where $b \in \{0, 1\}$
- **Multilinear**: $z = \prod_{i} x_i$

---

## Stair-Locatelli Cuts

The Stair-Locatelli module strengthens bilinear relaxations by:

1. **Domain Projection**: For each bilinear term $z = x \cdot y$, the feasible region is projected onto the $(x, y)$-plane by solving a series of optimization subproblems over a grid. This produces an orthogonal polygon that tightly contains the feasible domain.

2. **Cut Generation**: Valid cutting planes are derived from all combinations of three vertices of the projected polygon. Each cut is verified against boundary checkpoints to ensure validity as either an underestimator or overestimator of the bilinear term.

3. **Modes**:
   - **Locatelli** (`feature/stair_locatelli = 1`): Uses the convex hull of the projected polygon for cut generation.
   - **Stair-Locatelli** (`feature/stair_locatelli = 2`): Uses the full (non-convex) staircase polygon for tighter cuts.
   - **Indicator Locatelli** (`feature/stair_locatelli = 3`): Simplified projection for indicator-type bilinear constraints.

4. **Volume Evaluation**: The module computes the 3D volume of the McCormick relaxation over the projected domain to quantify the tightening effect of the cuts.

### References

The Locatelli cut generation approach is based on:

> Locatelli and Schoen (2014). *On convex envelopes for bivariate functions over polytopes*.
> [https://optimization-online.org/?p=26208](https://optimization-online.org/?p=26208)

The Stair-Locatelli extension to non-convex polygonal domains is described in:

> Göß, Krause, Kuchlbauer and Kuen (2026). *On convex envelopes of bivariate functions over polygons*.

---

## MPIP

The **Multipartite Implication Polytope (MPIP)** module exploits conditional relationships between sets of binary variables that arise in piecewise-linear relaxations. When continuous variables are discretized, each variable's domain is partitioned into intervals represented by binary indicator variables (subject to SOS1 constraints). The functional relationship between variables induces logical implications: if a certain combination of intervals is active for the input variables, only a subset of intervals can be active for the output variable.

The MPIP module:

1. **Extraction**: Automatically detects MPIP structures in the PWL relaxation by analyzing nonlinear expression trees, bilinear terms, and multilinear terms.

2. **Relation Computation**: For each MPIP instance, computes the implication relation by solving interval bounding problems — determining which output intervals are reachable for each combination of input intervals.

3. **Constraint Generation**: Adds various families of valid inequalities derived from the MPIP structure:
   - **McCormick constraints**: Basic implication constraints linking input and output binaries.
   - **Stripe constraints**: Aggregated constraints along single input dimensions.
   - **Bar constraints**: Row- and column-based aggregated constraints (bipartite case).
   - **Corner constraints**: Staircase-shaped constraints exploiting monotonicity (bipartite case).

4. **Separation**: A callback-based separation routine that dynamically generates violated MPIP cuts during branch-and-bound by solving a separation LP for each fractional solution.

### References

The theoretical foundation of implication polytopes over multiple sets of binary variables is described in:

> Burlacu, Gemander and Kuen (2024). *The Bipartite Implication Polytope: Conditional Relations over Multiple Sets of Binary Variables*.
> [https://optimization-online.org/?p=26208](https://optimization-online.org/?p=26208)

The identification and application of MPIP instances within piecewise-linear relaxations is presented in:

> Braun, Burlacu, Kuen and Rolsing (2026). *The Bipartite Implication Polytope: Modeling Binary Relations in Piecewise-Linear Approximations*.
> [https://optimization-online.org/?p=33773](https://optimization-online.org/?p=33773)

---

## External Solvers

Alpaca provides a unified interface for both Gurobi and SCIP:

```python
# Use Gurobi
alpaca.customize_settings({"external_solver": "gurobi"})

# Use SCIP
alpaca.customize_settings({"external_solver": "scip"})
```

Solver-specific features (e.g., callbacks, separation handlers, log parsing) are automatically configured based on the selected backend.

---

## References

If you use Alpaca in your research, please cite the following:

> Braun, Burlacu, Kuen and Rolsing (2026). *The Bipartite Implication Polytope: Modeling Binary Relations in Piecewise-Linear Approximations*.
> [https://optimization-online.org/?p=33773](https://optimization-online.org/?p=33773)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Authors

- **Tobias Kuen**
- **Robert Burlacu**
- **Dennis Cost**