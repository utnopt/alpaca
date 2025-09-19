# ALPACA

Code to ALPACA: Adaptive Linear Piecewise Approximation with Combinatorial Augmentation


## Features
PWL relaxation methods:
- multiple-choice method

Bilinear handling:
- McCormick envelopes
- reformulation to sum of squares
- piecewise constant relaxations

Separators:
- multipartite implication polytope

## Installation

create conda env with all required packages
```bash
        conda env create -f conda_env.yml
```

## Usage

- download osil files from www.minlplib.org/download.html
- put the files into data/import/instances
- adjust data/import/config.json
- run run.py
- get logs and results from export/

## Authors and acknowledgment
Code and model by: Robert Burlacu, Tobias Kuen, ...
