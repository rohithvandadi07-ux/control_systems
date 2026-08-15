
# Battery State Estimation and Equivalent Circuit Modelling

A physics-based and data-driven battery modelling framework developed using NASA battery cycling data. The project combines battery data preprocessing, State of Charge (SOC) estimation, Open Circuit Voltage (OCV) characterization, Equivalent Circuit Model (ECM) implementation in MATLAB/Simulink, and a TCN-LSTM-based data-driven SOC estimation pipeline.

> **Current milestone:** The core system-modelling stage has been completed to the required baseline level. The ECM has been implemented in Simulink and validated against NASA B0005 Cycle 42.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [SOC Estimation](#soc-estimation)
- [OCV-SOC Characterization](#ocv-soc-characterization)
- [Equivalent Circuit Model](#equivalent-circuit-model)
- [MATLAB and Simulink Implementation](#matlab-and-simulink-implementation)
- [ECM Validation](#ecm-validation)
- [Validation Results](#validation-results)
- [Deep Learning Component](#deep-learning-component)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Running the Project](#running-the-project)
- [Current Status](#current-status)
- [Future Work](#future-work)
- [Research Direction](#research-direction)
- [License](#license)

---

## Project Overview

Accurate battery state estimation and system modelling are important components of Battery Management Systems (BMS), particularly for electric vehicles, energy storage systems, and battery monitoring applications.

This project uses experimental battery cycling data from the NASA battery dataset to develop and evaluate a battery modelling pipeline.

The current workflow includes:

1. NASA battery data loading and preprocessing
2. Battery discharge-cycle extraction
3. Current and voltage analysis
4. Coulomb-counting-based SOC estimation
5. OCV-SOC relationship characterization
6. OCV-SOC lookup-table generation
7. Equivalent Circuit Model development
8. MATLAB/Simulink implementation
9. Experimental-versus-model voltage comparison
10. Quantitative ECM validation
11. TCN-LSTM-based SOC estimation experiments

---

## Objectives

The main objectives are:

- Process and analyze NASA battery cycling data.
- Extract useful current, voltage, time, and capacity measurements.
- Estimate battery State of Charge (SOC).
- Establish an OCV-SOC relationship.
- Develop an OCV lookup table for the ECM.
- Implement an Equivalent Circuit Model in MATLAB/Simulink.
- Validate the ECM against experimental NASA battery measurements.
- Evaluate model performance using MAE, RMSE, and R².
- Develop a TCN-LSTM pipeline for data-driven SOC estimation.
- Establish a baseline for future hybrid physics-based and data-driven battery modelling.

---

## System Architecture

The current project can be represented as:

```text
                    NASA Battery Dataset
                             |
                             v
                    Data Loading / Extraction
                             |
                             v
                     Data Preprocessing
                             |
              +--------------+--------------+
              |                             |
              v                             v
       Current / Voltage              Deep Learning
           Analysis                      Pipeline
              |                             |
              v                             v
       Coulomb Counting                 TCN-LSTM
              |                       SOC Estimation
              v
         SOC Estimation
              |
              v
       OCV-SOC Characterization
              |
              v
        OCV Lookup Table
              |
              v
      Equivalent Circuit Model
              |
              v
        MATLAB / Simulink
              |
              v
        ECM Terminal Voltage
              |
              v
   Interpolation to NASA Time Points
              |
              v
      Experimental vs ECM Voltage
              |
              v
       MAE / RMSE / R² Validation
```

---

## Dataset

The repository currently contains NASA battery datasets for:

- `B0005`
- `B0006`
- `B0007`
- `B0018`

The primary ECM validation performed in the current milestone uses:

**NASA B0005 — Cycle 42**

### B0005 Cycle 42

The extracted validation cycle contains:

| Parameter | Value |
|---|---:|
| Samples | 192 |
| Duration | 3591.734 s |
| Capacity | 1.847026 Ah |
| Current range | -2.016847 to 0.005955 A |
| Voltage range | 2.632143 to 4.187614 V |

The raw NASA datasets are stored under:

```text
data/raw/
```

---

## Data Preprocessing

The preprocessing stage extracts the measurements required for battery modelling.

The main signals used are:

- Time
- Measured current
- Measured voltage
- Battery capacity

The preprocessing workflow performs:

- Data extraction
- Numeric conversion
- Vector orientation normalization
- Length consistency checks
- Invalid-value removal
- Time normalization

For the B0005 Cycle 42 validation data, the resulting vectors contain 192 valid samples.

The main preprocessing script is:

```text
scripts/preprocess_nasa.py
```

---

## SOC Estimation

SOC is estimated using Coulomb counting.

For a discharge cycle, accumulated removed charge is obtained by integrating current with respect to time:

```text
Q_removed = integral(I dt) / 3600
```

where the division by `3600` converts ampere-seconds to ampere-hours.

The normalized SOC is then calculated as:

```text
SOC = 1 - Q_removed / Q
```

where:

- `SOC` is the normalized State of Charge.
- `Q_removed` is the accumulated discharged capacity in Ah.
- `Q` is the measured battery capacity in Ah.

The resulting SOC is constrained to the physical range:

```text
0 <= SOC <= 1
```

For the B0005 Cycle 42 validation cycle, the initial SOC used for the Simulink input is:

```text
SOC_initial = 1.0
```

The SOC decreases as charge is removed during the discharge process.

---

## OCV-SOC Characterization

The battery's Open Circuit Voltage (OCV) relationship is represented using an SOC-dependent lookup table.

The lookup table used by the current ECM contains 21 SOC points.

### OCV-SOC Lookup Table

| SOC | OCV (V) |
|---:|---:|
| 0.00 | 3.292744 |
| 0.05 | 3.403797 |
| 0.10 | 3.550025 |
| 0.15 | 3.604772 |
| 0.20 | 3.635909 |
| 0.25 | 3.659696 |
| 0.30 | 3.678855 |
| 0.35 | 3.696174 |
| 0.40 | 3.713818 |
| 0.45 | 3.733248 |
| 0.50 | 3.756212 |
| 0.55 | 3.781671 |
| 0.60 | 3.808977 |
| 0.65 | 3.838356 |
| 0.70 | 3.869335 |
| 0.75 | 3.902132 |
| 0.80 | 3.936341 |
| 0.85 | 3.972473 |
| 0.90 | 4.007215 |
| 0.95 | 4.041096 |
| 1.00 | 4.191711 |

The corresponding MATLAB lookup-table data is stored in:

```text
matlab/ocv_table.mat
```

---

## Equivalent Circuit Model

The core system-modelling component is an Equivalent Circuit Model implemented in MATLAB/Simulink.

The ECM represents the battery electrically using:

- SOC-dependent OCV
- Electrical resistance
- Dynamic polarization / RC behaviour
- Battery current
- Terminal voltage

The Simulink model is:

```text
simulink/battery/battery_ecm.slx
```

The ECM is used to reproduce the battery terminal-voltage response under the measured discharge current profile.

---

## MATLAB and Simulink Implementation

### MATLAB

MATLAB is used for:

- NASA dataset loading
- Cycle extraction
- Data preprocessing
- SOC calculation
- OCV-SOC characterization
- Lookup-table generation
- Validation
- Error calculation
- Visualization

The primary MATLAB script is:

```text
matlab/load_nasa_cycle.m
```

### Simulink

Simulink is used for:

- ECM implementation
- Dynamic battery simulation
- Terminal-voltage estimation
- Model-output generation

The main model is:

```text
simulink/battery/battery_ecm.slx
```

---

## ECM Validation

The ECM was validated using:

```text
NASA B0005
Cycle 42
```

The NASA validation data contains 192 measurement samples.

The Simulink ECM produces 66 simulation samples.

Because the two datasets do not have identical time grids, the ECM output is interpolated onto the NASA measurement time points before calculating the validation metrics.

Conceptually:

```text
NASA measurement time
        |
        | 192 points
        v
Experimental Voltage
        |
        | comparison
        |
ECM Simulation
        |
        | 66 simulation points
        v
Interpolation
        |
        v
ECM Voltage at NASA Times
```

This ensures that measured and simulated voltages are compared at corresponding time instants.

---

## Validation Results

The current ECM validation produced the following results:

| Metric | Result |
|---|---:|
| **MAE** | **0.027106 V** |
| **RMSE** | **0.052137 V** |
| **R²** | **0.950428** |

### Interpretation

The current baseline ECM achieves an R² of approximately **0.95**, indicating that it captures most of the voltage variation observed in the validation cycle.

The model follows the experimental voltage trajectory closely over most of the discharge period.

The largest mismatch occurs near the end-of-discharge region, where the measured battery voltage exhibits a sharp drop. The current ECM does not reproduce this drop completely.

This behaviour is retained as a known limitation and a target for future model refinement.

### Validation Error

The voltage prediction error is calculated as:

```text
error = V_measured - V_ECM
```

The current validation therefore provides both:

- A quantitative baseline using MAE, RMSE, and R²
- A visual comparison between experimental and simulated voltage

---

## Validation Outputs

The repository contains the generated validation data:

```text
results_ecm_validation_B0005.csv
```

The project also generated MATLAB figures for:

- Validation SOC
- Validation current
- Measured voltage
- Measured-versus-ECM voltage
- ECM voltage prediction error

These plots were used to inspect the model behaviour during validation.

---

## Deep Learning Component

In parallel with the physics-based ECM, the repository contains a TCN-LSTM-based SOC estimation pipeline.

The implementation is located at:

```text
models/tcn_lstm/
```

### Components

```text
models/tcn_lstm/
├── dataset.py
├── evaluate.py
├── model.py
└── train.py
```

The model combines:

- Temporal Convolutional Network (TCN) processing
- Long Short-Term Memory (LSTM) temporal modelling

The purpose of this component is to investigate data-driven SOC estimation from battery time-series data.

The current repository contains experimental TCN-LSTM results for NASA B0018.

### TCN-LSTM Outputs

```text
results/tcn_lstm/
├── best_model.pt
├── evaluation_B0018.csv
├── normalized_dataset.csv
├── soc_error_B0018.png
├── soc_prediction_B0018.png
└── training_history.csv
```

The deep-learning component is currently treated as a complementary data-driven branch rather than replacing the physics-based ECM.

---

## Repository Structure

```text
control_systems/
│
├── .gitignore
│
├── ECM_B0005_baseline.mat
│
├── data/
│   ├── raw/
│   │   ├── B0005.mat
│   │   ├── B0006.mat
│   │   ├── B0007.mat
│   │   └── B0018.mat
│   │
│   └── processed/
│       └── nasa_discharge_reference_soc.csv
│
├── matlab/
│   ├── load_nasa_cycle.m
│   └── ocv_table.mat
│
├── models/
│   ├── __init__.py
│   │
│   └── tcn_lstm/
│       ├── __init__.py
│       ├── dataset.py
│       ├── evaluate.py
│       ├── model.py
│       └── train.py
│
├── results/
│   └── tcn_lstm/
│       ├── best_model.pt
│       ├── evaluation_B0018.csv
│       ├── normalized_dataset.csv
│       ├── soc_error_B0018.png
│       ├── soc_prediction_B0018.png
│       └── training_history.csv
│
├── results_ecm_validation_B0005.csv
│
├── scripts/
│   └── preprocess_nasa.py
│
└── simulink/
    └── battery/
        └── battery_ecm.slx
```

### Ignored Files

The following types of generated or environment-specific files are intentionally excluded from version control:

```text
.venv/
__pycache__/
*.slxc
slprj/
```

This keeps the repository focused on source code, datasets, models, Simulink files, and reproducible results.

---

## Requirements

### MATLAB

The ECM implementation requires:

- MATLAB
- Simulink

Development and testing were performed using:

```text
MATLAB R2026a
```

### Python

The preprocessing and TCN-LSTM components require Python.

The project uses packages including:

```text
numpy
pandas
scikit-learn
torch
```

A virtual environment can be created using:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies according to the project's Python environment.

---

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd control_systems
```

### 2. MATLAB / Simulink

Open MATLAB and navigate to the project directory.

Run the MATLAB processing script:

```text
matlab/load_nasa_cycle.m
```

The Simulink ECM is located at:

```text
simulink/battery/battery_ecm.slx
```

The workflow extracts the NASA cycle, prepares the model inputs, runs the ECM, interpolates the simulated output to the NASA measurement time points, and evaluates the resulting voltage prediction.

### 3. Python TCN-LSTM

Activate the Python environment:

```bash
source .venv/bin/activate
```

The TCN-LSTM implementation is located in:

```text
models/tcn_lstm/
```

Training and evaluation scripts are provided through:

```text
models/tcn_lstm/train.py
models/tcn_lstm/evaluate.py
```

---

## Current Status

### Completed

- [x] NASA battery dataset integration
- [x] B0005, B0006, B0007 and B0018 data available
- [x] Battery cycle extraction
- [x] Current and voltage preprocessing
- [x] SOC calculation using Coulomb counting
- [x] OCV-SOC characterization
- [x] OCV lookup-table generation
- [x] Equivalent Circuit Model development
- [x] Simulink implementation
- [x] ECM simulation
- [x] NASA/Simulink time alignment
- [x] ECM output interpolation
- [x] B0005 Cycle 42 validation
- [x] MAE calculation
- [x] RMSE calculation
- [x] R² calculation
- [x] Validation plots
- [x] TCN-LSTM implementation
- [x] TCN-LSTM experimental evaluation

### Current Milestone

The **system-modelling baseline is complete**.

The ECM has been implemented and validated against NASA B0005 Cycle 42 with:

```text
MAE  = 0.027106 V
RMSE = 0.052137 V
R²   = 0.950428
```

This baseline is now preserved for subsequent refinement and extension.

---

## Future Work

The next development stages can include:

1. Investigating the end-of-discharge voltage mismatch.
2. Refining ECM electrical parameters.
3. Evaluating the ECM on additional NASA cycles.
4. Performing cross-cycle validation.
5. Comparing different ECM configurations.
6. Improving low-SOC voltage behaviour.
7. Integrating SOC estimation with the ECM.
8. Investigating joint SOC and battery degradation estimation.
9. Comparing physics-based and deep-learning approaches.
10. Developing a unified hybrid battery state-estimation framework.

The current baseline results should be preserved before parameter refinement so that future improvements can be measured against the current model.

---

## Research Direction

The long-term direction is to combine physics-based and data-driven battery modelling.

```text
                         Battery Data
                              |
              +---------------+---------------+
              |                               |
              v                               v
       Physics-Based                    Data-Driven
            ECM                           TCN-LSTM
              |                               |
              v                               v
       Voltage Modelling                 SOC Estimation
              |                               |
              +---------------+---------------+
                              |
                              v
                    Future Hybrid Estimator
```

The ECM provides an interpretable representation of battery electrical behaviour, while the TCN-LSTM branch provides a data-driven approach for learning temporal battery behaviour.

The eventual objective is to investigate whether the complementary strengths of both approaches can be combined into a more robust battery state-estimation framework.

---

## Project Status Summary

| Component | Status |
|---|---|
| NASA Dataset Integration | Completed |
| Data Preprocessing | Completed |
| SOC Estimation | Completed |
| OCV Characterization | Completed |
| OCV Lookup Table | Completed |
| ECM Development | Completed |
| Simulink Implementation | Completed |
| B0005 Cycle 42 Validation | Completed |
| ECM Performance Evaluation | Completed |
| TCN-LSTM Pipeline | Implemented |
| Cross-Cycle ECM Validation | Future Work |
| ECM Refinement | Future Work |
| Hybrid ECM + TCN-LSTM | Future Work |

---

## License

This project is intended for academic and research purposes.

---

## Acknowledgement

The project uses NASA battery cycling data for battery modelling, analysis, and validation.
