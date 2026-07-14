# Battery State of Health & RUL Prediction
## Explainable ML Across Discharge and Charge Domains

> **Course:** AI for Connected Industries
> **Module**: Fundamentals of Machine Learning  
> **Institution:** Universität Ulm  
> **Supervisor:** Prof. Dr. Reinhold von Schwerin  
> **Author:** Adithya Prasad Shivarpatna Venkatesh  
> **GitHub:** [adithyaprasadsv](https://github.com/adithyaprasadsv)

---

## Overview

This project predicts **State of Health (SoH)** and **Remaining Useful Life (RUL)**
of lithium-ion batteries using explainable machine learning on the
[NASA PCOE Battery Dataset](https://catalog.data.gov/dataset).
The methodology follows the **CRISP-DM** framework across two complementary domains:

| Part | Domain | Input | Method | Best RMSE |
|------|---------|-------|--------|-----------|
| 1 | Discharge curves | 14 physics-informed features | Ridge / RF / XGBoost + SHAP | **0.0041 Ah** |
| 2 | Charging curves | HF1, HF2, HF3, HF6 | Multi-output LSTM → XGBoost + SHAP | **0.0434 Ah** (Q75 calibrated) |

**Key finding:** SHAP analysis across both domains independently identifies
internal resistance as the dominant SoH driver and confirmed from discharge
transient response, charging curve time constants, and electrochemical
impedance spectroscopy.

---

## Project Structure

```
battery-soh-rul/
│
├── data/                          # NASA PCOE .mat files (not tracked)
│   ├── B0005.mat
│   ├── B0006.mat
│   ├── B0007.mat
│   └── B0018.mat
│
├── notebooks/
│   ├── battery_soh_EDA.ipynb      # Business Understanding, Data Understanding: EDA
│   └── battery_soh_script.ipynb   # Modeling: Pre-processsing, Build, Evaluation
│
├── results/
│   ├── B0018_SoH_RUL_all_models.csv   # Full prediction table
│   ├── phase_rmse_soh.csv             # Phase-wise SoH RMSE
│   └── phase_rmse_rul.csv             # Phase-wise RUL RMSE
│
├── assets/
│
├── report/
│   └── battery_soh_report.pdf    
│
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

**Batteries used:**

| Battery | Cycles | Discharge current | Cutoff | Ambient | Role |
|---------|--------|-------------------|--------|---------|------|
| B0005 | 168 | 2A constant | 2.7V | 24°C | Train |
| B0006 | 168 | 2A constant | 2.7V | 24°C | Train |
| B0007 | 168 | 2A constant | 2.7V | 24°C | Train |
| B0018 | 132 | 2A constant | 2.7V | 24°C | **Test (holdout)** |

> **Battery selection note:** 20 additional batteries (B0025–B0044) were
> evaluated against three protocol-consistency criteria (discharge current,
> cutoff voltage, ambient temperature). Only B0036 qualified; it was
> excluded from the final model to maintain a clean four-battery
> comparable study matching Zhao et al. (2022).

Download `.mat` files from the NASA repository and place in `data/`.

---

## Installation

```bash
git clone https://github.com/adithyaprasadsv/battery-soh-rul.git
cd battery-soh-rul
pip install -r requirements.txt
```

---

## Methodology: CRISP-DM

### Business Understanding
Predict SoH (capacity\_Ah) and RUL (cycles to EOL at 1.4 Ah) from signals
available in a BMS. Explainability is a primary deliverable alongside accuracy.

### Data Understanding: EDA Findings
- Capacity degrades non-monotonically; local recovery bumps require
  rolling features, not point-in-time statistics
- Voltage plateau shortens with cycle number directly motivates
  `discharge_dur_s` and `R_int_proxy` features
- B0018 diverges from training batteries by up to 0.20 Ah in cycles 45–88 acts as
  the primary source of cross-battery generalisation error
- **Split strategy matters:** random shuffle produces artificially low RMSE
  due to cycle-level leakage across batteries

### Data Preparation

**Part 1: Discharge features (14 total):**

| Category | Features |
|----------|----------|
| Statistical | `discharge_dur_s`, `voltage_mean`, `voltage_std`, `voltage_slope`, `voltage_at_80pct`, `temp_rise` |
| Physics-informed | `R_int_proxy` (ΔV/ΔI), `energy_Wh`, `Q_cum_Ah`, `Q_temp_compensated`, `dQdV_peak_height`, `dQdV_peak_voltage` |
| Trend | `rolling_mean_5`, `rolling_std_5` (shift(1) to prevent leakage) |

**Part 2: Charge health factors (4 selected from 6):**

| HF | Description | Spearman r | Selected |
|----|-------------|-----------|----------|
| HF1 | CC charge time 3.9→4.2V | 0.935 | ✓ |
| HF2 | Voltage rise within 600s | −0.919 | ✓ |
| HF3 | CV current drop within 900s | 0.598 | ✓ |
| HF4 | Exponential fit floor | −0.172 | ✗ |
| HF5 | Exponential fit amplitude | 0.030 | ✗ |
| HF6 | RC time constant | −0.871 | ✓ |

## Feature Glossary

### Discharge-Domain Features

| Feature | Description |
|---|---|
| `discharge_dur_s` | Total discharge duration (seconds) from cycle start to 2.7V cutoff. Strongest SoH predictor (r=0.95), at constant 2A, duration directly encodes charge delivered. |
| `voltage_mean` | Mean terminal voltage over the full discharge. Declines with age as internal resistance reduces the usable voltage window. |
| `voltage_std` | Standard deviation of terminal voltage during discharge. Increases with age as the voltage curve shape changes. |
| `voltage_slope` | Linear regression slope of voltage over time (V/s). Becomes more negative with aging as resistance accelerates voltage decay. |
| `voltage_at_80pct` | Terminal voltage at 80% of discharge duration. Position-specific snapshot capturing mid-discharge behaviour independent of total duration. |
| `temp_rise` | Peak minus start temperature within a cycle (°C). Used instead of `temp_max` to cancel inter-battery ambient offsets, rises with internal resistance as resistive heating increases. |
| `R_int_proxy` | Internal resistance proxy: ΔV/ΔI averaged over first 10 timesteps. Approximates R_int from the transient voltage drop under applied load. Rises monotonically with aging. |
| `energy_Wh` | Energy delivered per cycle: ∫\|V·I\|dt / 3600 (Wh). Captures both voltage and current contributions. Top SHAP feature in discharge domain. |
| `Q_cum_Ah` | Cumulative charge delivered per cycle: ∫\|I\|dt / 3600 (Ah). At constant 2A discharge, near-perfectly correlated with `discharge_dur_s`. |
| `Q_temp_compensated` | Temperature-compensated capacity: Q / (1 + α(T_amb − T_ref)) where α = 0.005 °C⁻¹. Removes thermal drift from the capacity signal. |
| `dQdV_peak_height` | Incremental capacity (dQ/dV) peak height extracted over the 2.7–3.7V voltage window. Peaks correspond to electrode phase transitions; amplitude decreases with aging. Normalised by discharge duration to remove early-cycle artifact. |
| `dQdV_peak_voltage` | Voltage position of the dQ/dV peak. Shifts toward lower voltages with aging as phase transition dynamics change. |
| `rolling_mean_5` | Rolling mean of capacity_Ah over the preceding 5 cycles, computed with shift(1) to prevent target leakage. Captures local degradation trend through non-monotonic recovery regions. |
| `rolling_std_5` | Rolling standard deviation of capacity_Ah over the preceding 5 cycles with shift(1). High values indicate proximity to a recovery bump or anomalous cycle. |

---

### Charge-Domain Health Factors

| Feature | Description |
|---|---|
| `HF1` | Time for terminal voltage to rise from 3.9V to 4.2V during CC charging (seconds). Increases with R_int as higher resistance slows the voltage rise rate. Spearman r = 0.935 with capacity. Degradation rate correlation r = 0.997 across batteries. |
| `HF2` | Voltage increase within 600s after reaching 3.9V during CC charging (V). Decreases with aging as the cell charges more slowly per unit time. Spearman r = −0.919. |
| `HF3` | Current reduction within 900s after entering the CV phase (A). Reflects how quickly the cell accepts charge at fixed voltage. Spearman r = 0.598. |
| `HF4` | Asymptotic current floor of exponential CV decay fit: I(t) = HF4 + HF5·exp(−t/HF6). Near-zero correlation with capacity (Spearman r = −0.172). Excluded from model. |
| `HF5` | Initial amplitude of exponential CV decay fit. Near-zero correlation with capacity (Spearman r = 0.030). Excluded from model. |
| `HF6` | RC time constant of exponential CV current decay (seconds). Directly measures the equivalent circuit time constant; increases with R_int as aging raises internal resistance. Top SHAP feature in charge domain. Spearman r = −0.871. |
| `HF6_rolling_mean5` | Rolling mean of HF6 over the preceding 5 cycles with shift(1). Highest SHAP rank in charge-domain XGBoost, local trend of RC time constant is more predictive than its instantaneous value. |
| `HF1_rolling_mean5` | Rolling mean of HF1 over the preceding 5 cycles with shift(1). Captures the rate of CC charge time increase rather than its absolute level. |

---

### EIS-Derived Features (Omitted due to missing cycles)

| Feature | Description |
|---|---|
| `Re` | Electrolyte resistance (Ω) from EIS. High-frequency intercept on the Nyquist plot. Increases with electrolyte decomposition and lithium salt depletion. Spearman r = −0.64 with capacity (n ≈ 170 cycles). |
| `Rct` | Charge transfer resistance (Ω) from EIS. Nyquist semicircle diameter. Increases with SEI layer growth and active material loss. Spearman r = −0.70 with capacity. Stronger predictor than Re, consistent with SEI growth being the dominant aging mechanism in LiCoO₂ at room temperature. |


### Modeling

**Part 1:** Ridge (baseline) → Random Forest → XGBoost (tuned via
`GridSearchCV` with `GroupKFold`, 50 iterations). All models in
`sklearn` Pipelines with `StandardScaler`.

**Part 2:** Multi-output LSTM (2 layers, hidden=32, dropout=0.3, Adam)
predicts future HF trajectories from 10-cycle sliding windows.
XGBoost maps predicted HFs to SoH. Q90 post-hoc calibration corrects
systematic under-prediction from inter-battery capacity offset.

**Critical implementation decisions:**
- `GroupKFold` by battery ID prevents cycle-level leakage in CV
- `shift(1)` on rolling features prevents target leakage
- Cross-battery validation split
- Cycle index excluded from XGBoost-HF features as it can leak future cycle info

---

## Results

### SoH Prediction: B0018 Holdout

| Model | RMSE (Ah) | Domain |
|-------|-----------|--------|
| **Ridge (discharge)** | **0.0041** | | Discharge |
| XGBoost (discharge) | 0.0069 | Discharge |
| Paper GPR+LSTM *(within-battery)* | 0.0150 | Charge |
| XGBoost-HF Q90 calibrated | 0.0579 | Charge |
| XGBoost-HF Q75 calibrated | 0.0434 | Charge |
| XGBoost-HF true HF | 0.0579 | Charge |
| XGBoost-HF raw | 0.0594 | Charge |

### RUL Prediction: B0018 Holdout (Linear Extrapolation)

| Model | RMSE (cycles) | MAE (cycles) |
|-------|--------------|--------------|
| **XGBoost (discharge)** | **7.56** | 5.03 |
| Ridge (discharge) | 7.64 | **4.94** |
| XGBoost-HF Q75 calibrated | **5.44** | 3.32 |
| XGBoost-HF Q90 calibrated | 7.45  | 6.38 |
| XGBoost-HF raw | 12.47 | 9.47 |
| Paper GPR+LSTM *(within-battery)* | 5.54 | **0.58** |

> **Note on paper comparison:** Zhao et al. (2022) uses a within-battery
> split (train on first 50% of each battery, test on last 50%).
> Our cross-battery holdout is a stricter evaluation as the model
> never sees B0018 during training. The 2.2× RUL gap is explained
> by evaluation protocol differences.

### Phase-wise RMSE Summary

| Model | Early SoH | Mid SoH | Late SoH | Early RUL | Mid RUL | Late RUL |
|-------|-----------|---------|----------|-----------|---------|----------|
| Ridge | 0.0040 | 0.0045 | **0.0039** | 13.05 | 6.47 | **0.94** |
| XGBoost (dis) | 0.0071 | 0.0067 | 0.0069 | 12.85 | 6.44 | 1.29 |
| HF Q75 | **0.0234** | 0.0464 | 0.0515 | 11.66 | 2.59 | **2.53** |
| HF Q90 | 0.0337 | 0.0489 | 0.0782 | 7.88 | 5.69 | 7.45 |
| HF raw | 0.0611 | 0.0770 | 0.0311 | 20.93 | 13.59 | 2.15 |

---

## Key Findings

### 1. Split strategy inflates reported performance
Random shuffle RMSE is `[0.0085]` Ah vs cross-battery 0.0075 Ah, 
a `[1.1]×` apparent improvement from sequential cycle ordering.

### 2. Cross-domain SHAP convergence
Both measurement domains independently identify internal resistance as
the dominant SoH driver:
Discharge domain →  energy_Wh, Q_cum_Ah     (capacity throughput ↓ with age)
Charge domain →  HF6_rolling_mean5, HF1  (resistance ↑ slows charging)
EIS →  Rct (r=−0.70), Re (r=−0.64)

Degradation rate correlation confirms this: HF1 evolution rate vs
capacity degradation rate achieves r=0.997 (p=0.003) across all four batteries.

### 3. LSTM smoothing accidentally improves RUL
True HFs produce catastrophic mid-phase RUL error (59.09 cycles) due to
anomalous B0018 experimental cycles 45–60. LSTM smoothing of these
anomalies reduces mid-phase RUL error to 13.59 cycles at the cost of
higher SoH RMSE, a fundamental trade-off between point accuracy and
trajectory stability.

### 4. Q75 calibration is better than Q90 for RUL
Q75 achieves lower overall RUL RMSE (5.76 vs 8.55 cycles) 
because Q90 over-corrects in late life,
making EOL projections too optimistic.

---

## Causal Roadmap

```
This project establishes the correlation structure that motivates a causal model:
Aging (cycle count)
     │
     ▼
R_int increase
┌────┴────┐
▼         ▼
Re ↑      Rct ↑     ← EIS confirmation
└────┬────┘
     │
┌────┴──────────────┐
▼                   ▼
energy_Wh ↓    HF6 ↑, HF1 ↑
(discharge)     (charge)
│                   │
└───────┬───────────┘
        ▼
SoH (capacity) ↓
```

**Example:** DoWhy DAG-based causal model to answer:
*"If R_int is held constant (via temperature control or formation cycling),
what is the counterfactual effect on capacity fade rate?"*

---

## References

1. CRISP-DM Consortium, "CRISP-DM 1.0: Step-by-Step Data Mining Guide" SPSS Inc., 2000. [Online]. 
    Available: [CRISP-DM](https://www.the-modeling-agency.com/crisp-dm.pdf)
2. J. Zhao et al., "Method of Predicting SOH and RUL of Lithium-Ion Battery
   Based on the Combination of LSTM and GPR," *Sustainability*, 2022.
3. B. Saha and K. Goebel, "Battery Data Set," NASA PCoE, 2007.
4. Zhang, Y., Tang, Q., Zhang, Y. et al., "Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning." Nat Commun 11, 1706 (2020). 
    Available: [EIS](https://doi.org/10.1038/s41467-020-15235-7)

---

## License

Dataset subject to NASA data usage terms.