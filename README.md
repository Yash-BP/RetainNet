# RetainNet

**Customer churn prediction, survival analysis, and customer lifetime value estimation**

RetainNet is a Python project that explores how survival analysis can help estimate customer churn risk and future customer value. Instead of predicting only whether a customer will leave, survival models estimate the probability that a customer remains active over time.

The project combines customer segmentation, a filtering stage, survival models, evaluation metrics, and a Streamlit dashboard.

## Features

- Customer segmentation using K-Means.
- A logistic-regression-based filtering stage.
- Survival modelling with DeepSurv and DeepHit.
- Baseline comparisons using Cox proportional hazards and Random Survival Forest, where supported by the installed dependencies.
- Evaluation using concordance index and Brier-score-based metrics.
- Customer lifetime value calculations using survival estimates and configurable financial assumptions.
- A Streamlit dashboard with customer analytics and a retention scenario simulator.
- Experimental modules for inference, conformal calibration, and data-drift detection.

**Note:** The advanced modules and retention scenarios should be treated as experimental unless independently validated. Simulated intervention results are sensitivity analyses, not proof of causal business impact.

## How it works

1. **Segmentation:** K-Means groups customers with similar observed characteristics.
2. **Filtering:** A logistic-regression stage selects or weights observations according to the project's filtering logic.
3. **Survival modelling:** Models estimate customer survival over time while accounting for right-censored observations.
4. **Evaluation:** Model performance is assessed using survival-analysis metrics.
5. **Customer value:** Survival estimates are combined with revenue, discounting, and acquisition-cost assumptions.
6. **Dashboard:** Results and configurable scenarios are presented through Streamlit.

## Technology

- Python
- pandas and NumPy
- scikit-learn
- PyTorch
- Streamlit
- Survival-analysis libraries, depending on the selected model and installed dependencies
- pytest

## Getting started

### Requirements

Install a Python version supported by the project's dependencies. Python 3.10–3.13 is the range specified for this project; verify compatibility with the versions listed in `requirements.txt`.

### Installation

```powershell
git clone https://github.com/Yash-BP/RetainNet.git
cd RetainNet

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the dashboard

```powershell
python -m streamlit run dashboard.py
```

Streamlit will provide a local URL for opening the dashboard in your browser.

### Run the tests

```powershell
python -m pytest -q
```

The repository's previously reported test run completed with 91 passing tests and two warnings. Run the suite in your own environment to verify the current result.

### Run a synthetic experiment

```powershell
python main.py --dataset synthetic --model all --epochs 20 --n-samples 500
```

Check `python main.py --help` for available options. Real-data experiments require a compatible dataset and the appropriate command-line arguments.

## Data and scientific assumptions

Survival analysis models time until an event while accounting for right-censoring. Their reliability depends on assumptions about censoring, data quality, feature construction, and model specification.

DeepSurv uses a proportional-hazards formulation. DeepHit provides a discrete-time alternative with a different modelling approach.

Customer lifetime value estimates depend on the revenue, discount-rate, time-horizon, and acquisition-cost assumptions used by the implementation. They should be interpreted as model-based estimates rather than guaranteed future revenue.

The retention simulator produces scenario-based estimates. It does not establish that a proposed intervention will cause the predicted improvement.

## Limitations

- Performance depends on the dataset, preprocessing, censoring patterns, and model configuration.
- Synthetic benchmark results do not establish performance on real customers.
- Some models may require additional dependencies.
- Advanced inference and calibration modules require separate validation before their outputs are used for business decisions.
- The project is an academic and portfolio implementation, not a production-validated decision system.

## Repository structure

```text
RetainNet/
├── dashboard.py
├── main.py
├── requirements.txt
├── README.md
├── RETAINNET_FULL_PROJECT_DOCUMENTATION.txt
├── retainnet/
│   ├── advanced/
│   ├── data/
│   ├── ltv/
│   ├── metrics/
│   ├── models/
│   └── pipeline/
└── tests/
```

## Project purpose

RetainNet was developed as a final-year computer engineering project to explore survival analysis, churn modelling, and customer lifetime value estimation in a single Python application.

Contributions, bug reports, and suggestions are welcome.
