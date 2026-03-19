# 📡 Customer Churn Prediction — Production ML System

End-to-end machine learning project predicting customer churn for a telecommunications company. Built to production standards with causal inference, A/B testing, distributed data processing, explainability, hyperparameter optimisation, experiment tracking, and REST API serving.

---

## 📌 Overview

Customer churn — when a subscriber cancels their service — is one of the most costly problems in the telecom industry. Acquiring a new customer costs **5–25× more** than retaining an existing one. This project builds a complete ML pipeline that:

- Processes data at scale using a **PySpark** pipeline mirroring production data engineering patterns
- Validates that retention interventions work using a **randomised A/B test**
- Identifies *why* customers churn using **causal inference** (Propensity Score Matching + DoWhy)
- Predicts at-risk customers before they leave using **XGBoost and a PyTorch neural network**
- Explains predictions globally and per-customer using **SHAP and LIME**
- Quantifies the business value of the model in **dollar terms**
- Serves predictions via a **FastAPI REST endpoint** ready for production deployment

---

## 📊 Results

| Model | ROC-AUC | PR-AUC | F2-Score |
|-------|---------|--------|----------|
| Logistic Regression (baseline) | 0.859 | 0.687 | 0.727 |
| XGBoost | 0.842 | 0.664 | 0.690 |
| Neural Network — PyTorch | 0.853 | 0.642 | 0.779 |
| Optimised NN — Optuna | 0.850 | 0.647 | 0.763 |

> **Why F2-Score?** Missing a churner (false negative) costs ~8× more than a wasted retention offer (false positive). F2 penalises false negatives more heavily, making it a better metric than accuracy or F1 for this problem.

**Business Impact (Test Set):**
- Churners correctly identified: **266 out of 280 (95% recall)**
- Expected profit: **$56,060**
- Model value add: **+$168,060** over no-model baseline

---

## 📋 Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Business Framing | Problem definition, cost parameters, success metrics |
| 2 | Data Loading & Quality Audit | Missing values, dtypes, target distribution |
| 3 | PySpark Pipeline | Production-scale data processing, feature engineering, Parquet handoff |
| 4 | Exploratory Data Analysis | 6 visualisation sections, churn drivers, correlations |
| 5 | A/B Test | Power analysis, SRM check, z-test, business impact of retention offer |
| 6 | Causal Inference | Propensity Score Matching + DoWhy DAG, refutation tests |
| 7 | Preprocessing Pipeline | sklearn Pipeline + ColumnTransformer, no data leakage |
| 8 | Baseline Models | Logistic Regression & XGBoost with full evaluation |
| 9 | Neural Network (PyTorch) | Deep NN with BatchNorm, Dropout, early stopping, LR scheduling |
| 10 | Autoencoder | Unsupervised anomaly detection trained on loyal customers only |
| 11 | Hyperparameter Optimisation | Optuna TPE search over 30 trials, pruning, importance analysis |
| 12 | Explainability | SHAP (global + waterfall) and LIME (individual customer audit) |
| 13 | MLflow Tracking | Experiment logging, model registry for all runs |
| 14 | Business Output | Profit curve, optimal threshold, ROI vs. baseline |
| 15 | Model Serving | FastAPI REST endpoint with health check and risk tier |

---

## 🧪 A/B Test — Validating Retention Interventions

Before building a churn model, this project validates that retention offers actually work using a rigorous randomised controlled experiment:

- **Power analysis** to determine required sample size before running the test
- **50/50 random assignment** of customers to treatment (offer) vs control (no offer)
- **Sample Ratio Mismatch (SRM) check** to verify randomisation integrity
- **Two-proportion z-test** (one-tailed): Z = −8.03, p ≈ 0.000
- Result: offer reduces churn from **26.3% → 18.3%** (8pp absolute reduction)

> The business impact calculation intentionally shows a negative ROI for untargeted offers (-$74,820). This is not a failure — it is the proof of why the ML model is needed. Targeting only high-risk customers turns negative ROI into positive ROI.

---

## 🔗 Causal Inference — Why Do Customers Churn?

Correlation ≠ causation. This section goes beyond feature importance to estimate the **Average Treatment Effect (ATE)** of key variables on churn.

**Method 1 — Propensity Score Matching (PSM):**
- Treatment: month-to-month contract vs long-term contract
- Matched treated/control customers on observable confounders (tenure, charges, services)
- Result: month-to-month contract **causally increases churn by ~31.6%** (95% CI: [29.8%, 33.5%])

**Method 2 — DoWhy Causal Graph:**
- Domain knowledge encoded as a Directed Acyclic Graph (DAG)
- Linear regression adjustment estimates causal effect of MonthlyCharges on churn
- Result: moving from low (~$35) to high (~$80) monthly charges **causally increases churn probability by 34.7%**
- Robustness confirmed via random common cause and placebo treatment refutation tests

---

## ⚡ PySpark Pipeline

Although this dataset has 7,043 rows, the data processing pipeline is built in PySpark to mirror production data engineering patterns at scale:

- Ingestion from CSV (production equivalent: `spark.read.parquet("s3://data-lake/...")`)
- Type casting, null handling, target encoding in Spark SQL
- Feature engineering using distributed operations and `approxQuantile`
- Window functions for CLV ranking within customer segments
- Output written to **Parquet**, read back into pandas for local modelling — the standard handoff pattern in production ML platforms

---

## 🔧 Feature Engineering

10 domain-informed features created on top of the 21 raw columns:

| Feature | Description |
|---------|-------------|
| `tenure_group` | Binned tenure: 0–6mo, 6–12mo, 1–2yr, 2–4yr, 4+yr |
| `num_services` | Count of active subscriptions (0–9) |
| `charges_per_tenure` | Monthly charges normalised by tenure |
| `has_security_bundle` | 1 if customer has OnlineSecurity + DeviceProtection + TechSupport |
| `is_long_term` | 1 if on One-year or Two-year contract |
| `is_autopay` | 1 if using automatic payment method |
| `clv_proxy` | tenure × MonthlyCharges (lifetime value estimate) |
| `is_high_value` | 1 if MonthlyCharges ≥ 75th percentile |
| `log_total_charges` | Log-transformed TotalCharges (reduces skew) |
| `log_clv` | Log-transformed CLV proxy |

---

## 🧠 Neural Network Architecture

**Default NN** (256→128→64→32, 4 layers):
```
Input (40) → Linear(256) → BatchNorm → ReLU → Dropout(0.4)
           → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(64)  → BatchNorm → ReLU → Dropout(0.2)
           → Linear(32)  → BatchNorm → ReLU → Dropout(0.1)
           → Linear(1)   → Sigmoid
```

**Optimised NN** (found by Optuna after 30 trials):
```
Input (40) → Linear(32) → BatchNorm → ReLU → Dropout(0.22)
           → Linear(64) → BatchNorm → ReLU → Dropout(0.31)
           → Linear(1)  → Sigmoid
```

Optuna found that a simpler 2-layer network performs comparably to the deeper default — suggesting the dataset does not require a deep network and that good feature engineering matters more than model complexity.

**Training details:**
- Optimiser: AdamW with weight decay
- Loss: BCEWithLogitsLoss with class-weight balancing
- Sampler: WeightedRandomSampler for imbalanced classes
- Regularisation: Dropout + L2 weight decay
- Early stopping: patience = 20 epochs
- LR Scheduler: ReduceLROnPlateau

---

## 🔬 Hyperparameter Optimisation

Optuna's Tree-structured Parzen Estimator (TPE) searched over:

- Number of layers (2–5)
- Hidden units per layer (32, 64, 128, 256)
- Dropout rates (0.1–0.5)
- Learning rate (1e-4 to 1e-2, log scale)
- Weight decay (1e-5 to 1e-2)
- Batch size (128, 256, 512)

30 trials with MedianPruner — unpromising trials stopped early to save compute.

---

## 🔍 Explainability

**SHAP (SHapley Additive exPlanations):**
- Global feature importance (mean |SHAP|)
- Beeswarm plot showing direction and magnitude per customer
- Dependence plot for tenure
- Waterfall plot for individual customer audit

**LIME (Local Interpretable Model-agnostic Explanations):**
- Per-customer explanation for the highest-confidence churner (98.3% predicted probability)
- Shows which features pushed the prediction toward or away from churn
- Model-agnostic — works independently of the model type

Key finding: `tenure`, `is_long_term`, `charges_per_tenure`, and `InternetService_Fiber optic` are the top churn drivers globally.

---

## 💰 Business ROI Analysis

The model's decision threshold is optimised using a **profit curve** based on real business cost parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Cost of missing a churner (FN) | $400 | Lost future revenue |
| Cost of wasted intervention (FP) | $50 | Discount or retention call |
| Retention rate after intervention | 30% | Conservative industry estimate |
| CLV saved if retained | $1,200 | 24 months × $50/month |

The optimal threshold maximises expected profit across all possible thresholds, rather than defaulting to 0.5.

---

## 🚀 Running the API

After running the notebook, three artefact files are saved: `preprocessor.pkl`, `churn_model.pt`, and `feature_names.json`.

**Install dependencies:**
```bash
pip install fastapi uvicorn
```

**Start the server:**
```bash
uvicorn serve:app --reload
```

**Interactive API docs:**
```
http://localhost:8000/docs
```

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0,
       "Contract": "Month-to-month", "InternetService": "Fiber optic",
       "PaymentMethod": "Electronic check"}'
```

**Example response:**
```json
{
  "churn_probability": 0.7823,
  "churn_prediction": true,
  "risk_tier": "HIGH",
  "threshold_used": 0.04
}
```

---

## ⚙️ Installation

```bash
git clone https://github.com/neshatsh/customer-churn-prediction-pytorch.git
cd customer-churn-prediction-pytorch
pip install torch xgboost optuna shap lime mlflow fastapi uvicorn scikit-learn \
            pandas numpy matplotlib seaborn pyspark dowhy econml statsmodels
```

**Dataset:**
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project root, or add it via Kaggle's "Add Data" button if running on Kaggle.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch |
| ML & Preprocessing | Scikit-learn, XGBoost |
| Distributed Processing | PySpark |
| Causal Inference | DoWhy, EconML |
| Statistical Testing | SciPy, Statsmodels |
| Hyperparameter Search | Optuna |
| Explainability | SHAP, LIME |
| Experiment Tracking | MLflow |
| API Serving | FastAPI, Uvicorn |
| Data & Visualisation | Pandas, NumPy, Matplotlib, Seaborn |

---

## 📁 Dataset

**IBM Telco Customer Churn**
- 7,043 customers | 21 features | Binary target (Churn)
- Churn rate: 26.5%
- Source: [Kaggle — blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📈 Key Insights

1. **Contract type is the #1 churn driver** — Month-to-month customers churn at 43% vs 11% for annual contracts
2. **First 6 months are the highest-risk window** — Tenure is the single most predictive feature
3. **Security bundle customers churn at 2× lower rate** — Bundled services create switching friction
4. **Fiber optic customers churn more than DSL** — Possible price/quality dissatisfaction signal
5. **Auto-pay customers are significantly more loyal** — Payment friction correlates with disengagement
6. **Monthly charges have a genuine causal effect** — Not just correlation; confirmed via DoWhy refutation tests
7. **Retention offers work, but only when targeted** — A/B test confirms statistical significance; untargeted offers lose money

---

This project demonstrates a complete ML workflow from raw data to deployed API, with emphasis on business value, causal understanding, explainability, and production readiness.
