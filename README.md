# 📡 Customer Churn Prediction — PyTorch & Beyond

> An industry-level, end-to-end machine learning project predicting customer churn for a telecommunications company. Built to production standards with explainability, hyperparameter optimisation, experiment tracking, and REST API serving.

---

## 📌 Overview

Customer churn — when a subscriber cancels their service — is one of the most costly problems in the telecom industry. Acquiring a new customer costs **5–25× more** than retaining an existing one. This project builds a complete ML pipeline that:

- Identifies at-risk customers **before** they leave
- Explains **why** a customer is likely to churn (per-customer and globally)
- Quantifies the **business value** of the model in dollar terms
- Serves predictions via a **REST API** ready for production deployment

---

## 📊 Results

| Model | ROC-AUC | PR-AUC | F2-Score |
|-------|---------|--------|----------|
| Logistic Regression (baseline) | 0.837 | 0.645 | 0.704 |
| XGBoost (baseline) | 0.831 | 0.641 | 0.697 |
| Neural Network — PyTorch | 0.835 | 0.632 | 0.741 |
| **Optimised NN — Optuna** | **0.855** | **0.660** | **0.754** |

> **Why F2-Score?** Missing a churner (false negative) costs ~8× more than a wasted retention offer (false positive). F2 penalises false negatives more heavily, making it a better metric than accuracy or F1 for this problem.

**Business Impact (Test Set):**
- Churners correctly identified: **272 out of 280** (97% recall)
- Expected profit: **$60,470** vs. baseline of **-\$112,000**
- **Model value add: +$172,470** over no-model baseline



[//]: # (## 🗂️ Project Structure)

[//]: # ()
[//]: # (```)

[//]: # (customer-churn-prediction-pytorch/)

[//]: # (│)

[//]: # (├── telco-customer-churn-prediction.ipynb   # Main notebook &#40;all sections&#41;)

[//]: # (├── serve.py                                # FastAPI serving script)

[//]: # (├── preprocessor.pkl                        # Fitted sklearn preprocessor)

[//]: # (├── churn_model.pt                          # Trained PyTorch model weights)

[//]: # (├── feature_names.json                      # Feature config & threshold)

[//]: # (└── README.md)

[//]: # (```)

---

## 📋 Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Business Framing | Problem definition, cost parameters, success metrics |
| 2 | Data Loading & Quality Audit | Missing values, dtypes, target distribution |
| 3 | Exploratory Data Analysis | 5 visualisation sections, churn drivers, correlations |
| 4 | Feature Engineering | 10 domain-informed features (CLV proxy, service bundles, tenure groups) |
| 5 | Preprocessing Pipeline | sklearn Pipeline + ColumnTransformer, no data leakage |
| 6 | Baseline Models | Logistic Regression & XGBoost with full evaluation |
| 7 | Neural Network (PyTorch) | Deep NN with BatchNorm, Dropout, early stopping, LR scheduling |
| 8 | Autoencoder | Unsupervised anomaly detection trained on loyal customers only |
| 9 | Hyperparameter Optimisation | Optuna TPE search over 30 trials, pruning, importance analysis |
| 10 | Explainability | SHAP (global + waterfall) and LIME (individual customer audit) |
| 11 | MLflow Tracking | Experiment logging, model registry for all runs |
| 12 | Business Output | Profit curve, optimal threshold, ROI vs. baseline |
| 13 | Model Serving | FastAPI REST endpoint with health check and risk tier |

---

## 🔧 Feature Engineering

Rather than using raw columns alone, 10 domain-informed features were created:

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

**Default NN** (256→128→64→32, 4 layers, 54,721 parameters):
```
Input (40) → Linear(256) → BatchNorm → ReLU → Dropout(0.4)
           → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(64)  → BatchNorm → ReLU → Dropout(0.2)
           → Linear(32)  → BatchNorm → ReLU → Dropout(0.1)
           → Linear(1)   → Sigmoid
```

**Optimised NN** (found by Optuna after 30 trials):
```
Input (40) → Linear(64) → BatchNorm → ReLU → Dropout(0.35)
           → Linear(32) → BatchNorm → ReLU → Dropout(0.35)
           → Linear(1)  → Sigmoid
```

> Optuna found that a simpler 64→32 (2 layers) performs comparably to the deeper default — suggesting the dataset does not require a deep network and that good feature engineering matters more than model complexity.

**Training details:**
- Optimiser: AdamW with weight decay
- Loss: BCEWithLogitsLoss with class-weight balancing
- Sampler: WeightedRandomSampler for imbalanced classes
- Regularisation: Dropout + L2 weight decay
- Early stopping: patience = 20 epochs
- LR Scheduler: ReduceLROnPlateau

---

## 🔬 Hyperparameter Optimisation

Optuna's **Tree-structured Parzen Estimator (TPE)** was used to search over:

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
- Per-customer explanation showing which features pushed the prediction up or down
- Model-agnostic — works independently of the model type

Key finding: `tenure`, `Contract_Two year`, `InternetService_Fiber optic`, and `MonthlyCharges` are the top churn drivers globally.

---

## 💰 Business ROI Analysis

The model's decision threshold is not fixed at 0.5 — it is optimised using a **profit curve** based on real business cost parameters:

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
  "threshold_used": 0.03
}
```

---

## ⚙️ Installation

```bash
git clone https://github.com/neshatsh/customer-churn-prediction-pytorch.git
cd customer-churn-prediction-pytorch
pip install torch xgboost optuna shap lime mlflow imbalanced-learn fastapi uvicorn scikit-learn pandas numpy matplotlib seaborn
```

**Dataset:**  
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project root, or add it via Kaggle's "Add Data" button if running on Kaggle.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch |
| ML & Preprocessing | Scikit-learn, XGBoost, imbalanced-learn |
| Hyperparameter Search | Optuna |
| Explainability | SHAP, LIME |
| Experiment Tracking | MLflow |
| API Serving | FastAPI, Uvicorn |
| Data & Visualisation | Pandas, NumPy, Matplotlib, Seaborn |

---

## 📁 Dataset

**IBM Telco Customer Churn**  
- 7,043 customers | 21 features | Binary target (`Churn`)  
- Churn rate: 26.5%  
- Source: [Kaggle — blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📈 Key Insights

1. **Contract type is the #1 churn driver** — Month-to-month customers churn at 43% vs 11% for annual contracts
2. **First 6 months are the highest-risk window** — Tenure is the single most predictive feature
3. **Security bundle customers churn at 2× lower rate** — Bundled services create switching friction
4. **Fiber optic customers churn more than DSL** — Possible price/quality dissatisfaction signal
5. **Auto-pay customers are significantly more loyal** — Payment friction correlates with disengagement

---

*This project demonstrates a complete ML workflow from raw data to deployed API, with emphasis on business value, explainability, and production readiness.*
