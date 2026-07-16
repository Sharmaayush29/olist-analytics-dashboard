# 🛍️ Olist E-Commerce Analytics Dashboard

A **production-grade Business Intelligence & ML platform** built on the Brazilian Olist E-Commerce Public Dataset.

---

## 📊 Live Demo

👉 **[Launch the Dashboard](https://olist-analytics-dashboard-ayushsharma.streamlit.app/)**

---

## ✨ Features

### 📈 Analytics Pages
| Page | Description |
|---|---|
| 📊 Executive Dashboard | KPIs, Revenue Growth, Order Trends, Moving Averages |
| 📈 Sales Analytics | Monthly Revenue & Orders with trend lines and growth % |
| 👥 Customer Analytics | State distribution, loyalty segmentation, review patterns |
| 📦 Product Analytics | Top products, weight distribution, category heatmaps |
| 💳 Payment Analytics | Payment method share, installment analysis, AOV |
| 🚚 Delivery Analytics | SLA performance, delivery spread, ±1σ trend bands |

### 🤖 Machine Learning
| Page | Description |
|---|---|
| 🧠 ML Review Predictor | Predict customer review scores with confidence & risk |
| 📉 Model Performance | Confusion matrix, ROC curves, F1, precision, recall |

### 💡 Intelligence
| Page | Description |
|---|---|
| 💡 Business Insights | Automated findings, correlation matrix, KPI callouts |
| 🎯 Strategic Action Plan | Data-driven recommendations with actionable steps |

---

## 🎯 Page-Specific Filters

Every page shows only its relevant filters in the sidebar:
- **Sales**: Date range + Category + Seller State
- **Customers**: Customer State + Review Score range
- **Products**: Category + Price range
- **Payments**: Payment Method + Max Installments
- **Delivery**: Customer State + On-Time/Late toggle

---

## 📥 Download Options

Every data page includes a download button:
- Filtered Dataset (CSV)
- Page-specific data exports
- Prediction results (CSV)
- Business report (TXT)
- Model evaluation report (TXT)

---

## 🛠️ Technology Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Streamlit** | Dashboard frontend |
| **Plotly** | Interactive charts |
| **Pandas / NumPy** | Data wrangling |
| **Scikit-learn** | Random Forest ML model |
| **GitHub Actions** | CI/CD via Streamlit Cloud |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Sharmaayush29/olist-analytics-dashboard.git
cd olist-analytics-dashboard

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

---

## 📂 Project Structure

```
olist-analytics-dashboard/
├── app.py                      # Main Streamlit application
├── etl_pipeline.py             # ETL: raw Olist CSVs → processed dataset
├── processed_olist_data.csv    # Cleaned, feature-engineered dataset
├── best_model.pkl              # Trained Random Forest classifier
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🤖 ML Model Details

- **Algorithm**: Random Forest Classifier
- **Target**: Customer Review Score (1–5 stars)
- **Features**: Payment Value, Installments, Delivery Days, Category, Seller State, Customer State
- **Evaluation**: Accuracy, Precision, Recall, F1 (Macro), ROC-AUC per class

---

## 📊 Dataset

**Brazilian E-Commerce Public Dataset by Olist** — available on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

~100,000 orders from 2016–2018 across multiple Brazilian states.

---

## 👤 Author

**Ayush Sharma**  
Data Scientist · ML Engineer · Streamlit Developer  
ReadyNest Internship — Week 5 Project
👉 **[Linkedwin profile](https://www.linkedin.com/in/ayush-sharma-014763319?utm_source=share_via&utm_content=profile&utm_medium=member_android/)**

[![GitHub](https://img.shields.io/badge/GitHub-Sharmaayush29-black?logo=github)](https://github.com/Sharmaayush29)
