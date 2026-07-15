# Olist Brazilian E-Commerce End-to-End Analytics Pipeline & Dashboard

This repository contains a professional-grade business intelligence dashboard and predictive model trained on the Brazilian Olist E-Commerce dataset. It is built as part of the ReadyNest Internship program (Week 5).

## Folder Structure

```
cool-pasteur/
│
├── app.py                      # Main Streamlit dashboard application
├── etl_pipeline.py            # Extraction, Transformation, Loading & model training script
├── processed_olist_data.csv    # Cleaned, optimized subset of the merged Olist dataset (generated)
├── best_model.pkl              # Trained Random Forest classifier and state/category label encoders (generated)
├── requirements.txt            # Python library dependencies
└── README.md                   # Project documentation (this file)
```

## Setup & Execution

### 1. Pre-requisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Generate Processed Data and Model (ETL)
Run the ETL pipeline to process the raw datasets located under `C:\Users\ayush\Desktop\Week5_End_to_End_Pipeline\data\raw` and train the predictive classifier:
```bash
python etl_pipeline.py
```
This will generate `processed_olist_data.csv` and `best_model.pkl` in the root folder.

### 4. Run the Streamlit Dashboard
Launch the local development server:
```bash
streamlit run app.py
```

## Features

- **Page 1: Executive Dashboard**: Operational and revenue overview with real-time KPI metrics and sales timeline.
- **Page 2: Sales Analytics**: Analysis of sales distribution, top performing categories, and average order values.
- **Page 3: Customer Analytics**: Visualizes customer state distribution, repeat purchase patterns, and satisfaction rates.
- **Page 4: Product Analytics**: In-depth catalog metrics, product dimension distributions, and product sizes.
- **Page 5: Payment Analytics**: Visualizes payment methods share, installments count distributions, and payment values.
- **Page 6: Delivery Analytics**: Analysis of delivery logistics speeds, SLAs, and monthly delivery times.
- **Page 7: Machine Learning Prediction**: Input order values, installments, delivery days, and location context to predict review scores (1-5 ★) and identify risks.
- **Page 8: Business Insights**: Automatically generated findings on top performance areas and bottlenecks.
- **Page 9: Strategic Action Plan**: Actionable recommendations for regional fulfillment and marketing strategy.
- **Page 10: About Project**: Technology stack details, ETL pipeline structure, and developer credentials.
