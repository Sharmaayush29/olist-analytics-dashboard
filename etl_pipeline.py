import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Raw data directory path
raw_dir = r"C:\Users\ayush\Desktop\Week5_End_to_End_Pipeline\data\raw"

print("Loading raw datasets...")
customers = pd.read_csv(os.path.join(raw_dir, "olist_customers_dataset.csv"))
orders = pd.read_csv(os.path.join(raw_dir, "olist_orders_dataset.csv"))
order_items = pd.read_csv(os.path.join(raw_dir, "olist_order_items_dataset.csv"))
payments = pd.read_csv(os.path.join(raw_dir, "olist_order_payments_dataset.csv"))
reviews = pd.read_csv(os.path.join(raw_dir, "olist_order_reviews_dataset.csv"))
products = pd.read_csv(os.path.join(raw_dir, "olist_products_dataset.csv"))
sellers = pd.read_csv(os.path.join(raw_dir, "olist_sellers_dataset.csv"))
translations = pd.read_csv(os.path.join(raw_dir, "product_category_name_translation.csv"))

print("Merging datasets...")
# Merge orders and customers
df = pd.merge(orders, customers, on="customer_id", how="inner")

# Merge order items
df = pd.merge(df, order_items, on="order_id", how="inner")

# Merge payments
# Group payments to avoid duplicating order items if there are multiple payment methods, 
# or just take the main payment
payments_agg = payments.groupby("order_id").agg({
    "payment_value": "sum",
    "payment_installments": "max",
    "payment_type": lambda x: x.iloc[0] # Take first payment type
}).reset_index()
df = pd.merge(df, payments_agg, on="order_id", how="inner")

# Merge reviews
# Take the first review score per order to avoid duplicating items
reviews_agg = reviews.groupby("order_id").agg({
    "review_score": "first"
}).reset_index()
df = pd.merge(df, reviews_agg, on="order_id", how="inner")

# Merge products
df = pd.merge(df, products, on="product_id", how="inner")

# Merge sellers
df = pd.merge(df, sellers, on="seller_id", how="inner")

# Translate category names
df = pd.merge(df, translations, on="product_category_name", how="left")
df["product_category_name_english"] = df["product_category_name_english"].fillna("other")

print("Engineering features...")
# Convert datetime columns
datetime_cols = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", 
                 "order_delivered_customer_date", "order_estimated_delivery_date"]
for col in datetime_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Calculate delivery days
df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
# Fill missing delivery days with median
df["delivery_days"] = df["delivery_days"].fillna(df["delivery_days"].median())
df["delivery_days"] = df["delivery_days"].clip(lower=0)

# Calculate late delivery
df["is_late"] = (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(int)

# Extract month of purchase
df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

# Filter out rows with missing key values
df = df.dropna(subset=["review_score", "payment_value", "payment_installments"])

# Keep relevant columns
columns_to_keep = [
    "order_id", "order_purchase_timestamp", "order_month", "payment_value", "payment_installments", "payment_type",
    "delivery_days", "is_late", "product_category_name_english", "seller_id", "seller_state",
    "customer_id", "customer_unique_id", "customer_state", "customer_city", "review_score",
    "product_id", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"
]
df = df[columns_to_keep]

# Sample down for Streamlit performance
print(f"Total merged dataset rows: {len(df)}")
df_sampled = df.sample(n=min(30000, len(df)), random_state=42).reset_index(drop=True)

# Save to processed CSV
output_csv = r"C:\Users\ayush\Documents\antigravity\cool-pasteur\processed_olist_data.csv"
df_sampled.to_csv(output_csv, index=False)
print(f"Saved processed dataset to {output_csv}")

print("Training Machine Learning Model...")
# Prepare training data
# Predict review score (1 to 5)
X_df = df_sampled[["payment_value", "payment_installments", "delivery_days", "product_category_name_english", "seller_state", "customer_state"]].copy()
y = df_sampled["review_score"].astype(int)

# Label Encoding for categoricals
le_category = LabelEncoder()
X_df["product_category_name_english"] = le_category.fit_transform(X_df["product_category_name_english"].astype(str))

le_seller_state = LabelEncoder()
X_df["seller_state"] = le_seller_state.fit_transform(X_df["seller_state"].astype(str))

le_customer_state = LabelEncoder()
X_df["customer_state"] = le_customer_state.fit_transform(X_df["customer_state"].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model test accuracy: {accuracy:.4f}")

# Save model and encoders
output_model = r"C:\Users\ayush\Documents\antigravity\cool-pasteur\best_model.pkl"
model_data = {
    "model": model,
    "le_category": le_category,
    "le_seller_state": le_seller_state,
    "le_customer_state": le_customer_state
}

with open(output_model, "wb") as f:
    pickle.dump(model_data, f)
print(f"Saved trained model package to {output_model}")
print("ETL and Training Pipeline completed successfully!")
