import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import math

# Page Configuration
st.set_page_config(
    page_title="Telco Churn — Advanced Predictive Analytics & Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek Dark Mode Styling (Zinc / Obsidian theme matching the premium dashboard)
st.markdown("""
<style>
    /* Main Background and Text */
    .stApp {
        background-color: #09090b;
        color: #fafafa;
    }
    
    /* Headers and Title */
    h1, h2, h3, p {
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    }
    
    /* Card Styles */
    div.metric-container {
        background-color: #141416;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.45);
        margin-bottom: 10px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #141416;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    
    /* Selected Tab Color */
    button[data-baseweb="tab"] {
        color: #a1a1aa !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom-color: #3b82f6 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 6px;
        border: none;
        transition: background-color 0.15s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        color: white;
    }
    
    /* Simulator Output styling */
    .sim-card {
        background-color: #141416;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load DATASET from HTML file
@st.cache_data
def load_dataset():
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Regex to locate const DATASET = [[...]];
        match = re.search(r'const DATASET\s*=\s*(\[.*?\]);', content, re.DOTALL)
        if match:
            dataset_str = match.group(1)
            dataset = json.loads(dataset_str)
            columns = [
                "Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                "MonthlyCharges", "TotalCharges", "Churn", "CLTV", "ChurnScore", "ChurnReason"
            ]
            df = pd.DataFrame(dataset, columns=columns)
            # Data preprocessing for Python usability
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
            df["Churn"] = df["Churn"].astype(int)
            df["Tenure"] = df["Tenure"].astype(int)
            df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0.0)
            df["CLTV"] = df["CLTV"].astype(float)
            df["ChurnScore"] = df["ChurnScore"].astype(float)
            return df
    except Exception as e:
        st.error(f"Error loading dataset from dashboard.html: {e}")
    return None

df_raw = load_dataset()

# Sidebar Navigation (Deployment Mode Selector)
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
    <div style="width: 28px; height: 28px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 6px; display: flex; align-items: center; justify-content: center;">
        <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width: 14px; height: 14px;">
            <path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>
        </svg>
    </div>
    <div>
        <h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #fafafa;">Churn Predict</h3>
        <p style="margin: 0; font-size: 10px; color: #71717a;">B.Tech Streamlit App</p>
    </div>
</div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox(
    "Choose Deployment View:",
    ["Native Streamlit Dashboard", "Original HTML Dashboard (Embedded)"]
)

if app_mode == "Original HTML Dashboard (Embedded)":
    # -------------------------------------------------------------
    # VIEW 1: Direct HTML component embedding
    # -------------------------------------------------------------
    st.markdown("### 📊 Original Premium Analytics Dashboard")
    st.markdown("This view renders the beautiful offline HTML/JS dashboard with fully interactive cross-filtering inside Streamlit.")
    
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=1300, scrolling=True)
    except Exception as e:
        st.error(f"Could not load dashboard.html: {e}")

else:
    # -------------------------------------------------------------
    # VIEW 2: Native Streamlit Re-implementation
    # -------------------------------------------------------------
    if df_raw is None:
        st.error("No dataset available to build Native Streamlit Dashboard.")
        st.stop()

    st.markdown("<h1 style='margin-bottom:0;'>Interactive Churn Predictive System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa; font-size:14px; margin-top:2px;'>Fully dynamic exploration dashboard driven by raw customer profiles.</p>", unsafe_allow_html=True)

    # Sidebar Filter Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filter Segment Cohort")
    
    # 1. Profile Filters
    with st.sidebar.expander("Customer Profile", expanded=True):
        f_senior = st.checkbox("Senior Citizen Only")
        f_partner = st.checkbox("Has Partner")
        f_dependents = st.checkbox("Has Dependents")
        f_gender = st.radio("Gender Selection", ["All", "Male", "Female"])

    # 2. Service Filters
    with st.sidebar.expander("Core Services", expanded=False):
        f_phone = st.checkbox("Phone Service Enabled")
        f_multiple = st.checkbox("Multiple Lines")
        f_internet = st.multiselect("Internet Service Technology", ["DSL", "Fiber optic", "No"])

    # 3. Add-on Subscription Filters
    with st.sidebar.expander("Add-on Subscriptions", expanded=False):
        f_security = st.checkbox("No Online Security")
        f_backup = st.checkbox("No Online Backup")
        f_protection = st.checkbox("No Device Protection")
        f_support = st.checkbox("No Tech Support")
        f_streamTV = st.checkbox("Streaming TV Enabled")
        f_streamMovies = st.checkbox("Streaming Movies Enabled")

    # Apply Sidebar Filters to Dataframe
    df = df_raw.copy()
    if f_senior:
        df = df[df["SeniorCitizen"] == 1]
    if f_partner:
        df = df[df["Partner"] == "Yes"]
    if f_dependents:
        df = df[df["Dependents"] == "Yes"]
    if f_gender != "All":
        df = df[df["Gender"] == f_gender]
    if f_phone:
        df = df[df["PhoneService"] == "Yes"]
    if f_multiple:
        df = df[df["MultipleLines"] == "Yes"]
    if f_internet:
        df = df[df["InternetService"].isin(f_internet)]
    if f_security:
        df = df[df["OnlineSecurity"] == "No"]
    if f_backup:
        df = df[df["OnlineBackup"] == "No"]
    if f_protection:
        df = df[df["DeviceProtection"] == "No"]
    if f_support:
        df = df[df["TechSupport"] == "No"]
    if f_streamTV:
        df = df[df["StreamingTV"] == "Yes"]
    if f_streamMovies:
        df = df[df["StreamingMovies"] == "Yes"]

    # Filter Reset Button
    if st.sidebar.button("Reset Filters"):
        st.rerun()

    # KPI Strip Row
    total_cohort = len(df)
    global_size = len(df_raw)
    pct_cohort = (total_cohort / global_size * 100) if global_size > 0 else 0
    
    churn_count = len(df[df["Churn"] == 1])
    retained_count = total_cohort - churn_count
    churn_rate = (churn_count / total_cohort * 100) if total_cohort > 0 else 0
    retained_rate = 100 - churn_rate

    avg_monthly = df["MonthlyCharges"].mean() if total_cohort > 0 else 0
    total_mrr = df["MonthlyCharges"].sum() if total_cohort > 0 else 0
    avg_tenure = df["Tenure"].mean() if total_cohort > 0 else 0
    avg_cltv = df["CLTV"].mean() if total_cohort > 0 else 0
    avg_score = df["ChurnScore"].mean() if total_cohort > 0 else 0

    kpi_cols = st.columns(6)
    
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="metric-container" style="border-top: 3px solid #3b82f6;">
            <p style="margin:0; font-size:10px; color:#a1a1aa; font-weight:700; text-transform:uppercase;">Active Cohort Size</p>
            <h2 style="margin:0; font-size:24px; color:#93c5fd; font-weight:800;">{total_cohort:,}</h2>
            <p style="margin:0; font-size:11px; color:#52525b;">{pct_cohort:.1f}% of dataset</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[1]:
        trend_label = "↑ Critical Risk" if churn_rate > 30 else "Stable"
        trend_color = "#f43f5e" if churn_rate > 30 else "#10b981"
        st.markdown(f"""
        <div class="metric-container" style="border-top: 3px solid #f43f5e;">
            <p style="margin:0; font-size:10px; color:#a1a1aa; font-weight:700; text-transform:uppercase;">Churned Customers</p>
            <h2 style="margin:0; font-size:24px; color:#fca5a5; font-weight:800;">{churn_count:,}</h2>
            <p style="margin:0; font-size:11px; color:{trend_color}; font-weight:600;">{churn_rate:.2f}% ({trend_label})</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[2]:
        st.markdown(f"""
        <div class="metric-container" style="border-top: 3px solid #10b981;">
            <p style="margin:0; font-size:10px; color:#a1a1aa; font-weight:700; text-transform:uppercase;">Retained Customers</p>
            <h2 style="margin:0; font-size:24px; color:#86efac; font-weight:800;">{retained_count:,}</h2>
            <p style="margin:0; font-size:11px; color:#86efac;">{retained_rate:.2f}% retention</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[3]:
        st.markdown(f"""
        <div class="metric-container" style="border-top: 3px solid #f59e0b;">
            <p style="margin:0; font-size:10px; color:#a1a1aa; font-weight:700; text-transform:uppercase;">Avg Monthly Charge</p>
            <h2 style="margin:0; font-size:24px; color:#fde047; font-weight:800;">${avg_monthly:.2f}</h2>
            <p style="margin:0; font-size:11px; color:#52525b;">MRR: ${int(total_mrr):,}</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[4]:
        st.markdown(f"""
        <div class="metric-container" style="border-top: 3px solid #06b6d4;">
            <p style="margin:0; font-size:10px; color:#a1a1aa; font-weight:700; text-transform:uppercase;">Average Tenure</p>
            <h2 style="margin:0; font-size:24px; color:#67e8f9; font-weight:800;">{avg_tenure:.1f}</h2>
            <p style="margin:0; font-size:11px; color:#52525b;">Months active</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[5]:
        st.markdown(f"""
        <div class="metric-container" style="border-top: 3px solid #8b5cf6;">
            <p style="margin:0; font-size:10px; color:#a1a1aa; font-weight:700; text-transform:uppercase;">Average CLTV</p>
            <h2 style="margin:0; font-size:24px; color:#c084fc; font-weight:800;">${int(avg_cltv):,}</h2>
            <p style="margin:0; font-size:11px; color:#a1a1aa;">Avg Churn Score: {avg_score:.1f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Tabs Configuration
    tabs = st.tabs([
        "Overview & Cohort Stats", 
        "Billing & Lifecycle", 
        "Subscribers & Add-ons", 
        "Demographics Analysis", 
        "ML Performance & Simulator"
    ])

    # ------------------ TAB 1: OVERVIEW ------------------
    with tabs[0]:
        st.markdown("### Cohort Insights & Segment Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Churn Status Distribution**")
            fig_churn = px.pie(
                df, names="Churn", hole=0.72,
                color="Churn", color_discrete_map={0: "#10b981", 1: "#f43f5e"},
                labels={"Churn": "Status"}
            )
            fig_churn.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa")
            )
            st.plotly_chart(fig_churn, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Contract Term**")
            fig_contract = px.pie(
                df, names="Contract", hole=0.72,
                color="Contract", color_discrete_map={"Month-to-month": "#3b82f6", "Two year": "#10b981", "One year": "#8b5cf6"}
            )
            fig_contract.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa")
            )
            st.plotly_chart(fig_contract, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Internet Service Technology**")
            internet_counts = df["InternetService"].value_counts().reset_index()
            fig_internet = px.bar(
                internet_counts, x="InternetService", y="count",
                color_discrete_sequence=["#8b5cf6"]
            )
            fig_internet.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=False,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_internet, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        col1_2, col2_2 = st.columns([1.2, 1])
        
        with col1_2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06); height:100%;'>", unsafe_allow_html=True)
            st.write("**Top Churn Reasons (Survey Responses)**")
            reasons_df = df[(df["Churn"] == 1) & (df["ChurnReason"] != "")]
            if len(reasons_df) > 0:
                reasons_counts = reasons_df["ChurnReason"].value_counts().reset_index().head(8)
                fig_reasons = px.bar(
                    reasons_counts, x="count", y="ChurnReason", orientation='h',
                    color_discrete_sequence=["#f43f5e"]
                )
                fig_reasons.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10), height=260, showlegend=False,
                    xaxis_title="", yaxis_title="", yaxis=dict(autorange="reversed"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#fafafa"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)")
                )
                st.plotly_chart(fig_reasons, use_container_width=True)
            else:
                st.info("No churn records found for the selected cohort.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2_2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06); height:100%;'>", unsafe_allow_html=True)
            st.write("**Payment Methods**")
            pm_counts = df["PaymentMethod"].value_counts().reset_index()
            fig_pm = px.bar(
                pm_counts, x="count", y="PaymentMethod", orientation='h',
                color_discrete_sequence=["#06b6d4"]
            )
            fig_pm.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=260, showlegend=False,
                xaxis_title="", yaxis_title="", yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_pm, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ TAB 2: BILLING & LIFECYCLE ------------------
    with tabs[1]:
        st.markdown("### Billing Profiles & Lifecycle Durations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Customer Tenure (Months)**")
            
            # Recreate brackets
            bins = [0, 12, 24, 36, 48, 60, 100]
            labels = ['0–12 mo', '13–24 mo', '25–36 mo', '37–48 mo', '49–60 mo', '60+ mo']
            df['TenureGroup'] = pd.cut(df['Tenure'], bins=bins, labels=labels)
            tenure_counts = df['TenureGroup'].value_counts().reindex(labels).reset_index()
            
            fig_t = px.bar(
                tenure_counts, x="TenureGroup", y="count",
                color_discrete_sequence=["#a855f7"]
            )
            fig_t.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=False,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_t, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Monthly Charge Bracket**")
            
            charge_bins = [0, 30, 50, 70, 90, 1000]
            charge_labels = ['< $30', '$30–50', '$50–70', '$70–90', '$90+']
            df['ChargeGroup'] = pd.cut(df['MonthlyCharges'], bins=charge_bins, labels=charge_labels)
            charge_counts = df['ChargeGroup'].value_counts().reindex(charge_labels).reset_index()
            
            fig_c = px.bar(
                charge_counts, x="ChargeGroup", y="count",
                color_discrete_sequence=["#f59e0b"]
            )
            fig_c.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=False,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_c, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        col1_2, col2_2 = st.columns(2)
        
        with col1_2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Average Monthly Charge by Tenure**")
            
            tenure_charge_trend = df.groupby('TenureGroup', observed=False)['MonthlyCharges'].mean().reset_index()
            fig_trend = px.line(
                tenure_charge_trend, x="TenureGroup", y="MonthlyCharges",
                markers=True
            )
            fig_trend.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=210, showlegend=False,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2_2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Paperless Billing Churn Impact**")
            
            fig_pb = px.histogram(
                df, x="PaperlessBilling", color="Churn", barmode="group",
                color_discrete_map={0: "#10b981", 1: "#f43f5e"}
            )
            fig_pb.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=210,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_pb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ TAB 3: SUBSCRIBERS & ADD-ONS ------------------
    with tabs[2]:
        st.markdown("### Core Services & Add-on Bundling Impacts")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Phone & Multiple Lines**")
            
            fig_phone = px.histogram(
                df, x="MultipleLines", color="Churn", barmode="stack",
                color_discrete_map={0: "#10b981", 1: "#f43f5e"}
            )
            fig_phone.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_phone, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Add-on Service Churn Rates (%)**")
            
            # Compute Churn percentages for Security, TechSupport, Backup, Protection
            addons = ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection']
            yes_rates = []
            no_rates = []
            
            for add in addons:
                yes_df = df[df[add] == 'Yes']
                no_df = df[df[add] == 'No']
                
                yr = (len(yes_df[yes_df["Churn"] == 1]) / len(yes_df) * 100) if len(yes_df) > 0 else 0
                nr = (len(no_df[no_df["Churn"] == 1]) / len(no_df) * 100) if len(no_df) > 0 else 0
                yes_rates.append(yr)
                no_rates.append(nr)
                
            fig_add = go.Figure(data=[
                go.Bar(name='Service: Yes Churn %', x=addons, y=yes_rates, marker_color='rgba(16, 185, 129, 0.6)'),
                go.Bar(name='Service: No Churn %', x=addons, y=no_rates, marker_color='rgba(244, 63, 94, 0.65)')
            ])
            fig_add.update_layout(
                barmode='group', margin=dict(t=10, b=10, l=10, r=10), height=200,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", suffix="%")
            )
            st.plotly_chart(fig_add, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Streaming Churn Impact**")
            
            stream_labels = ['TV: Yes', 'TV: No', 'Movies: Yes', 'Movies: No']
            tv_yes = df[df['StreamingTV'] == 'Yes']
            tv_no = df[df['StreamingTV'] == 'No']
            mov_yes = df[df['StreamingMovies'] == 'Yes']
            mov_no = df[df['StreamingMovies'] == 'No']
            
            stream_churns = [
                (len(tv_yes[tv_yes["Churn"] == 1]) / len(tv_yes) * 100) if len(tv_yes) > 0 else 0,
                (len(tv_no[tv_no["Churn"] == 1]) / len(tv_no) * 100) if len(tv_no) > 0 else 0,
                (len(mov_yes[mov_yes["Churn"] == 1]) / len(mov_yes) * 100) if len(mov_yes) > 0 else 0,
                (len(mov_no[mov_no["Churn"] == 1]) / len(mov_no) * 100) if len(mov_no) > 0 else 0
            ]
            fig_stream = px.bar(
                x=stream_labels, y=stream_churns,
                color=stream_labels,
                color_discrete_map={'TV: Yes': '#3b82f6', 'TV: No': '#52525b', 'Movies: Yes': '#8b5cf6', 'Movies: No': '#52525b'}
            )
            fig_stream.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=False,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_stream, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        col1_2, col2_2 = st.columns(2)
        
        with col1_2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Churn Correlation: Contract Terms**")
            fig_c_corr = px.histogram(
                df, x="Contract", color="Churn", barmode="stack",
                color_discrete_map={0: "#10b981", 1: "#f43f5e"}
            )
            fig_c_corr.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_c_corr, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2_2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Churn Correlation: Internet Service Type**")
            fig_i_corr = px.histogram(
                df, x="InternetService", color="Churn", barmode="stack",
                color_discrete_map={0: "#10b981", 1: "#f43f5e"}
            )
            fig_i_corr.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_i_corr, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ TAB 4: DEMOGRAPHICS ANALYSIS ------------------
    with tabs[3]:
        st.markdown("### Demographic Segment Risk Assessment")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Gender Churn Rate**")
            fig_gen = px.histogram(
                df, x="Gender", color="Churn", barmode="group",
                color_discrete_map={0: "#10b981", 1: "#f43f5e"}
            )
            fig_gen.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_gen, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Senior Citizen Churn Rate**")
            fig_sen = px.histogram(
                df, x="SeniorCitizen", color="Churn", barmode="group",
                color_discrete_map={0: "#10b981", 1: "#f43f5e"}
            )
            fig_sen.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_sen, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Partner & Dependents Churn Impact (%)**")
            
            p_yes = df[df['Partner'] == 'Yes']
            p_no = df[df['Partner'] == 'No']
            d_yes = df[df['Dependents'] == 'Yes']
            d_no = df[df['Dependents'] == 'No']
            
            pd_labels = ['Partner: Yes', 'Partner: No', 'Dependents: Yes', 'Dependents: No']
            pd_churns = [
                (len(p_yes[p_yes["Churn"] == 1]) / len(p_yes) * 100) if len(p_yes) > 0 else 0,
                (len(p_no[p_no["Churn"] == 1]) / len(p_no) * 100) if len(p_no) > 0 else 0,
                (len(d_yes[d_yes["Churn"] == 1]) / len(d_yes) * 100) if len(d_yes) > 0 else 0,
                (len(d_no[d_no["Churn"] == 1]) / len(d_no) * 100) if len(d_no) > 0 else 0
            ]
            fig_pd = px.bar(
                x=pd_labels, y=pd_churns,
                color=pd_labels,
                color_discrete_map={'Partner: Yes': '#06b6d4', 'Partner: No': '#52525b', 'Dependents: Yes': '#10b981', 'Dependents: No': '#52525b'}
            )
            fig_pd.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=200, showlegend=False,
                xaxis_title="", yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_pd, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ TAB 5: ML & SIMULATOR ------------------
    with tabs[4]:
        st.markdown("### Machine Learning Validation & Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Predictor Importance Score**")
            
            features = ['Contract', 'Tenure Months', 'Monthly Charges', 'Internet Service', 'Total Charges', 'Payment Method', 'Online Security', 'Tech Support', 'Senior Citizen']
            importance = [0.218, 0.174, 0.158, 0.119, 0.104, 0.082, 0.058, 0.048, 0.039]
            
            fig_f = px.bar(
                x=importance, y=features, orientation='h',
                color_discrete_sequence=["#8b5cf6"]
            )
            fig_f.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=220, showlegend=False,
                xaxis_title="", yaxis_title="", yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)")
            )
            st.plotly_chart(fig_f, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='background-color:#141416; padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
            st.write("**Model ROC Curves**")
            
            fig_roc = go.Figure()
            # XGBoost
            fig_roc.add_trace(go.Scatter(
                x=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                y=[0, 0.45, 0.68, 0.78, 0.83, 0.88, 0.91, 0.94, 0.97, 0.99, 1],
                mode='lines', name='XGBoost (AUC 0.867)',
                line=dict(color='#8b5cf6', width=2)
            ))
            # Random Forest
            fig_roc.add_trace(go.Scatter(
                x=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                y=[0, 0.42, 0.65, 0.75, 0.81, 0.85, 0.89, 0.92, 0.95, 0.98, 1],
                mode='lines', name='Random Forest (AUC 0.855)',
                line=dict(color='#3b82f6', width=1.5)
            ))
            # Baseline
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines', name='Baseline (AUC 0.500)',
                line=dict(color='#52525b', width=1, dash='dash')
            ))
            
            fig_roc.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), height=220,
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                legend=dict(x=0.5, y=0.1, xanchor='center', yanchor='bottom', bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("### Real-Time Customer Churn Risk Calculator (Logistic Regression)")
        
        sim_col1, sim_col2 = st.columns([1.2, 1])
        
        with sim_col1:
            s_tenure = st.slider("Tenure Months", min_value=1, max_value=72, value=24)
            s_charges = st.slider("Monthly Charges ($)", min_value=18, max_value=120, value=65)
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                s_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                s_security = st.selectbox("Online Security", ["No", "Yes"])
                s_payment = st.selectbox("Payment Method", ["Electronic check", "Others (Auto/Mail)"])
            with sub_col2:
                s_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                s_support = st.selectbox("Tech Support", ["No", "Yes"])
                
            # LogReg Equation calculation
            z = -3.1038389604616
            z += s_tenure * -0.030491610122415365
            z += s_charges * 0.012189431210984791
            
            if s_contract == 'Month-to-month':
                z += 1.3857182666646537
            elif s_contract == 'One year':
                z += 0.6576894325510663
                
            if s_internet == 'Fiber optic':
                z += 0.4995417554463798
            elif s_internet == 'DSL':
                z += -0.057752432881235924
                
            if s_security == 'No':
                z += 0.49681526610849097
            if s_support == 'No':
                z += 0.4182502774281507
            if s_payment == 'Electronic check':
                z += 0.4346891266087697
                
            prob = 1 / (1 + math.exp(-z))
            prob_pct = int(round(prob * 100))

        with sim_col2:
            st.markdown("<div class='sim-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='margin:0; color:#a1a1aa;'>Calculated Risk Probability</h4>", unsafe_allow_html=True)
            
            if prob_pct < 30:
                risk_lbl = "Low Risk"
                risk_color = "#10b981"
            elif prob_pct < 65:
                risk_lbl = "Medium Risk"
                risk_color = "#f59e0b"
            else:
                risk_lbl = "High Risk"
                risk_color = "#f43f5e"
                
            st.markdown(f"<h1 style='font-size:72px; font-weight:800; color:{risk_color}; margin:10px 0;'>{prob_pct}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<span style='background-color:{risk_color}22; color:{risk_color}; font-weight:800; font-size:16px; padding:6px 16px; border-radius:20px; text-transform:uppercase;'>{risk_lbl}</span>", unsafe_allow_html=True)
            st.markdown("<p style='color:#52525b; font-size:12px; margin-top:20px;'>Logistic Regression Inference Engine</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ FOOTER ------------------
    st.markdown("---")
    st.markdown("""
    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #52525b;">
        <span>Telco Churn Predictive System · B.Tech Submission Portfolio</span>
        <span>Dataset: Telco_customer_churn.xlsx (7,043 rows) · California, United States</span>
    </div>
    """, unsafe_allow_html=True)
