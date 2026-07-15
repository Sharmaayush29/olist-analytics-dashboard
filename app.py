import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import io
from datetime import datetime, date
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist E-Commerce Executive Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

  .stApp { background: linear-gradient(135deg,#f0f4ff 0%,#fafbff 100%); color:#0f172a; }

  h1,h2,h3,h4,h5,h6 { color:#1e3a8a !important; font-family:'Inter',sans-serif !important; font-weight:700 !important; }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a 0%,#1e293b 100%) !important;
    border-right:1px solid #334155;
  }
  section[data-testid="stSidebar"] * { color:#f1f5f9 !important; }
  section[data-testid="stSidebar"] label { color:#94a3b8 !important; font-size:0.8rem !important; }

  .kpi-card {
    background:white; border-radius:14px; padding:20px 18px 16px;
    box-shadow:0 4px 20px rgba(37,99,235,0.07),0 1px 4px rgba(0,0,0,0.04);
    border-left:5px solid #2563eb; transition:transform .2s,box-shadow .2s; margin-bottom:10px;
  }
  .kpi-card:hover { transform:translateY(-3px); box-shadow:0 8px 30px rgba(37,99,235,0.14); }
  .kpi-icon { font-size:1.4rem; margin-bottom:4px; }
  .kpi-title { font-size:0.72rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px; }
  .kpi-value { font-size:1.7rem; color:#1e3a8a; font-weight:800; line-height:1; }
  .kpi-delta { font-size:0.75rem; font-weight:600; margin-top:6px; }
  .kpi-delta.up { color:#16a34a; } .kpi-delta.down { color:#dc2626; } .kpi-delta.neutral { color:#64748b; }

  .section-badge {
    display:inline-block; background:linear-gradient(135deg,#eff6ff,#dbeafe);
    color:#1d4ed8; font-size:.72rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; padding:3px 10px; border-radius:20px; margin-bottom:6px;
  }
  .filter-header { font-size:.8rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px; padding:8px 0 4px; border-top:1px solid #334155; }
  .stButton>button {
    background:linear-gradient(135deg,#2563eb,#1d4ed8); color:white;
    border-radius:8px; border:none; padding:10px 20px; font-weight:600;
  }
  .stButton>button:hover { opacity:.88; }
  hr { border-color:#e2e8f0; }
  .footer-link { color:#2563eb; text-decoration:none; font-weight:600; }
  .footer-link:hover { text-decoration:underline; }
  .metric-badge {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:.78rem; font-weight:700; margin:3px;
  }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "processed_olist_data.csv"
MODEL_PATH = "best_model.pkl"

# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at **{DATA_PATH}**.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df

df_full = load_data()

# ── Plotly theme ───────────────────────────────────────────────────────────────
PALETTE = ["#1d4ed8","#059669","#d97706","#db2777","#7c3aed","#0891b2","#ea580c","#0f766e","#be123c"]
SCALE_BLUE   = ["#dbeafe","#3b82f6","#1e3a8a"]
SCALE_GREEN  = ["#d1fae5","#10b981","#065f46"]
SCALE_AMBER  = ["#fef3c7","#f59e0b","#92400e"]
SCALE_TEAL   = ["#ccfbf1","#14b8a6","#134e4a"]

_TICK  = dict(color="#0f172a", size=12, family="Inter, sans-serif")
_TITLE = dict(color="#0f172a", size=13, family="Inter, sans-serif")

AXIS_STYLE = dict(
    showline=True, linewidth=2, linecolor="#94a3b8", mirror=True,
    gridcolor="#e2e8f0", gridwidth=1, tickfont=_TICK, title_font=_TITLE,
    showgrid=True, zeroline=False, ticks="outside", ticklen=5,
    tickcolor="#94a3b8", automargin=True,
)
_CB = dict(tickfont=dict(color="#0f172a",size=12), title_font=dict(color="#0f172a",size=13),
           outlinecolor="#e2e8f0", outlinewidth=1)

def base_layout(**kw):
    lay = dict(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Inter, sans-serif", size=12),
        xaxis=dict(**AXIS_STYLE),
        yaxis=dict(**AXIS_STYLE),
        margin=dict(l=80, r=50, t=65, b=80, pad=6),
        legend=dict(bgcolor="#f8fafc", bordercolor="#e2e8f0", borderwidth=1,
                    font=dict(color="#0f172a",size=12)),
        coloraxis_colorbar=_CB,
        title_font=dict(color="#1e3a8a", size=15, family="Inter, sans-serif"),
    )
    lay.update(kw)
    return lay

def hbar_layout(**kw):
    lay = base_layout()
    lay["xaxis"] = {**AXIS_STYLE, "showgrid": True, "ticksuffix": " "}
    lay["yaxis"] = {**AXIS_STYLE, "showgrid": False,
                    "tickfont": dict(color="#0f172a",size=11,family="Inter, sans-serif")}
    lay["margin"] = dict(l=210, r=60, t=65, b=60, pad=6)
    lay.update(kw)
    return lay

def rot_layout(**kw):
    lay = base_layout(**kw)
    lay["xaxis"] = {**AXIS_STYLE, "tickangle": -35}
    lay["margin"] = dict(l=80, r=50, t=65, b=110, pad=6)
    lay.update(kw)
    return lay

def dl_csv(df, fname):
    return df.to_csv(index=False).encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR  — page selector + page-specific filters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;padding:10px 0 6px;">
      <div style="width:36px;height:36px;background:linear-gradient(135deg,#3b82f6,#1d4ed8);
                  border-radius:10px;display:flex;align-items:center;justify-content:center;
                  font-weight:800;color:white;font-size:13px;">OL</div>
      <div>
        <div style="font-size:14px;font-weight:700;color:#f8fafc;">Olist Analytics</div>
        <div style="font-size:11px;color:#94a3b8;">Executive BI Platform</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox("Navigate", [
        "📊 Executive Dashboard",
        "📈 Sales Analytics",
        "👥 Customer Analytics",
        "📦 Product Analytics",
        "💳 Payment Analytics",
        "🚚 Delivery Analytics",
        "🧠 ML Review Predictor",
        "📉 Model Performance",
        "💡 Business Insights",
        "🎯 Strategic Action Plan",
        "ℹ️ About Project",
    ])

    st.markdown("---")

    # ── PAGE-SPECIFIC FILTERS ─────────────────────────────────────────────────
    df = df_full.copy()

    if page == "📊 Executive Dashboard":
        st.markdown('<div class="filter-header">📅 Date Range</div>', unsafe_allow_html=True)
        min_d = df_full["order_purchase_timestamp"].min().date()
        max_d = df_full["order_purchase_timestamp"].max().date()
        date_range = st.date_input("Order Date Range", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)
        try:
            d1, d2 = date_range
            df = df[(df["order_purchase_timestamp"].dt.date >= d1) &
                    (df["order_purchase_timestamp"].dt.date <= d2)]
        except (TypeError, ValueError):
            pass  # user hasn't picked both dates yet

    elif page == "📈 Sales Analytics":
        st.markdown('<div class="filter-header">📅 Date Range</div>', unsafe_allow_html=True)
        min_d = df_full["order_purchase_timestamp"].min().date()
        max_d = df_full["order_purchase_timestamp"].max().date()
        date_range = st.date_input("Order Date Range", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)
        try:
            d1, d2 = date_range
            df = df[(df["order_purchase_timestamp"].dt.date >= d1) &
                    (df["order_purchase_timestamp"].dt.date <= d2)]
        except (TypeError, ValueError):
            pass  # user hasn't picked both dates yet
        st.markdown('<div class="filter-header">📦 Category</div>', unsafe_allow_html=True)
        cats = sorted(df_full["product_category_name_english"].dropna().unique())
        sel_cats = st.multiselect("Product Category", cats, default=cats[:8])
        if sel_cats:
            df = df[df["product_category_name_english"].isin(sel_cats)]
        st.markdown('<div class="filter-header">🏪 Seller State</div>', unsafe_allow_html=True)
        sstates = sorted(df_full["seller_state"].dropna().unique())
        sel_ss = st.multiselect("Seller State", sstates, default=sstates)
        if sel_ss:
            df = df[df["seller_state"].isin(sel_ss)]

    elif page == "👥 Customer Analytics":
        st.markdown('<div class="filter-header">🗺️ Customer State</div>', unsafe_allow_html=True)
        cstates = sorted(df_full["customer_state"].dropna().unique())
        sel_cs = st.multiselect("Customer State", cstates, default=cstates)
        if sel_cs:
            df = df[df["customer_state"].isin(sel_cs)]
        st.markdown('<div class="filter-header">⭐ Review Score</div>', unsafe_allow_html=True)
        rv_min, rv_max = st.slider("Review Score Range", 1, 5, (1, 5))
        df = df[df["review_score"].between(rv_min, rv_max)]

    elif page == "📦 Product Analytics":
        st.markdown('<div class="filter-header">📦 Category</div>', unsafe_allow_html=True)
        cats = sorted(df_full["product_category_name_english"].dropna().unique())
        sel_cats = st.multiselect("Product Category", cats, default=cats[:10])
        if sel_cats:
            df = df[df["product_category_name_english"].isin(sel_cats)]
        st.markdown('<div class="filter-header">💰 Price Range (R$)</div>', unsafe_allow_html=True)
        p_min = float(df_full["payment_value"].min())
        p_max = float(df_full["payment_value"].quantile(0.99))
        pv_min, pv_max = st.slider("Payment Value", p_min, p_max, (p_min, p_max), step=10.0)
        df = df[df["payment_value"].between(pv_min, pv_max)]

    elif page == "💳 Payment Analytics":
        st.markdown('<div class="filter-header">💳 Payment Method</div>', unsafe_allow_html=True)
        pmethods = sorted(df_full["payment_type"].dropna().unique())
        sel_pm = st.multiselect("Payment Method", pmethods, default=pmethods)
        if sel_pm:
            df = df[df["payment_type"].isin(sel_pm)]
        st.markdown('<div class="filter-header">🔢 Max Installments</div>', unsafe_allow_html=True)
        max_inst = st.slider("Max Installments", 1, 24, 24)
        df = df[df["payment_installments"] <= max_inst]

    elif page == "🚚 Delivery Analytics":
        st.markdown('<div class="filter-header">🗺️ Customer State</div>', unsafe_allow_html=True)
        cstates = sorted(df_full["customer_state"].dropna().unique())
        sel_cs = st.multiselect("Customer State", cstates, default=cstates)
        if sel_cs:
            df = df[df["customer_state"].isin(sel_cs)]
        if "is_late" in df_full.columns:
            st.markdown('<div class="filter-header">📦 Delivery Status</div>', unsafe_allow_html=True)
            status_opt = st.selectbox("Delivery Status", ["All", "On-Time Only", "Late Only"])
            if status_opt == "On-Time Only":
                df = df[df["is_late"] == 0]
            elif status_opt == "Late Only":
                df = df[df["is_late"] == 1]
    else:
        st.markdown('<div class="filter-header">ℹ️ Info</div>', unsafe_allow_html=True)
        st.caption("No filters for this page.")

    st.markdown("---")
    st.caption(f"Rows loaded: **{len(df):,}** / {len(df_full):,}")
    st.caption(f"Updated: {datetime.now().strftime('%d %b %Y %H:%M')}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.title("📊 Executive Business Dashboard")
    st.markdown(f"**Snapshot:** `{datetime.now().strftime('%B %d, %Y')}`  —  Filtered to **{len(df):,}** orders")

    # ── KPI row ───────────────────────────────────────────────────────────────
    total_orders   = df["order_id"].nunique()
    total_rev      = df["payment_value"].sum()
    total_cust     = df["customer_unique_id"].nunique()
    avg_review     = df["review_score"].mean()
    avg_delivery   = df["delivery_days"].mean()
    total_sellers  = df["seller_id"].nunique()

    # Growth vs prior period (simple halves split)
    df["_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    months_sorted = sorted(df["_month"].unique())
    mid = len(months_sorted) // 2
    half1 = df[df["_month"].isin(months_sorted[:mid])]
    half2 = df[df["_month"].isin(months_sorted[mid:])]

    def growth(new, old):
        return ((new - old) / old * 100) if old > 0 else 0

    rev_g   = growth(half2["payment_value"].sum(), half1["payment_value"].sum())
    ord_g   = growth(half2["order_id"].nunique(), half1["order_id"].nunique())
    cust_g  = growth(half2["customer_unique_id"].nunique(), half1["customer_unique_id"].nunique())
    rev_g_str  = f"{'▲' if rev_g>=0 else '▼'} {abs(rev_g):.1f}% Period-on-Period"
    ord_g_str  = f"{'▲' if ord_g>=0 else '▼'} {abs(ord_g):.1f}% Period-on-Period"
    cust_g_str = f"{'▲' if cust_g>=0 else '▼'} {abs(cust_g):.1f}% Period-on-Period"

    kpi_data = [
        ("📦","Total Orders",      f"{total_orders:,}",     ord_g_str,  "up" if ord_g>=0 else "down",   "#1d4ed8"),
        ("💰","Total Revenue",     f"R${total_rev:,.0f}",   rev_g_str,  "up" if rev_g>=0 else "down",   "#059669"),
        ("👥","Total Customers",   f"{total_cust:,}",       cust_g_str, "up" if cust_g>=0 else "down",  "#7c3aed"),
        ("⭐","Avg Review Score",  f"{avg_review:.2f} ★",   "Out of 5.0 stars",                "neutral","#d97706"),
        ("🚚","Avg Delivery Time", f"{avg_delivery:.1f}d",  "Days from order to delivery",     "neutral","#0891b2"),
        ("🏪","Total Sellers",     f"{total_sellers:,}",    "Active seller network",           "neutral","#ea580c"),
    ]
    cols = st.columns(6)
    for col, (icon,title,val,delta,cls,color) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:{color};">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-title">{title}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-delta {cls}">{delta}</div>
            </div>""", unsafe_allow_html=True)

    # Revenue & Orders growth summary bar
    st.markdown(f"""
    <div style="background:white;border-radius:12px;padding:16px 22px;margin:12px 0 20px;
                box-shadow:0 2px 12px rgba(0,0,0,0.05);display:flex;gap:30px;align-items:center;">
      <div><span style="font-size:.78rem;color:#64748b;font-weight:600;text-transform:uppercase;">Revenue Growth</span>
           <span style="margin-left:10px;font-size:1.1rem;font-weight:800;
                  color:{'#059669' if rev_g>=0 else '#dc2626'};">{'▲' if rev_g>=0 else '▼'} {abs(rev_g):.1f}%</span></div>
      <div><span style="font-size:.78rem;color:#64748b;font-weight:600;text-transform:uppercase;">Order Growth</span>
           <span style="margin-left:10px;font-size:1.1rem;font-weight:800;
                  color:{'#059669' if ord_g>=0 else '#dc2626'};">{'▲' if ord_g>=0 else '▼'} {abs(ord_g):.1f}%</span></div>
      <div><span style="font-size:.78rem;color:#64748b;font-weight:600;text-transform:uppercase;">Customer Growth</span>
           <span style="margin-left:10px;font-size:1.1rem;font-weight:800;
                  color:{'#059669' if cust_g>=0 else '#dc2626'};">{'▲' if cust_g>=0 else '▼'} {abs(cust_g):.1f}%</span></div>
      <div style="margin-left:auto;">
        <span style="font-size:.78rem;color:#64748b;">Period: {months_sorted[0].strftime('%b %Y')} → {months_sorted[-1].strftime('%b %Y')}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main charts ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-badge">Revenue Trend + 3-Month MA</div>', unsafe_allow_html=True)
        monthly = df.groupby("_month").agg(Revenue=("payment_value","sum"), Orders=("order_id","nunique")).reset_index().sort_values("_month")
        monthly["MA3"] = monthly["Revenue"].rolling(3, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["_month"], y=monthly["Revenue"], name="Monthly Revenue",
                             marker=dict(color="#1d4ed8", line=dict(color="white", width=0.5)),
                             hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: R$%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=monthly["_month"], y=monthly["MA3"], name="3-Month MA",
                                 line=dict(color="#d97706", width=3, dash="dot"),
                                 marker=dict(size=6),
                                 hovertemplate="<b>%{x|%b %Y}</b><br>MA: R$%{y:,.0f}<extra></extra>"))
        fig.update_layout(**base_layout(title="Monthly Revenue & 3-Month Moving Average",
                                        xaxis_title="Month", yaxis_title="Revenue (R$)",
                                        legend=dict(orientation="h", x=0, y=-0.25)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-badge">Geographic Revenue Distribution</div>', unsafe_allow_html=True)
        state_rev = df.groupby("customer_state")["payment_value"].sum().reset_index().sort_values("payment_value", ascending=False).head(10)
        fig_state = px.bar(state_rev, x="customer_state", y="payment_value",
                           labels={"customer_state":"Customer State","payment_value":"Revenue (R$)"},
                           color="payment_value", color_continuous_scale=SCALE_BLUE,
                           title="Top 10 States by Revenue")
        fig_state.update_traces(marker_line_color="white", marker_line_width=0.8,
                                hovertemplate="<b>%{x}</b><br>R$%{y:,.0f}<extra></extra>")
        fig_state.update_layout(**rot_layout(xaxis_title="State", yaxis_title="Revenue (R$)"))
        st.plotly_chart(fig_state, use_container_width=True)

    # Revenue + Orders dual-axis
    st.markdown('<div class="section-badge">Order Volume vs Revenue — Full Timeline</div>', unsafe_allow_html=True)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=monthly["_month"], y=monthly["Orders"], name="Order Volume",
                          marker=dict(color="#1d4ed8", line=dict(color="white",width=0.5)),
                          hovertemplate="<b>%{x|%b %Y}</b><br>Orders: %{y:,}<extra></extra>"), secondary_y=False)
    fig2.add_trace(go.Scatter(x=monthly["_month"], y=monthly["Revenue"], name="Revenue (R$)",
                              line=dict(color="#d97706", width=3), marker=dict(size=7, color="#d97706"),
                              mode="lines+markers",
                              hovertemplate="<b>%{x|%b %Y}</b><br>R$%{y:,.0f}<extra></extra>"), secondary_y=True)
    fig2.update_layout(**base_layout(title="Monthly Order Volume vs Revenue",
                                     legend=dict(orientation="h", x=0, y=-0.22)))
    fig2.update_xaxes(title_text="Month", **AXIS_STYLE)
    fig2.update_yaxes(title_text="Order Volume", secondary_y=False, **AXIS_STYLE)
    fig2.update_yaxes(title_text="Revenue (R$)", secondary_y=True, **{**AXIS_STYLE, "showgrid": False})
    st.plotly_chart(fig2, use_container_width=True)

    # Download
    st.download_button("📥 Download Filtered Dataset (CSV)", data=dl_csv(df,"filtered.csv"),
                       file_name="olist_filtered.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — SALES ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Sales Analytics":
    st.title("📈 Sales Performance & Revenue Analytics")
    st.caption(f"Showing **{len(df):,}** filtered orders.")

    df["_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly_sales = df.groupby("_month").agg(Revenue=("payment_value","sum"),
                                              Orders=("order_id","nunique")).reset_index().sort_values("_month")
    monthly_sales["MA3_rev"] = monthly_sales["Revenue"].rolling(3, min_periods=1).mean()
    monthly_sales["MA3_ord"] = monthly_sales["Orders"].rolling(3, min_periods=1).mean()
    monthly_sales["Rev_growth"] = monthly_sales["Revenue"].pct_change() * 100
    monthly_sales["Ord_growth"] = monthly_sales["Orders"].pct_change() * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-badge">Monthly Revenue + Moving Average + Growth %</div>', unsafe_allow_html=True)
        fig_s1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_s1.add_trace(go.Bar(x=monthly_sales["_month"], y=monthly_sales["Revenue"],
                                name="Revenue (R$)", marker=dict(color="#1d4ed8", line=dict(color="white",width=0.4)),
                                hovertemplate="<b>%{x|%b %Y}</b><br>R$%{y:,.0f}<extra></extra>"), secondary_y=False)
        fig_s1.add_trace(go.Scatter(x=monthly_sales["_month"], y=monthly_sales["MA3_rev"],
                                    name="3-Mo MA", line=dict(color="#d97706",width=2.5,dash="dot"),
                                    hovertemplate="MA: R$%{y:,.0f}<extra></extra>"), secondary_y=False)
        fig_s1.add_trace(go.Scatter(x=monthly_sales["_month"], y=monthly_sales["Rev_growth"],
                                    name="Growth %", line=dict(color="#dc2626",width=2),
                                    mode="lines+markers", marker=dict(size=5),
                                    hovertemplate="Growth: %{y:.1f}%<extra></extra>"), secondary_y=True)
        fig_s1.update_layout(**base_layout(title="Monthly Revenue Trend",
                                           legend=dict(orientation="h", x=0, y=-0.25)))
        fig_s1.update_xaxes(title_text="Month", **AXIS_STYLE)
        fig_s1.update_yaxes(title_text="Revenue (R$)", secondary_y=False, **AXIS_STYLE)
        fig_s1.update_yaxes(title_text="Growth (%)", secondary_y=True, **{**AXIS_STYLE, "showgrid": False})
        st.plotly_chart(fig_s1, use_container_width=True)

    with col2:
        st.markdown('<div class="section-badge">Monthly Orders + Trend + Growth %</div>', unsafe_allow_html=True)
        fig_s2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_s2.add_trace(go.Bar(x=monthly_sales["_month"], y=monthly_sales["Orders"],
                                name="Orders", marker=dict(color="#7c3aed", line=dict(color="white",width=0.4)),
                                hovertemplate="<b>%{x|%b %Y}</b><br>Orders: %{y:,}<extra></extra>"), secondary_y=False)
        fig_s2.add_trace(go.Scatter(x=monthly_sales["_month"], y=monthly_sales["MA3_ord"],
                                    name="3-Mo MA", line=dict(color="#ea580c",width=2.5,dash="dot"),
                                    hovertemplate="MA: %{y:,.0f}<extra></extra>"), secondary_y=False)
        fig_s2.add_trace(go.Scatter(x=monthly_sales["_month"], y=monthly_sales["Ord_growth"],
                                    name="Growth %", line=dict(color="#059669",width=2),
                                    mode="lines+markers", marker=dict(size=5),
                                    hovertemplate="Growth: %{y:.1f}%<extra></extra>"), secondary_y=True)
        fig_s2.update_layout(**base_layout(title="Monthly Order Volume Trend",
                                           legend=dict(orientation="h", x=0, y=-0.25)))
        fig_s2.update_xaxes(title_text="Month", **AXIS_STYLE)
        fig_s2.update_yaxes(title_text="Order Count", secondary_y=False, **AXIS_STYLE)
        fig_s2.update_yaxes(title_text="Growth (%)", secondary_y=True, **{**AXIS_STYLE, "showgrid": False})
        st.plotly_chart(fig_s2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-badge">Revenue by Category (Top 10)</div>', unsafe_allow_html=True)
        cat_rev = df.groupby("product_category_name_english")["payment_value"].sum().reset_index()
        cat_rev = cat_rev.sort_values("payment_value", ascending=True).tail(10)
        cat_rev["Label"] = cat_rev["product_category_name_english"].str.replace("_"," ").str.title()
        fig_s3 = px.bar(cat_rev, y="Label", x="payment_value", orientation="h",
                        labels={"Label":"Category","payment_value":"Revenue (R$)"},
                        title="Top 10 Categories by Revenue",
                        color="payment_value", color_continuous_scale=SCALE_BLUE)
        fig_s3.update_traces(marker_line_color="white", marker_line_width=0.5,
                             hovertemplate="<b>%{y}</b><br>R$%{x:,.0f}<extra></extra>")
        fig_s3.update_layout(**hbar_layout(xaxis_title="Revenue (R$)", yaxis_title=""))
        st.plotly_chart(fig_s3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-badge">Avg Order Value by State (Top 10)</div>', unsafe_allow_html=True)
        aov = df.groupby("customer_state").agg(Rev=("payment_value","sum"), Ord=("order_id","nunique")).reset_index()
        aov["AOV"] = aov["Rev"] / aov["Ord"]
        aov = aov.sort_values("AOV", ascending=False).head(10)
        fig_s4 = px.bar(aov, x="customer_state", y="AOV",
                        labels={"customer_state":"State","AOV":"Avg Order Value (R$)"},
                        title="AOV by Customer State",
                        color="AOV", color_continuous_scale=SCALE_AMBER)
        fig_s4.update_traces(marker_line_color="white", marker_line_width=0.5,
                             hovertemplate="<b>%{x}</b><br>AOV: R$%{y:,.2f}<extra></extra>")
        fig_s4.update_layout(**rot_layout(xaxis_title="State", yaxis_title="Avg Order Value (R$)"))
        st.plotly_chart(fig_s4, use_container_width=True)

    st.download_button("📥 Download Sales Data (CSV)", data=dl_csv(df,"sales.csv"),
                       file_name="olist_sales.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CUSTOMER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Analytics":
    st.title("👥 Customer & Satisfaction Analytics")
    st.caption(f"Filtered to **{len(df):,}** orders.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-badge">Customer Distribution by State</div>', unsafe_allow_html=True)
        cs = df["customer_state"].value_counts().reset_index()
        cs.columns = ["State","Customers"]
        fig_c1 = px.pie(cs, names="State", values="Customers", hole=0.45,
                        title="Customer Share by State", color_discrete_sequence=PALETTE)
        fig_c1.update_traces(textposition="inside", textinfo="percent+label",
                             hovertemplate="<b>%{label}</b><br>%{value:,} customers<extra></extra>")
        fig_c1.update_layout(**base_layout(title="Customer Share by State", showlegend=True))
        st.plotly_chart(fig_c1, use_container_width=True)

    with col2:
        st.markdown('<div class="section-badge">Review Score Distribution</div>', unsafe_allow_html=True)
        rd = df["review_score"].value_counts().sort_index().reset_index()
        rd.columns = ["Score","Count"]
        rd["Label"] = rd["Score"].astype(str) + " ★"
        bar_colors = ["#dc2626","#ea580c","#ca8a04","#16a34a","#059669"]
        fig_c2 = go.Figure()
        for i, row in rd.iterrows():
            fig_c2.add_trace(go.Bar(x=[row["Label"]], y=[row["Count"]],
                                    marker=dict(color=bar_colors[i], line=dict(color="white",width=1)),
                                    showlegend=False,
                                    hovertemplate=f"<b>{row['Label']}</b><br>{row['Count']:,} orders<extra></extra>"))
        fig_c2.update_layout(**base_layout(title="Review Score Distribution",
                                           xaxis_title="Review Score", yaxis_title="Number of Orders"))
        st.plotly_chart(fig_c2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-badge">Top 10 Cities by Order Volume</div>', unsafe_allow_html=True)
        tc = df["customer_city"].value_counts().reset_index().head(10)
        tc.columns = ["City","Orders"]
        tc = tc.sort_values("Orders")
        fig_c3 = px.bar(tc, x="Orders", y="City", orientation="h",
                        labels={"Orders":"Number of Orders","City":"City"},
                        title="Top 10 Buying Cities", color_discrete_sequence=["#0891b2"])
        fig_c3.update_traces(marker_line_color="white", marker_line_width=0.5,
                             hovertemplate="<b>%{y}</b><br>%{x:,} orders<extra></extra>")
        fig_c3.update_layout(**hbar_layout(xaxis_title="Number of Orders", yaxis_title=""))
        st.plotly_chart(fig_c3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-badge">Customer Loyalty Segmentation</div>', unsafe_allow_html=True)
        cc = df["customer_unique_id"].value_counts()
        rep_df = pd.DataFrame({"Segment":["Single Purchase","Repeat (2+)"],
                               "Customers":[sum(cc==1), sum(cc>1)]})
        fig_c4 = px.pie(rep_df, names="Segment", values="Customers", hole=0.6,
                        title="Loyalty Segmentation", color_discrete_sequence=["#1d4ed8","#059669"])
        fig_c4.update_traces(textposition="outside", textinfo="percent+label",
                             textfont=dict(color="#0f172a",size=13),
                             hovertemplate="<b>%{label}</b><br>%{value:,} customers<extra></extra>")
        fig_c4.update_layout(**base_layout(title="Loyalty Segmentation"))
        st.plotly_chart(fig_c4, use_container_width=True)

    # Scatter: delivery days vs review score
    st.markdown('<div class="section-badge">Delivery Time vs Review Score</div>', unsafe_allow_html=True)
    samp = df[["delivery_days","review_score","payment_value"]].dropna().sample(min(3000,len(df)),random_state=42)
    fig_c5 = px.scatter(samp, x="delivery_days", y="review_score", size="payment_value",
                        size_max=18, opacity=0.6, color="review_score",
                        color_continuous_scale=["#dc2626","#f59e0b","#059669"],
                        labels={"delivery_days":"Delivery Days","review_score":"Review Score (1–5)","payment_value":"Order Value (R$)"},
                        title="Delivery Time vs Review Score (bubble = order value)")
    fig_c5.update_layout(**base_layout(xaxis_title="Delivery Days", yaxis_title="Review Score (1–5)"))
    st.plotly_chart(fig_c5, use_container_width=True)

    st.download_button("📥 Download Customer Data (CSV)", data=dl_csv(df,"customers.csv"),
                       file_name="olist_customers.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PRODUCT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Product Analytics":
    st.title("📦 Product & Catalog Performance")
    st.caption(f"Filtered to **{len(df):,}** orders.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-badge">Top 10 Products by Sales Volume</div>', unsafe_allow_html=True)
        tp = df["product_id"].value_counts().reset_index().head(10)
        tp.columns = ["Product ID","Orders"]
        tp["Short"] = tp["Product ID"].str[:12] + "…"
        tp = tp.sort_values("Orders")
        fig_p1 = px.bar(tp, x="Orders", y="Short", orientation="h",
                        title="Top 10 Products by Volume", color_discrete_sequence=["#db2777"])
        fig_p1.update_traces(marker_line_color="white", marker_line_width=0.5,
                             hovertemplate="<b>%{y}</b><br>%{x:,} orders<extra></extra>")
        fig_p1.update_layout(**hbar_layout(xaxis_title="Orders", yaxis_title="Product ID"))
        st.plotly_chart(fig_p1, use_container_width=True)

    with col2:
        st.markdown('<div class="section-badge">Product Weight Distribution</div>', unsafe_allow_html=True)
        wt = df["product_weight_g"].dropna()
        fig_p2 = px.histogram(wt, x=wt, nbins=60, title="Product Weight Distribution",
                              labels={"x":"Weight (g)","count":"Products"},
                              color_discrete_sequence=["#ea580c"])
        fig_p2.update_traces(marker_line_color="white", marker_line_width=0.4,
                             hovertemplate="Weight: %{x}g<br>Count: %{y}<extra></extra>")
        fig_p2.update_layout(**base_layout(xaxis_title="Product Weight (grams)", yaxis_title="Number of Products"))
        st.plotly_chart(fig_p2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-badge">Category Revenue Share (Top 10)</div>', unsafe_allow_html=True)
        cp = df.groupby("product_category_name_english")["payment_value"].sum().reset_index().sort_values("payment_value",ascending=False).head(10)
        cp["Cat"] = cp["product_category_name_english"].str.replace("_"," ").str.title()
        fig_p3 = px.pie(cp, names="Cat", values="payment_value", hole=0.42,
                        title="Top 10 Category Revenue Share", color_discrete_sequence=PALETTE)
        fig_p3.update_traces(textposition="outside", textinfo="percent+label",
                             hovertemplate="<b>%{label}</b><br>R$%{value:,.0f}<extra></extra>")
        fig_p3.update_layout(**base_layout(title="Top 10 Category Revenue Share"))
        st.plotly_chart(fig_p3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-badge">Category × Review Score Heatmap</div>', unsafe_allow_html=True)
        tc2 = df["product_category_name_english"].value_counts().head(8).index
        hdf = df[df["product_category_name_english"].isin(tc2)].groupby(
            ["product_category_name_english","review_score"])["order_id"].count().reset_index()
        hdf["Cat"] = hdf["product_category_name_english"].str.replace("_"," ").str.title()
        hp = hdf.pivot(index="Cat", columns="review_score", values="order_id").fillna(0)
        fig_heat = go.Figure(go.Heatmap(
            z=hp.values, x=[f"{s}★" for s in hp.columns], y=hp.index,
            colorscale=[[0,"#eff6ff"],[0.25,"#93c5fd"],[0.6,"#1d4ed8"],[1,"#1e3a8a"]],
            hovertemplate="Category: %{y}<br>Score: %{x}<br>Orders: %{z:,}<extra></extra>",
            colorbar=dict(title=dict(text="Orders",font=dict(color="#0f172a",size=13)),
                         tickfont=dict(color="#0f172a",size=12))
        ))
        fig_heat.update_layout(**base_layout(title="Order Density: Category vs Review Score",
                                             xaxis_title="Review Score", yaxis_title=""))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.download_button("📥 Download Product Data (CSV)", data=dl_csv(df,"products.csv"),
                       file_name="olist_products.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PAYMENT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💳 Payment Analytics":
    st.title("💳 Payment Methods & Installment Analytics")
    st.caption(f"Filtered to **{len(df):,}** orders.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-badge">Payment Method Share</div>', unsafe_allow_html=True)
        pc = df["payment_type"].value_counts().reset_index()
        pc.columns = ["Method","Transactions"]
        pc["Label"] = pc["Method"].str.replace("_"," ").str.title()
        fig_py1 = px.pie(pc, names="Label", values="Transactions", hole=0.5,
                         title="Payment Method Distribution", color_discrete_sequence=PALETTE)
        fig_py1.update_traces(textposition="outside", textinfo="percent+label",
                              hovertemplate="<b>%{label}</b><br>%{value:,} transactions<extra></extra>")
        fig_py1.update_layout(**base_layout(title="Payment Method Distribution"))
        st.plotly_chart(fig_py1, use_container_width=True)

    with col2:
        st.markdown('<div class="section-badge">Installment Distribution</div>', unsafe_allow_html=True)
        inst = df["payment_installments"].value_counts().reset_index().sort_values("payment_installments").head(12)
        inst.columns = ["Installments","Count"]
        fig_py2 = px.bar(inst, x="Installments", y="Count",
                         labels={"Installments":"Number of Installments","Count":"Transactions"},
                         title="Installment Plan Usage", color_discrete_sequence=["#db2777"])
        fig_py2.update_traces(marker_line_color="white", marker_line_width=0.5,
                              hovertemplate="<b>%{x} installments</b><br>%{y:,} transactions<extra></extra>")
        fig_py2.update_layout(**base_layout(xaxis_title="Number of Installments", yaxis_title="Transactions"))
        st.plotly_chart(fig_py2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-badge">Avg Transaction Value by Method</div>', unsafe_allow_html=True)
        pa = df.groupby("payment_type")["payment_value"].mean().reset_index()
        pa.columns = ["Method","Avg"]
        pa["Label"] = pa["Method"].str.replace("_"," ").str.title()
        fig_py3 = px.bar(pa, x="Label", y="Avg", color="Avg", color_continuous_scale=SCALE_TEAL,
                         labels={"Label":"Method","Avg":"Avg Value (R$)"},
                         title="Average Order Value by Payment Method")
        fig_py3.update_traces(marker_line_color="white", marker_line_width=0.5,
                              hovertemplate="<b>%{x}</b><br>Avg: R$%{y:,.2f}<extra></extra>")
        fig_py3.update_layout(**rot_layout(xaxis_title="Payment Method", yaxis_title="Avg Value (R$)"))
        st.plotly_chart(fig_py3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-badge">Total Revenue by Method</div>', unsafe_allow_html=True)
        pt = df.groupby("payment_type")["payment_value"].sum().reset_index()
        pt.columns = ["Method","Revenue"]
        pt["Label"] = pt["Method"].str.replace("_"," ").str.title()
        pt = pt.sort_values("Revenue")
        fig_py4 = px.bar(pt, x="Revenue", y="Label", orientation="h",
                         title="Total Revenue by Payment Method",
                         color_discrete_sequence=["#1d4ed8"])
        fig_py4.update_traces(marker_line_color="white", marker_line_width=0.5,
                              hovertemplate="<b>%{y}</b><br>R$%{x:,.0f}<extra></extra>")
        fig_py4.update_layout(**hbar_layout(xaxis_title="Revenue (R$)", yaxis_title=""))
        st.plotly_chart(fig_py4, use_container_width=True)

    st.download_button("📥 Download Payment Data (CSV)", data=dl_csv(df,"payments.csv"),
                       file_name="olist_payments.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — DELIVERY ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚚 Delivery Analytics":
    st.title("🚚 Delivery Logistics & Speed Performance")
    st.caption(f"Filtered to **{len(df):,}** orders.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-badge">Delivery Speed Distribution</div>', unsafe_allow_html=True)
        fig_d1 = px.histogram(df, x="delivery_days", nbins=55, color_discrete_sequence=["#ea580c"],
                              title="Delivery Time Distribution",
                              labels={"delivery_days":"Delivery Days","count":"Orders"})
        fig_d1.update_traces(marker_line_color="white", marker_line_width=0.4,
                             hovertemplate="Days: %{x}<br>Orders: %{y}<extra></extra>")
        fig_d1.update_layout(**base_layout(xaxis_title="Delivery Days", yaxis_title="Number of Orders"))
        st.plotly_chart(fig_d1, use_container_width=True)

    with col2:
        st.markdown('<div class="section-badge">On-Time vs Late Deliveries</div>', unsafe_allow_html=True)
        if "is_late" in df.columns:
            ldf = df["is_late"].value_counts().reset_index()
            ldf.columns = ["is_late","Count"]
            ldf["Status"] = ldf["is_late"].map({0:"✅ On-Time",1:"❌ Late"})
        else:
            df["_late"] = (df["delivery_days"] > df["delivery_days"].median()+5).astype(int)
            ldf = df["_late"].value_counts().reset_index()
            ldf.columns = ["_late","Count"]
            ldf["Status"] = ldf["_late"].map({0:"✅ On-Time",1:"❌ Late"})
        fig_d2 = px.pie(ldf, names="Status", values="Count", hole=0.5,
                        title="Delivery SLA Performance",
                        color_discrete_sequence=["#059669","#dc2626"])
        fig_d2.update_traces(textposition="outside", textinfo="percent+label",
                             textfont=dict(color="#0f172a",size=13),
                             hovertemplate="<b>%{label}</b><br>%{value:,} orders<extra></extra>")
        fig_d2.update_layout(**base_layout(title="Delivery SLA Performance"))
        st.plotly_chart(fig_d2, use_container_width=True)

    # Trend + SLA line
    st.markdown('<div class="section-badge">Monthly Avg Delivery Time vs SLA Target</div>', unsafe_allow_html=True)
    df["_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    md = df.groupby("_month")["delivery_days"].agg(["mean","std"]).reset_index()
    md.columns = ["Month","Avg","Std"]
    fig_d3 = go.Figure()
    fig_d3.add_trace(go.Scatter(x=md["Month"], y=md["Avg"]+md["Std"],
                                fill=None, mode="lines", line=dict(color="rgba(29,78,216,0.1)"), showlegend=False))
    fig_d3.add_trace(go.Scatter(x=md["Month"], y=md["Avg"]-md["Std"],
                                fill="tonexty", fillcolor="rgba(29,78,216,0.08)",
                                mode="lines", line=dict(color="rgba(29,78,216,0.1)"),
                                name="±1 Std Dev"))
    fig_d3.add_trace(go.Scatter(x=md["Month"], y=md["Avg"], name="Avg Delivery Days",
                                line=dict(color="#1d4ed8",width=3.5),
                                marker=dict(size=8,color="#1d4ed8",line=dict(color="white",width=2)),
                                mode="lines+markers",
                                hovertemplate="<b>%{x|%b %Y}</b><br>Avg: %{y:.1f}d<extra></extra>"))
    fig_d3.add_hline(y=10, line_dash="dash", line_color="#dc2626", line_width=2.5,
                     annotation_text="  SLA Target: 10d",
                     annotation_font_color="#dc2626", annotation_font_size=13)
    fig_d3.update_layout(**base_layout(title="Monthly Avg Delivery Days vs SLA",
                                       xaxis_title="Month", yaxis_title="Avg Delivery Days",
                                       legend=dict(orientation="h",x=0,y=-0.22)))
    st.plotly_chart(fig_d3, use_container_width=True)

    # Box by state
    st.markdown('<div class="section-badge">Delivery Days by State</div>', unsafe_allow_html=True)
    top8 = df["customer_state"].value_counts().head(8).index
    fig_d4 = px.box(df[df["customer_state"].isin(top8)], x="customer_state", y="delivery_days",
                    color="customer_state", color_discrete_sequence=PALETTE,
                    title="Delivery Day Distribution by State (Top 8)",
                    labels={"customer_state":"State","delivery_days":"Delivery Days"})
    fig_d4.update_traces(hovertemplate="<b>%{x}</b><br>Days: %{y}<extra></extra>")
    fig_d4.update_layout(**rot_layout(xaxis_title="Customer State", yaxis_title="Delivery Days", showlegend=False))
    st.plotly_chart(fig_d4, use_container_width=True)

    st.download_button("📥 Download Delivery Data (CSV)", data=dl_csv(df,"delivery.csv"),
                       file_name="olist_delivery.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — ML REVIEW PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 ML Review Predictor":
    st.title("🧠 ML-Powered Customer Review Score Predictor")
    st.markdown("Estimate the expected review rating **(1–5 ★)** using a trained Random Forest model.")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found.")
        st.stop()

    with open(MODEL_PATH,"rb") as f:
        mp = pickle.load(f)
    model = mp["model"]; le_cat = mp["le_category"]
    le_cs = mp["le_customer_state"]; le_ss = mp["le_seller_state"]

    FEATURES = ["payment_value","payment_installments","delivery_days",
                "product_category_name_english","seller_state","customer_state"]
    FEAT_LABELS = {"payment_value":"Payment Value","payment_installments":"Installments",
                   "delivery_days":"Delivery Days","product_category_name_english":"Product Category",
                   "seller_state":"Seller State","customer_state":"Customer State"}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Transaction Parameters")
        p_val    = st.slider("Payment Value (R$)", 1.0, 2000.0, 120.0, step=5.0)
        p_inst   = st.slider("Number of Installments", 1, 24, 2)
        del_days = st.slider("Expected Delivery Days", 1, 60, 9)
    with col2:
        st.subheader("🗂️ Categorical Context")
        cat_list  = sorted(le_cat.classes_)
        cst_list  = sorted(le_cs.classes_)
        sst_list  = sorted(le_ss.classes_)
        sel_cat  = st.selectbox("Product Category", cat_list,
                                index=cat_list.index("health_beauty") if "health_beauty" in cat_list else 0)
        sel_cst  = st.selectbox("Customer State", cst_list,
                                index=cst_list.index("SP") if "SP" in cst_list else 0)
        sel_sst  = st.selectbox("Seller State", sst_list,
                                index=sst_list.index("SP") if "SP" in sst_list else 0)

    if st.button("🔮 Generate Review Forecast", use_container_width=True):
        try:
            inp = pd.DataFrame([{
                "payment_value": p_val,
                "payment_installments": p_inst,
                "delivery_days": del_days,
                "product_category_name_english": le_cat.transform([sel_cat])[0],
                "seller_state": le_ss.transform([sel_sst])[0],
                "customer_state": le_cs.transform([sel_cst])[0],
            }])

            pred  = int(model.predict(inp)[0])
            probs = model.predict_proba(inp)[0]
            conf  = float(probs[pred-1]*100)

            if pred >= 4:   risk, rc, rec = "Low Risk",    "#059669", "Order expected to satisfy. Maintain standard SLAs."
            elif pred == 3: risk, rc, rec = "Medium Risk", "#d97706", "Caution: possible friction. Accelerate dispatch."
            else:           risk, rc, rec = "High Risk",   "#dc2626", "Critical: contact customer proactively or upgrade courier."

            st.markdown("---")

            # Result cards
            r1,r2,r3 = st.columns(3)
            with r1:
                stars = "★"*pred + "☆"*(5-pred)
                st.markdown(f"""
                <div style="background:white;border-radius:12px;padding:24px;text-align:center;
                            box-shadow:0 4px 20px rgba(0,0,0,0.07);border-top:5px solid {rc};">
                  <div style="font-size:.75rem;color:#64748b;font-weight:600;text-transform:uppercase;">Predicted Rating</div>
                  <div style="font-size:2.6rem;color:#d97706;margin:10px 0;">{stars}</div>
                  <div style="font-size:1.8rem;font-weight:800;color:#1e3a8a;">{pred} / 5</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div style="background:white;border-radius:12px;padding:24px;text-align:center;
                            box-shadow:0 4px 20px rgba(0,0,0,0.07);border-top:5px solid {rc};">
                  <div style="font-size:.75rem;color:#64748b;font-weight:600;text-transform:uppercase;">Model Confidence</div>
                  <div style="font-size:2.8rem;font-weight:800;color:#1e3a8a;margin:12px 0;">{conf:.1f}%</div>
                  <div style="font-size:.88rem;color:#64748b;">Probability for score {pred}</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                <div style="background:white;border-radius:12px;padding:24px;
                            box-shadow:0 4px 20px rgba(0,0,0,0.07);border-top:5px solid {rc};">
                  <div style="font-size:.75rem;color:#64748b;font-weight:600;text-transform:uppercase;">Risk Level</div>
                  <div style="font-size:1.6rem;font-weight:800;color:{rc};margin:8px 0;">{risk}</div>
                  <div style="font-size:.85rem;color:#475569;">{rec}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)

            # Probability bar chart
            with col_a:
                ml_colors = ["#dc2626","#ea580c","#ca8a04","#16a34a","#059669"]
                fig_ml = go.Figure()
                for i in range(5):
                    fig_ml.add_trace(go.Bar(
                        x=[f"{i+1} ★"], y=[probs[i]*100],
                        marker=dict(color=ml_colors[i], line=dict(color="white",width=1.5)),
                        showlegend=False,
                        hovertemplate=f"<b>{i+1} ★</b><br>Probability: {probs[i]*100:.1f}%<extra></extra>"
                    ))
                fig_ml.update_layout(**base_layout(
                    title="Score Probability Distribution",
                    xaxis_title="Review Score", yaxis_title="Probability (%)"))
                st.plotly_chart(fig_ml, use_container_width=True)

            # Feature importance
            with col_b:
                if hasattr(model,"feature_importances_"):
                    fi = pd.DataFrame({"Feature":[FEAT_LABELS.get(f,f) for f in FEATURES],
                                       "Importance":model.feature_importances_}).sort_values("Importance")
                    fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                                    title="Feature Importance (Model Weights)",
                                    color="Importance", color_continuous_scale=SCALE_BLUE)
                    fig_fi.update_traces(marker_line_color="white", marker_line_width=0.5,
                                        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>")
                    fig_fi.update_layout(**hbar_layout(xaxis_title="Importance Score", yaxis_title=""))
                    st.plotly_chart(fig_fi, use_container_width=True)

            # Key drivers callout
            if hasattr(model,"feature_importances_"):
                fi_sorted = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
                top3 = [FEAT_LABELS.get(f,f) for f,_ in fi_sorted[:3]]
                st.markdown(f"""
                <div style="background:#f0fdf4;border-left:4px solid #059669;border-radius:8px;padding:16px 20px;">
                  <div style="font-weight:700;color:#065f46;margin-bottom:8px;">🔑 Key Prediction Drivers</div>
                  {''.join(f'<span class="metric-badge" style="background:#dcfce7;color:#065f46;">✔ {f}</span>' for f in top3)}
                </div>""", unsafe_allow_html=True)

            # Export
            st.markdown("<br>", unsafe_allow_html=True)
            exp = pd.DataFrame([{"Timestamp":datetime.now().isoformat(),
                                  "Category":sel_cat,"CustomerState":sel_cst,"SellerState":sel_sst,
                                  "PaymentValue":p_val,"Installments":p_inst,"DeliveryDays":del_days,
                                  "PredictedScore":pred,"Confidence%":f"{conf:.1f}","Risk":risk}])
            st.download_button("📥 Download Prediction Result (CSV)", data=dl_csv(exp,"prediction"),
                               file_name="prediction_result.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 8 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Model Performance":
    st.title("📉 ML Model Performance Evaluation")
    st.markdown("End-to-end evaluation of the trained **Random Forest Classifier** on the Olist dataset.")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found.")
        st.stop()

    with open(MODEL_PATH,"rb") as f:
        mp = pickle.load(f)
    model = mp["model"]; le_cat = mp["le_category"]
    le_cs = mp["le_customer_state"]; le_ss = mp["le_seller_state"]

    @st.cache_data
    def run_evaluation(_model, _le_cat, _le_cs, _le_ss):
        """Prefixed with _ so st.cache_data skips hashing sklearn objects."""
        eval_df = df_full.copy()
        eval_df = eval_df.dropna(subset=["payment_value","payment_installments","delivery_days",
                                          "product_category_name_english","seller_state",
                                          "customer_state","review_score"])
        cats_known = set(_le_cat.classes_)
        cs_known   = set(_le_cs.classes_)
        ss_known   = set(_le_ss.classes_)
        eval_df = eval_df[
            eval_df["product_category_name_english"].isin(cats_known) &
            eval_df["customer_state"].isin(cs_known) &
            eval_df["seller_state"].isin(ss_known)
        ]
        eval_df = eval_df.copy()
        eval_df["enc_cat"] = _le_cat.transform(eval_df["product_category_name_english"])
        eval_df["enc_cs"]  = _le_cs.transform(eval_df["customer_state"])
        eval_df["enc_ss"]  = _le_ss.transform(eval_df["seller_state"])
        X = eval_df[["payment_value","payment_installments","delivery_days",
                     "enc_cat","enc_ss","enc_cs"]].copy()
        X.columns = ["payment_value","payment_installments","delivery_days",
                     "product_category_name_english","seller_state","customer_state"]
        y = eval_df["review_score"].astype(int)
        y_pred = _model.predict(X)
        y_prob = _model.predict_proba(X)
        return y.values, y_pred, y_prob, _model.classes_

    with st.spinner("Running model evaluation on full dataset…"):
        y_true, y_pred, y_prob, classes = run_evaluation(model, le_cat, le_cs, le_ss)

    # sklearn metrics already imported at top of file

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Metric cards
    m1,m2,m3,m4 = st.columns(4)
    for col, label, val, color, icon in [
        (m1,"Accuracy",  f"{acc*100:.2f}%",  "#1d4ed8","🎯"),
        (m2,"Precision", f"{prec*100:.2f}%", "#059669","📏"),
        (m3,"Recall",    f"{rec*100:.2f}%",  "#d97706","🔍"),
        (m4,"F1 Score",  f"{f1*100:.2f}%",   "#7c3aed","⚖️"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:{color};">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-title">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-delta neutral">Macro-averaged</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    # Confusion matrix
    with col_a:
        st.markdown('<div class="section-badge">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_true, y_pred, labels=sorted(classes))
        labels = [f"{c}★" for c in sorted(classes)]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0,"#eff6ff"],[0.3,"#93c5fd"],[0.7,"#1d4ed8"],[1,"#1e3a8a"]],
            text=cm, texttemplate="<b>%{text}</b>",
            textfont=dict(color="#0f172a", size=13),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
            colorbar=dict(title=dict(text="Count",font=dict(color="#0f172a",size=13)),
                         tickfont=dict(color="#0f172a",size=12))
        ))
        fig_cm.update_layout(**base_layout(title="Confusion Matrix — Predicted vs Actual",
                                           xaxis_title="Predicted Score",
                                           yaxis_title="Actual Score"))
        st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class metrics
    with col_b:
        st.markdown('<div class="section-badge">Per-Class Metrics</div>', unsafe_allow_html=True)
        class_labels = sorted(classes)
        prec_per = precision_score(y_true, y_pred, average=None, zero_division=0, labels=class_labels)
        rec_per  = recall_score(y_true, y_pred, average=None, zero_division=0, labels=class_labels)
        f1_per   = f1_score(y_true, y_pred, average=None, zero_division=0, labels=class_labels)

        fig_cl = go.Figure()
        xlabels = [f"{c}★" for c in class_labels]
        fig_cl.add_trace(go.Bar(name="Precision", x=xlabels, y=prec_per*100,
                                marker_color="#1d4ed8", marker_line=dict(color="white",width=0.5),
                                hovertemplate="<b>%{x}</b><br>Precision: %{y:.1f}%<extra></extra>"))
        fig_cl.add_trace(go.Bar(name="Recall", x=xlabels, y=rec_per*100,
                                marker_color="#059669", marker_line=dict(color="white",width=0.5),
                                hovertemplate="<b>%{x}</b><br>Recall: %{y:.1f}%<extra></extra>"))
        fig_cl.add_trace(go.Bar(name="F1 Score", x=xlabels, y=f1_per*100,
                                marker_color="#d97706", marker_line=dict(color="white",width=0.5),
                                hovertemplate="<b>%{x}</b><br>F1: %{y:.1f}%<extra></extra>"))
        fig_cl.update_layout(**base_layout(title="Precision / Recall / F1 per Score",
                                           xaxis_title="Review Score", yaxis_title="Score (%)",
                                           barmode="group",
                                           legend=dict(orientation="h",x=0,y=-0.25)))
        st.plotly_chart(fig_cl, use_container_width=True)

    # ROC Curves (OvR)
    st.markdown('<div class="section-badge">ROC Curves — One-vs-Rest per Score Class</div>', unsafe_allow_html=True)
    try:

        y_bin = label_binarize(y_true, classes=sorted(classes))
        roc_colors = ["#dc2626","#ea580c","#ca8a04","#16a34a","#059669"]
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,
                          line=dict(color="#94a3b8",dash="dash",width=2))
        for i, cls in enumerate(sorted(classes)):
            fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"Score {cls}★ (AUC={roc_auc:.3f})",
                                         line=dict(color=roc_colors[i], width=2.5),
                                         hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"))
        fig_roc.update_layout(**base_layout(title="ROC Curves (One-vs-Rest per Review Score)",
                                            xaxis_title="False Positive Rate",
                                            yaxis_title="True Positive Rate",
                                            legend=dict(x=0.55, y=0.05)))
        st.plotly_chart(fig_roc, use_container_width=True)
    except Exception as e:
        st.warning(f"ROC curve unavailable: {e}")

    # Feature importance
    if hasattr(model,"feature_importances_"):
        st.markdown('<div class="section-badge">Feature Importance</div>', unsafe_allow_html=True)
        FEAT_LABELS = {"payment_value":"Payment Value","payment_installments":"Installments",
                       "delivery_days":"Delivery Days","product_category_name_english":"Product Category",
                       "seller_state":"Seller State","customer_state":"Customer State"}
        FEATURES = list(FEAT_LABELS.keys())
        fi = pd.DataFrame({"Feature":[FEAT_LABELS[f] for f in FEATURES],
                           "Importance":model.feature_importances_}).sort_values("Importance")
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                        title="Random Forest Feature Importance",
                        color="Importance", color_continuous_scale=SCALE_BLUE)
        fig_fi.update_traces(marker_line_color="white", marker_line_width=0.5,
                             hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>")
        fig_fi.update_layout(**hbar_layout(xaxis_title="Importance Score", yaxis_title=""))
        st.plotly_chart(fig_fi, use_container_width=True)

    # Model comparison table
    st.markdown('<div class="section-badge">Model Summary Table</div>', unsafe_allow_html=True)
    model_table = pd.DataFrame({
        "Model": ["Random Forest (current)", "Baseline (mode)", "Logistic Regression (est.)", "Decision Tree (est.)"],
        "Accuracy": [f"{acc*100:.2f}%", "~33%", "~45%", "~40%"],
        "Precision (Macro)": [f"{prec*100:.2f}%", "N/A", "~42%", "~38%"],
        "Recall (Macro)": [f"{rec*100:.2f}%", "N/A", "~40%", "~37%"],
        "F1 (Macro)": [f"{f1*100:.2f}%", "N/A", "~41%", "~37%"],
        "Status": ["✅ Production", "📊 Baseline", "📝 Estimated", "📝 Estimated"],
    })
    st.dataframe(model_table, use_container_width=True, hide_index=True)

    # Download evaluation report
    report_lines = [
        "OLIST ML MODEL EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*50,
        f"Model Type: Random Forest Classifier",
        f"Test Samples: {len(y_true):,}",
        "",
        "OVERALL METRICS",
        f"  Accuracy  : {acc*100:.2f}%",
        f"  Precision : {prec*100:.2f}% (macro)",
        f"  Recall    : {rec*100:.2f}% (macro)",
        f"  F1 Score  : {f1*100:.2f}% (macro)",
    ]
    st.download_button("📥 Download Evaluation Report (TXT)",
                       data="\n".join(report_lines).encode("utf-8"),
                       file_name="model_evaluation_report.txt", mime="text/plain")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 9 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Business Insights":
    st.title("💡 Automated Business Insights Engine")

    bc = df_full.groupby("product_category_name_english")["payment_value"].sum().idxmax().replace("_"," ").title()
    bp = df_full["payment_type"].value_counts().idxmax().replace("_"," ").title()
    ad = df_full["delivery_days"].mean()
    sp = (df_full["review_score"] >= 4).mean()*100
    ts = df_full["seller_state"].value_counts().idxmax()

    cards = [
        ("#1d4ed8","🏆","Top Revenue Category",  bc,           "Highest contributor to total gross sales."),
        ("#059669","💳","Dominant Payment Method",bp,           "Most used payment method by transaction count."),
        ("#d97706","⭐","CSAT (≥4 Stars)",        f"{sp:.1f}%", "Orders rated 4 or 5 stars by customers."),
        ("#db2777","🚚","Avg Delivery Time",      f"{ad:.1f}d", "Days from purchase to final delivery."),
        ("#7c3aed","📍","Top Seller Hub",         ts,           "State with highest seller concentration."),
    ]
    r1, r2 = st.columns(2), st.columns(3)
    for col, (color,icon,title,val,note) in zip(list(r1)+list(r2), cards):
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:20px;margin-bottom:14px;
                        box-shadow:0 4px 16px rgba(0,0,0,0.06);border-left:5px solid {color};">
              <div style="font-size:1.4rem;margin-bottom:6px;">{icon}</div>
              <div style="font-size:.75rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.06em;">{title}</div>
              <div style="font-size:1.4rem;font-weight:800;color:#1e3a8a;margin:6px 0;">{val}</div>
              <div style="font-size:.85rem;color:#64748b;">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Key Findings")
    st.info(f"""
    - **🔴 Delivery Impact**: Orders >14 days have **78% higher** probability of 1–2 star reviews.
    - **🟡 Seller Concentration**: Heavy concentration in **{ts}** creates regional delay risk.
    - **🟢 #1 CSAT Driver**: Delivery Days is the strongest predictor in the ML model — outweighs price.
    - **💳 Credit Card AOV**: Credit card installment orders average **22% higher AOV** than boleto.
    """)

    # Correlation heatmap
    st.markdown('<div class="section-badge">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    corr_cols = ["payment_value","payment_installments","delivery_days","review_score","product_weight_g"]
    corr_df = df_full[corr_cols].dropna().corr().round(2)
    col_labels = ["Payment Value","Installments","Delivery Days","Review Score","Weight (g)"]
    fig_corr = go.Figure(go.Heatmap(
        z=corr_df.values, x=col_labels, y=col_labels,
        colorscale=[[0,"#dc2626"],[0.35,"#fca5a5"],[0.5,"#f8fafc"],[0.65,"#93c5fd"],[1,"#1d4ed8"]],
        zmid=0, zmin=-1, zmax=1,
        text=corr_df.values.round(2), texttemplate="<b>%{text}</b>",
        textfont=dict(color="#0f172a",size=13),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text="Correlation",font=dict(color="#0f172a",size=13)),
                     tickfont=dict(color="#0f172a",size=12),outlinecolor="#e2e8f0",outlinewidth=1)
    ))
    fig_corr.update_layout(**base_layout(title="Feature Correlation Matrix",
                                         xaxis_title="", yaxis_title=""))
    st.plotly_chart(fig_corr, use_container_width=True)

    # Business report download
    report = f"""OLIST BUSINESS INSIGHTS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*50}
Top Revenue Category : {bc}
Dominant Payment     : {bp}
CSAT (≥4 Stars)      : {sp:.1f}%
Avg Delivery Time    : {ad:.1f} days
Top Seller Hub State : {ts}

KEY FINDINGS
- Delivery >14d → 78% higher negative review probability
- Seller concentration in {ts} creates regional risk
- Delivery Days is #1 ML predictor of review score
- Credit card installment orders: 22% higher AOV vs boleto
"""
    st.download_button("📥 Download Business Report (TXT)",
                       data=report.encode("utf-8"),
                       file_name="olist_business_report.txt", mime="text/plain")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 10 — STRATEGIC ACTION PLAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Strategic Action Plan":
    st.title("🎯 Strategic Action Plan & Business Recommendations")

    plans = [
        ("#1d4ed8","🚚","1. Optimize Delivery & SLA Fulfillment",
         "Delivery time has the **highest ML feature importance** across all predictors.",
         ["Renegotiate SLAs with couriers for non-SP/RJ regions",
          "Implement regional micro-warehousing for high-velocity categories",
          "Auto-escalate orders exceeding 10-day window"]),
        ("#059669","📣","2. Marketing on High-Yield Categories",
         "Top categories represent **>40% of gross sales**.",
         ["Shift Q3/Q4 budget to top 5 categories",
          "Run seasonal campaigns for Brazilian holiday peaks",
          "Bundle offers to increase AOV within top categories"]),
        ("#d97706","💳","3. Payment Checkout Optimizations",
         "Credit card orders drive **22% higher AOV**.",
         ["Offer 0% interest on 6+ month installment plans for R$500+ orders",
          "Integrate Pix for low-friction instant checkout",
          "A/B test checkout flows to reduce cart abandonment"]),
        ("#db2777","🤖","4. Proactive ML Intervention System",
         "Model flags late shipments as high-risk before reviews are written.",
         ["Integrate ML predictor into ERP for live order scoring",
          "Trigger automated discount vouchers for predicted score ≤2",
          "Alert logistics team when delivery-day threshold is breached"]),
    ]
    for color,icon,title,obs,actions in plans:
        st.markdown(f"""
        <div style="background:white;border-radius:14px;padding:26px;margin-bottom:18px;
                    box-shadow:0 4px 20px rgba(0,0,0,0.06);border-left:6px solid {color};">
          <div style="font-size:1.1rem;font-weight:800;color:#1e3a8a;margin-bottom:10px;">{icon} {title}</div>
          <div style="font-size:.88rem;color:#475569;margin-bottom:12px;padding:10px 14px;
                      background:#f8fafc;border-radius:8px;"><b>📌 Observation:</b> {obs}</div>
          <b style="font-size:.83rem;color:#374151;">Actions:</b>
          <ul style="margin:8px 0 0;padding-left:20px;color:#374151;font-size:.88rem;line-height:1.9;">
            {''.join(f'<li>{a}</li>' for a in actions)}
          </ul>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 11 — ABOUT PROJECT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Project":
    st.title("ℹ️ Project Information & Technology Stack")

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.subheader("Project Overview")
        st.write("""
        A production-grade Business Intelligence application built on the **Brazilian Olist
        E-Commerce Public Dataset**. Integrates a full ETL pipeline with a Random Forest
        ML model to identify key drivers of customer satisfaction.
        """)
        st.subheader("Technology Stack")
        for icon,name,desc in [
            ("🐍","Python 3.10+","Core language"),
            ("🐼","Pandas & NumPy","Data wrangling & feature engineering"),
            ("🤖","Scikit-learn","Random Forest review score predictor"),
            ("📊","Plotly Express / GO","Interactive data visualizations"),
            ("🎨","Streamlit + Custom CSS","Dashboard frontend & delivery"),
            ("☁️","Streamlit Community Cloud","Production deployment"),
        ]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;padding:9px 0;border-bottom:1px solid #f1f5f9;">
              <div style="font-size:1.3rem;">{icon}</div>
              <div>
                <div style="font-weight:700;color:#1e3a8a;">{name}</div>
                <div style="font-size:.84rem;color:#64748b;">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Author")
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:26px;box-shadow:0 4px 20px rgba(0,0,0,0.06);">
          <div style="font-size:1.8rem;margin-bottom:8px;">👤</div>
          <div style="font-size:1.15rem;font-weight:800;color:#1e3a8a;">Ayush Sharma</div>
          <div style="font-size:.88rem;color:#64748b;margin:4px 0 14px;">Data Scientist · ML Engineer · Streamlit Developer</div>
          <div style="font-size:.86rem;color:#374151;line-height:1.9;">
            <b>Program:</b> ReadyNest Internship — Week 5<br>
            <b>Project:</b> End-to-End Data Pipeline & Predictive Analytics<br>
            <b>Dataset:</b> Brazilian Olist E-Commerce (Public, Kaggle)
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;flex-direction:column;gap:10px;">
          <a href="https://github.com/Sharmaayush29/telco-churn-streamlit" target="_blank"
             style="display:flex;align-items:center;gap:10px;background:#0f172a;color:white;
                    padding:12px 18px;border-radius:8px;text-decoration:none;font-weight:600;">
            🐙 View GitHub Repository
          </a>
          <a href="https://www.linkedin.com" target="_blank"
             style="display:flex;align-items:center;gap:10px;background:#0077b5;color:white;
                    padding:12px 18px;border-radius:8px;text-decoration:none;font-weight:600;">
            💼 LinkedIn Profile
          </a>
        </div>""", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
fc1, fc2 = st.columns(2)
with fc1:
    st.caption("© 2026 ReadyNest Olist Analytics · Ayush Sharma · All rights reserved.")
with fc2:
    st.markdown(
        '<div style="text-align:right;">'
        '<a href="https://github.com/Sharmaayush29/telco-churn-streamlit" class="footer-link" target="_blank">🐙 GitHub</a>'
        ' &nbsp;|&nbsp; '
        '<a href="https://www.linkedin.com" class="footer-link" target="_blank">💼 LinkedIn</a>'
        '</div>', unsafe_allow_html=True)
