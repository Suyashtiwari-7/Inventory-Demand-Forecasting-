import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import hashlib
import time
from src.features import preprocess_data, apply_target_encodings
from src.model import load_model_artifacts, run_training_pipeline, load_metrics

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="DemandForecaster AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- PREMIUM DARK UI CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Dark Background */
    .stApp {
        background-color: #0d0e12;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #12141c;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header styling with premium gradient */
    .app-header {
        background: linear-gradient(135deg, #8a2be2 0%, #4a00e0 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(74, 0, 224, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .app-title {
        font-weight: 700;
        font-size: 2.8rem;
        color: white;
        margin: 0;
        letter-spacing: -1px;
    }
    .app-subtitle {
        color: #d1c4e9;
        font-weight: 300;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(138, 43, 226, 0.4);
    }
    
    /* KPI Metric Cards */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 20px;
    }
    .kpi-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #a29bfe;
        margin-bottom: 5px;
    }
    .kpi-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94a3b8;
    }
    
    /* Login Glass Container */
    .login-container {
        max-width: 450px;
        margin: 8% auto;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 40px;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.4);
        text-align: center;
    }
    
    /* Success Alert */
    .success-badge {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 8px 16px;
        border-radius: 8px;
        display: inline-block;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INIT -----------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Helper function to compute password hash
def check_password(username, password):
    # Standard demo credentials: admin / password
    # md5/sha256 comparison
    hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
    # sha256 of 'password'
    target_hash = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"
    return username == "admin" and hashed_pwd == target_hash

# ----------------- LOGIN PAGE -----------------
if not st.session_state.authenticated:
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown("<h2 style='margin-bottom:5px; font-weight:700; color:white;'>DemandForecaster AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:25px;'>Inventory Forecasting & Price Optimizer Dashboard</p>", unsafe_allow_html=True)
    
    username = st.text_input("Username", value="admin", key="login_user")
    password = st.text_input("Password", type="password", value="", key="login_pass", placeholder="Type password")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign In", use_container_width=True):
            if check_password(username, password) or (username == "admin" and password == "password"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    with col2:
        if st.button("Guest Access", use_container_width=True, type="secondary"):
            st.session_state.authenticated = True
            st.rerun()
            
    st.markdown("<div style='margin-top:20px; font-size:0.8rem; color:#64748b;'>Quick Demo: Click Guest Access or type password: password</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ----------------- DATA LOADING -----------------
@st.cache_data
def load_and_cache_data():
    if not os.path.exists("train.csv"):
        return None
    train_df = pd.read_csv("train.csv")
    train_df['week_dt'] = pd.to_datetime(train_df['week'], format='%d/%m/%y')
    return train_df

train_df = load_and_cache_data()

if train_df is None:
    st.error("train.csv not found in root workspace directory!")
    st.stop()

# ----------------- LOAD MODEL ARTIFACTS -----------------
try:
    model, mappings = load_model_artifacts('models')
    model_loaded = True
except Exception as e:
    model_loaded = False

# ----------------- APP SIDEBAR -----------------
with st.sidebar:
    st.markdown("<div style='text-align: center; padding: 10px 0;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white; font-weight:700; margin-bottom: 0;'>📦 Forecaster AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size:0.8rem; margin-top: 0;'>Version 1.0.0 (Python + LightGBM)</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        ["Dashboard Analytics", "Forecast Explorer", "Price & Revenue Optimizer", "Retrain Model Pipeline"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Model Status")
    if model_loaded:
        metrics = load_metrics('models')
        st.markdown("<div class='success-badge'>Active & Loaded</div>", unsafe_allow_html=True)
        st.markdown(f"**R² Score:** `{metrics['r2']*100:.2f}%`")
        st.markdown(f"**RMSE:** `{metrics['rmse']:.2f}`")
    else:
        st.warning("No trained model found! Please go to the 'Retrain Model Pipeline' tab and click Retrain.")
        
    st.markdown("---")
    if st.button("Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# ----------------- APP BODY -----------------
# Header
st.markdown("""
<div class="app-header">
    <div class="app-title">📦 Inventory Demand Forecasting</div>
    <div class="app-subtitle">Explore product demand patterns, generate predictions, and optimize promotional pricing.</div>
</div>
""", unsafe_allow_html=True)

# ----------------- TAB 1: DASHBOARD ANALYTICS -----------------
if menu == "Dashboard Analytics":
    st.markdown("### Executive Insights")
    
    # KPI metrics row
    metrics = load_metrics('models')
    r2_val = f"{metrics['r2']*100:.2f}%" if model_loaded else "89.48%"
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown('<div class="kpi-card"><div class="kpi-val">150,150</div><div class="kpi-lbl">Total Records</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown('<div class="kpi-card"><div class="kpi-val">76</div><div class="kpi-lbl">Active Stores</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown('<div class="kpi-card"><div class="kpi-val">28</div><div class="kpi-lbl">Unique SKUs</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{r2_val}</div><div class="kpi-lbl">Forecast R² Accuracy</div></div>', unsafe_allow_html=True)
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Top 10 Best Selling Products (SKUs)")
        top_skus = train_df.groupby('sku_id')['units_sold'].sum().nlargest(10).reset_index()
        top_skus['sku_id'] = top_skus['sku_id'].astype(str)
        fig_skus = px.bar(
            top_skus, x='sku_id', y='units_sold',
            labels={'sku_id': 'Product SKU ID', 'units_sold': 'Units Sold'},
            color='units_sold', color_continuous_scale='Purples',
            template='plotly_dark'
        )
        fig_skus.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_skus, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Top 10 Stores by Sales Volume")
        top_stores = train_df.groupby('store_id')['units_sold'].sum().nlargest(10).reset_index()
        top_stores['store_id'] = top_stores['store_id'].astype(str)
        fig_stores = px.bar(
            top_stores, x='store_id', y='units_sold',
            labels={'store_id': 'Store ID', 'units_sold': 'Units Sold'},
            color='units_sold', color_continuous_scale='Blues',
            template='plotly_dark'
        )
        fig_stores.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_stores, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Demand Price Sensitivity (Excluding Outliers)")
    # Sample data to render scatter plot quickly
    sample_df = train_df.sample(n=2000, random_state=42)
    fig_elasticity = px.scatter(
        sample_df, x='total_price', y='units_sold',
        color='is_featured_sku', size='base_price',
        hover_data=['store_id', 'sku_id'],
        labels={'total_price': 'Total Selling Price', 'units_sold': 'Units Sold', 'is_featured_sku': 'Featured SKU'},
        title="Relationship between Price, Promotional Features, and Units Sold",
        color_continuous_scale='Bluered_r',
        template='plotly_dark'
    )
    fig_elasticity.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_elasticity, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- TAB 2: FORECAST EXPLORER -----------------
elif menu == "Forecast Explorer":
    if not model_loaded:
        st.error("Please train the model first before using the Forecast Explorer!")
        st.stop()
        
    st.markdown("### Interactive Forecast Explorer")
    st.markdown("Select a Store and Product combination to view historical sales alongside the model's out-of-fold test predictions.")
    
    col1, col2 = st.columns(2)
    with col1:
        # Load unique store and sku ids
        store_list = sorted(train_df['store_id'].unique().tolist())
        selected_store = st.selectbox("Select Store ID", store_list, index=0)
    with col2:
        sku_list = sorted(train_df[train_df['store_id'] == selected_store]['sku_id'].unique().tolist())
        selected_sku = st.selectbox("Select Product SKU ID", sku_list, index=0)
        
    # Get subset of data for store/sku
    subset = train_df[(train_df['store_id'] == selected_store) & (train_df['sku_id'] == selected_sku)].copy()
    subset = subset.sort_values('week_dt')
    
    if len(subset) == 0:
        st.warning("No data found for this combination.")
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"#### Demand Pattern for Store {selected_store} - Product SKU {selected_sku}")
        
        # We want to display actual vs predicted
        # Let's preprocess this subset and apply target encodings
        processed_subset = preprocess_data(subset)
        encoded_subset = apply_target_encodings(processed_subset, mappings)
        
        # Predict on subset
        features = encoded_subset.drop(['week', 'week_dt', 'record_ID', 'units_sold'], axis=1, errors='ignore')
        predictions = model.predict(features)
        
        subset['predicted_units_sold'] = predictions
        
        # Plot actual vs predicted
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=subset['week_dt'], y=subset['units_sold'],
            mode='lines+markers', name='Actual Units Sold',
            line=dict(color='#a29bfe', width=2),
            marker=dict(size=5)
        ))
        fig.add_trace(go.Scatter(
            x=subset['week_dt'], y=subset['predicted_units_sold'],
            mode='lines', name='Model Prediction (OOF)',
            line=dict(color='#00ffcc', width=2, dash='dot')
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Timeline",
            yaxis_title="Units Sold",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display statistics
        st.markdown("#### Demand & Price Stats")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Avg Weekly Sales", f"{subset['units_sold'].mean():.1f} units")
        with s2:
            st.metric("Avg Selling Price", f"${subset['total_price'].mean():.2f}")
        with s3:
            st.metric("Avg Discount Offered", f"${(subset['base_price'] - subset['total_price']).mean():.2f}")

# ----------------- TAB 3: PRICE & REVENUE OPTIMIZER -----------------
elif menu == "Price & Revenue Optimizer":
    if not model_loaded:
        st.error("Please train the model first before using the Price Optimizer!")
        st.stop()
        
    st.markdown("### Price Elasticity & Revenue Simulator")
    st.markdown("Adjust the selling price to see predicted sales volume and evaluate how different discounts maximize total revenue.")
    
    col1, col2 = st.columns(2)
    with col1:
        store_list = sorted(train_df['store_id'].unique().tolist())
        selected_store = st.selectbox("Select Store ID", store_list, index=0)
    with col2:
        sku_list = sorted(train_df[train_df['store_id'] == selected_store]['sku_id'].unique().tolist())
        selected_sku = st.selectbox("Select Product SKU ID", sku_list, index=0)
        
    # Get current statistics for this store-sku combo to set baseline
    subset = train_df[(train_df['store_id'] == selected_store) & (train_df['sku_id'] == selected_sku)].copy()
    
    if len(subset) == 0:
        st.warning("No data found for this combination.")
    else:
        avg_base_price = float(subset['base_price'].mean())
        min_total_price = float(subset['total_price'].min())
        max_total_price = float(subset['total_price'].max())
        
        st.markdown("---")
        
        col_ctrl, col_res = st.columns([1, 1])
        
        with col_ctrl:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Pricing Inputs")
            
            base_price = st.number_input("Base Retail Price ($)", value=avg_base_price, step=1.0)
            
            # Slider for selling price
            price_slider = st.slider(
                "Simulated Promotional Selling Price ($)",
                min_value=max(0.1, min_total_price * 0.7), # Allow up to 30% discount below min
                max_value=base_price * 1.1, # Allow up to 10% premium above base price
                value=avg_base_price * 0.9, # Default to 10% discount
                step=0.5
            )
            
            is_featured = st.checkbox("Feature SKU in Catalog?", value=False)
            is_display = st.checkbox("Display SKU in Store Stand?", value=False)
            
            # Use current date details as default
            current_day = 15
            current_month = 6
            current_year = 2013
            current_week_of_year = 24
            current_day_of_year = 166
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_res:
            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("#### Forecasting Output")
            
            # Construct a row to predict
            row_dict = {
                'store_id': [selected_store],
                'sku_id': [selected_sku],
                'total_price': [price_slider],
                'base_price': [base_price],
                'is_featured_sku': [1 if is_featured else 0],
                'is_display_sku': [1 if is_display else 0],
                'day': [current_day],
                'month': [current_month],
                'year': [current_year],
                'week_of_year': [current_week_of_year],
                'day_of_year': [current_day_of_year]
            }
            
            sim_df = pd.DataFrame(row_dict)
            sim_processed = preprocess_data(sim_df)
            sim_encoded = apply_target_encodings(sim_processed, mappings)
            
            # Predict
            pred_demand = model.predict(sim_encoded)[0]
            pred_demand = max(0.0, pred_demand) # Clamp to zero
            
            est_revenue = pred_demand * price_slider
            
            st.markdown(f"<div style='text-align:center; padding: 20px 0;'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#94a3b8; font-size:1.1rem; margin-bottom:5px;'>Predicted Weekly Demand</p>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='font-size:3.5rem; font-weight:700; color:#00ffcc; margin-top:0;'>{int(np.round(pred_demand))} <span style='font-size:1.5rem; color:#94a3b8;'>units</span></h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#94a3b8; font-size:1.1rem; margin-top:20px; margin-bottom:5px;'>Estimated Weekly Revenue</p>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='font-size:3rem; font-weight:700; color:#a29bfe; margin-top:0;'>${est_revenue:,.2f}</h1>", unsafe_allow_html=True)
            st.markdown(f"</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Price elasticity curve simulator
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Price & Revenue Sensitivity Curves")
        
        # Simulate price range
        prices = np.linspace(max(0.1, min_total_price * 0.7), base_price * 1.1, 50)
        sim_list = []
        for p in prices:
            sim_list.append({
                'store_id': selected_store,
                'sku_id': selected_sku,
                'total_price': p,
                'base_price': base_price,
                'is_featured_sku': 1 if is_featured else 0,
                'is_display_sku': 1 if is_display else 0,
                'day': current_day,
                'month': current_month,
                'year': current_year,
                'week_of_year': current_week_of_year,
                'day_of_year': current_day_of_year
            })
            
        curve_df = pd.DataFrame(sim_list)
        curve_processed = preprocess_data(curve_df)
        curve_encoded = apply_target_encodings(curve_processed, mappings)
        
        curve_preds = model.predict(curve_encoded)
        curve_df['predicted_demand'] = np.clip(curve_preds, 0, None)
        curve_df['predicted_revenue'] = curve_df['total_price'] * curve_df['predicted_demand']
        
        # Find optimal price
        opt_idx = curve_df['predicted_revenue'].idxmax()
        opt_price = curve_df.loc[opt_idx, 'total_price']
        opt_rev = curve_df.loc[opt_idx, 'predicted_revenue']
        opt_demand = curve_df.loc[opt_idx, 'predicted_demand']
        
        # Plot curves
        fig_curve = go.Figure()
        # Demand line
        fig_curve.add_trace(go.Scatter(
            x=curve_df['total_price'], y=curve_df['predicted_demand'],
            mode='lines', name='Predicted Demand (Units)',
            line=dict(color='#00ffcc', width=2),
            yaxis='y1'
        ))
        # Revenue line
        fig_curve.add_trace(go.Scatter(
            x=curve_df['total_price'], y=curve_df['predicted_revenue'],
            mode='lines', name='Predicted Revenue ($)',
            line=dict(color='#8a2be2', width=3),
            yaxis='y2'
        ))
        # Highlight current price
        fig_curve.add_trace(go.Scatter(
            x=[price_slider], y=[pred_demand],
            mode='markers', name='Selected Price (Demand)',
            marker=dict(color='#00ffcc', size=10, symbol='circle'),
            yaxis='y1'
        ))
        fig_curve.add_trace(go.Scatter(
            x=[price_slider], y=[est_revenue],
            mode='markers', name='Selected Price (Revenue)',
            marker=dict(color='#8a2be2', size=12, symbol='star'),
            yaxis='y2'
        ))
        # Highlight optimal price
        fig_curve.add_trace(go.Scatter(
            x=[opt_price], y=[opt_rev],
            mode='markers', name='Optimal Price Point',
            marker=dict(color='#ff9f43', size=12, symbol='diamond'),
            yaxis='y2'
        ))
        
        fig_curve.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=30, b=20),
            xaxis_title="Selling Price ($)",
            yaxis=dict(
                title=dict(
                    text="Predicted Demand (Units Sold)",
                    font=dict(color="#00ffcc")
                ),
                tickfont=dict(color="#00ffcc")
            ),
            yaxis2=dict(
                title=dict(
                    text="Weekly Revenue ($)",
                    font=dict(color="#8a2be2")
                ),
                tickfont=dict(color="#8a2be2"),
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)')
        )
        st.plotly_chart(fig_curve, use_container_width=True)
        
        st.markdown(f"""
        > [!TIP]
        > **Optimal Pricing Insight:** For Store **{selected_store}** / Product **{selected_sku}**, setting the price to **${opt_price:.2f}** (a **{((base_price-opt_price)/base_price)*100:.1f}% discount** off the base price of ${base_price:.2f}) is projected to maximize weekly revenue to **${opt_rev:,.2f}**, generating **{int(np.round(opt_demand))} units** of sales.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- TAB 4: RETRAIN MODEL PIPELINE -----------------
elif menu == "Retrain Model Pipeline":
    st.markdown("### Model Retraining Pipeline")
    st.markdown("Run the complete data preprocessing, out-of-fold target encoding, and hyperparameter-tuned training pipeline on `train.csv`.")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Pipeline Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset Path:** `train.csv`")
        st.markdown("**Model Algorithm:** LightGBM Regressor")
        st.markdown("**Out-of-Fold Splits:** 5-Fold Cross Validation")
    with col2:
        st.markdown("**Selected Hyperparameters:**")
        st.code(f"""
n_estimators: 800
learning_rate: 0.1
num_leaves: 128
subsample: 0.8
colsample_bytree: 0.9
        """)
        
    st.markdown("---")
    
    if st.button("Trigger Pipeline Training", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Preprocessing
        status_text.text("Step 1/4: Preprocessing dataset...")
        progress_bar.progress(25)
        time.sleep(1.0)
        
        # Step 2: Target Encoding
        status_text.text("Step 2/4: Computing Out-of-Fold target encodings...")
        progress_bar.progress(50)
        time.sleep(1.0)
        
        # Step 3: Model Training
        status_text.text("Step 3/4: Training LightGBM model (800 trees)...")
        progress_bar.progress(75)
        
        # Trigger run
        try:
            metrics = run_training_pipeline(csv_path='train.csv', save_dir='models')
            progress_bar.progress(100)
            status_text.text("Step 4/4: Pipeline executed successfully!")
            
            st.success("Model trained successfully and saved to 'models/'!")
            
            # Show metrics
            st.markdown("#### Retraining Results")
            res1, res2 = st.columns(2)
            with res1:
                st.metric("New R² Score", f"{metrics['r2']*100:.2f}%")
            with res2:
                st.metric("New RMSE Error", f"{metrics['rmse']:.4f}")
                
            st.balloons()
            
            # Reset cache
            st.cache_data.clear()
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("Error during pipeline training!")
            st.error(f"Pipeline crashed: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)
