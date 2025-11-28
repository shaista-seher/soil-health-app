#streamlit_soil_health_app.py

#Soil Health & Fertilizer Recommender - Streamlit Web App

#Requirements: streamlit, pandas, numpy, scikit-learn, matplotlib

import streamlit as st 
import pandas as pd 
import numpy as np 
import base64 
import io 
import os 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator

st.set_page_config(page_title="Soil  Spark", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

#-------------------------
# Add light mint green background and styling
#-------------------------
def add_mint_background():
    css = """
    <style>
    .stApp {
        background-color: #dcfce7;
    }
    
    /* Style text inputs with green background */
    .stTextInput > div > div > input {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
        border: 2px solid #059669 !important;
        font-weight: 500;
    }
    
    /* Style text input labels */
    .stTextInput > label {
        color: #065f46 !important;
        font-weight: 600;
    }
    
    /* Style metric containers */
    .stMetric {
        background-color: #d1fae5;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #059669;
    }
    
    /* Style metric labels and values */
    .stMetric > label {
        color: #065f46 !important;
    }
    
    .stMetric > div {
        color: #065f46 !important;
    }
    
    /* Make all text darker green */
    p, span, div {
        color: #065f46;
    }
    
    /* Style info boxes */
    .stAlert {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
        border: 1px solid #059669 !important;
    }
    
    /* Style success boxes */
    .stSuccess {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_mint_background()

#---------------------------
#Load & clean columns
#---------------------------

def clean_columns(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy() 
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False) 
    return df

#---------------------------
#Load CSVs (with fallback upload)
#---------------------------

@st.cache_resource
def load_datasets():
    soil_df = None 
    fert_df = None 
    if os.path.exists("soilhealth.csv"): 
        soil_df = pd.read_csv("soilhealth.csv") 
    if os.path.exists("fertilizer.csv"): 
        fert_df = pd.read_csv("fertilizer.csv") 
    return soil_df, fert_df

soil_df, fert_df = load_datasets()

#---------------------------
#Sidebar: Assets and dataset upload
#---------------------------

with st.sidebar: 
    st.header("Soil  Spark") 
    st.write("Upload datasets if not present in project folder") 
    soil_upload = st.file_uploader("Upload soilhealth.csv", type=['csv']) 
    fert_upload = st.file_uploader("Upload fertilizer.csv", type=['csv'])

    if soil_upload is not None:
        soil_df = pd.read_csv(soil_upload)
    if fert_upload is not None:
        fert_df = pd.read_csv(fert_upload)

    st.markdown("---")
    
    # Advanced options in sidebar
    st.subheader("Advanced Options") 
    show_dataset = st.checkbox("Show dataset preview", value=False) 
    show_charts = st.checkbox("Show nutrient charts", value=True)
    
    st.caption("Developed: AI Soil Health & Fertilizer Recommender")

#---------------------------
#Validate datasets
#---------------------------

if soil_df is None or fert_df is None: 
    st.warning("Soil or fertilizer dataset not found. Upload CSVs in the sidebar or place 'soilhealth.csv' and 'fertilizer.csv' in the app folder.")

if soil_df is not None: 
    soil_df = clean_columns(soil_df) 
if fert_df is not None: 
    fert_df = clean_columns(fert_df)

if fert_df is not None and all(col in fert_df.columns for col in ['nitrogen','phosphorus','potassium']): 
    fert_df[['nitrogen','phosphorus','potassium']] = fert_df[['nitrogen','phosphorus','potassium']] * 2

#---------------------------
#Model training functions (cached)
#---------------------------

@st.cache_data(show_spinner=False) 
def train_soil_model(df: pd.DataFrame): 
    df2 = df.copy() 
    df2 = df2[['n','p','k','ph','output']].dropna() 
    df2['n_p_ratio'] = df2['n'] / (df2['p'] + 1e-6) 
    le = LabelEncoder() 
    y = le.fit_transform(df2['output']) 
    X = df2[['n','p','k','ph','n_p_ratio']] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    model = RandomForestClassifier(n_estimators=200, random_state=42) 
    model.fit(X_train, y_train) 
    acc = accuracy_score(y_test, model.predict(X_test)) 
    return model, le, acc

@st.cache_data(show_spinner=False) 
def train_fert_model(df: pd.DataFrame): 
    df2 = df.copy() 
    cols_needed = ['nitrogen','phosphorus','potassium','fertilizer_recommended'] 
    df2 = df2[[c for c in cols_needed if c in df2.columns]].dropna() 
    df2['n_p_ratio'] = df2['nitrogen'] / (df2['phosphorus'] + 1e-6) 
    le = LabelEncoder() 
    df2['fert_enc'] = le.fit_transform(df2['fertilizer_recommended']) 
    X = df2[['nitrogen','phosphorus','potassium','n_p_ratio']] 
    y = df2['fert_enc'] 
    scaler = StandardScaler() 
    Xs = scaler.fit_transform(X) 
    model = RandomForestClassifier(n_estimators=200, random_state=42) 
    model.fit(Xs, y) 
    cv = cross_val_score(model, Xs, y, cv=5).mean() 
    return model, le, scaler, cv

soil_model = None 
fert_model = None 
le_soil = None 
le_fert = None 
scaler = None 
soil_acc = None 
fert_acc = None

if soil_df is not None: 
    try: 
        soil_model, le_soil, soil_acc = train_soil_model(soil_df) 
    except Exception as e: 
        st.error(f"Failed to train soil model: {e}")

if fert_df is not None: 
    try: 
        fert_model, le_fert, scaler, fert_acc = train_fert_model(fert_df) 
    except Exception as e: 
        st.error(f"Failed to train fertilizer model: {e}")

#---------------------------
#Utility functions
#---------------------------

def ph_category_and_text(pH): 
    if pH < 5.5: 
        return "Highly acidic", "Soil is highly acidic — mix agricultural lime." 
    if 5.5 <= pH < 6.5: 
        return "Slightly acidic", "Soil slightly acidic — add agricultural lime." 
    if 6.5 <= pH <= 7.5: 
        return "Neutral", "Soil is neutral — maintain with compost." 
    if 7.5 < pH <= 8.5: 
        return "Slightly alkaline", "Soil slightly alkaline — apply gypsum." 
    return "Highly alkaline", "Soil highly alkalic — add gypsum + compost."

def nutrient_level(val, nut): 
    if nut == 'N': 
        return "Low" if val < 200 else "Medium" if val <= 400 else "High" 
    if nut == 'P': 
        return "Low" if val < 15 else "Medium" if val <= 35 else "High" 
    if nut == 'K': 
        return "Low" if val < 110 else "Medium" if val <= 280 else "High"

def nutrient_warnings(N, P, K): 
    msgs = [] 
    mapping_name = {"N":"Nitrogen","P":"Phosphorus","K":"Potassium"} 
    mapping_rec = {"N":"Urea","P":"DAP","K":"MOP"} 
    for val, nut in zip([N, P, K], ['N','P','K']): 
        level = nutrient_level(val, nut) 
        if level == "Low": 
            msgs.append(f"{mapping_name[nut]} ({level}): Add {mapping_rec[nut]}") 
        elif level == "Medium":
            msgs.append(f"{mapping_name[nut]} ({level}): Balanced") 
        else: 
            high_msg = { 
                "N": "Avoid extra urea; too much reduces flowering.", 
                "P": "Avoid extra P; excess affects micronutrient uptake.", 
                "K": "Avoid extra potash; excess reduces Mg/Ca uptake." 
            }[nut] 
            msgs.append(f"{mapping_name[nut]} ({level}): {high_msg}") 
    return msgs

def icar_tip(primary, soil_health):
    if soil_health == 'Low':
         return f"Apply {primary} with compost/FYM in 2–3 splits as per ICAR guidelines." 
    elif soil_health == 'Moderate':
         return f"Apply {primary} in 2 splits and include compost." 
    else: 
         return f"Apply {primary} once and maintain crop rotation."

#---------------------------
#Prediction logic
#---------------------------

def predict_soil_health(N, P, K, pH):
    if soil_model is None:
        return "Unknown", "Model not loaded"
    
    try:
        n_p_ratio = N / (P + 1e-6)
        pred_num = soil_model.predict([[N, P, K, pH, n_p_ratio]])[0]
        mapping = {0: "Low", 1: "Moderate", 2: "Healthy"}
        pred = mapping.get(pred_num, "Unknown")

        if pred == "Healthy":
            reason = "Your soil has good nutrient balance and suitable pH levels."
        elif pred == "Moderate":
            if pH < 5.5 or pH > 8.5:
                reason = "Extreme pH level affecting soil health."
            elif N < 200:
                reason = "Slight nitrogen deficiency detected."
            elif P < 15:
                reason = "Slight phosphorus deficiency detected."
            elif K < 120:
                reason = "Slight potassium deficiency detected."
            else:
                reason = "Your soil shows slight nutrient imbalance. Consider mild correction."
        else:
            if pH < 5.5 or pH > 8.5:
                reason = "Extreme pH level - immediate correction needed."
            elif N < 200:
                reason = "Severe nitrogen deficiency detected."
            elif P < 15:
                reason = "Severe phosphorus deficiency detected."
            elif K < 120:
                reason = "Severe potassium deficiency detected."
            else:
                reason = "Your soil nutrients are imbalanced; improvement is needed."

        return pred, reason

    except Exception as e:
        return "Unknown", f"Prediction error: {e}"

def recommend_fertilizer(N, P, K, soil_health): 
    if fert_model is None or scaler is None: 
        if N < 200: 
            primary = 'Urea' 
        elif P < 15: 
            primary = 'DAP' 
        elif K < 110: 
            primary = 'MOP' 
        else: 
            primary = 'Balanced NPK' 
        return primary, 0.65 
    user_df = pd.DataFrame([[N, P, K, N/(P+1e-6)]], columns=['nitrogen','phosphorus','potassium','n_p_ratio']) 
    user_scaled = scaler.transform(user_df) 
    fert_enc = fert_model.predict(user_scaled)[0] 
    prob = fert_model.predict_proba(user_scaled).max() 
    primary = le_fert.inverse_transform([fert_enc])[0] 
    if soil_health == 'Low' and 'organic' not in primary.lower(): 
        primary = primary + ' + Organic matter' 
    return primary, prob

#---------------------------
# Initialize session state
#---------------------------

if 'page' not in st.session_state:
    st.session_state.page = 'start'
if 'N' not in st.session_state:
    st.session_state.N = None
if 'P' not in st.session_state:
    st.session_state.P = None
if 'K' not in st.session_state:
    st.session_state.K = None
if 'pH' not in st.session_state:
    st.session_state.pH = None
if 'show_details' not in st.session_state:
    st.session_state.show_details = False

#---------------------------
# START PAGE
#---------------------------

if st.session_state.page == 'start':
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Center content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <h1 style='text-align: center; color: #059669; font-size: 72px; font-family: Georgia, serif; margin-bottom: 20px;'>
            SOILS PARK
        </h1>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <p style='text-align: center; color: #10b981; font-size: 20px; margin-bottom: 50px;'>
            🌱 AI-Powered Soil Health & Fertilizer Guidance 🌱
        </p>
        """, unsafe_allow_html=True)
        
        # Start button
        if st.button("🚀 Start Journey", use_container_width=True, type="primary"):
            st.session_state.page = 'input'
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align: center; color: #6b7280; font-size: 14px;'>
            Smart recommendations powered by machine learning
        </p>
        """, unsafe_allow_html=True)

#---------------------------
# INPUT PAGE
#---------------------------

elif st.session_state.page == 'input':
    st.markdown("<h1 style='text-align:center; color:#059669;'>🌾 SOILS PARK</h1>", unsafe_allow_html=True) 
    st.markdown("<p style='text-align:center; font-size:16px; color:#10b981;'>Smart farm recommendations powered by ML — enter your soil test values to get instant guidance.</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the input form
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown("### 📝 Enter Soil Test Values")
        
        with st.form("soil_input_form"):
            N_input = st.text_input("Nitrogen (N) - kg/ha", placeholder="e.g., 200", help="Range: 0-600 kg/ha")
            P_input = st.text_input("Phosphorus (P) - kg/ha", placeholder="e.g., 30", help="Range: 0-120 kg/ha")
            K_input = st.text_input("Potassium (K) - kg/ha", placeholder="e.g., 150", help="Range: 0-800 kg/ha")
            pH_input = st.text_input("pH Value", placeholder="e.g., 6.5", help="Range: 3.5-10.0")
            
            submit_button = st.form_submit_button("🔍 Analyze Soil", use_container_width=True)
            
            if submit_button:
                # Validate inputs
                try:
                    N_val = float(N_input)
                    P_val = float(P_input)
                    K_val = float(K_input)
                    pH_val = float(pH_input)
                    
                    # Range validation
                    if not (0 <= N_val <= 600):
                        st.error("❌ Nitrogen must be between 0 and 600 kg/ha")
                    elif not (0 <= P_val <= 120):
                        st.error("❌ Phosphorus must be between 0 and 120 kg/ha")
                    elif not (0 <= K_val <= 800):
                        st.error("❌ Potassium must be between 0 and 800 kg/ha")
                    elif not (3.5 <= pH_val <= 10.0):
                        st.error("❌ pH must be between 3.5 and 10.0")
                    else:
                        # Store in session state and navigate
                        st.session_state.N = N_val
                        st.session_state.P = P_val
                        st.session_state.K = K_val
                        st.session_state.pH = pH_val
                        st.session_state.page = 'output'
                        st.session_state.show_details = False
                        st.rerun()
                        
                except ValueError:
                    st.error("❌ Please enter valid numeric values for all fields")
    
    # Show dataset preview if enabled
    if show_dataset:
        st.markdown("---")
        st.markdown("### 📊 Dataset Preview") 
        col1, col2 = st.columns(2)
        with col1:
            if soil_df is not None: 
                st.write("**Soil Dataset (first 5 rows)**") 
                st.dataframe(soil_df.head(), use_container_width=True) 
        with col2:
            if fert_df is not None: 
                st.write("**Fertilizer Dataset (first 5 rows)**") 
                st.dataframe(fert_df.head(), use_container_width=True)
    
    # Show model accuracy
    if soil_acc is not None or fert_acc is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        if soil_acc is not None:
            col1.info(f"🎯 Soil health model accuracy: {soil_acc*100:.2f}%") 
        if fert_acc is not None:
            col2.info(f"🎯 Fertilizer model accuracy: {fert_acc*100:.2f}%")

#---------------------------
# OUTPUT PAGE
#---------------------------

elif st.session_state.page == 'output':
    # Get values from session state
    N = st.session_state.N
    P = st.session_state.P
    K = st.session_state.K
    pH = st.session_state.pH
    
    # Run predictions
    soil_health, reason = predict_soil_health(N, P, K, pH)
    primary, conf = recommend_fertilizer(N, P, K, soil_health)
    ph_cat, ph_text = ph_category_and_text(pH)
    
    # Header
    st.markdown("<h1 style='text-align:center; color:#059669;'>📊 SOILS PARK - Analysis Results</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display input values
    st.markdown("<h3 style='color:#065f46;'>📥 Input Values</h3>", unsafe_allow_html=True)
    input_col1, input_col2, input_col3, input_col4 = st.columns(4)
    input_col1.metric("Nitrogen (N)", f"{N} kg/ha")
    input_col2.metric("Phosphorus (P)", f"{P} kg/ha")
    input_col3.metric("Potassium (K)", f"{K} kg/ha")
    input_col4.metric("pH", f"{pH}")
    
    st.markdown("---")
    
    # Main results
    st.markdown("<h3 style='color:#065f46;'>🎯 Primary Results</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    # Color coding for soil health
    health_color = {"Healthy": "🟢", "Moderate": "🟡", "Low": "🔴", "Unknown": "⚪"}
    c1.markdown(f"<h4 style='color:#065f46;'>{health_color.get(soil_health, '⚪')} Soil Health</h4>", unsafe_allow_html=True)
    c1.markdown(f"<h3 style='color:#065f46;'><strong>{soil_health}</strong></h3>", unsafe_allow_html=True)
    c1.markdown(f"<p style='color:#065f46;'><em>{reason}</em></p>", unsafe_allow_html=True)
    
    c2.markdown("<h4 style='color:#065f46;'>💊 Recommended Fertilizer</h4>", unsafe_allow_html=True)
    c2.markdown(f"<h3 style='color:#065f46;'><strong>{primary}</strong></h3>", unsafe_allow_html=True)
    c2.markdown(f"<p style='color:#065f46;'><em>Confidence: {conf*100:.2f}%</em></p>", unsafe_allow_html=True)
    
    c3.markdown("<h4 style='color:#065f46;'>🧪 pH Category</h4>", unsafe_allow_html=True)
    c3.markdown(f"<h3 style='color:#065f46;'><strong>{ph_cat}</strong></h3>", unsafe_allow_html=True)
    c3.markdown(f"<p style='color:#065f46;'><em>{ph_text}</em></p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Get Recommendations button (expandable)
    if st.button("📋 Get Detailed Recommendations", use_container_width=True, type="primary"):
        st.session_state.show_details = not st.session_state.show_details
    
    # Show details if expanded
    if st.session_state.show_details:
        st.markdown("---")
        
        # ICAR Tips
        st.markdown("<h3 style='color:#065f46;'>🌱 ICAR Action Plan</h3>", unsafe_allow_html=True)
        st.info(icar_tip(primary, soil_health))
        st.markdown(f"<p style='color:#065f46;'><strong>pH Management:</strong> {ph_text}</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Nutrient Warnings
        st.markdown("<h3 style='color:#065f46;'>⚠️ Nutrient Analysis & Quick Actions</h3>", unsafe_allow_html=True)
        warnings = nutrient_warnings(N, P, K)
        for msg in warnings:
            st.markdown(f"<p style='color:#065f46;'>• {msg}</p>", unsafe_allow_html=True)
        
        # Charts
        if show_charts:
            st.markdown("---")
            st.markdown("<h3 style='color:#065f46;'>📈 Visual Analysis</h3>", unsafe_allow_html=True)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("<p style='color:#065f46;'><strong>Nutrient Distribution</strong></p>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6,4))
                nutrients = ['Nitrogen', 'Phosphorus', 'Potassium']
                vals = [N, P, K]
                colors = ['#2ecc71', '#3498db', '#e74c3c']
                bars = ax.bar(nutrients, vals, color=colors, alpha=0.7)
                ax.set_ylabel('kg/ha', fontsize=12)
                ax.set_title('NPK Levels', fontsize=14, weight='bold')
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                
                # Add value labels on bars
                for bar, val in zip(bars, vals):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1f}',
                           ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with chart_col2:
                st.markdown("<p style='color:#065f46;'><strong>pH Status</strong></p>", unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(6,4))
                
                # pH scale visualization
                ph_range = np.linspace(3.5, 10, 100)
                colors_scale = plt.cm.RdYlGn(np.linspace(0, 1, len(ph_range)))
                
                for i in range(len(ph_range)-1):
                    ax2.barh(0, 0.065, left=ph_range[i], height=0.5, 
                            color=colors_scale[i], edgecolor='none')
                
                # Mark current pH
                ax2.plot([pH, pH], [-0.3, 0.3], 'k-', linewidth=3, marker='v', 
                        markersize=12, label=f'Your pH: {pH}')
                
                ax2.set_xlim(3.5, 10)
                ax2.set_ylim(-0.5, 0.5)
                ax2.set_xlabel('pH Value', fontsize=12)
                ax2.set_title(f'pH Level: {ph_cat}', fontsize=14, weight='bold')
                ax2.set_yticks([])
                ax2.legend(loc='upper right')
                ax2.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig2)
        
        st.markdown("---")
        st.success("✅ Detailed recommendations generated. Use these results as guidance and cross-check with local agronomists for field-scale implementation.")
    
    # Navigation buttons
    st.markdown("<br>", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    
    with nav_col2:
        if st.button("🔄 Analyze New Sample", use_container_width=True):
            st.session_state.page = 'input'
            st.session_state.show_details = False
            st.rerun()

#---------------------------
# Footer
#---------------------------

st.markdown("---") 
st.caption("Built for educational & prototyping purposes. Always validate recommendations with local soil labs and agronomists.")
