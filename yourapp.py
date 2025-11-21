#streamlit_soil_health_app.py

#Soil Health & Fertilizer Recommender - Streamlit Web App (GIF background)

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

st.set_page_config(page_title="AI in Soil Health & Fertilizer Recommendation", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

#-------------------------

#Helper: add GIF background via base64 encoded CSS

#Place a GIF named 'bg.gif' inside an 'assets' folder next to this file.

#-------------------------

def add_gif_background(gif_path: str, opacity: float = 0.18): 
    if not os.path.exists(gif_path): 
        return 
    with open(gif_path, "rb") as f: 
        data = f.read() 
    b64 = base64.b64encode(data).decode() 
    css = f""" 
    <style> 
    .stApp {{ 
        background-image: url('data:image/gif;base64,{b64}'); 
        background-size: cover; 
        background-attachment: fixed; 
        background-repeat: no-repeat; 
        opacity: 1; 
    }} 
    /* Add a translucent layer for contrast */ 
    .app-overlay {{ 
        position: fixed; 
        top: 0; left: 0; right: 0; bottom: 0; 
        background: rgba(255,255,255,{opacity}); 
        pointer-events: none; 
        z-index: 0; 
    }} 
    </style> 
    <div class="app-overlay"></div> 
    """ 
    st.markdown(css, unsafe_allow_html=True)

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

@st.experimental_singleton 
def load_datasets(): 
    soil_df = None 
    fert_df = None 
    # try local files first 
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
    st.header("AI in Soil Health") 
    st.write("Upload datasets if not present in project folder") 
    soil_upload = st.file_uploader("Upload soilhealth.csv", type=['csv']) 
    fert_upload = st.file_uploader("Upload fertilizer.csv", type=['csv'])

    if soil_upload is not None:
        soil_df = pd.read_csv(soil_upload)
    if fert_upload is not None:
        fert_df = pd.read_csv(fert_upload)

    st.markdown("---")
    st.markdown("*Assets*")
    st.write("Place your GIF at: assets/bg.gif (recommended). If present, the app will use it as the animated background.")
    st.markdown("---")
    st.caption("Developed: AI Soil Health & Fertilizer Recommender")

#Try to add gif background if exists

add_gif_background(os.path.join('assets', 'bg.gif'))

#---------------------------

#Validate datasets

#---------------------------

if soil_df is None or fert_df is None: 
    st.warning("Soil or fertilizer dataset not found. Upload CSVs in the sidebar or place 'soilhealth.csv' and 'fertilizer.csv' in the app folder.")

if soil_df is not None: 
    soil_df = clean_columns(soil_df) 
if fert_df is not None: 
    fert_df = clean_columns(fert_df)

#convert fert ppm -> kg/ha if salts (simple heuristic)

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

#UI Layout

#---------------------------

st.markdown("<h1 style='text-align:center; color:#2b6a2f;'>🌾 AI in Soil Health Prediction & Fertilizer Recommendation</h1>", unsafe_allow_html=True) 
st.markdown("<p style='text-align:center; font-size:14px; color:#4b7a3b;'>Smart farm recommendations powered by ML — enter your soil test values to get instant guidance.</p>", unsafe_allow_html=True)

#two-column layout for input & outputs

col1, col2 = st.columns([1,1.1])

with col1: 
    st.subheader("Input Soil Test Values") 
    N = st.slider("Nitrogen (N) - kg/ha", min_value=0.0, max_value=600.0, value=200.0, step=1.0) 
    P = st.slider("Phosphorus (P) - kg/ha", min_value=0.0, max_value=120.0, value=30.0, step=1.0) 
    K = st.slider("Potassium (K) - kg/ha", min_value=0.0, max_value=800.0, value=150.0, step=1.0) 
    pH = st.number_input("pH", min_value=3.5, max_value=10.0, value=6.5, step=0.1, format="%.1f") 
    st.markdown("---") 
    st.subheader("Advanced Options") 
    show_dataset = st.checkbox("Show dataset preview", value=False) 
    show_charts = st.checkbox("Show nutrient charts", value=True) 
    run_button = st.button("Get Recommendation")

with col2: 
    st.subheader("Output") 
    placeholder = st.empty()

#---------------------------

#Utility functions (same logic)

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
    return "Highly alkaline", "Soil highly alkaline — add gypsum + compost."

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
    # Fallback if model not trained 
    if soil_model is None: 
        # simple rule-based fallback 
        if pH < 5.5 or pH > 8.5: 
            return 'Low', 'Extreme pH level' 
        if N < 200: 
            return 'Low', 'Nitrogen deficiency'
        if P < 15: 
            return 'Low', 'Phosphorus deficiency' 
        if K < 120: 
            return 'Low', 'Potassium deficiency' 
        return 'Moderate', 'Rule-based fallback: nutrients OK'

user_df = pd.DataFrame([[N, P, K, pH, N/(P+1e-6)]], columns=['n','p','k','ph','n_p_ratio'])
pred_enc = soil_model.predict(user_df)[0]
mapping = {0:'Low', 1:'Moderate', 2:'Healthy'}
soil_health = mapping.get(pred_enc, 'Low')
# reason
if pH < 5.5 or pH > 8.5:
    reason = 'Extreme pH level'
elif N < 200:
    reason = 'Nitrogen deficiency'
elif P < 15:
    reason = 'Phosphorus deficiency'
elif K < 120:
    reason = 'Potassium deficiency'
else:
    reason = 'All nutrients in optimal range'
return soil_health, reason

def recommend_fertilizer(N, P, K, soil_health): 
    if fert_model is None or scaler is None: 
        # simple fallback 
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

#Show dataset preview and models metrics

#---------------------------

if show_dataset: 
    st.markdown("### Dataset Preview") 
    if soil_df is not None: 
        st.write("Soil dataset (first 5 rows)") 
        st.dataframe(soil_df.head()) 
    if fert_df is not None: 
        st.write("Fertilizer dataset (first 5 rows)") 
        st.dataframe(fert_df.head()) 
    st.markdown("---")

if soil_acc is not None: 
    st.info(f"Soil health model accuracy: {soil_acc100:.2f}%") 
if fert_acc is not None: 
    st.info(f"Fertilizer model CV accuracy: {fert_acc100:.2f}%")

#---------------------------

#Run predictions when clicked

#---------------------------

if run_button: 
    with placeholder.container(): 
        st.markdown("### Results") 
        soil_health, reason = predict_soil_health(N, P, K, pH) 
        primary, conf = recommend_fertilizer(N, P, K, soil_health)

# Output cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Soil Health", soil_health)
        c1.write(f"*Reason:* {reason}")
        c2.metric("Recommended Fertilizer", primary)
        c2.write(f"*Confidence:* {conf*100:.2f}%")
        ph_cat, ph_text = ph_category_and_text(pH)
        c3.metric("pH Category", ph_cat)
        c3.write(ph_text)

        st.markdown("---")
        st.subheader("ICAR Tip & Action Plan")
        st.write(icar_tip(primary, soil_health))
        st.write(ph_text)

        st.markdown("---")
        st.subheader("Nutrient Warnings & Quick Actions")
        for msg in nutrient_warnings(N, P, K):
            st.write("- "+msg)

        # Charts
        if show_charts:
            st.markdown("---")
            st.subheader("Nutrient Bar Chart")
            fig, ax = plt.subplots(figsize=(6,3))
            nutrients = ['N','P','K']
            vals = [N, P, K]
            ax.bar(nutrients, vals)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel('kg/ha')
            st.pyplot(fig)

            st.subheader("pH Gauge (simple)")
            fig2, ax2 = plt.subplots(figsize=(6,1.2))
            ax2.axis('off')
            ax2.text(0.02, 0.5, f'pH = {pH}', fontsize=18, weight='bold')
            ax2.text(0.40, 0.5, f'({ph_cat})', fontsize=14)
            st.pyplot(fig2)

        st.success("Recommendation generated — use results as guidance and cross-check with local agronomists for field-scale plans.")

#---------------------------

#Footer + extra resources

#---------------------------

st.markdown("---") 
st.markdown("Extra features: Dataset preview, charts, tips, and an action plan. Export results by copying or taking screenshots.") 
st.markdown("How to run:\n1. Place your GIF at ./assets/bg.gif (optional).\n2. Put soilhealth.csv and fertilizer.csv in the same folder or upload via sidebar.\n3. Install dependencies: pip install -r requirements.txt (see requirements below).\n4. Run: streamlit run streamlit_soil_health_app.py.")

st.markdown("### Requirements (example)\n``` streamlit pandas numpy scikit-learn matplotlib```")

st.caption("Built for educational & prototyping purposes. Always validate recommendations with local soil labs and agronomists.")
