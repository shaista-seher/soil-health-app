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
    
    /* Style language selection buttons */
    .stButton > button {
        background-color: #bbf7d0 !important;
        color: #065f46 !important;
        border: 2px solid #059669 !important;
        font-weight: 600;
        font-size: 16px;
        padding: 10px 20px;
    }
    
    .stButton > button:hover {
        background-color: #86efac !important;
        border: 2px solid #047857 !important;
        color: #064e3b !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
add_mint_background()

#---------------------------
# Language Translations
#---------------------------

translations = {
    'english': {
        'app_title': 'SOILS PARK',
        'app_subtitle': '🌱 AI-Powered Soil Health & Fertilizer Guidance 🌱',
        'start_button': '🚀 Start Journey',
        'language_page_title': 'Choose Your Language',
        'language_page_subtitle': 'Select your preferred language for the application',
        'continue_button': 'Continue',
        'input_page_title': '🌾 SOILS PARK',
        'input_page_subtitle': 'Smart farm recommendations powered by ML — enter your soil test values to get instant guidance.',
        'enter_values': '📝 Enter Soil Test Values',
        'nitrogen': 'Nitrogen (N) - kg/ha',
        'phosphorus': 'Phosphorus (P) - kg/ha',
        'potassium': 'Potassium (K) - kg/ha',
        'ph_value': 'pH Value',
        'analyze_button': '🔍 Analyze Soil',
        'output_page_title': '📊 SOILS PARK - Analysis Results',
        'input_values': '📥 Input Values',
        'soil_health': 'Soil Health',
        'recommended_fertilizer': 'Recommended Fertilizer',
        'ph_category': 'pH Category',
        'detailed_recommendations': '📋 Get Detailed Recommendations',
        'icar_plan': '🌱 ICAR Action Plan',
        'nutrient_analysis': '⚠️ Nutrient Analysis & Quick Actions',
        'visual_analysis': '📈 Visual Analysis',
        'nutrient_distribution': 'Nutrient Distribution',
        'ph_status': 'pH Status',
        'analyze_new': '🔄 Analyze New Sample',
        'footer': 'Built for educational & prototyping purposes. Always validate recommendations with local soil labs and agronomists.',
        'soil_health_reasons_Healthy': 'Your soil has good nutrient balance and suitable pH levels.',
        'soil_health_reasons_Moderate': 'Your soil shows slight nutrient imbalance. Consider mild correction.',
        'soil_health_reasons_Low': 'Your soil nutrients are imbalanced; improvement is needed.',
        'ph_Highly acidic': 'Highly acidic',
        'ph_Slightly acidic': 'Slightly acidic', 
        'ph_Neutral': 'Neutral',
        'ph_Slightly alkaline': 'Slightly alkaline',
        'ph_Highly alkaline': 'Highly alkaline',
        'ph_text_Highly acidic': 'Soil is highly acidic — mix agricultural lime.',
        'ph_text_Slightly acidic': 'Soil slightly acidic — add agricultural lime.',
        'ph_text_Neutral': 'Soil is neutral — maintain with compost.',
        'ph_text_Slightly alkaline': 'Soil slightly alkaline — apply gypsum.',
        'ph_text_Highly alkaline': 'Soil highly alkalic — add gypsum + compost.',
        'Primary Results': 'Primary Results'
    },
    'hindi': {
        'app_title': 'मृदा पार्क',
        'app_subtitle': '🌱 एआई-संचालित मृदा स्वास्थ्य और उर्वरक मार्गदर्शन 🌱',
        'start_button': '🚀 यात्रा शुरू करें',
        'language_page_title': 'अपनी भाषा चुनें',
        'language_page_subtitle': 'एप्लिकेशन के लिए अपनी पसंदीदा भाषा चुनें',
        'continue_button': 'जारी रखें',
        'input_page_title': '🌾 मृदा पार्क',
        'input_page_subtitle': 'एमएल द्वारा संचालित स्मार्ट फार्म सिफारिशें — त्वरित मार्गदर्शन प्राप्त करने के लिए अपने मृदा परीक्षण मान दर्ज करें।',
        'enter_values': '📝 मृदा परीक्षण मान दर्ज करें',
        'nitrogen': 'नाइट्रोजन (N) - किग्रा/हेक्टेयर',
        'phosphorus': 'फॉस्फोरस (P) - किग्रा/हेक्टेयर',
        'potassium': 'पोटैशियम (K) - किग्रा/हेक्टेयर',
        'ph_value': 'pH मान',
        'analyze_button': '🔍 मृदा का विश्लेषण करें',
        'output_page_title': '📊 मृदा पार्क - विश्लेषण परिणाम',
        'input_values': '📥 इनपुट मान',
        'soil_health': 'मृदा स्वास्थ्य',
        'recommended_fertilizer': 'सुझाया गया उर्वरक',
        'ph_category': 'pH श्रेणी',
        'detailed_recommendations': '📋 विस्तृत सिफारिशें प्राप्त करें',
        'icar_plan': '🌱 आईसीएआर कार्य योजना',
        'nutrient_analysis': '⚠️ पोषक तत्व विश्लेषण और त्वरित कार्रवाई',
        'visual_analysis': '📈 दृश्य विश्लेषण',
        'nutrient_distribution': 'पोषक तत्व वितरण',
        'ph_status': 'pH स्थिति',
        'analyze_new': '🔄 नया नमूना विश्लेषण करें',
        'footer': 'शैक्षिक और प्रोटोटाइप उद्देश्यों के लिए बनाया गया। स्थानीय मृदा प्रयोगशालाओं और कृषि विशेषज्ञों के साथ सिफारिशों को हमेशा सत्यापित करें।',
        'soil_health_reasons_Healthy': 'आपकी मिट्टी में अच्छा पोषक तत्व संतुलन और उपयुक्त पीएच स्तर है।',
        'soil_health_reasons_Moderate': 'आपकी मिट्टी में थोड़ा पोषक तत्व असंतुलन दिखता है। हल्के सुधार पर विचार करें।',
        'soil_health_reasons_Low': 'आपके मृदा पोषक तत्व असंतुलित हैं; सुधार की आवश्यकता है।',
        'ph_Highly acidic': 'अत्यधिक अम्लीय',
        'ph_Slightly acidic': 'थोड़ा अम्लीय', 
        'ph_Neutral': 'तटस्थ',
        'ph_Slightly alkaline': 'थोड़ा क्षारीय',
        'ph_Highly alkaline': 'अत्यधिक क्षारीय',
        'ph_text_Highly acidic': 'मिट्टी अत्यधिक अम्लीय है — कृषि चूना मिलाएं।',
        'ph_text_Slightly acidic': 'मिट्टी थोड़ी अम्लीय है — कृषि चूना डालें।',
        'ph_text_Neutral': 'मिट्टी तटस्थ है — कम्पोस्ट के साथ बनाए रखें।',
        'ph_text_Slightly alkaline': 'मिट्टी थोड़ी क्षारीय है — जिप्सम लगाएं।',
        'ph_text_Highly alkaline': 'मिट्टी अत्यधिक क्षारीय है — जिप्सम + कम्पोस्ट डालें।',
        'Primary Results': 'प्राथमिक परिणाम'
    },
    'telugu': {
        'app_title': 'సాయిల్ పార్క్',
        'app_subtitle': '🌱 AI-నడిచే నేల ఆరోగ్యం & ఎరువు మార్గదర్శకత్వం 🌱',
        'start_button': '🚀 ప్రయాణం ప్రారంభించండి',
        'language_page_title': 'మీ భాషను ఎంచుకోండి',
        'language_page_subtitle': 'అప్లికేషన్ కోసం మీకు నచ్చిన భాషను ఎంచుకోండి',
        'continue_button': 'కొనసాగించు',
        'input_page_title': '🌾 సాయిల్ పార్క్',
        'input_page_subtitle': 'ML ద్వారా నడిచే స్మార్ట్ ఫార్మ్ సిఫార్సులు — తక్షణ మార్గదర్శకత్వం పొందడానికి మీ నేల పరీక్ష విలువలను నమోదు చేయండి.',
        'enter_values': '📝 నేల పరీక్ష విలువలను నమోదు చేయండి',
        'nitrogen': 'నత్రజని (N) - kg/ha',
        'phosphorus': 'భాస్వరం (P) - kg/ha',
        'potassium': 'పొటాషియం (K) - kg/ha',
        'ph_value': 'pH విలువ',
        'analyze_button': '🔍 నేల విశ్లేషించండి',
        'output_page_title': '📊 సాయిల్ పార్క్ - విశ్లేషణ ఫలితాలు',
        'input_values': '📥 ఇన్పుట్ విలువలు',
        'soil_health': 'నేల ఆరోగ్యం',
        'recommended_fertilizer': 'సిఫారసు చేసిన ఎరువు',
        'ph_category': 'pH వర్గం',
        'detailed_recommendations': '📋 వివరణాత్మక సిఫార్సులను పొందండి',
        'icar_plan': '🌱 ఐసిఎఆర్ యాక్షన్ ప్లాన్',
        'nutrient_analysis': '⚠️ పోషక విశ్లేషణ & త్వరిత చర్యలు',
        'visual_analysis': '📈 దృశ్య విశ్లేషణ',
        'nutrient_distribution': 'పోషక పంపిణీ',
        'ph_status': 'pH స్థితి',
        'analyze_new': '🔄 కొత్త నమూనా విశ్లేషించండి',
        'footer': 'విద్యా & ప్రోటోటైప్ ప్రయోజనాల కోసం నిర్మించబడింది. స్థానిక నేల ప్రయోగశాలలు మరియు వ్యవసాయ నిపుణులతో సిఫార్సులను ఎల్లప్పుడూ ధృవీకరించండి.',
        'soil_health_reasons_Healthy': 'మీ నేలలో మంచి పోషక సమతుల్యత మరియు తగిన pH స్థాయిలు ఉన్నాయి.',
        'soil_health_reasons_Moderate': 'మీ నేల స్వల్ప పోషక అసమతుల్యతను చూపుతుంది. తేలికపాటి దిద్దుబాటు పరిగణించండి.',
        'soil_health_reasons_Low': 'మీ నేల పోషకాలు అసమతుల్యంగా ఉన్నాయి; మెరుగుదల అవసరం.',
        'ph_Highly acidic': 'అత్యంత ఆమ్లం',
        'ph_Slightly acidic': 'కొంచెం ఆమ్లం', 
        'ph_Neutral': 'తటస్థం',
        'ph_Slightly alkaline': 'కొంచెం క్షారం',
        'ph_Highly alkaline': 'అత్యంత క్షారం',
        'ph_text_Highly acidic': 'నేల అత్యంత ఆమ్లం — వ్యవసాయ సున్నాని కలపండి.',
        'ph_text_Slightly acidic': 'నేల కొంచెం ఆమ్లం — వ్యవసాయ సున్నాని జోడించండి.',
        'ph_text_Neutral': 'నేల తటస్థంగా ఉంది — కంపోస్ట్తో నిర్వహించండి.',
        'ph_text_Slightly alkaline': 'నేల కొంచెం క్షారం — జిప్సం వర్తించండి.',
        'ph_text_Highly alkaline': 'నేల అత్యంత క్షారం — జిప్సం + కంపోస్ట్ జోడించండి.',
        'Primary Results': 'ప్రాథమిక ఫలితాలు'
    },
    'tamil': {
        'app_title': 'மண் பூங்கா',
        'app_subtitle': '🌱 AI-இயக்கும் மண் ஆரோக்கியம் & உர பரிந்துரை 🌱',
        'start_button': '🚀 பயணத்தை தொடங்கவும்',
        'language_page_title': 'உங்கள் மொழியை தேர்ந்தெடுக்கவும்',
        'language_page_subtitle': 'விண்ணப்பத்திற்கான உங்கள் விருப்ப மொழியைத் தேர்ந்தெடுக்கவும்',
        'continue_button': 'தொடரவும்',
        'input_page_title': '🌾 மண் பூங்கா',
        'input_page_subtitle': 'ML-இயக்கும் ஸ்மார்ட் பண்ணை பரிந்துரைகள் — உடனடி வழிகாட்டுதலுக்கு உங்கள் மண் சோதனை மதிப்புகளை உள்ளிடவும்.',
        'enter_values': '📝 மண் சோதனை மதிப்புகளை உள்ளிடவும்',
        'nitrogen': 'நைட்ரஜன் (N) - kg/ha',
        'phosphorus': 'பாஸ்பரஸ் (P) - kg/ha',
        'potassium': 'பொட்டாசியம் (K) - kg/ha',
        'ph_value': 'pH மதிப்பு',
        'analyze_button': '🔍 மண்ணை பகுப்பாய்வு செய்யவும்',
        'output_page_title': '📊 மண் பூங்கா - பகுப்பாய்வு முடிவுகள்',
        'input_values': '📥 உள்ளீட்டு மதிப்புகள்',
        'soil_health': 'மண் ஆரோக்கியம்',
        'recommended_fertilizer': 'பரிந்துரைக்கப்பட்ட உரம்',
        'ph_category': 'pH வகை',
        'detailed_recommendations': '📋 விரிவான பரிந்துரைகளைப் பெறுக',
        'icar_plan': '🌾 ஐசிஏஆர் செயல் திட்டம்',
        'nutrient_analysis': '⚠️ ஊட்டச்சத்து பகுப்பாய்வு & விரைவு நடவடிக்கைகள்',
        'visual_analysis': '📈 காட்சி பகுப்பாய்வு',
        'nutrient_distribution': 'ஊட்டச்சத்து விநியோகம்',
        'ph_status': 'pH நிலை',
        'analyze_new': '🔄 புதிய மாதிரியை பகுப்பாய்வு செய்யவும்',
        'footer': 'கல்வி & முன்மாதிரி நோக்கங்களுக்காக கட்டப்பட்டது. உள்ளூர் மண் ஆய்வகங்கள் மற்றும் விவசாய நிபுணர்களுடன் பரிந்துரைகளை எப்போதும் சரிபார்க்கவும்.',
        'soil_health_reasons_Healthy': 'உங்கள் மண்ணில் நல்ல ஊட்டச்சத்து சமநிலை மற்றும் பொருத்தமான pH நிலைகள் உள்ளன.',
        'soil_health_reasons_Moderate': 'உங்கள் மண் சிறிய ஊட்டச்சத்து சமநிலையின்மையைக் காட்டுகிறது. லேசான திருத்தத்தைக் கவனியுங்கள்.',
        'soil_health_reasons_Low': 'உங்கள் மண் ஊட்டச்சத்துக்கள் சமநிலையற்றவை; மேம்பாடு தேவை.',
        'ph_Highly acidic': 'மிகவும் அமிலமான',
        'ph_Slightly acidic': 'சற்று அமிலமான', 
        'ph_Neutral': 'நடுநிலையான',
        'ph_Slightly alkaline': 'சற்று காரமான',
        'ph_Highly alkaline': 'மிகவும் காரமான',
        'ph_text_Highly acidic': 'மண் மிகவும் அமிலமானது — விவசாய சுண்ணாம்பை கலக்கவும்.',
        'ph_text_Slightly acidic': 'மண் சற்று அமிலமானது — விவசாய சுண்ணாம்பை சேர்க்கவும்.',
        'ph_text_Neutral': 'மண் நடுநிலையானது — கம்போஸ்ட்டுடன் பராமரிக்கவும்.',
        'ph_text_Slightly alkaline': 'மண் சற்று காரமானது — ஜிப்சம் பயன்படுத்தவும்.',
        'ph_text_Highly alkaline': 'மண் மிகவும் காரமானது — ஜிப்சம் + கம்போஸ்ட் சேர்க்கவும்.',
        'Primary Results': 'முதன்மை முடிவுகள்'
    },
    'kannada': {
        'app_title': 'ಮಣ್ಣಿನ ಪಾರ್ಕ್',
        'app_subtitle': '🌱 AI-ಚಾಲಿತ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಮತ್ತು ಎರುವ ಹುಡುಕಾಟ 🌱',
        'start_button': '🚀 ಪ್ರಯಾಣ ಪ್ರಾರಂಭಿಸಿ',
        'language_page_title': 'ನಿಮ್ಮ ಭಾಷೆಯನ್ನು ಆರಿಸಿ',
        'language_page_subtitle': 'ಅಪ್ಲಿಕೇಶನ್ಗಾಗಿ ನಿಮ್ಮ ಆದ್ಯತೆಯ ಭಾಷೆಯನ್ನು ಆರಿಸಿ',
        'continue_button': 'ಮುಂದುವರಿಸಿ',
        'input_page_title': '🌾 ಮಣ್ಣಿನ ಪಾರ್ಕ್',
        'input_page_subtitle': 'ML-ಚಾಲಿತ ಸ್ಮಾರ್ಟ್ ಫಾರ್ಮ್ ಶಿಫಾರಸುಗಳು — ತ್ವರಿತ ಮಾರ್ಗದರ್ಶನ ಪಡೆಯಲು ನಿಮ್ಮ ಮಣ್ಣಿನ ಪರೀಕ್ಷಾ ಮೌಲ್ಯಗಳನ್ನು ನಮೂದಿಸಿ.',
        'enter_values': '📝 ಮಣ್ಣಿನ ಪರೀಕ್ಷಾ ಮೌಲ್ಯಗಳನ್ನು ನಮೂದಿಸಿ',
        'nitrogen': 'ನೈಟ್ರೋಜನ್ (N) - kg/ha',
        'phosphorus': 'ಫಾಸ್ಫರಸ್ (P) - kg/ha',
        'potassium': 'ಪೊಟಾಶಿಯಂ (K) - kg/ha',
        'ph_value': 'pH ಮೌಲ್ಯ',
        'analyze_button': '🔍 ಮಣ್ಣನ್ನು ವಿಶ್ಲೇಷಿಸಿ',
        'output_page_title': '📊 ಮಣ್ಣಿನ ಪಾರ್ಕ್ - ವಿಶ್ಲೇಷಣೆ ಫಲಿತಾಂಶಗಳು',
        'input_values': '📥 ಇನ್ಪುಟ್ ಮೌಲ್ಯಗಳು',
        'soil_health': 'ಮಣ್ಣಿನ ಆರೋಗ್ಯ',
        'recommended_fertilizer': 'ಶಿಫಾರಸು ಮಾಡಿದ ಎರುವು',
        'ph_category': 'pH ವರ್ಗ',
        'detailed_recommendations': '📋 ವಿವರವಾದ ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಿರಿ',
        'icar_plan': '🌾 ಐಸಿಎಆರ್ ಕ್ರಿಯಾ ಯೋಜನೆ',
        'nutrient_analysis': '⚠️ ಪೋಷಕ ವಿಶ್ಲೇಷಣೆ ಮತ್ತು ತ್ವರಿತ ಕ್ರಮಗಳು',
        'visual_analysis': '📈 ದೃಶ್ಯ ವಿಶ್ಲೇಷಣೆ',
        'nutrient_distribution': 'ಪೋಷಕ ವಿತರಣೆ',
        'ph_status': 'pH ಸ್ಥಿತಿ',
        'analyze_new': '🔄 ಹೊಸ ಮಾದರಿ ವಿಶ್ಲೇಷಿಸಿ',
        'footer': 'ಶೈಕ್ಷಣಿಕ ಮತ್ತು ಪ್ರೋಟೋಟೈಪ್ ಉದ್ದೇಶಗಳಿಗಾಗಿ ನಿರ್ಮಿಸಲಾಗಿದೆ. ಸ್ಥಳೀಯ ಮಣ್ಣಿನ ಪ್ರಯೋಗಶಾಲೆಗಳು ಮತ್ತು ಕೃಷಿ ತಜ್ಞರೊಂದಿಗೆ ಶಿಫಾರಸುಗಳನ್ನು ಯಾವಾಗಲೂ ಪರಿಶೀಲಿಸಿ.',
        'soil_health_reasons_Healthy': 'ನಿಮ್ಮ ಮಣ್ಣಿನಲ್ಲಿ ಉತ್ತಮ ಪೋಷಕ ಸಮತೋಲನ ಮತ್ತು ಸೂಕ್ತ pH ಮಟ್ಟಗಳಿವೆ.',
        'soil_health_reasons_Moderate': 'ನಿಮ್ಮ ಮಣ್ಣು ಸ್ವಲ್ಪ ಪೋಷಕ ಅಸಮತೋಲನವನ್ನು ತೋರಿಸುತ್ತದೆ. ಸೌಮ್ಯ ತಿದ್ದುಪಡಿಯನ್ನು ಪರಿಗಣಿಸಿ.',
        'soil_health_reasons_Low': 'ನಿಮ್ಮ ಮಣ್ಣಿನ ಪೋಷಕಗಳು ಅಸಮತೋಲಿತವಾಗಿವೆ; ಸುಧಾರಣೆ ಅಗತ್ಯವಿದೆ.',
        'ph_Highly acidic': 'ಅತ್ಯಂತ ಆಮ್ಲೀಯ',
        'ph_Slightly acidic': 'ಸ್ವಲ್ಪ ಆಮ್ಲೀಯ', 
        'ph_Neutral': 'ತಟಸ್ಥ',
        'ph_Slightly alkaline': 'ಸ್ವಲ್ಪ ಕ್ಷಾರೀಯ',
        'ph_Highly alkaline': 'ಅತ್ಯಂತ ಕ್ಷಾರೀಯ',
        'ph_text_Highly acidic': 'ಮಣ್ಣು ಅತ್ಯಂತ ಆಮ್ಲೀಯವಾಗಿದೆ — ಕೃಷಿ ಸುಣ್ಣವನ್ನು ಬೆರೆಸಿ.',
        'ph_text_Slightly acidic': 'ಮಣ್ಣು ಸ್ವಲ್ಪ ಆಮ್ಲೀಯವಾಗಿದೆ — ಕೃಷಿ ಸುಣ್ಣವನ್ನು ಸೇರಿಸಿ.',
        'ph_text_Neutral': 'ಮಣ್ಣು ತಟಸ್ಥವಾಗಿದೆ — ಕಂಪೋಸ್ಟ್ನೊಂದಿಗೆ ನಿರ್ವಹಿಸಿ.',
        'ph_text_Slightly alkaline': 'ಮಣ್ಣು ಸ್ವಲ್ಪ ಕ್ಷಾರೀಯವಾಗಿದೆ — ಜಿಪ್ಸಂ ಅನ್ನು ಅನ್ವಯಿಸಿ.',
        'ph_text_Highly alkaline': 'ಮಣ್ಣು ಅತ್ಯಂತ ಕ್ಷಾರೀಯವಾಗಿದೆ — ಜಿಪ್ಸಂ + ಕಂಪೋಸ್ಟ್ ಸೇರಿಸಿ.',
        'Primary Results': 'ಪ್ರಾಥಮಿಕ ಫಲಿತಾಂಶಗಳು'
    }
}

def get_translation(key, language='english'):
    """Get translation for a given key in the specified language"""
    try:
        if language in translations and key in translations[language]:
            return translations[language][key]
        # Fallback to English
        if key in translations['english']:
            return translations['english'][key]
        return key  # Return the key itself if not found
    except:
        return key  # Return the key itself in case of any error

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

def ph_category_and_text(pH, language='english'): 
    if pH < 5.5: 
        return get_translation('ph_Highly acidic', language), get_translation('ph_text_Highly acidic', language)
    if 5.5 <= pH < 6.5: 
        return get_translation('ph_Slightly acidic', language), get_translation('ph_text_Slightly acidic', language)
    if 6.5 <= pH <= 7.5: 
        return get_translation('ph_Neutral', language), get_translation('ph_text_Neutral', language)
    if 7.5 < pH <= 8.5: 
        return get_translation('ph_Slightly alkaline', language), get_translation('ph_text_Slightly alkaline', language)
    return get_translation('ph_Highly alkaline', language), get_translation('ph_text_Highly alkaline', language)

def nutrient_level(val, nut): 
    if nut == 'N': 
        return "Low" if val < 200 else "Medium" if val <= 400 else "High" 
    if nut == 'P': 
        return "Low" if val < 15 else "Medium" if val <= 35 else "High" 
    if nut == 'K': 
        return "Low" if val < 110 else "Medium" if val <= 280 else "High"

def nutrient_warnings(N, P, K, language='english'): 
    msgs = [] 
    mapping_name = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium"}
    mapping_rec = {"N": "Urea", "P": "DAP", "K": "MOP"} 
    
    # Translations for nutrient names and recommendations
    if language != 'english':
        nutrient_translations = {
            'hindi': {"Nitrogen": "नाइट्रोजन", "Phosphorus": "फॉस्फोरस", "Potassium": "पोटैशियम", "Urea": "यूरिया", "DAP": "डीएपी", "MOP": "एमओपी"},
            'telugu': {"Nitrogen": "నత్రజని", "Phosphorus": "భాస్వరం", "Potassium": "పొటాషియం", "Urea": "యూరియా", "DAP": "డీఏపీ", "MOP": "ఎమ్ఓపీ"},
            'tamil': {"Nitrogen": "நைட்ரஜன்", "Phosphorus": "பாஸ்பரஸ்", "Potassium": "பொட்டாசியம்", "Urea": "யூரியா", "DAP": "டிஏபி", "MOP": "எம்ஓபி"},
            'kannada': {"Nitrogen": "ನೈಟ್ರೋಜನ్", "Phosphorus": "ಫಾಸ್ಫರಸ್", "Potassium": "ಪೊಟಾಶಿಯಂ", "Urea": "ಯೂರಿಯಾ", "DAP": "ಡಿಎಪಿ", "MOP": "ಎಮ್ಒಪಿ"}
        }
        if language in nutrient_translations:
            mapping_name = {"N": nutrient_translations[language]["Nitrogen"], 
                          "P": nutrient_translations[language]["Phosphorus"], 
                          "K": nutrient_translations[language]["Potassium"]}
            mapping_rec = {"N": nutrient_translations[language]["Urea"], 
                         "P": nutrient_translations[language]["DAP"], 
                         "K": nutrient_translations[language]["MOP"]}
    
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

def icar_tip(primary, soil_health, language='english'):
    if soil_health == 'Low':
         return f"Apply {primary} with compost/FYM in 2–3 splits as per ICAR guidelines." 
    elif soil_health == 'Moderate':
         return f"Apply {primary} in 2 splits and include compost." 
    else: 
         return f"Apply {primary} once and maintain crop rotation."

#---------------------------
#Prediction logic
#---------------------------

def predict_soil_health(N, P, K, pH, language='english'):
    if soil_model is None:
        return "Unknown", "Model not loaded"
    
    try:
        n_p_ratio = N / (P + 1e-6)
        pred_num = soil_model.predict([[N, P, K, pH, n_p_ratio]])[0]
        mapping = {0: "Low", 1: "Moderate", 2: "Healthy"}
        pred = mapping.get(pred_num, "Unknown")

        # Get translated reason
        reason_key = f'soil_health_reasons_{pred}'
        reason = get_translation(reason_key, language)
        
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
if 'language' not in st.session_state:
    st.session_state.language = 'english'

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
            st.session_state.page = 'language'
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align: center; color: #6b7280; font-size: 14px;'>
            Smart recommendations powered by machine learning
        </p>
        """, unsafe_allow_html=True)

#---------------------------
# LANGUAGE SELECTION PAGE
#---------------------------

elif st.session_state.page == 'language':
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Center content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <h1 style='text-align: center; color: #059669; font-size: 48px; font-family: Georgia, serif; margin-bottom: 20px;'>
            {get_translation('language_page_title', 'english')}
        </h1>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <p style='text-align: center; color: #10b981; font-size: 18px; margin-bottom: 50px;'>
            {get_translation('language_page_subtitle', 'english')}
        </p>
        """, unsafe_allow_html=True)
        
        # Language selection buttons
        languages = [
            ('English', 'english'),
            ('हिन्दी (Hindi)', 'hindi'),
            ('తెలుగు (Telugu)', 'telugu'),
            ('தமிழ் (Tamil)', 'tamil'),
            ('ಕನ್ನಡ (Kannada)', 'kannada')
        ]
        
        for lang_name, lang_code in languages:
            if st.button(lang_name, use_container_width=True, key=lang_code):
                st.session_state.language = lang_code
                st.session_state.page = 'input'
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Back button
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 'start'
            st.rerun()

#---------------------------
# INPUT PAGE
#---------------------------

elif st.session_state.page == 'input':
    current_lang = st.session_state.language
    
    st.markdown(f"<h1 style='text-align:center; color:#059669;'>{get_translation('input_page_title', current_lang)}</h1>", unsafe_allow_html=True) 
    st.markdown(f"<p style='text-align:center; font-size:16px; color:#10b981;'>{get_translation('input_page_subtitle', current_lang)}</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the input form
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown(f"### {get_translation('enter_values', current_lang)}")
        
        with st.form("soil_input_form"):
            N_input = st.text_input(get_translation('nitrogen', current_lang), placeholder="e.g., 200", help="Range: 0-600 kg/ha")
            P_input = st.text_input(get_translation('phosphorus', current_lang), placeholder="e.g., 30", help="Range: 0-120 kg/ha")
            K_input = st.text_input(get_translation('potassium', current_lang), placeholder="e.g., 150", help="Range: 0-800 kg/ha")
            pH_input = st.text_input(get_translation('ph_value', current_lang), placeholder="e.g., 6.5", help="Range: 3.5-10.0")
            
            submit_button = st.form_submit_button(get_translation('analyze_button', current_lang), use_container_width=True)
            
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
         # Back to language selection button
    if st.button("🌐 Change Language", use_container_width=True):
        st.session_state.page = 'language'
        st.rerun()             
    
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
    current_lang = st.session_state.language
    
    # Get values from session state
    N = st.session_state.N
    P = st.session_state.P
    K = st.session_state.K
    pH = st.session_state.pH
    
    # Run predictions
    soil_health, reason = predict_soil_health(N, P, K, pH, current_lang)
    primary, conf = recommend_fertilizer(N, P, K, soil_health)
    ph_cat, ph_text = ph_category_and_text(pH, current_lang)
    
    # Header
    st.markdown(f"<h1 style='text-align:center; color:#059669;'>{get_translation('output_page_title', current_lang)}</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display input values
    st.markdown(f"<h3 style='color:#065f46;'>{get_translation('input_values', current_lang)}</h3>", unsafe_allow_html=True)
    input_col1, input_col2, input_col3, input_col4 = st.columns(4)
    input_col1.metric(get_translation('nitrogen', current_lang), f"{N} kg/ha")
    input_col2.metric(get_translation('phosphorus', current_lang), f"{P} kg/ha")
    input_col3.metric(get_translation('potassium', current_lang), f"{K} kg/ha")
    input_col4.metric(get_translation('ph_value', current_lang), f"{pH}")
    
    st.markdown("---")
    
    # Main results
    st.markdown(f"<h3 style='color:#065f46;'>🎯 {get_translation('Primary Results', current_lang)}</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    # Color coding for soil health
    health_color = {"Healthy": "🟢", "Moderate": "🟡", "Low": "🔴", "Unknown": "⚪"}
    c1.markdown(f"<h4 style='color:#065f46;'>{health_color.get(soil_health, '⚪')} {get_translation('soil_health', current_lang)}</h4>", unsafe_allow_html=True)
    c1.markdown(f"<h3 style='color:#065f46;'><strong>{soil_health}</strong></h3>", unsafe_allow_html=True)
    c1.markdown(f"<p style='color:#065f46;'><em>{reason}</em></p>", unsafe_allow_html=True)
    
    c2.markdown(f"<h4 style='color:#065f46;'>💊 {get_translation('recommended_fertilizer', current_lang)}</h4>", unsafe_allow_html=True)
    c2.markdown(f"<h3 style='color:#065f46;'><strong>{primary}</strong></h3>", unsafe_allow_html=True)
    c2.markdown(f"<p style='color:#065f46;'><em>Confidence: {conf*100:.2f}%</em></p>", unsafe_allow_html=True)
    
    c3.markdown(f"<h4 style='color:#065f46;'>🧪 {get_translation('ph_category', current_lang)}</h4>", unsafe_allow_html=True)
    c3.markdown(f"<h3 style='color:#065f46;'><strong>{ph_cat}</strong></h3>", unsafe_allow_html=True)
    c3.markdown(f"<p style='color:#065f46;'><em>{ph_text}</em></p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Get Recommendations button (expandable)
    if st.button(get_translation('detailed_recommendations', current_lang), use_container_width=True, type="primary"):
        st.session_state.show_details = not st.session_state.show_details
    
    # Show details if expanded
    if st.session_state.show_details:
        st.markdown("---")
        
        # ICAR Tips
        st.markdown(f"<h3 style='color:#065f46;'>{get_translation('icar_plan', current_lang)}</h3>", unsafe_allow_html=True)
        st.info(icar_tip(primary, soil_health, current_lang))
        st.markdown(f"<p style='color:#065f46;'><strong>pH Management:</strong> {ph_text}</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Nutrient Warnings
        st.markdown(f"<h3 style='color:#065f46;'>{get_translation('nutrient_analysis', current_lang)}</h3>", unsafe_allow_html=True)
        warnings = nutrient_warnings(N, P, K, current_lang)
        for msg in warnings:
            st.markdown(f"<p style='color:#065f46;'>• {msg}</p>", unsafe_allow_html=True)
        
        # Charts
        if show_charts:
            st.markdown("---")
            st.markdown(f"<h3 style='color:#065f46;'>{get_translation('visual_analysis', current_lang)}</h3>", unsafe_allow_html=True)
            
            chart_col1, chart_col2 = st.columns(2)

            # Soil Health vs Nutrient Levels
            st.markdown("---")
            st.markdown(f"<h3 style='color:#065f46;'>🌱 Soil Health vs Nutrient Levels</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#065f46;'>How different nutrient ranges affect soil health classification</p>", unsafe_allow_html=True)
                
            # Create sample data based on user's input and general soil science
            health_categories = ['Low', 'Moderate', 'Healthy']
                
                # Nutrient ranges for each health category (simplified for demonstration)
            nutrient_ranges = {
                 'Nitrogen (kg/ha)': {'Low': [0, 200], 'Moderate': [200, 400], 'Healthy': [400, 600]},
                 'Phosphorus (kg/ha)': {'Low': [0, 15], 'Moderate': [15, 35], 'Healthy': [35, 120]},
                 'Potassium (kg/ha)': {'Low': [0, 110], 'Moderate': [110, 280], 'Healthy': [280, 800]}
            }
                
            fig_health, axes = plt.subplots(1, 3, figsize=(15, 5))
            colors = ['#e74c3c', '#f39c12', '#2ecc71']  # Red, Orange, Green
                
            nutrients = ['Nitrogen (kg/ha)', 'Phosphorus (kg/ha)', 'Potassium (kg/ha)']
            user_values = [N, P, K]
                
            for i, (nutrient, ax) in enumerate(zip(nutrients, axes)):
                ranges = nutrient_ranges[nutrient]
                # Create horizontal bars for each health category
                y_pos = np.arange(len(health_categories))
                bar_values = [ranges[cat][1] - ranges[cat][0] for cat in health_categories]
                bars = ax.barh(y_pos, bar_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
                # Add range labels
                for j, (cat, bar) in enumerate(zip(health_categories, bars)):
                     width = bar.get_width()
                     ax.text(width/2, bar.get_y() + bar.get_height()/2, 
                           f'{ranges[cat][0]}-{ranges[cat][1]}', 
                           ha='center', va='center', fontweight='bold', fontsize=10)
                    
                # Mark user's current value
                user_val = user_values[i]
                user_category = None
                for cat in health_categories:
                    if ranges[cat][0] <= user_val <= ranges[cat][1]:
                        user_category = cat
                        break
                    
                if user_category:
                    cat_index = health_categories.index(user_category)
                    ax.axhline(y=cat_index, color='red', linestyle='--', linewidth=2, 
                            label=f'Your {nutrient.split()[0]}: {user_val}')
                    
                ax.set_yticks(y_pos)
                ax.set_yticklabels(health_categories)
                ax.set_xlabel(nutrient, fontsize=12)
                ax.set_title(f'{nutrient.split()[0]} Ranges', fontsize=13, weight='bold')
                ax.grid(axis='x', alpha=0.3)
                ax.legend()
                
            plt.tight_layout()
            st.pyplot(fig_health)
                
                # Interpretation
            st.markdown(f"""
            <div style='background-color: #d1fae5; padding: 15px; border-radius: 10px; border: 1px solid #059669; margin-top: 15px;'>
            <strong>📊 Interpretation Guide:</strong><br>
              • <span style='color:#e74c3c'><strong>Red (Low)</strong></span>: Nutrient deficiency - requires immediate attention<br>
              • <span style='color:#f39c12'><strong>Orange (Moderate)</strong></span>: Acceptable but could be improved<br>
              • <span style='color:#2ecc71'><strong>Green (Healthy)</strong></span>: Optimal range for plant growth
              </div>
              """, unsafe_allow_html=True)
            
            with chart_col1:
                        # pH Effect Chart
                st.markdown("---")
                st.markdown(f"<h3 style='color:#065f46;'>🧪 pH Impact on Nutrient Availability</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:#065f46;'>How soil pH affects nutrient absorption by plants</p>", unsafe_allow_html=True)
                
                fig_ph, ax_ph = plt.subplots(figsize=(10, 5))
                
                # pH ranges
                ph_categories = ['Highly Acidic\n(4.0-5.5)', 'Slightly Acidic\n(5.5-6.5)', 'Neutral\n(6.5-7.5)', 
                                'Slightly Alkaline\n(7.5-8.5)', 'Highly Alkaline\n(8.5-9.0)']
                
                # Nutrient availability scores (relative)
                nitrogen_avail = [30, 80, 100, 70, 40]
                phosphorus_avail = [20, 90, 100, 60, 20]
                potassium_avail = [50, 90, 100, 80, 50]
                micronutrient_avail = [90, 70, 50, 30, 20]
                
                x_pos = np.arange(len(ph_categories))
                width = 0.2
                
                ax_ph.bar(x_pos - width*1.5, nitrogen_avail, width, label='Nitrogen', color='#2ecc71', alpha=0.8)
                ax_ph.bar(x_pos - width/2, phosphorus_avail, width, label='Phosphorus', color='#3498db', alpha=0.8)
                ax_ph.bar(x_pos + width/2, potassium_avail, width, label='Potassium', color='#e74c3c', alpha=0.8)
                ax_ph.bar(x_pos + width*1.5, micronutrient_avail, width, label='Micronutrients', color='#f39c12', alpha=0.8)
                
                ax_ph.set_xlabel('pH Range', fontsize=12)
                ax_ph.set_ylabel('Relative Availability (%)', fontsize=12)
                ax_ph.set_title('Nutrient Availability at Different pH Levels', fontsize=14, weight='bold')
                ax_ph.set_xticks(x_pos)
                ax_ph.set_xticklabels(ph_categories, rotation=45, ha='right')
                ax_ph.legend()
                ax_ph.grid(axis='y', alpha=0.3)
                ax_ph.set_ylim(0, 110)
                
                # Highlight current pH range
                current_ph_range = None
                if pH < 5.5:
                    current_ph_range = 0
                elif 5.5 <= pH < 6.5:
                    current_ph_range = 1
                elif 6.5 <= pH <= 7.5:
                    current_ph_range = 2
                elif 7.5 < pH <= 8.5:
                    current_ph_range = 3
                else:
                    current_ph_range = 4
                    
                ax_ph.axvline(x=current_ph_range, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                              label=f'Your pH: {pH}')
                ax_ph.legend()
                
                plt.tight_layout()
                st.pyplot(fig_ph)
                
                st.markdown("""
                <div style='background-color: #d1fae5; padding: 15px; border-radius: 10px; border: 1px solid #059669;'>
                <strong>💡 pH Insight:</strong> Your current pH level (<strong>{:.1f}</strong>) falls in the <strong>{}</strong> range. 
                Most nutrients are optimally available in neutral pH (6.5-7.5).
                </div>
                """.format(pH, ph_cat), unsafe_allow_html=True)
           
            with chart_col2:
                 
                            # Feature Importance Plot
                st.markdown("---")
                st.markdown(f"<h3 style='color:#065f46;'>🔍 Feature Importance</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:#065f46;'>Which nutrients most affect soil health predictions</p>", unsafe_allow_html=True)
                
                if soil_model is not None:
                    try:
                        # Get feature importances
                        importances = soil_model.feature_importances_
                        feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'N/P Ratio']
                        
                        # Create feature importance plot
                        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                        y_pos = np.arange(len(feature_names))
                        ax_imp.barh(y_pos, importances, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
                        ax_imp.set_yticks(y_pos)
                        ax_imp.set_yticklabels(feature_names)
                        ax_imp.set_xlabel('Importance Score', fontsize=12)
                        ax_imp.set_title('Feature Importance in Soil Health Prediction', fontsize=14, weight='bold')
                        ax_imp.grid(axis='x', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig_imp)
                        
                        # Display importance percentages
                        imp_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance (%)': (importances * 100).round(2)
                        })
                        st.dataframe(imp_df, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Could not generate feature importance: {e}")
                # Confusion Matrix
        st.markdown("---")
        st.markdown(f"<h3 style='color:#065f46;'>📊 Model Performance</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#065f46;'>How accurately the model predicts different soil health levels</p>", unsafe_allow_html=True)
        
        # Note: This requires test data - you might need to modify based on your data availability
        st.info("Model accuracy metrics available during training. Current soil health prediction is based on trained Random Forest model.")
        
        if soil_acc is not None:
            st.metric("Overall Model Accuracy", f"{soil_acc*100:.2f}%")

        
        st.markdown("---")
        st.success("✅ Detailed recommendations generated. Use these results as guidance and cross-check with local agronomists for field-scale implementation.")
    
    # Navigation buttons
    st.markdown("<br>", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    
    with nav_col2:
        if st.button(get_translation('analyze_new', current_lang), use_container_width=True):
            st.session_state.page = 'input'
            st.session_state.show_details = False
            st.rerun()

#---------------------------
# Footer
#---------------------------

st.markdown("---") 
st.caption(get_translation('footer', st.session_state.language))
