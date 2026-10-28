import os
import datetime
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌾", layout="wide")

model = joblib.load('crop_model.pkl')

# ---------------------------------------------------------------------------
# TRANSLATIONS
# ---------------------------------------------------------------------------
T = {
    "en": {
        "title": "🌾 Smart Crop Recommendation System",
        "subtitle": "Enter your farm's soil and weather details below, and we will suggest the best crop for you.",
        "soil_weather": "🧪 Soil & Weather Details",
        "n": "🌿 Nitrogen (N)", "p": "🧬 Phosphorus (P)", "k": "🧂 Potassium (K)", "ph": "⚗️ Soil pH",
        "temp": "🌡️ Temperature (°C)", "humidity": "💧 Humidity (%)", "rainfall": "🌧️ Rainfall (mm)",
        "predict_btn": "🔍 Predict Best Crop",
        "analyzing": "Analyzing your soil and weather data...",
        "best_crop": "Best Recommended Crop:",
        "confidence": "Confidence",
        "quick_fert_tip": "🧪 Quick Fertilizer Tip",
        "show_profile": "📋 Show full {crop} profile & fertilizer advice",
        "overall_suit": "Overall suitability to your inputs:",
        "value_vs_ideal": "📊 Your value vs. ideal range",
        "fert_reco": "🧪 Fertilizer recommendation",
        "fert_note": "General guidance only — always confirm exact dosage with a soil test / local agri office.",
        "conditions_vs_ideal": "📈 Your Conditions vs. Ideal for {crop}",
        "confidence_gauge": "🎯 Prediction Confidence",
        "match_overview": "🌱 All {n} Crops — Match Overview",
        "match_hint": "Click **Details** to expand a crop's full profile, fertilizer advice and graphs right below it.",
        "details_btn": "Details — {crop}",
        "match_pct": "{v:.1f}% match",
        "conf_chart": "📊 Confidence Chart — All Crops",
        "full_table": "📋 Full Details Table",
        "col_rank": "Rank", "col_crop": "Crop", "col_icon": "Icon", "col_conf": "Confidence (%)",
        "nav_predict": "🔍 Predict",
        "nav_coach": "🧑‍🌾 Crop Coach",
        "nav_encyclopedia": "🌿 Crop Info",
        "nav_insights": "📊 Insights",
        "nav_whatif": "🧪 What-If",
        "sidebar_lang": "🌐 Language / भाषा",
        "sidebar_enc": "🌿 Crop Info",
        "enc_no_prediction": "Run a prediction on the **Predict** page first to see the full match overview, chart and table here.",
        "reset_btn": "🔄 Reset Inputs",
        "download_csv": "📥 Download Full Table (CSV)",
        "share_label": "🔗 Share this result (tap the copy icon)",
        "print_tip": "🖨️ To save as PDF: on computer press **Ctrl+P** and choose 'Save as PDF'. On mobile, open your browser menu → **Share / Print** → 'Save as PDF'.",
        "share_text": "Smart Crop Recommendation Result\\nBest Crop: {crop}\\nConfidence: {conf:.2f}%\\nInputs — N:{n}, P:{p}, K:{k}, Temp:{temp}°C, Humidity:{hum}%, pH:{ph}, Rainfall:{rain}mm",
        "insights_title": "📊 Insights & Extras",
        "tab_history": "📜 History",
        "tab_feat_imp": "🧠 Why This Model Works",
        "tab_market": "💰 Market Price (Approx.)",
        "tab_seasonal": "📅 Seasonal Calendar",
        "no_history": "No predictions yet in this session. Run one on the **Predict** page.",
        "history_time": "Time", "history_crop": "Crop", "history_conf": "Confidence (%)",
        "clear_history": "🗑️ Clear History",
        "feat_imp_desc": "This chart shows how much each feature (on average) influences the Random Forest model's decisions across all 22 crops. Features with taller bars matter more to the prediction.",
        "market_note": "⚠️ These are illustrative, approximate price ranges only (per quintal) — always check your local mandi/market rate before making decisions.",
        "col_price": "Approx. Price Range (₹/quintal)",
        "seasonal_note": "Crops grouped by their typical growing season in India (approximate).",
        "season_kharif": "☔ Kharif (Jun–Oct)", "season_rabi": "❄️ Rabi (Oct–Mar)",
        "season_zaid": "☀️ Zaid / Summer (Mar–Jun)", "season_year": "🔁 Year-round",
        "whatif_title": "🧪 What-If Simulator",
        "whatif_sub": "Move the sliders and watch the prediction update instantly — no button needed. Great for understanding how each factor affects the recommendation.",
        "whatif_top_crop": "Live Top Crop:",
        "sidebar_lookup": "Look up a crop",
        "sidebar_ideal": "Ideal growing conditions (approximate)",
        "coach_header": "🧑‍🌾 Crop Coach",
        "coach_sub": "A free, built-in farming assistant — no sign-up or API key needed.",
        "coach_greeting": "Hi! Based on your inputs, here's my advice for growing **{crop}**:",
        "coach_no_prediction": "⚠️ Please run a prediction on the **Predict** page first, so Crop Coach knows which crop to advise you on.",
        "coach_ask_label": "Ask Crop Coach about fertilizer, water, soil, weather, pests or harvest:",
        "coach_ask_placeholder": "e.g. how much water does this crop need?",
        "coach_ask_btn": "Ask",
        "coach_fallback": "I can help with: **fertilizer**, **water/irrigation**, **soil pH**, **weather**, **pests/disease**, and **harvest** questions for {crop}. Try asking about one of these!",
        "warn_temp": "⚠️ {v}°C is an unusually extreme temperature — double-check this value.",
        "warn_humidity": "⚠️ {v}% humidity is very low — double-check this value.",
        "warn_rainfall": "⚠️ {v}mm rainfall is very low for most crops — double-check this value.",
        "warn_ph": "⚠️ Soil pH {v} is far from the normal 4.5–9 farming range — double-check this value.",
        "n_low": "🟢 **Nitrogen low** — apply **Urea** (~{amt:.0f} kg/ha extra) to close the gap.",
        "n_high": "🟡 **Nitrogen high** — cut back on urea/nitrogen-rich manure this season.",
        "n_ok": "✅ **Nitrogen** is already in a good range.",
        "p_low": "🟢 **Phosphorus low** — apply **DAP / SSP** (~{amt:.0f} kg/ha extra).",
        "p_high": "🟡 **Phosphorus high** — reduce phosphate fertilizer application.",
        "p_ok": "✅ **Phosphorus** is already in a good range.",
        "k_low": "🟢 **Potassium low** — apply **MOP (Muriate of Potash)** (~{amt:.0f} kg/ha extra).",
        "k_high": "🟡 **Potassium high** — reduce potash application.",
        "k_ok": "✅ **Potassium** is already in a good range.",
        "feat_N": "Nitrogen", "feat_P": "Phosphorus", "feat_K": "Potassium", "feat_temperature": "Temperature",
        "feat_humidity": "Humidity", "feat_ph": "Soil pH", "feat_rainfall": "Rainfall",
        "your_value": "Your value",
        "coach_water_more": "💧 Your rainfall/humidity is on the lower side for {crop} — plan for extra irrigation.",
        "coach_water_less": "💧 Your rainfall/humidity is on the higher side for {crop} — watch out for waterlogging and ensure drainage.",
        "coach_water_ok": "💧 Your water conditions look well matched for {crop}.",
        "coach_ph_off": "⚗️ Your soil pH ({v}) is outside {crop}'s ideal range ({lo}–{hi}) — consider lime (to raise pH) or sulfur (to lower it) after a proper soil test.",
        "coach_ph_ok": "⚗️ Your soil pH ({v}) suits {crop} well.",
        "coach_temp_off": "🌡️ Your temperature ({v}°C) is outside {crop}'s ideal range ({lo}–{hi}°C) for this season.",
        "coach_temp_ok": "🌡️ Your temperature ({v}°C) suits {crop} well.",
        "coach_pest": "🐛 For any crop: inspect leaves/stems weekly, remove affected parts early, rotate crops each season, and consult your local Krishi Vigyan Kendra / agri office before using any pesticide.",
        "coach_harvest": "🌾 Harvest timing depends on the crop variety and local season calendar — your nearest agriculture extension office can give the exact days-to-harvest for {crop} in your region.",
    },
    "hi": {
        "title": "🌾 स्मार्ट फसल सिफारिश प्रणाली",
        "subtitle": "नीचे अपने खेत की मिट्टी और मौसम की जानकारी भरें, हम आपके लिए सबसे अच्छी फसल सुझाएंगे।",
        "soil_weather": "🧪 मिट्टी और मौसम का विवरण",
        "n": "🌿 नाइट्रोजन (N)", "p": "🧬 फॉस्फोरस (P)", "k": "🧂 पोटैशियम (K)", "ph": "⚗️ मिट्टी का pH",
        "temp": "🌡️ तापमान (°C)", "humidity": "💧 आर्द्रता (%)", "rainfall": "🌧️ वर्षा (mm)",
        "predict_btn": "🔍 सबसे अच्छी फसल जानें",
        "analyzing": "आपकी मिट्टी और मौसम के आंकड़ों का विश्लेषण किया जा रहा है...",
        "best_crop": "सबसे अच्छी सुझाई गई फसल:",
        "confidence": "विश्वसनीयता",
        "quick_fert_tip": "🧪 त्वरित खाद सुझाव",
        "show_profile": "📋 {crop} की पूरी जानकारी और खाद सलाह देखें",
        "overall_suit": "आपके इनपुट के अनुसार कुल उपयुक्तता:",
        "value_vs_ideal": "📊 आपका मान बनाम आदर्श सीमा",
        "fert_reco": "🧪 खाद सुझाव",
        "fert_note": "यह केवल सामान्य मार्गदर्शन है — सटीक मात्रा के लिए मिट्टी जांच / स्थानीय कृषि कार्यालय से पुष्टि करें।",
        "conditions_vs_ideal": "📈 आपकी परिस्थितियाँ बनाम {crop} के लिए आदर्श स्थिति",
        "confidence_gauge": "🎯 भविष्यवाणी की विश्वसनीयता",
        "match_overview": "🌱 सभी {n} फसलें — मिलान अवलोकन",
        "match_hint": "किसी भी फसल की पूरी जानकारी, खाद सलाह और ग्राफ देखने के लिए **Details** पर क्लिक करें — वो उसी के नीचे खुलेगी।",
        "details_btn": "विवरण — {crop}",
        "match_pct": "{v:.1f}% मिलान",
        "conf_chart": "📊 विश्वसनीयता चार्ट — सभी फसलें",
        "full_table": "📋 पूरी विवरण तालिका",
        "col_rank": "क्रम", "col_crop": "फसल", "col_icon": "चिन्ह", "col_conf": "विश्वसनीयता (%)",
        "nav_predict": "🔍 अनुमान",
        "nav_coach": "🧑‍🌾 क्रॉप कोच",
        "nav_encyclopedia": "🌿 फसल जानकारी",
        "nav_insights": "📊 जानकारियाँ",
        "nav_whatif": "🧪 व्हाट-इफ",
        "sidebar_lang": "🌐 Language / भाषा",
        "sidebar_enc": "🌿 फसल जानकारी",
        "enc_no_prediction": "यहाँ पूरा मिलान अवलोकन, चार्ट और तालिका देखने के लिए पहले **Predict** पेज पर जाकर अनुमान लगाएं।",
        "reset_btn": "🔄 इनपुट रीसेट करें",
        "download_csv": "📥 पूरी तालिका डाउनलोड करें (CSV)",
        "share_label": "🔗 यह परिणाम शेयर करें (कॉपी आइकन दबाएं)",
        "print_tip": "🖨️ PDF सेव करने के लिए: कंप्यूटर पर **Ctrl+P** दबाएं और 'Save as PDF' चुनें। मोबाइल पर ब्राउज़र मेन्यू → **Share / Print** → 'Save as PDF' चुनें।",
        "share_text": "स्मार्ट फसल सिफारिश परिणाम\\nसर्वश्रेष्ठ फसल: {crop}\\nविश्वसनीयता: {conf:.2f}%\\nइनपुट — N:{n}, P:{p}, K:{k}, तापमान:{temp}°C, आर्द्रता:{hum}%, pH:{ph}, वर्षा:{rain}mm",
        "insights_title": "📊 जानकारियाँ और अतिरिक्त सुविधाएं",
        "tab_history": "📜 इतिहास",
        "tab_feat_imp": "🧠 यह मॉडल कैसे काम करता है",
        "tab_market": "💰 बाज़ार भाव (अनुमानित)",
        "tab_seasonal": "📅 मौसमी कैलेंडर",
        "no_history": "इस सेशन में अभी तक कोई अनुमान नहीं लगाया गया। **Predict** पेज पर जाकर एक लगाएं।",
        "history_time": "समय", "history_crop": "फसल", "history_conf": "विश्वसनीयता (%)",
        "clear_history": "🗑️ इतिहास साफ़ करें",
        "feat_imp_desc": "यह चार्ट दिखाता है कि हर फीचर (औसतन) Random Forest मॉडल के फैसलों को सभी 22 फसलों में कितना प्रभावित करता है। ऊंची बार वाले फीचर अनुमान में ज़्यादा मायने रखते हैं।",
        "market_note": "⚠️ ये केवल अनुमानित मूल्य सीमा हैं (प्रति क्विंटल) — निर्णय लेने से पहले हमेशा अपनी स्थानीय मंडी का भाव जांचें।",
        "col_price": "अनुमानित मूल्य सीमा (₹/क्विंटल)",
        "seasonal_note": "फसलें उनके सामान्य भारतीय बुवाई मौसम के अनुसार समूहीकृत (अनुमानित)।",
        "season_kharif": "☔ खरीफ (जून–अक्टूबर)", "season_rabi": "❄️ रबी (अक्टूबर–मार्च)",
        "season_zaid": "☀️ ज़ायद / गर्मी (मार्च–जून)", "season_year": "🔁 साल भर",
        "whatif_title": "🧪 व्हाट-इफ सिम्युलेटर",
        "whatif_sub": "स्लाइडर हिलाएं और तुरंत अनुमान बदलते हुए देखें — किसी बटन की ज़रूरत नहीं। यह समझने के लिए बेहतरीन कि हर कारक सिफारिश को कैसे प्रभावित करता है।",
        "whatif_top_crop": "लाइव सर्वश्रेष्ठ फसल:",
        "sidebar_lookup": "फसल खोजें",
        "sidebar_ideal": "आदर्श उगाने की परिस्थितियाँ (अनुमानित)",
        "coach_header": "🧑‍🌾 क्रॉप कोच",
        "coach_sub": "एक मुफ़्त, बिल्ट-इन खेती सहायक — किसी साइन-अप या API key की ज़रूरत नहीं।",
        "coach_greeting": "नमस्ते! आपके इनपुट के आधार पर, **{crop}** उगाने के लिए मेरी सलाह:",
        "coach_no_prediction": "⚠️ कृपया पहले **Predict** पेज पर जाकर अनुमान लगाएं, ताकि Crop Coach को पता चले कि किस फसल के बारे में सलाह देनी है।",
        "coach_ask_label": "खाद, पानी, मिट्टी, मौसम, कीट या कटाई के बारे में क्रॉप कोच से पूछें:",
        "coach_ask_placeholder": "जैसे: इस फसल को कितने पानी की ज़रूरत है?",
        "coach_ask_btn": "पूछें",
        "coach_fallback": "मैं {crop} के लिए इनमें मदद कर सकता हूँ: **खाद**, **पानी/सिंचाई**, **मिट्टी pH**, **मौसम**, **कीट/रोग**, और **कटाई**। इनमें से कोई पूछ के देखिए!",
        "warn_temp": "⚠️ {v}°C तापमान असामान्य रूप से अधिक/कम है — कृपया मान दोबारा जांचें।",
        "warn_humidity": "⚠️ {v}% आर्द्रता बहुत कम है — कृपया मान दोबारा जांचें।",
        "warn_rainfall": "⚠️ {v}mm वर्षा ज़्यादातर फसलों के लिए बहुत कम है — कृपया मान दोबारा जांचें।",
        "warn_ph": "⚠️ मिट्टी का pH {v} सामान्य 4.5–9 की खेती सीमा से बहुत दूर है — कृपया मान दोबारा जांचें।",
        "n_low": "🟢 **नाइट्रोजन कम है** — कमी पूरी करने के लिए **यूरिया** डालें (~{amt:.0f} किग्रा/हेक्टेयर अतिरिक्त)।",
        "n_high": "🟡 **नाइट्रोजन ज़्यादा है** — इस सीज़न यूरिया/नाइट्रोजन युक्त खाद कम करें।",
        "n_ok": "✅ **नाइट्रोजन** पहले से ही सही मात्रा में है।",
        "p_low": "🟢 **फॉस्फोरस कम है** — **DAP / SSP** डालें (~{amt:.0f} किग्रा/हेक्टेयर अतिरिक्त)।",
        "p_high": "🟡 **फॉस्फोरस ज़्यादा है** — फॉस्फेट खाद कम करें।",
        "p_ok": "✅ **फॉस्फोरस** पहले से ही सही मात्रा में है।",
        "k_low": "🟢 **पोटैशियम कम है** — **MOP (म्यूरेट ऑफ पोटाश)** डालें (~{amt:.0f} किग्रा/हेक्टेयर अतिरिक्त)।",
        "k_high": "🟡 **पोटैशियम ज़्यादा है** — पोटाश खाद कम करें।",
        "k_ok": "✅ **पोटैशियम** पहले से ही सही मात्रा में है।",
        "feat_N": "नाइट्रोजन", "feat_P": "फॉस्फोरस", "feat_K": "पोटैशियम", "feat_temperature": "तापमान",
        "feat_humidity": "आर्द्रता", "feat_ph": "मिट्टी pH", "feat_rainfall": "वर्षा",
        "your_value": "आपका मान",
        "coach_water_more": "💧 {crop} के लिए आपकी वर्षा/आर्द्रता थोड़ी कम है — अतिरिक्त सिंचाई की योजना बनाएं।",
        "coach_water_less": "💧 {crop} के लिए आपकी वर्षा/आर्द्रता थोड़ी ज़्यादा है — जलभराव से बचें और जल निकासी सुनिश्चित करें।",
        "coach_water_ok": "💧 आपकी पानी की स्थिति {crop} के लिए अच्छी तरह मेल खाती है।",
        "coach_ph_off": "⚗️ आपकी मिट्टी का pH ({v}) {crop} की आदर्श सीमा ({lo}–{hi}) से बाहर है — उचित मिट्टी जांच के बाद चूना (pH बढ़ाने) या सल्फर (pH घटाने) पर विचार करें।",
        "coach_ph_ok": "⚗️ आपकी मिट्टी का pH ({v}) {crop} के लिए उपयुक्त है।",
        "coach_temp_off": "🌡️ आपका तापमान ({v}°C) इस मौसम में {crop} की आदर्श सीमा ({lo}–{hi}°C) से बाहर है।",
        "coach_temp_ok": "🌡️ आपका तापमान ({v}°C) {crop} के लिए उपयुक्त है।",
        "coach_pest": "🐛 किसी भी फसल के लिए: हर हफ्ते पत्तियों/तनों की जांच करें, प्रभावित हिस्सों को जल्दी हटाएं, हर सीज़न फसल चक्र अपनाएं, और कोई भी कीटनाशक इस्तेमाल करने से पहले अपने नज़दीकी कृषि विज्ञान केंद्र / कृषि कार्यालय से सलाह लें।",
        "coach_harvest": "🌾 कटाई का समय फसल की किस्म और स्थानीय मौसम कैलेंडर पर निर्भर करता है — आपका नज़दीकी कृषि विस्तार कार्यालय आपके क्षेत्र में {crop} के लिए सटीक दिन बता सकता है।",
    },
}

crop_data = {
    'rice':        {'icon': '🌾', 'ideal': {'N': (60, 100), 'P': (35, 60), 'K': (35, 45), 'temperature': (20, 35), 'humidity': (70, 90), 'ph': (5.5, 7.0), 'rainfall': (150, 300)},
                     'desc': {'en': "A staple cereal that thrives in flooded fields with high humidity.", 'hi': "एक मुख्य अनाज जो जलभराव वाले खेतों और उच्च आर्द्रता में अच्छी तरह उगता है।"}},
    'maize':       {'icon': '🌽', 'ideal': {'N': (60, 100), 'P': (35, 60), 'K': (15, 25), 'temperature': (18, 27), 'humidity': (50, 75), 'ph': (5.5, 7.5), 'rainfall': (60, 110)},
                     'desc': {'en': "A versatile cereal grown widely as food, feed, and industrial raw material.", 'hi': "एक बहुउपयोगी अनाज जिसे भोजन, चारे और औद्योगिक उपयोग के लिए उगाया जाता है।"}},
    'jute':        {'icon': '🧵', 'ideal': {'N': (60, 100), 'P': (35, 60), 'K': (35, 45), 'temperature': (24, 35), 'humidity': (70, 90), 'ph': (6.0, 7.5), 'rainfall': (150, 200)},
                     'desc': {'en': "A fibre crop grown in warm, humid conditions, mainly for making sacking and rope.", 'hi': "एक रेशेदार फसल जो गर्म, आर्द्र परिस्थितियों में बोरी और रस्सी बनाने के लिए उगाई जाती है।"}},
    'cotton':      {'icon': '☁️', 'ideal': {'N': (100, 140), 'P': (35, 60), 'K': (15, 25), 'temperature': (22, 30), 'humidity': (60, 85), 'ph': (5.5, 8.0), 'rainfall': (60, 110)},
                     'desc': {'en': "A fibre crop that needs a long, warm growing season and moderate rainfall.", 'hi': "एक रेशेदार फसल जिसे लंबे, गर्म मौसम और मध्यम वर्षा की आवश्यकता होती है।"}},
    'coconut':     {'icon': '🥥', 'ideal': {'N': (5, 30), 'P': (5, 30), 'K': (25, 35), 'temperature': (25, 30), 'humidity': (90, 100), 'ph': (5.5, 7.0), 'rainfall': (130, 230)},
                     'desc': {'en': "A tropical palm that loves sandy soil, high humidity and coastal climates.", 'hi': "एक उष्णकटिबंधीय ताड़ जो रेतीली मिट्टी, उच्च आर्द्रता और तटीय जलवायु पसंद करता है।"}},
    'papaya':      {'icon': '🍑', 'ideal': {'N': (30, 70), 'P': (45, 70), 'K': (45, 55), 'temperature': (23, 35), 'humidity': (85, 95), 'ph': (6.0, 7.0), 'rainfall': (40, 150)},
                     'desc': {'en': "A fast-growing fruit tree that prefers warm weather and well-drained soil.", 'hi': "तेज़ी से बढ़ने वाला फलदार पेड़ जो गर्म मौसम और अच्छी जल निकासी वाली मिट्टी पसंद करता है।"}},
    'orange':      {'icon': '🍊', 'ideal': {'N': (5, 30), 'P': (5, 30), 'K': (5, 15), 'temperature': (10, 30), 'humidity': (90, 100), 'ph': (6.0, 8.0), 'rainfall': (100, 120)},
                     'desc': {'en': "A citrus fruit that needs mild winters and well-drained, slightly acidic soil.", 'hi': "एक खट्टे फल जिसे हल्की सर्दी और अच्छी जल निकासी वाली, थोड़ी अम्लीय मिट्टी चाहिए।"}},
    'apple':       {'icon': '🍎', 'ideal': {'N': (0, 40), 'P': (120, 145), 'K': (195, 205), 'temperature': (15, 25), 'humidity': (90, 100), 'ph': (5.5, 6.5), 'rainfall': (100, 120)},
                     'desc': {'en': "A temperate fruit tree that needs cold winters to set fruit well.", 'hi': "एक समशीतोष्ण फलदार पेड़ जिसे अच्छी तरह फल देने के लिए ठंडी सर्दियों की आवश्यकता होती है।"}},
    'muskmelon':   {'icon': '🍈', 'ideal': {'N': (90, 100), 'P': (5, 25), 'K': (45, 55), 'temperature': (25, 35), 'humidity': (85, 95), 'ph': (6.0, 7.0), 'rainfall': (20, 30)},
                     'desc': {'en': "A warm-season melon that needs hot days and low humidity to ripen sweetly.", 'hi': "एक गर्म मौसम का खरबूजा जिसे मीठा पकने के लिए गर्म दिन और कम आर्द्रता चाहिए।"}},
    'watermelon':  {'icon': '🍉', 'ideal': {'N': (90, 100), 'P': (5, 25), 'K': (45, 55), 'temperature': (22, 32), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (40, 50)},
                     'desc': {'en': "A sprawling summer vine crop that needs warm soil and consistent moisture.", 'hi': "गर्मियों की एक फैलने वाली बेल फसल जिसे गर्म मिट्टी और लगातार नमी चाहिए।"}},
    'grapes':      {'icon': '🍇', 'ideal': {'N': (0, 40), 'P': (120, 145), 'K': (195, 205), 'temperature': (10, 40), 'humidity': (80, 90), 'ph': (5.5, 6.5), 'rainfall': (60, 70)},
                     'desc': {'en': "A vine fruit that does best with warm days, cool nights and good drainage.", 'hi': "एक बेल फल जो गर्म दिनों, ठंडी रातों और अच्छी जल निकासी में सबसे अच्छा होता है।"}},
    'mango':       {'icon': '🥭', 'ideal': {'N': (0, 40), 'P': (15, 40), 'K': (25, 35), 'temperature': (27, 38), 'humidity': (45, 55), 'ph': (5.5, 7.5), 'rainfall': (80, 100)},
                     'desc': {'en': "A tropical fruit tree that needs a distinct dry season to flower well.", 'hi': "एक उष्णकटिबंधीय फलदार पेड़ जिसे अच्छी तरह फूलने के लिए स्पष्ट शुष्क मौसम चाहिए।"}},
    'banana':      {'icon': '🍌', 'ideal': {'N': (80, 120), 'P': (70, 100), 'K': (45, 55), 'temperature': (25, 30), 'humidity': (75, 85), 'ph': (5.5, 6.5), 'rainfall': (90, 120)},
                     'desc': {'en': "A tropical fruit crop that needs rich soil, warmth and steady moisture.", 'hi': "एक उष्णकटिबंधीय फल फसल जिसे उपजाऊ मिट्टी, गर्मी और लगातार नमी चाहिए।"}},
    'pomegranate': {'icon': '🍒', 'ideal': {'N': (0, 40), 'P': (5, 30), 'K': (35, 45), 'temperature': (18, 25), 'humidity': (85, 95), 'ph': (5.5, 7.2), 'rainfall': (100, 110)},
                     'desc': {'en': "A hardy fruit shrub that tolerates drought once established.", 'hi': "एक मजबूत फलदार झाड़ी जो स्थापित होने के बाद सूखे को सहन करती है।"}},
    'lentil':      {'icon': '🟤', 'ideal': {'N': (0, 30), 'P': (60, 80), 'K': (15, 25), 'temperature': (18, 30), 'humidity': (60, 70), 'ph': (5.5, 7.5), 'rainfall': (40, 50)},
                     'desc': {'en': "A cool-season pulse crop that fixes nitrogen and needs little water.", 'hi': "एक ठंडे मौसम की दलहन फसल जो नाइट्रोजन स्थिर करती है और कम पानी चाहती है।"}},
    'blackgram':   {'icon': '⚫', 'ideal': {'N': (20, 60), 'P': (55, 75), 'K': (15, 25), 'temperature': (25, 35), 'humidity': (60, 70), 'ph': (6.0, 7.5), 'rainfall': (60, 70)},
                     'desc': {'en': "A short-duration pulse suited to warm climates and light soils.", 'hi': "एक कम अवधि की दलहन फसल जो गर्म जलवायु और हल्की मिट्टी के लिए उपयुक्त है।"}},
    'mungbean':    {'icon': '🟢', 'ideal': {'N': (0, 40), 'P': (30, 70), 'K': (35, 55), 'temperature': (27, 32), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (40, 60)},
                     'desc': {'en': "A fast-growing pulse that prefers warm, humid conditions.", 'hi': "एक तेज़ी से बढ़ने वाली दलहन फसल जो गर्म, आर्द्र परिस्थितियों को पसंद करती है।"}},
    'mothbeans':   {'icon': '🟡', 'ideal': {'N': (0, 40), 'P': (30, 60), 'K': (15, 25), 'temperature': (24, 32), 'humidity': (40, 65), 'ph': (3.5, 9.0), 'rainfall': (25, 65)},
                     'desc': {'en': "A drought-tolerant pulse that grows well in arid, sandy soils.", 'hi': "एक सूखा-सहनशील दलहन फसल जो शुष्क, रेतीली मिट्टी में अच्छी तरह उगती है।"}},
    'pigeonpeas':  {'icon': '🟠', 'ideal': {'N': (0, 40), 'P': (55, 80), 'K': (15, 25), 'temperature': (18, 37), 'humidity': (30, 70), 'ph': (4.5, 7.5), 'rainfall': (90, 200)},
                     'desc': {'en': "A deep-rooted pulse that copes well with poor, dry soils.", 'hi': "एक गहरी जड़ों वाली दलहन फसल जो खराब, शुष्क मिट्टी में भी अच्छी तरह टिकती है।"}},
    'kidneybeans': {'icon': '🔴', 'ideal': {'N': (0, 40), 'P': (55, 80), 'K': (15, 25), 'temperature': (15, 25), 'humidity': (18, 25), 'ph': (5.5, 6.0), 'rainfall': (60, 150)},
                     'desc': {'en': "A cool-season pulse that dislikes waterlogging and high humidity.", 'hi': "एक ठंडे मौसम की दलहन फसल जिसे जलभराव और उच्च आर्द्रता पसंद नहीं है।"}},
    'chickpea':    {'icon': '🟨', 'ideal': {'N': (20, 60), 'P': (55, 80), 'K': (75, 85), 'temperature': (17, 25), 'humidity': (14, 20), 'ph': (5.5, 8.0), 'rainfall': (65, 105)},
                     'desc': {'en': "A drought-tolerant pulse best suited to cool, dry growing seasons.", 'hi': "एक सूखा-सहनशील दलहन फसल जो ठंडे, शुष्क मौसम के लिए सबसे उपयुक्त है।"}},
    'coffee':      {'icon': '☕', 'ideal': {'N': (80, 120), 'P': (15, 40), 'K': (25, 35), 'temperature': (23, 28), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (150, 200)},
                     'desc': {'en': "A shade-loving shrub that needs a warm climate and steady rainfall.", 'hi': "एक छाया-पसंद झाड़ी जिसे गर्म जलवायु और लगातार वर्षा चाहिए।"}},
}
crop_icons = {c: v['icon'] for c, v in crop_data.items()}

# Illustrative, approximate reference data — NOT live/real-time. For real
# decisions, check the local mandi and the official sowing calendar.
CROP_SEASON = {
    'rice': 'kharif', 'maize': 'kharif', 'jute': 'kharif', 'cotton': 'kharif',
    'mungbean': 'kharif', 'mothbeans': 'kharif', 'pigeonpeas': 'kharif', 'blackgram': 'kharif',
    'lentil': 'rabi', 'chickpea': 'rabi', 'kidneybeans': 'rabi', 'apple': 'rabi', 'grapes': 'rabi',
    'muskmelon': 'zaid', 'watermelon': 'zaid',
    'coconut': 'year', 'papaya': 'year', 'orange': 'year', 'mango': 'year', 'banana': 'year',
    'pomegranate': 'year', 'coffee': 'year',
}
CROP_PRICE = {
    'rice': (1800, 2200), 'maize': (1700, 2100), 'jute': (4500, 5200), 'cotton': (6000, 7200),
    'coconut': (2000, 3500), 'papaya': (800, 1500), 'orange': (2500, 4500), 'apple': (4000, 8000),
    'muskmelon': (900, 1800), 'watermelon': (500, 1200), 'grapes': (3000, 6000), 'mango': (2000, 5000),
    'banana': (800, 1800), 'pomegranate': (5000, 9000), 'lentil': (6000, 7500), 'blackgram': (6500, 8000),
    'mungbean': (7000, 8500), 'mothbeans': (5500, 7000), 'pigeonpeas': (6500, 8000), 'kidneybeans': (7000, 9500),
    'chickpea': (4800, 5800), 'coffee': (8000, 15000),
}

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
FEATURE_SCALE = {'N': (0, 140), 'P': (0, 145), 'K': (0, 205), 'temperature': (0, 50),
                  'humidity': (0, 100), 'ph': (0, 14), 'rainfall': (0, 300)}
ASSETS_DIR = "assets"

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'page' not in st.session_state:
    st.session_state.page = 'predict'


def crop_image_path(crop):
    for ext in ("jpg", "jpeg", "png", "webp"):
        path = os.path.join(ASSETS_DIR, f"{crop}.{ext}")
        if os.path.exists(path):
            return path
    return None


def fertilizer_advice(t, crop, n, p, k):
    ideal = crop_data[crop]['ideal']
    tips = []
    n_lo, n_hi = ideal['N']
    p_lo, p_hi = ideal['P']
    k_lo, k_hi = ideal['K']
    tips.append(t['n_low'].format(amt=(n_lo - n) * 2) if n < n_lo else (t['n_high'] if n > n_hi else t['n_ok']))
    tips.append(t['p_low'].format(amt=(p_lo - p) * 2) if p < p_lo else (t['p_high'] if p > p_hi else t['p_ok']))
    tips.append(t['k_low'].format(amt=(k_lo - k) * 2) if k < k_lo else (t['k_high'] if k > k_hi else t['k_ok']))
    return tips


def match_score(crop, inp):
    ideal = crop_data[crop]['ideal']
    hits = sum(1 for f in FEATURES if ideal[f][0] <= inp[f] <= ideal[f][1])
    return hits / len(FEATURES)


def feature_bar_html(t, feature, ideal_lo, ideal_hi, user_val):
    scale_lo, scale_hi = FEATURE_SCALE[feature]
    span = max(scale_hi - scale_lo, 1e-6)
    left_pct = max(0, min(100, (ideal_lo - scale_lo) / span * 100))
    width_pct = max(1, min(100 - left_pct, (ideal_hi - ideal_lo) / span * 100))
    user_pct = max(0, min(100, (user_val - scale_lo) / span * 100))
    in_range = ideal_lo <= user_val <= ideal_hi
    marker_color = "#1f7a3d" if in_range else "#a83232"
    return f"""
    <div style="margin-bottom:9px;">
      <div style="display:flex; justify-content:space-between; font-size:12px; color:#5a4632; margin-bottom:2px;">
        <span>{t['feat_' + feature]}</span><span>{t['your_value']}: {user_val:g}</span>
      </div>
      <div style="position:relative; height:10px; background:#e2d9c3; border-radius:5px;">
        <div style="position:absolute; left:{left_pct:.1f}%; width:{width_pct:.1f}%; height:100%; background:#5b8fae; opacity:0.55; border-radius:5px;"></div>
        <div style="position:absolute; left:calc({user_pct:.1f}% - 2px); width:4px; height:14px; top:-2px; background:{marker_color}; border-radius:2px; transition: left 0.4s ease;"></div>
      </div>
    </div>
    """


def render_crop_detail(t, crop, inp):
    info = crop_data[crop]
    score = match_score(crop, inp)
    img_path = crop_image_path(crop)

    top_l, top_r = st.columns([1, 2])
    with top_l:
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.markdown(f"<div style='font-size:70px; text-align:center;'>{info['icon']}</div>", unsafe_allow_html=True)
    with top_r:
        st.markdown(f"**{crop.capitalize()}**")
        st.write(info['desc'][st.session_state.lang])
        st.markdown(f"**{t['overall_suit']}** {score*100:.0f}%")
        st.markdown(f"""<div class="suit-bar-bg"><div class="suit-bar-fill" style="width:{score*100:.0f}%"></div></div>""", unsafe_allow_html=True)

    st.write("")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**{t['value_vs_ideal']}**")
        for f in FEATURES:
            lo, hi = info['ideal'][f]
            st.markdown(feature_bar_html(t, f, lo, hi, inp[f]), unsafe_allow_html=True)
    with d2:
        st.markdown(f"**{t['fert_reco']}**")
        for tip in fertilizer_advice(t, crop, inp['N'], inp['P'], inp['K']):
            st.markdown(f"- {tip}")
        st.caption(t['fert_note'])


def crop_coach_reply(t, msg, crop, inp):
    msg_l = msg.lower()
    for c in crop_data.keys():
        if c in msg_l:
            crop = c
            break
    ideal = crop_data[crop]['ideal']

    if any(w in msg_l for w in ["fertil", "khad", "urea", "dap", "mop", "nutri", "उर्वरक", "खाद"]):
        return "\n\n".join(fertilizer_advice(t, crop, inp['N'], inp['P'], inp['K']))
    if any(w in msg_l for w in ["water", "irrigat", "rain", "पानी", "सिंचाई", "वर्षा"]):
        lo, hi = ideal['humidity']
        if inp['humidity'] < lo:
            return t['coach_water_more'].format(crop=crop.capitalize())
        elif inp['humidity'] > hi:
            return t['coach_water_less'].format(crop=crop.capitalize())
        return t['coach_water_ok'].format(crop=crop.capitalize())
    if any(w in msg_l for w in ["soil", "ph", "मिट्टी"]):
        lo, hi = ideal['ph']
        if lo <= inp['ph'] <= hi:
            return t['coach_ph_ok'].format(v=inp['ph'], crop=crop.capitalize())
        return t['coach_ph_off'].format(v=inp['ph'], crop=crop.capitalize(), lo=lo, hi=hi)
    if any(w in msg_l for w in ["weather", "temp", "climate", "मौसम", "तापमान"]):
        lo, hi = ideal['temperature']
        if lo <= inp['temperature'] <= hi:
            return t['coach_temp_ok'].format(v=inp['temperature'], crop=crop.capitalize())
        return t['coach_temp_off'].format(v=inp['temperature'], crop=crop.capitalize(), lo=lo, hi=hi)
    if any(w in msg_l for w in ["pest", "disease", "insect", "bug", "कीट", "रोग", "बीमारी"]):
        return t['coach_pest']
    if any(w in msg_l for w in ["harvest", "yield", "cut", "कटाई", "पैदावार"]):
        return t['coach_harvest'].format(crop=crop.capitalize())
    return t['coach_fallback'].format(crop=crop.capitalize())


def input_warnings(t, inp):
    warns = []
    if inp['temperature'] < 5 or inp['temperature'] > 45:
        warns.append(t['warn_temp'].format(v=inp['temperature']))
    if inp['humidity'] < 15:
        warns.append(t['warn_humidity'].format(v=inp['humidity']))
    if inp['rainfall'] < 10:
        warns.append(t['warn_rainfall'].format(v=inp['rainfall']))
    if inp['ph'] < 4.5 or inp['ph'] > 9:
        warns.append(t['warn_ph'].format(v=inp['ph']))
    return warns


# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lato:wght@400;600&display=swap');

@keyframes fadeIn { from {opacity:0; transform: translateY(10px);} to {opacity:1; transform: translateY(0);} }
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 0 rgba(201,162,39,0.45); }
    70% { box-shadow: 0 0 0 14px rgba(201,162,39,0); }
    100% { box-shadow: 0 0 0 0 rgba(201,162,39,0); }
}

html, body, [class*="css"] { font-family: 'Lato', sans-serif; }
h1, h2, h3, .main-title, .crop-title, .hero-sub { font-family: 'Playfair Display', 'Georgia', serif; }

.stApp { background: radial-gradient(circle at top left, #16261d 0%, #0e1712 55%, #0b120e 100%); }

/* Make all headings/subheaders/captions clearly readable on the dark background */
section.main h1, section.main h2, section.main h3, section.main h4 {
    color: #f5f0e1 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}
section.main p, section.main label, section.main span,
section.main li, section.main strong, section.main .stMarkdown,
section.main div[data-testid="stMarkdownContainer"] * {
    color: #efe9d6 !important;
    opacity: 1 !important;
}
div[data-testid="stCaptionContainer"], section.main small { color: #cfc7ae !important; opacity: 1 !important; }

/* ---- Top nav bar ---- */
.topnav-wrap {
    display: flex;
    justify-content: center;
    gap: 14px;
    padding: 14px 10px;
    margin-bottom: 18px;
    border-bottom: 2px solid #c9a227;
    animation: fadeIn 0.6s ease-out;
}

.hero-banner {
    text-align: center;
    padding: 34px 20px 22px 20px;
    border-bottom: 3px solid #c9a227;
    margin-bottom: 22px;
    background:
        radial-gradient(circle at 15% 30%, rgba(201,162,39,0.10) 0%, transparent 45%),
        radial-gradient(circle at 85% 70%, rgba(139,94,60,0.12) 0%, transparent 45%),
        linear-gradient(180deg, rgba(31,61,43,0.35), transparent);
    border-radius: 14px;
    animation: fadeIn 0.7s ease-out;
}
.hero-banner h1 { font-size: 42px; color: #f5f0e1; margin-bottom: 8px; letter-spacing: 0.5px;
                   text-shadow: 0 2px 10px rgba(0,0,0,0.4); }
.hero-banner p { color: #d8cdb0; font-size: 15.5px; }

.input-card {
    background: linear-gradient(160deg, #182920 0%, #131f18 100%);
    border: 1px solid #3a4d3f;
    border-radius: 14px;
    padding: 22px 26px 10px 26px;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
    animation: fadeIn 0.8s ease-out;
}
.input-card h4 { color: #c9a227; margin-top: 0; font-size: 19px; border-bottom: 1px solid #3a4d3f; padding-bottom: 10px; }

.crop-card { background: #f5f0e1; border: 1px solid #8b5e3c; border-radius: 10px; padding: 16px 10px; text-align: center;
             color: #2f3e2f; margin-bottom: 6px; box-shadow: 0 3px 8px rgba(0,0,0,0.35);
             transition: transform 0.2s ease, box-shadow 0.2s ease; animation: fadeIn 0.5s ease-out; }
.crop-card:hover { transform: translateY(-5px) scale(1.03); box-shadow: 0 8px 18px rgba(201,162,39,0.4); }
.crop-icon { font-size: 34px; }
.crop-name { font-size: 15px; font-weight: 700; margin-top: 6px; }
.crop-conf-chip { display:inline-block; margin-top:4px; font-size: 11.5px; font-weight:700; color:#fff;
                   background:#5b8fae; padding:2px 9px; border-radius:12px; }

.top-card { background: linear-gradient(160deg, #1f3d2b 0%, #16291c 100%); border: 2px solid #c9a227; border-radius: 12px;
            padding: 28px; text-align: center; color: #f5f0e1; box-shadow: 0 6px 18px rgba(0,0,0,0.45);
            animation: fadeIn 0.6s ease-out, pulseGlow 2.5s ease-out 1; }
.top-card .crop-title { font-size: 26px; font-weight: 700; margin-top: 8px; letter-spacing: 0.5px; }

.fert-quick { background: #22301f; border-left: 5px solid #c9a227; border-radius: 8px; padding: 14px 18px; margin-top: 14px; color: #f5f0e1; animation: fadeIn 0.6s ease-out; }

.coach-card { background: #182920; border: 1px solid #5b8fae; border-radius: 12px; padding: 18px 22px; margin-top: 18px; color: #f5f0e1; animation: fadeIn 0.7s ease-out; }
.coach-card h3 { color: #c9a227; margin-top: 0; }
.coach-reply { background: #0e1712; border-left: 4px solid #5b8fae; border-radius: 8px; padding: 12px 16px; margin-top: 10px; color: #e6e2d3; animation: fadeIn 0.4s ease-out; }

.suit-bar-bg { background: #e2d9c3; border-radius: 5px; height: 12px; width: 100%; overflow: hidden; }
.suit-bar-fill { background: linear-gradient(90deg, #8b5e3c, #1f3d2b); border-radius: 5px; height: 12px; transition: width 0.6s ease; }

div.stButton > button {
    border: 1px solid #c9a227;
    border-radius: 8px;
    background: linear-gradient(135deg, #1f3d2b, #16291c);
    color: #f5f0e1;
    font-weight: 700;
    padding: 10px 0;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #c9a227, #a9744f);
    color: #16291c;
    border: 1px solid #c9a227;
    box-shadow: 0 4px 14px rgba(201,162,39,0.35);
    transform: translateY(-2px);
}

/* active nav button look */
.nav-active button {
    background: linear-gradient(135deg, #c9a227, #a9744f) !important;
    color: #16291c !important;
}

div[data-testid="stExpander"] { background: #faf7ef; border: 1px solid #8b5e3c; border-radius: 10px; margin-top: 6px; }
div[data-testid="stExpander"] summary { font-weight: 700; color: #1f3d2b; }

@media print {
    .topnav-wrap, section[data-testid="stSidebar"], div.stButton { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
for key, default in [('prediction_made', False), ('pred_crop', None), ('pred_probs', None),
                      ('pred_classes', None), ('last_inputs', None), ('coach_qa', []), ('history', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

def reset_inputs():
    """Callback: runs BEFORE the page re-renders, so widget values can be reset safely."""
    st.session_state["in_n"] = 50
    st.session_state["in_p"] = 50
    st.session_state["in_k"] = 50
    st.session_state["in_ph"] = 6.5
    st.session_state["in_temp"] = 25.0
    st.session_state["in_hum"] = 60.0
    st.session_state["in_rain"] = 100.0


t = T[st.session_state.lang]

# ---------------------------------------------------------------------------
# TOP NAVIGATION — replaces the sidebar. Icons across the top switch pages.
# ---------------------------------------------------------------------------
nav_l, nav_m1, nav_m2, nav_m3, nav_m4, nav_m5, nav_r = st.columns([1, 2, 2, 2, 2, 2, 1.3])
with nav_m1:
    cls = "nav-active" if st.session_state.page == "predict" else ""
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button(t['nav_predict'], use_container_width=True, key="nav_predict_btn"):
        st.session_state.page = "predict"
    st.markdown('</div>', unsafe_allow_html=True)
with nav_m2:
    cls = "nav-active" if st.session_state.page == "coach" else ""
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button(t['nav_coach'], use_container_width=True, key="nav_coach_btn"):
        st.session_state.page = "coach"
    st.markdown('</div>', unsafe_allow_html=True)
with nav_m3:
    cls = "nav-active" if st.session_state.page == "encyclopedia" else ""
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button(t['nav_encyclopedia'], use_container_width=True, key="nav_enc_btn"):
        st.session_state.page = "encyclopedia"
    st.markdown('</div>', unsafe_allow_html=True)
with nav_m4:
    cls = "nav-active" if st.session_state.page == "insights" else ""
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button(t['nav_insights'], use_container_width=True, key="nav_insights_btn"):
        st.session_state.page = "insights"
    st.markdown('</div>', unsafe_allow_html=True)
with nav_m5:
    cls = "nav-active" if st.session_state.page == "whatif" else ""
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button(t['nav_whatif'], use_container_width=True, key="nav_whatif_btn"):
        st.session_state.page = "whatif"
    st.markdown('</div>', unsafe_allow_html=True)
with nav_r:
    lang_choice = st.selectbox(" ", ["English", "हिंदी"], label_visibility="collapsed",
                                index=0 if st.session_state.lang == 'en' else 1, key="lang_select")
    st.session_state.lang = 'en' if lang_choice == "English" else 'hi'

t = T[st.session_state.lang]  # refresh after nav may have changed language

# ===========================================================================
# PAGE: PREDICT
# ===========================================================================
if st.session_state.page == "predict":
    st.markdown(f"""
    <div class="hero-banner">
        <h1>{t['title']}</h1>
        <p>{t['subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="input-card"><h4>{t["soil_weather"]}</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input(t['n'], min_value=0, max_value=140, value=50, key="in_n")
        p = st.number_input(t['p'], min_value=0, max_value=145, value=50, key="in_p")
        k = st.number_input(t['k'], min_value=0, max_value=205, value=50, key="in_k")
        ph = st.number_input(t['ph'], min_value=0.0, max_value=14.0, value=6.5, key="in_ph")
    with col2:
        temperature = st.number_input(t['temp'], min_value=0.0, max_value=50.0, value=25.0, key="in_temp")
        humidity = st.number_input(t['humidity'], min_value=0.0, max_value=100.0, value=60.0, key="in_hum")
        rainfall = st.number_input(t['rainfall'], min_value=0.0, max_value=300.0, value=100.0, key="in_rain")

    live_inputs = {'N': n, 'P': p, 'K': k, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
    for w in input_warnings(t, live_inputs):
        st.warning(w)

    pb_col, rb_col = st.columns([3, 1])
    with pb_col:
        predict_clicked = st.button(t['predict_btn'], use_container_width=True)
    with rb_col:
        st.button(t['reset_btn'], use_container_width=True, on_click=reset_inputs)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        with st.spinner(t['analyzing']):
            time.sleep(1)
            input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]
            classes = model.classes_

        st.session_state.prediction_made = True
        st.session_state.pred_crop = prediction
        st.session_state.pred_probs = probs.tolist()
        st.session_state.pred_classes = list(classes)
        st.session_state.last_inputs = live_inputs
        st.session_state.coach_qa = []
        st.session_state.history.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "crop": prediction.capitalize(),
            "conf": round(float(probs.max()) * 100, 2),
        })

    if st.session_state.prediction_made:
        prediction = st.session_state.pred_crop
        probs = np.array(st.session_state.pred_probs)
        classes = np.array(st.session_state.pred_classes)
        inp = st.session_state.last_inputs

        best_icon = crop_icons.get(prediction, '🌱')
        st.markdown(f"""
        <div class="top-card">
            <div style="font-size:48px;">{best_icon}</div>
            <div class="crop-title">{t['best_crop']} {prediction.upper()}</div>
            <div style="font-size:15px; margin-top:6px; color:#c9a227;">{t['confidence']}: {probs.max()*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        quick_tips = fertilizer_advice(t, prediction, inp['N'], inp['P'], inp['K'])
        st.markdown(f"""
        <div class="fert-quick">
            <b>{t['quick_fert_tip']}</b><br>{"<br>".join(quick_tips)}
        </div>
        """, unsafe_allow_html=True)

        with st.expander(t['show_profile'].format(crop=prediction.capitalize()), expanded=False):
            render_crop_detail(t, prediction, inp)

        st.write("")

        g1, g2 = st.columns([1, 1])
        with g1:
            st.subheader(t['confidence_gauge'])
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probs.max() * 100,
                number={'suffix': "%", 'font': {'color': '#f5f0e1'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': '#f5f0e1'},
                       'bar': {'color': "#c9a227"},
                       'bgcolor': "#182920",
                       'steps': [{'range': [0, 40], 'color': '#3a2323'}, {'range': [40, 70], 'color': '#5b4a1f'}, {'range': [70, 100], 'color': '#1f3d2b'}],
                       'threshold': {'line': {'color': "#5b8fae", 'width': 4}, 'thickness': 0.8, 'value': probs.max() * 100}}))
            gauge.update_layout(height=280, margin=dict(t=20, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': '#f5f0e1'})
            st.plotly_chart(gauge, use_container_width=True)

        sorted_idx = np.argsort(probs)[::-1]
        all_crops = [classes[i] for i in sorted_idx]
        all_conf = [probs[i] * 100 for i in sorted_idx]

        with g2:
            st.subheader(t['conditions_vs_ideal'].format(crop=prediction.capitalize()))
            user_vals, ideal_vals = [], []
            ideal_ranges = crop_data.get(prediction, {}).get('ideal', {})
            for f in FEATURES:
                v = inp[f]
                lo, hi = ideal_ranges.get(f, (0, 1))
                span = max(hi - lo, 1e-6)
                user_vals.append(np.clip((v - lo) / span, -0.5, 1.5) * 100)
                ideal_vals.append(50)
            radar_labels = [t['feat_' + f] for f in FEATURES]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=ideal_vals + [ideal_vals[0]], theta=radar_labels + [radar_labels[0]], name="Ideal", line=dict(color="#1f3d2b")))
            fig.add_trace(go.Scatterpolar(r=user_vals + [user_vals[0]], theta=radar_labels + [radar_labels[0]], name="You", fill='toself', line=dict(color="#c9a227")))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=280, margin=dict(t=20, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': '#f5f0e1'})
            st.plotly_chart(fig, use_container_width=True)

        st.write("")
        st.subheader(t['conf_chart'])
        bar_fig = go.Figure(go.Bar(
            x=all_conf, y=[c.capitalize() for c in all_crops], orientation='h',
            marker=dict(color=all_conf, colorscale=[[0, "#8b5e3c"], [1, "#1f3d2b"]]),
        ))
        bar_fig.update_layout(height=550, xaxis_title=t['confidence'] + " (%)", yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': '#f5f0e1'})
        st.plotly_chart(bar_fig, use_container_width=True)

        st.subheader(t['full_table'])
        table_df = pd.DataFrame({
            t['col_rank']: list(range(1, len(all_crops) + 1)),
            t['col_crop']: [c.capitalize() for c in all_crops],
            t['col_icon']: [crop_icons.get(c, '🌱') for c in all_crops],
            t['col_conf']: [f"{c:.2f}%" for c in all_conf]
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.download_button(
            t['download_csv'],
            data=table_df.drop(columns=[t['col_icon']]).to_csv(index=False).encode("utf-8-sig"),
            file_name="crop_prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.write("")
        st.markdown(f"**{t['share_label']}**")
        share_text = t['share_text'].format(
            crop=prediction.capitalize(), conf=probs.max() * 100,
            n=inp['N'], p=inp['P'], k=inp['K'], temp=inp['temperature'],
            hum=inp['humidity'], ph=inp['ph'], rain=inp['rainfall']
        ).replace("\\n", "\n")
        st.code(share_text, language=None)
        st.info(t['print_tip'])

# ===========================================================================
# PAGE: CROP COACH
# ===========================================================================
elif st.session_state.page == "coach":
    st.markdown(f"""
    <div class="hero-banner">
        <h1>{t['coach_header']}</h1>
        <p>{t['coach_sub']}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.prediction_made:
        # A prediction already exists — use it, but still let the user switch crop
        default_crop = st.session_state.pred_crop
        inp = st.session_state.last_inputs
    else:
        # No prediction yet — let the user pick a crop manually and chat right away
        default_crop = None
        inp = {'N': 50, 'P': 50, 'K': 50, 'temperature': 25.0, 'humidity': 60.0, 'ph': 6.5, 'rainfall': 100.0}

    crop_options = sorted(crop_data.keys())
    default_index = crop_options.index(default_crop) if default_crop in crop_options else 0
    chosen_crop = st.selectbox(t['sidebar_lookup'], crop_options,
                                index=default_index,
                                format_func=lambda c: f"{crop_data[c]['icon']} {c.capitalize()}",
                                key="coach_crop_select")

    quick_tips = fertilizer_advice(t, chosen_crop, inp['N'], inp['P'], inp['K'])

    st.markdown(f"""<div class="coach-card">
        <p>{t['coach_greeting'].format(crop=chosen_crop.capitalize())}</p></div>""", unsafe_allow_html=True)
    for tip in quick_tips:
        st.markdown(f"- {tip}")

    coach_msg = st.text_input(t['coach_ask_label'], placeholder=t['coach_ask_placeholder'], key="coach_input")
    if st.button(t['coach_ask_btn'], key="coach_ask"):
        if coach_msg.strip():
            reply = crop_coach_reply(t, coach_msg, chosen_crop, inp)
            st.session_state.coach_qa.append((coach_msg, reply))
    for q, a in reversed(st.session_state.coach_qa):
        st.markdown(f"**🧑 {q}**")
        st.markdown(f'<div class="coach-reply">🧑‍🌾 {a}</div>', unsafe_allow_html=True)

# ===========================================================================
# PAGE: ENCYCLOPEDIA
# ===========================================================================
elif st.session_state.page == "encyclopedia":
    st.markdown(f"""
    <div class="hero-banner">
        <h1>{t['sidebar_enc']}</h1>
    </div>
    """, unsafe_allow_html=True)

    enc_crop = st.selectbox(t['sidebar_lookup'], sorted(crop_data.keys()), format_func=lambda c: f"{crop_data[c]['icon']} {c.capitalize()}")
    info = crop_data[enc_crop]
    img_path = crop_image_path(enc_crop)

    ecol1, ecol2 = st.columns([1, 2])
    with ecol1:
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.markdown(f"<div style='font-size:90px; text-align:center;'>{info['icon']}</div>", unsafe_allow_html=True)
    with ecol2:
        st.markdown(f"### {info['icon']} {enc_crop.capitalize()}")
        st.write(info['desc'][st.session_state.lang])
        st.caption(t['sidebar_ideal'])
        for f in FEATURES:
            lo, hi = info['ideal'][f]
            st.write(f"**{t['feat_' + f]}:** {lo} – {hi}")

    st.write("")
    if not st.session_state.prediction_made:
        st.info(t['enc_no_prediction'])
    else:
        prediction = st.session_state.pred_crop
        probs = np.array(st.session_state.pred_probs)
        classes = np.array(st.session_state.pred_classes)
        inp = st.session_state.last_inputs

        sorted_idx = np.argsort(probs)[::-1]
        all_crops = [classes[i] for i in sorted_idx]
        all_conf = [probs[i] * 100 for i in sorted_idx]

        st.subheader(t['match_overview'].format(n=len(all_crops)))
        st.caption(t['match_hint'])
        n_cols = 4
        for row_start in range(0, len(all_crops), n_cols):
            cols = st.columns(n_cols)
            for j, crop in enumerate(all_crops[row_start:row_start + n_cols]):
                idx_global = row_start + j
                icon = crop_icons.get(crop, '🌱')
                conf = all_conf[idx_global]
                img_path2 = crop_image_path(crop)
                with cols[j]:
                    if img_path2:
                        st.image(img_path2, use_container_width=True)
                        st.markdown(f"""<div style="text-align:center; margin-top:-8px;">
                            <div class="crop-name" style="color:#f5f0e1;">{crop.capitalize()}</div>
                            <div class="crop-conf-chip">{t['match_pct'].format(v=conf)}</div></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="crop-card">
                            <div class="crop-icon">{icon}</div>
                            <div class="crop-name">{crop.capitalize()}</div>
                            <div class="crop-conf-chip">{t['match_pct'].format(v=conf)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with st.expander(t['details_btn'].format(crop=crop.capitalize())):
                        render_crop_detail(t, crop, inp)

# ===========================================================================
# PAGE: INSIGHTS (History / Feature Importance / Market Price / Seasonal)
# ===========================================================================
elif st.session_state.page == "insights":
    st.markdown(f"""
    <div class="hero-banner">
        <h1>{t['insights_title']}</h1>
    </div>
    """, unsafe_allow_html=True)

    tab_h, tab_f, tab_m, tab_s = st.tabs([t['tab_history'], t['tab_feat_imp'], t['tab_market'], t['tab_seasonal']])

    with tab_h:
        if not st.session_state.history:
            st.info(t['no_history'])
        else:
            hist_df = pd.DataFrame(st.session_state.history)
            hist_df.columns = [t['history_time'], t['history_crop'], t['history_conf']]
            st.dataframe(hist_df.iloc[::-1], use_container_width=True, hide_index=True)
            if st.button(t['clear_history']):
                st.session_state.history = []
                st.rerun()

    with tab_f:
        st.caption(t['feat_imp_desc'])
        importances = model.feature_importances_
        feat_names = [t['feat_' + f] for f in FEATURES]
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=True)
        imp_fig = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"], orientation='h',
            marker=dict(color=imp_df["Importance"], colorscale=[[0, "#8b5e3c"], [1, "#c9a227"]]),
        ))
        imp_fig.update_layout(height=380, margin=dict(t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': '#f5f0e1'})
        st.plotly_chart(imp_fig, use_container_width=True)

    with tab_m:
        st.warning(t['market_note'])
        price_df = pd.DataFrame({
            t['col_crop']: [c.capitalize() for c in sorted(CROP_PRICE.keys())],
            t['col_icon']: [crop_icons.get(c, '🌱') for c in sorted(CROP_PRICE.keys())],
            t['col_price']: [f"₹{CROP_PRICE[c][0]:,} – ₹{CROP_PRICE[c][1]:,}" for c in sorted(CROP_PRICE.keys())],
        })
        st.dataframe(price_df, use_container_width=True, hide_index=True)

    with tab_s:
        st.caption(t['seasonal_note'])
        season_labels = {'kharif': t['season_kharif'], 'rabi': t['season_rabi'],
                          'zaid': t['season_zaid'], 'year': t['season_year']}
        for season_key, season_label in season_labels.items():
            crops_in_season = [c for c, s in CROP_SEASON.items() if s == season_key]
            if not crops_in_season:
                continue
            st.markdown(f"**{season_label}**")
            chips = " &nbsp; ".join(f"{crop_icons.get(c,'🌱')} {c.capitalize()}" for c in sorted(crops_in_season))
            st.markdown(chips)
            st.write("")

# ===========================================================================
# PAGE: WHAT-IF SIMULATOR
# ===========================================================================
elif st.session_state.page == "whatif":
    st.markdown(f"""
    <div class="hero-banner">
        <h1>{t['whatif_title']}</h1>
        <p>{t['whatif_sub']}</p>
    </div>
    """, unsafe_allow_html=True)

    wc1, wc2 = st.columns(2)
    with wc1:
        wi_n = st.slider(t['n'], 0, 140, 50, key="wi_n")
        wi_p = st.slider(t['p'], 0, 145, 50, key="wi_p")
        wi_k = st.slider(t['k'], 0, 205, 50, key="wi_k")
        wi_ph = st.slider(t['ph'], 0.0, 14.0, 6.5, key="wi_ph")
    with wc2:
        wi_temp = st.slider(t['temp'], 0.0, 50.0, 25.0, key="wi_temp")
        wi_hum = st.slider(t['humidity'], 0.0, 100.0, 60.0, key="wi_hum")
        wi_rain = st.slider(t['rainfall'], 0.0, 300.0, 100.0, key="wi_rain")

    wi_input = np.array([[wi_n, wi_p, wi_k, wi_temp, wi_hum, wi_ph, wi_rain]])
    wi_pred = model.predict(wi_input)[0]
    wi_probs = model.predict_proba(wi_input)[0]
    wi_icon = crop_icons.get(wi_pred, '🌱')

    st.write("")
    st.markdown(f"""
    <div class="top-card">
        <div style="font-size:44px;">{wi_icon}</div>
        <div class="crop-title">{t['whatif_top_crop']} {wi_pred.upper()}</div>
        <div style="font-size:14px; margin-top:6px; color:#c9a227;">{t['confidence']}: {wi_probs.max()*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    wi_sorted = np.argsort(wi_probs)[::-1][:5]
    wi_cols = st.columns(5)
    wi_classes = model.classes_
    for i, idx in enumerate(wi_sorted):
        c = wi_classes[idx]
        conf = wi_probs[idx] * 100
        with wi_cols[i]:
            st.markdown(f"""
            <div class="crop-card">
                <div class="crop-icon">{crop_icons.get(c,'🌱')}</div>
                <div class="crop-name">{c.capitalize()}</div>
                <div class="crop-conf-chip">{t['match_pct'].format(v=conf)}</div>
            </div>
            """, unsafe_allow_html=True)