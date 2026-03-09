import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Aset yang Sudah Disimpan ---
@st.cache_resource # Agar model tidak di-load berulang kali setiap klik tombol
def load_assets():
    model = joblib.load('xgboost_diamond_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('label_encoders.pkl')
    return model, scaler, encoders

model, scaler, encoders = load_assets()

# --- 2. Tampilan Antarmuka (UI) ---
st.title("💎 Diamond Price Predictor")
st.write("Masukkan spesifikasi berlian di bawah ini untuk mendapatkan estimasi harga.")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat (Berat)", min_value=0.1, max_value=5.0, value=0.7)
    cut = st.selectbox("Cut (Kualitas Potongan)", options=encoders['cut'].classes_)
    color = st.selectbox("Color (Warna)", options=encoders['color'].classes_)

with col2:
    clarity = st.selectbox("Clarity (Kejernihan)", options=encoders['clarity'].classes_)
    depth = st.number_input("Depth %", min_value=40.0, max_value=80.0, value=61.0)
    table = st.number_input("Table %", min_value=40.0, max_value=90.0, value=57.0)

x = st.number_input("Length (x) in mm", min_value=0.1, value=5.0)
y = st.number_input("Width (y) in mm", min_value=0.1, value=5.0)
z = st.number_input("Depth (z) in mm", min_value=0.1, value=3.0)

# --- 3. Proses Prediksi ---
if st.button("Predict Price"):
    # Buat DataFrame dari input user
    input_df = pd.DataFrame([{
        'carat': carat, 'cut': cut, 'color': color, 
        'clarity': clarity, 'depth': depth, 'table': table,
        'x': x, 'y': y, 'z': z
    }])

    # Preprocessing: Encoding Kolom Kategorikal
    for col in ['cut', 'color', 'clarity']:
        input_df[col] = encoders[col].transform(input_df[col])

    # Preprocessing: Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)
    
    # --- 4. Tampilkan Hasil ---
    st.divider()
    st.subheader(f"Estimasi Harga Berlian:")
    st.header(f"USD ${prediction[0]:,.2f}")
    st.balloons()