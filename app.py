import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
st.title("Emlak Değerleme AI")

m2 = st.number_input("Metrekare", min_value=10)
oda = st.number_input("Oda Sayısı", min_value=1)
bina_yasi = st.number_input("Bina Yaşı", min_value=0)
ilce = st.selectbox("İlçe", df['ilce'].unique())
ilce_kod = df['ilce'].astype('category').cat.categories.get_loc(ilce)

if st.button("Tahmini Fiyatı Hesapla"):
    girdi = pd.DataFrame({
        'm2': [m2],
        'oda_sayisi': [oda],
        'bina_yasi': [bina_yasi],
        'ilce_kod': [ilce_kod]
    })
    tahmin = model.predict(girdi)[0]

    st.success(f"Tahmini Fiyat: {tahmin:,.0f} TL")
