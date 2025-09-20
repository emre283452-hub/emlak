import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import schedule
import time

# 🔄 Web scraping + veri temizleme
def ilan_verisi_cek(site='sahibinden', sayfa=1):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.sahibinden.com/satilik-daire/istanbul?page={sayfa}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    ilanlar = []
    for ilan in soup.select("tr.searchResultsItem"):
        try:
            baslik = ilan.select_one("td.searchResultsTitleValue").get_text(strip=True)
            fiyat = ilan.select_one("td.searchResultsPriceValue").get_text(strip=True)
            konum = ilan.select_one("td.searchResultsLocationValue").get_text(strip=True)
            ilanlar.append({"Başlık": baslik, "Fiyat": fiyat, "Konum": konum})
        except:
            continue
    return ilanlar

def temizle_ve_donustur(ilanlar):
    df = pd.DataFrame(ilanlar)
    df["Fiyat"] = df["Fiyat"].str.replace("TL", "").str.replace(".", "").str.strip()
    df["Fiyat"] = pd.to_numeric(df["Fiyat"], errors="coerce")
    df[["İl", "İlçe"]] = df["Konum"].str.split("/", expand=True)
    df = df.dropna(subset=["Fiyat", "İl", "İlçe"])
    return df

# 🔁 Otomatik veri güncelleme
def veri_guncelle():
    ilanlar = ilan_verisi_cek()
    df_scraped = temizle_ve_donustur(ilanlar)
    df_scraped.to_csv("guncel_ilanlar.csv", index=False)

schedule.every().day.at("06:00").do(veri_guncelle)
# Arka planda çalıştırmak için ayrı thread gerekebilir

# 📊 Modelleme
veri = {
    'm2': [100, 80, 120, 90, 150],
    'oda_sayisi': [3, 2, 4, 3, 5],
    'bina_yasi': [10, 5, 20, 15, 2],
    'ilce': ['Kadıköy', 'Beşiktaş', 'Üsküdar', 'Şişli', 'Ataşehir'],
    'fiyat': [2000000, 1800000, 2200000, 1900000, 2500000]
}
df = pd.DataFrame(veri)

numeric_features = ['m2', 'oda_sayisi', 'bina_yasi']
categorical_features = ['ilce']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model = Pipeline([
    ('prep', preprocessor),
    ('reg', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1))
])

X = df[numeric_features + categorical_features]
y = df['fiyat']
model.fit(X, y)

# 🌍 Coğrafi görselleştirme
def harita_uret():
    try:
        geo_df = gpd.read_file("ilce_shapefile.shp")
        ort_fiyat = df.groupby('ilce')['fiyat'].mean().reset_index()
        geo_df = geo_df.merge(ort_fiyat, on='ilce')
        fig, ax = plt.subplots(figsize=(10, 8))
        geo_df.plot(column='fiyat', cmap='OrRd', legend=True, ax=ax, edgecolor='black')
        plt.title("İlçe Bazlı Ortalama Fiyat Haritası")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("fiyat_haritasi.png")
    except:
        pass

harita_uret()

# 🖥️ Streamlit arayüzü
# 🌍 Sabit il/ilçe verisi (JSON yerine)
iller_ilceler = {
    "İstanbul": ["Kadıköy", "Beşiktaş", "Üsküdar", "Şişli", "Ataşehir"],
    "Ankara": ["Çankaya", "Keçiören", "Yenimahalle", "Mamak", "Etimesgut"],
    "İzmir": ["Bornova", "Karşıyaka", "Konak", "Buca", "Bayraklı"]
}

# 🖥️ Streamlit arayüzü
st.title("Üst Düzey Emlak Değerleme AI")

il = st.selectbox("İl", list(iller_ilceler.keys()))
ilce = st.selectbox("İlçe", iller_ilceler[il])
m2 = st.number_input("Metrekare", min_value=10)
oda = st.number_input("Oda Sayısı", min_value=1)
bina_yasi = st.number_input("Bina Yaşı", min_value=0)

if st.button("Tahmini Fiyatı Hesapla"):
    girdi = pd.DataFrame({
        'm2': [m2],
        'oda_sayisi': [oda],
        'bina_yasi': [bina_yasi],
        'ilce': [ilce]
    })
    tahmin = model.predict(girdi)[0]
    st.success(f"Tahmini Fiyat: {tahmin:,.0f} TL")

    st.subheader("İlçe Bazlı Ortalama Fiyat Haritası")
    st.image("fiyat_haritasi.png")


