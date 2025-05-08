# 🏠 Prediksi Harga Rumah Indonesia

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Beta-lightgrey)]()

Aplikasi interaktif berbasis **Streamlit** untuk memprediksi harga properti di Indonesia menggunakan **Machine Learning**. Cocok untuk analisis pasar real estate, pengambilan keputusan investasi, dan edukasi data science.

---

## 🚀 Fitur Unggulan

- 📤 Upload Dataset (CSV, Excel, JSON) dan otomatis dipetakan
- 🧹 Pembersihan & Standarisasi data properti
- 🤖 Prediksi harga rumah berdasarkan parameter pengguna
- 📈 Visualisasi tren historis harga dan distribusi
- 💾 Ekspor hasil prediksi & grafik visualisasi

---

## 🧠 Model Machine Learning

Model prediktif menggunakan algoritma regresi terlatih, dengan fitur:

- Luas tanah (`m²`)
- Jumlah kamar tidur & kamar mandi
- Jarak ke pusat kota
- Tahun pembangunan

Model disimpan dalam format `.pkl` dan siap digunakan ulang.

---

## 📊 Visualisasi Interaktif

- Tren harga berdasarkan tahun
- Distribusi harga properti
- Korelasi luas tanah vs harga
- Rata-rata harga per jumlah kamar

> Semua grafik dapat diekspor ke folder `/visualizations` secara otomatis.

---

## 📁 Struktur Folder

```

.
├── app.py                    # Aplikasi utama Streamlit
├── api.py                   # (Opsional) API endpoint
├── train\_model.py           # Script pelatihan ulang model
├── data/
│   ├── raw\_dataset.csv      # Data mentah
│   └── cleaned\_dataset.csv  # Data bersih
├── models/
│   ├── house\_price\_model.pkl
│   └── scaler.pkl
├── visualizations/          # Output visualisasi
├── data\_importer.py         # Modul pemetaan & preprocessing
├── requirements.txt
└── README.md

````

---

## 🔧 Instalasi Lokal

1. **Clone repositori**:
   ```bash
   git clone https://github.com/username/prediksi-harga-rumah.git
   cd prediksi-harga-rumah
   ```

2. **Siapkan environment & install dependensi**:

   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux/Mac
   venv\Scripts\activate          # Windows
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi Streamlit**:

   ```bash
   streamlit run app.py
   ```

---

## ⚙️ Teknologi yang Digunakan

| Teknologi    | Deskripsi                  |
| ------------ | -------------------------- |
| Python       | Bahasa utama               |
| Streamlit    | UI web interaktif          |
| Pandas       | Analisis & manipulasi data |
| Scikit-learn | Algoritma machine learning |
| Seaborn      | Visualisasi statistik      |
| Matplotlib   | Plotting dasar             |
| Joblib       | Menyimpan model ML `.pkl`  |

---

## 🤝 Kontribusi

Kontribusi terbuka untuk semua!
Silakan buat **issue** atau **pull request** jika kamu memiliki saran atau menemukan bug.

---

## 📄 Lisensi

Distribusi dengan lisensi [MIT License](LICENSE).

