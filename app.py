import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from data_importer import DatasetImporter

# Konfigurasi halaman
st.set_page_config(page_title="PropertyParams Importer & Predictor", layout="wide")
st.title("📥 Upload Dataset & 🏠 Prediksi Harga Rumah Indonesia")

# Sidebar untuk navigasi
page = st.sidebar.selectbox("Pilih Halaman", ["Upload Dataset", "Prediksi Harga"])

if page == "Upload Dataset":
    st.header("📂 Upload & Preview Dataset")
    importer = DatasetImporter()
    uploaded_file = st.file_uploader(
        "Pilih file CSV, Excel, atau JSON (contoh: harga_rumah.csv)",
        type=['csv', 'xlsx', 'json']
    )
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_type = os.path.splitext(file_name)[1].lower()
        with st.spinner(f"Mengimpor dan membersihkan {file_name}..."):
            df_clean = importer.process_uploaded_file(uploaded_file, file_type)
            if df_clean is not None:
                st.markdown("### 🔍 Kolom yang Dipetakan")
                mapping_info = pd.DataFrame({
                    "Kolom Asli": [syn for synonyms in importer.column_mapping.values() for syn in synonyms],
                    "Kolom Target": [target for target, synonyms in importer.column_mapping.items() for _ in synonyms]
                }).drop_duplicates().reset_index(drop=True)
                st.dataframe(mapping_info, use_container_width=True)

                st.markdown("### 📊 25 Baris Pertama Dataset Bersih")
                st.dataframe(df_clean.head(25), use_container_width=True)

                st.markdown("### 📈 Statistik Dasar")
                st.dataframe(df_clean.describe(), use_container_width=True)

                csv = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Unduh Dataset Bersih (CSV)", csv, "dataset_bersih.csv", "text/csv")

                os.makedirs("data", exist_ok=True)
                df_clean.to_csv("data/cleaned_dataset.csv", index=False)
                st.success("✅ Dataset berhasil disimpan untuk prediksi")
            else:
                st.error("❌ Gagal memproses file. Pastikan format data sesuai.")

elif page == "Prediksi Harga":
    st.header("🏡 Prediksi Harga Rumah")

    importer = DatasetImporter()
    cleaned_path = "data/cleaned_dataset.csv"
    raw_path = "data/raw_dataset.csv"
    df = None

    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    elif os.path.exists(raw_path):
        try:
            df_raw = pd.read_csv(raw_path)
            df = importer.normalize_columns(df_raw)
            df = importer.validate_required_columns(df)
            df = importer.parse_location(df)
            df = importer.clean_price_column(df)
            df = df.dropna().drop_duplicates()
            os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
            df.to_csv(cleaned_path, index=False)
        except Exception as e:
            st.error(f"❌ Gagal membersihkan raw_dataset.csv: {e}")

    if df is None or df.empty:
        st.warning("Silakan upload dataset terlebih dahulu di 'Upload Dataset'.")
        st.stop()

    input_col, output_col = st.columns([1, 3])

    with input_col:
        st.subheader("🛠️ Input Parameter")
        provinsi = st.selectbox("Provinsi", df['provinsi'].unique())
        kota = st.selectbox("Kota", df[df['provinsi'] == provinsi]['kota'].unique())
        luas = st.slider("Luas Tanah (m²)", int(df['luas_meter2'].min()), int(df['luas_meter2'].max()), int(df['luas_meter2'].median()))
        tidur = st.slider("Jumlah Kamar Tidur", int(df['kamar_tidur'].min()), int(df['kamar_tidur'].max()), int(df['kamar_tidur'].median()))
        mandi = st.slider("Jumlah Kamar Mandi", int(df['kamar_mandi'].min()), int(df['kamar_mandi'].max()), int(df['kamar_mandi'].median()))
        jarak = st.slider("Jarak ke Kota (km)", int(df['jarak_ke_kota'].min()), int(df['jarak_ke_kota'].max()), int(df['jarak_ke_kota'].median()))
        tahun = st.slider("Tahun Bangun", int(df['tahun_bangun'].min()), int(df['tahun_bangun'].max()), int(df['tahun_bangun'].median()))

    with output_col:
        try:
            model = joblib.load("models/house_price_model.pkl")
            scaler = joblib.load("models/scaler.pkl")
        except Exception as e:
            st.error(f"❌ Gagal memuat model atau scaler: {e}")
            st.stop()

        try:
            inp = scaler.transform([[luas, tidur, mandi, jarak, tahun]])
            pred = model.predict(inp)[0]
        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")
            st.stop()

        st.subheader("📊 Hasil Prediksi")
        st.write(f"**Harga Prediksi:** Rp {pred:,.0f}")

        st.subheader("📈 Analisis Data")
        hist_data = df[(df['provinsi'] == provinsi) & (df['kota'] == kota)]

        if hist_data.empty:
            st.info("⚠️ Tidak ada data historis untuk area ini.")
        else:
            chart_type = st.selectbox(
                "Pilih Jenis Visualisasi",
                ["Tren Harga per Tahun", "Distribusi Harga", "Hubungan Luas-Harga", "Perbandingan Kamar"]
            )
            fig, ax = plt.subplots()
            if chart_type == "Tren Harga per Tahun":
                tahun_harga = hist_data.groupby('tahun_bangun')['harga'].mean()
                ax.plot(tahun_harga.index, tahun_harga.values/1e9, marker='o')
                ax.set_xlabel("Tahun Bangun")
                ax.set_ylabel("Harga Rata-Rata (Miliar IDR)")
                ax.set_title("Tren Harga Properti per Tahun")

            elif chart_type == "Distribusi Harga":
                ax.hist(hist_data['harga']/1e9, bins=15, edgecolor='k')
                ax.set_xlabel("Harga (Miliar IDR)")
                ax.set_ylabel("Jumlah Properti")
                ax.set_title("Distribusi Harga Properti")

            elif chart_type == "Hubungan Luas-Harga":
                ax.scatter(hist_data['luas_meter2'], hist_data['harga']/1e9, alpha=0.6)
                ax.set_xlabel("Luas Tanah (m²)")
                ax.set_ylabel("Harga (Miliar IDR)")
                ax.set_title("Hubungan Luas Tanah vs Harga")

            elif chart_type == "Perbandingan Kamar":
                kamar_data = hist_data.groupby('kamar_tidur')['harga'].mean()
                ax.bar(kamar_data.index.astype(str), kamar_data.values/1e9)
                ax.set_xlabel("Jumlah Kamar Tidur")
                ax.set_ylabel("Harga Rata-Rata (Miliar IDR)")
                ax.set_title("Harga Rata-Rata Berdasarkan Jumlah Kamar Tidur")

            ax.grid(True)
            st.pyplot(fig)

        if st.button("⬇️ Ekspor Hasil Prediksi dan Visualisasi"):
            os.makedirs("visualizations", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"hasil_prediksi_{timestamp}"

            # Ekspor CSV hasil prediksi input
            result_df = pd.DataFrame({
                'parameter': ['Luas', 'Kamar Tidur', 'Kamar Mandi', 'Jarak ke Kota', 'Tahun Bangun'],
                'nilai': [luas, tidur, mandi, jarak, tahun]
            })
            result_df.to_csv(f"{filename_base}.csv", index=False)

            # Simpan visualisasi sesuai chart_type
            fig, ax = plt.subplots()
            if chart_type == "Tren Harga per Tahun":
                tahun_harga = hist_data.groupby('tahun_bangun')['harga'].mean()
                ax.plot(tahun_harga.index, tahun_harga.values / 1e9, marker='o')
                ax.set_xlabel("Tahun Bangun")
                ax.set_ylabel("Harga Rata-Rata (Miliar IDR)")
                ax.set_title("Tren Harga Properti per Tahun")
            elif chart_type == "Distribusi Harga":
                sns.histplot(hist_data['harga'] / 1e9, bins=15, edgecolor='k', ax=ax)
                ax.set_xlabel("Harga (Miliar IDR)")
                ax.set_ylabel("Jumlah Properti")
                ax.set_title("Distribusi Harga Properti")
            elif chart_type == "Hubungan Luas-Harga":
                sns.scatterplot(x=hist_data['luas_meter2'], y=hist_data['harga'] / 1e9, alpha=0.6, ax=ax)
                ax.set_xlabel("Luas Tanah (m²)")
                ax.set_ylabel("Harga (Miliar IDR)")
                ax.set_title("Hubungan Luas Tanah vs Harga")
            elif chart_type == "Perbandingan Kamar":
                kamar_data = hist_data.groupby('kamar_tidur')['harga'].mean()
                sns.barplot(x=kamar_data.index.astype(str), y=kamar_data.values / 1e9, ax=ax)
                ax.set_xlabel("Jumlah Kamar Tidur")
                ax.set_ylabel("Harga Rata-Rata (Miliar IDR)")
                ax.set_title("Harga Berdasarkan Jumlah Kamar Tidur")

            fig.tight_layout()
            plot_path = f"visualizations/{filename_base}_{chart_type.replace(' ', '_').lower()}.png"
            plt.savefig(plot_path)
            plt.close()

            st.success(f"Hasil dan visualisasi berhasil diekspor ke:\n- `{filename_base}.csv`\n- `{plot_path}`")