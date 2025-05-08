import os
import pandas as pd
import logging
from io import BytesIO

# Setup logging untuk debugging
logging.basicConfig(level=logging.INFO)

class DatasetImporter:
    def __init__(self):
        """Inisialisasi mapping kolom fleksibel untuk berbagai format dataset"""
        self.column_mapping = {
            'luas_meter2': ['luas_meter2', 'luas', 'area', 'luas_bangunan', 'size'],
            'kamar_tidur': ['kamar_tidur', 'kamar', 'bedrooms', 'jumlah_kamar', 'room'],
            'kamar_mandi': ['kamar_mandi', 'toilet', 'jumlah_toilet', 'bathroom'],
            'jarak_ke_kota': ['jarak_ke_kota', 'jarak', 'distance', 'proximity'],
            'tahun_bangun': ['tahun_bangun', 'tahun_dibangun', 'construction_year', 'year'],
            'lokasi': ['lokasi', 'area', 'wilayah', 'alamat', 'location'],
            'harga': ['harga', 'harga_rumah', 'price', 'nilai_properti', 'cost']
        }
    
    def normalize_columns(self, df):
        """Menyamakan nama kolom berdasarkan mapping yang didefinisikan"""
        rev_map = {syn: target for target, syns in self.column_mapping.items() for syn in syns}
        df = df.rename(columns=rev_map)
        # Pilih hanya kolom target yang unik
        cols_to_keep = list(self.column_mapping.keys())
        # Drop duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]
        return df[cols_to_keep]
    
    def parse_location(self, df):
        """Memecah kolom lokasi menjadi provinsi, kota, dan kecamatan"""
        if 'lokasi' in df.columns:
            try:
                df['provinsi'] = df['lokasi'].astype(str).str.split().str[0]
                df['kota'] = df['lokasi'].astype(str).str.split().str[1].str.replace(r'\(.*\)', '', regex=True)
                df['kecamatan'] = df['lokasi'].astype(str).str.extract(r"\((.*?)\)")
            except Exception as e:
                logging.warning(f"Gagal parse lokasi: {e}")
        return df
    
    def clean_price_column(self, df):
        """Membersihkan dan mengonversi kolom harga"""
        if 'harga' in df.columns:
            try:
                df['harga'] = (
                    df['harga'].astype(str)
                      .str.replace(r'\.', '', regex=True)
                      .str.replace(r',', '', regex=True)
                      .astype(int)
                )
            except Exception as e:
                logging.warning(f"Gagal membersihkan harga: {e}")
        return df
    
    def validate_required_columns(self, df):
        """Memastikan kolom penting tersedia"""
        required_cols = list(self.column_mapping.keys())
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom wajib hilang setelah mapping: {missing}")
        return df
    
    def process_uploaded_file(self, file, file_type, raw_filepath="data/raw_dataset.csv"):
        """Memproses file upload dan menambahkan data ke raw tanpa mengganti"""
        try:
            # Baca file
            if file_type == '.csv':
                df = pd.read_csv(file, on_bad_lines='skip')
            elif file_type == '.xlsx':
                df = pd.read_excel(file)
            elif file_type == '.json':
                df = pd.read_json(file)
            else:
                raise ValueError(f"Format {file_type} tidak didukung")
            
            # Proses
            df = self.normalize_columns(df)
            df = self.validate_required_columns(df)
            df = self.parse_location(df)
            df = self.clean_price_column(df)
            df = df.dropna().drop_duplicates()

            # Simpan ke raw, gabungkan dengan data lama
            os.makedirs(os.path.dirname(raw_filepath), exist_ok=True)
            if os.path.exists(raw_filepath):
                df_raw = pd.read_csv(raw_filepath)
                # Gabungkan
                df_combined = pd.concat([df_raw, df], ignore_index=True)
                # Drop duplicate rows & columns
                df_combined = df_combined.drop_duplicates()
                df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
            else:
                df_combined = df
            df_combined.to_csv(raw_filepath, index=False)
            return df_combined
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return None
