# main_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
import os

# Konfigurasi halaman Streamlit
# Mengatur judul halaman web, tata letak lebar, dan menyembunyikan sidebar secara default.
st.set_page_config(
    page_title="Klasifikasi Obesitas", 
    layout="wide", 
    initial_sidebar_state="collapsed" 
)

# --- CSS Kustom untuk Tampilan UI yang Lebih Baik ---
# Bagian ini berisi kode CSS untuk menyesuaikan tampilan elemen-elemen di aplikasi Streamlit.
# Ini membantu membuat UI lebih menarik, responsif, dan sesuai dengan gaya yang diinginkan.
st.markdown("""
<style>
    /* Mengubah warna latar belakang aplikasi dan font dasar */
    .stApp {
        background-color: #1a1a2e; /* Warna latar belakang biru gelap */
        color: #e0e0e0; /* Warna teks utama abu-abu terang */
        font-family: 'Inter', sans-serif; /* Menggunakan font 'Inter' untuk konsistensi */
    }

    /* Gaya untuk judul utama aplikasi (h1) */
    h1 {
        color: #e94560; /* Warna merah muda keunguan */
        text-align: center; /* Teks di tengah */
        font-size: 3em; /* Ukuran font lebih besar */
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3); /* Efek bayangan pada teks */
    }
    /* Gaya untuk judul sub-bagian (h2) */
    h2 {
        color: #0f3460; /* Warna biru gelap */
        font-size: 2em;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid #e94560; /* Garis bawah dengan warna aksen */
    }
    /* Gaya untuk judul di dalam form (h3) */
    h3 {
        color: #e94560; /* Warna merah muda keunguan */
        font-size: 1.5em;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Gaya untuk kontainer form input */
    .stForm {
        background-color: #0f3460; /* Latar belakang form biru gelap */
        padding: 30px;
        border-radius: 15px; /* Sudut membulat */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4); /* Efek bayangan yang lebih dalam */
        margin-bottom: 40px;
    }

    /* Gaya untuk label dari selectbox dan number input */
    .stSelectbox > label, .stNumberInput > label {
        font-size: 1.1em; /* Ukuran font label sedikit lebih besar */
        font-weight: bold;
        color: #e0e0e0; /* Warna teks label abu-abu terang */
        margin-bottom: 8px;
    }

    /* Gaya untuk elemen selectbox dan number input itu sendiri */
    .stSelectbox div[data-baseweb="select"], .stNumberInput div[data-baseweb="input"] {
        border-radius: 10px; /* Sudut membulat */
        border: 1px solid #334d6e; /* Border abu-abu kebiruan */
        background-color: #2e3b5e; /* Latar belakang input biru keunguan */
        padding: 8px 15px;
        color: #ffffff; /* Warna teks input putih */
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2); /* Bayangan dalam untuk efek kedalaman */
    }
    /* Efek hover pada selectbox dan number input */
    .stSelectbox div[data-baseweb="select"]:hover, .stNumberInput div[data-baseweb="input"]:hover {
        border-color: #e94560; /* Border berubah warna saat di-hover */
    }
    /* Memastikan teks di dalam input berwarna putih */
    .stSelectbox div[data-baseweb="select"] input, .stNumberInput div[data-baseweb="input"] input {
        color: #ffffff !important; 
    }

    /* Gaya untuk tombol submit */
    .stButton > button {
        background-color: #e94560; /* Warna merah muda keunguan */
        color: white;
        border-radius: 10px; /* Sudut membulat */
        padding: 12px 25px;
        font-size: 1.2em;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease; /* Transisi halus untuk efek hover */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); /* Bayangan pada tombol */
    }
    /* Efek hover pada tombol submit */
    .stButton > button:hover {
        background-color: #d13a50; /* Warna sedikit lebih gelap saat hover */
        transform: translateY(-3px); /* Tombol sedikit terangkat */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4); /* Bayangan lebih jelas saat hover */
    }

    /* Gaya untuk tampilan DataFrame */
    .stDataFrame {
        border-radius: 10px; /* Sudut membulat */
        overflow: hidden; /* Memastikan sudut membulat terlihat sempurna */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); /* Bayangan pada DataFrame */
        margin-top: 20px;
        margin-bottom: 30px;
    }
    /* Latar belakang dan warna teks tabel di DataFrame */
    .stDataFrame table {
        background-color: #2e3b5e; 
        color: #e0e0e0; 
    }
    /* Gaya untuk header tabel di DataFrame */
    .stDataFrame th {
        background-color: #0f3460; 
        color: #ffffff;
        font-weight: bold;
    }
    /* Garis antar sel di DataFrame */
    .stDataFrame td {
        border-top: 1px solid #334d6e; 
    }

    /* Gaya untuk pesan sukses/error/warning Streamlit */
    .stSuccess, .stError, .stWarning {
        border-radius: 10px; /* Sudut membulat */
        padding: 15px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2); /* Bayangan pada pesan */
        color: #ffffff; /* Teks putih untuk pesan */
        font-weight: bold;
    }
    /* Warna latar belakang untuk pesan sukses */
    .stSuccess {
        background-color: #28a745; 
    }
    /* Warna latar belakang untuk pesan error */
    .stError {
        background-color: #dc3545; 
    }
    /* Warna latar belakang dan teks untuk pesan warning */
    .stWarning {
        background-color: #ffc107; 
        color: #333333; 
    }

    /* Mengatur padding horizontal pada kolom untuk menjaga jarak antar input */
    .css-1offfwp { 
        padding: 0 1.5rem; 
    }
</style>
""", unsafe_allow_html=True)


# Fungsi preprocess_data: Mengolah data mentah menjadi siap pakai
# Fungsi ini bertanggung jawab untuk membersihkan, mengubah, dan menormalisasi data.
# Ini juga mengembalikan objek scaler yang telah dilatih, penting untuk memproses input baru.
def preprocess_data(df, show=False):
    df_processed = df.copy()
    # Mengganti semua tanda tanya '?' dengan nilai NaN (Not a Number)
    df_processed.replace('?', np.nan, inplace=True)
    
    # Mendefinisikan daftar kolom yang berisi data numerik
    numerical_cols_pre = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    # Mendefinisikan daftar kolom yang berisi data kategorikal
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

    # Mengonversi kolom numerik ke tipe data numerik.
    # 'errors='coerce' akan mengubah nilai yang tidak bisa dikonversi menjadi NaN.
    for col in numerical_cols_pre:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
    # Menghapus baris yang memiliki nilai NaN setelah konversi atau penggantian '?'
    df_processed.dropna(inplace=True) 

    # Memisahkan fitur (X) dan variabel target (y)
    X = df_processed.drop("NObeyesdad", axis=1)
    y = df_processed["NObeyesdad"]

    # Menginisialisasi dan melatih LabelEncoder untuk variabel target (NObeyesdad).
    # Ini mengubah label teks menjadi angka.
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # Menginisialisasi dan melatih LabelEncoder untuk fitur kategorikal.
    # Setiap kolom kategorikal akan memiliki encoder-nya sendiri.
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Memastikan kolom ada di DataFrame X sebelum mencoba melatih dan mengubahnya.
        if col in X.columns:
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    # Menginisialisasi dan melatih StandardScaler untuk fitur numerik.
    # Ini menormalisasi fitur numerik agar memiliki skala yang seragam.
    scaler = StandardScaler()
    # Memastikan semua kolom numerik ada di DataFrame X sebelum melakukan penskalaan.
    if all(col in X.columns for col in numerical_cols_pre):
        X[numerical_cols_pre] = scaler.fit_transform(X[numerical_cols_pre])

    # Mengembalikan data yang sudah diproses, encoder, dan scaler.
    # X, X, y_encoded, y_encoded merepresentasikan X_train, X_test, y_train, y_test
    # dalam konteks fungsi ini karena tidak ada pembagian data di sini.
    return (X, X, y_encoded, y_encoded), label_encoders, target_encoder, scaler 

# Fungsi utama aplikasi Streamlit yang berisi UI dan logika klasifikasi.
def run_text_classification():
    st.title("Aplikasi Klasifikasi Obesitas") # Judul utama yang terlihat di aplikasi
    st.markdown("### 📥 Masukkan Data Anda untuk Prediksi Tingkat Obesitas") # Sub-judul

    # Mendefinisikan path ke file dataset ObesityDataSet.csv.
    # os.path.dirname(__file__) mendapatkan direktori script saat ini,
    # os.path.join menggabungkan path secara aman lintas sistem operasi.
    data_path = os.path.join(os.path.dirname(__file__), "data", "ObesityDataSet.csv")

    # Memuat dan pra-memproses data sekali untuk mendapatkan encoder dan scaler yang sudah dilatih.
    # Ini penting agar input pengguna baru dapat diproses dengan cara yang sama seperti data pelatihan.
    try:
        df_full_data = pd.read_csv(data_path)
    except FileNotFoundError:
        # Menampilkan pesan error jika file dataset tidak ditemukan.
        st.error(f"Error: File 'ObesityDataSet.csv' tidak ditemukan di {data_path}.")
        st.info("Pastikan folder 'data' ada di direktori yang sama dengan 'main_app.py' dan 'ObesityDataSet.csv' ada di dalamnya.")
        st.stop() # Menghentikan eksekusi Streamlit jika file penting tidak ada.

    # Memanggil fungsi preprocess_data untuk mendapatkan data pelatihan yang sudah siap,
    # serta objek encoder dan scaler yang akan digunakan untuk inferensi.
    (X_train_full, X_test_full, y_train_full, y_test_full), label_encoders_preprocessed, target_encoder_preprocessed, scaler_for_inference = preprocess_data(df_full_data, show=False)

    # Mendefinisikan ulang kolom numerik dan kategorikal untuk input agar konsisten dengan proses pra-pemrosesan.
    numerical_cols_input = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_cols_input = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']


    # Formulir input pengguna menggunakan Streamlit
    # Semua input pengguna dikelompokkan dalam satu form.
    with st.form("manual_input_form"):
        # Menggunakan kolom untuk tata letak input yang lebih rapi dan terorganisir, mirip dengan contoh gambar.
        col1, col2, col3 = st.columns(3) # Membuat tiga kolom dengan lebar yang sama

        # Kolom 1: Informasi Pribadi
        with col1:
            st.markdown("#### Informasi Pribadi") # Sub-judul untuk kelompok input ini
            Gender = st.selectbox("Jenis Kelamin", options=["Male", "Female"])
            Age = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=23)
            Height = st.number_input("Tinggi Badan (meter)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
            Weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=300.0, value=80.0, step=0.5)
        
        # Kolom 2: Kebiasaan Makan
        with col2:
            st.markdown("#### Kebiasaan Makan") # Sub-judul
            family_history = st.selectbox("Riwayat Keluarga Obesitas", options=["yes", "no"])
            FAVC = st.selectbox("Sering Makanan Tinggi Kalori", options=["yes", "no"])
            FCVC = st.number_input("Frekuensi Konsumsi Sayuran (1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            NCP = st.number_input("Jumlah Makanan Utama per Hari", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
            CAEC = st.selectbox("Konsumsi Makanan Antar Waktu Makan", options=["no", "Sometimes", "Frequently", "Always"])

        # Kolom 3: Gaya Hidup & Lain-lain
        with col3:
            st.markdown("#### Gaya Hidup & Lain-lain") # Sub-judul
            SMOKE = st.selectbox("Merokok", options=["yes", "no"])
            CH2O = st.number_input("Konsumsi Air Harian (liter)", min_value=0.0, max_value=3.0, value=2.0, step=0.1)
            SCC = st.selectbox("Mengamati Konsumsi Kalori Sendiri", options=["yes", "no"])
            FAF = st.number_input("Frekuensi Aktivitas Fisik Mingguan", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            TUE = st.number_input("Waktu Penggunaan Perangkat Elektronik Harian", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            CALC = st.selectbox("Konsumsi Alkohol", options=["no", "Sometimes", "Frequently", "Always"])
            MTRANS = st.selectbox("Transportasi Utama", options=[
                "Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

        st.markdown("<br>", unsafe_allow_html=True) # Menambahkan spasi vertikal untuk estetika
        submitted = st.form_submit_button("🔍 Klasifikasikan") # Tombol untuk submit form

    # Logika yang dijalankan setelah tombol submit ditekan
    if submitted:
        # Mengumpulkan semua input pengguna ke dalam sebuah dictionary, lalu mengubahnya menjadi DataFrame.
        input_dict = {
            'Gender': [Gender], 'Age': [Age], 'Height': [Height], 'Weight': [Weight],
            'family_history_with_overweight': [family_history], 'FAVC': [FAVC],
            'FCVC': [FCVC], 'NCP': [NCP], 'CAEC': [CAEC], 'SMOKE': [SMOKE],
            'CH2O': [CH2O], 'SCC': [SCC], 'FAF': [FAF], 'TUE': [TUE],
            'CALC': [CALC], 'MTRANS': [MTRANS]
        }
        df_input_original = pd.DataFrame(input_dict)

        # Memastikan kolom numerik di DataFrame input pengguna diubah ke tipe numerik.
        for col in numerical_cols_input:
            df_input_original[col] = pd.to_numeric(df_input_original[col])
        
        st.markdown("### 📊 Inputan yang Digunakan untuk Prediksi")
        st.dataframe(df_input_original) # Menampilkan DataFrame input pengguna

        st.markdown("<br>", unsafe_allow_html=True) # Spasi


        # --- MODEL MENTAH (RAW MODEL) ---
        # Bagian ini melatih dan memprediksi menggunakan model tanpa pra-pemrosesan data lengkap dan tanpa tuning.
        st.markdown("## 🔷 Hasil Prediksi Model Tanpa Pra-pemrosesan & Tuning")
        # Memuat data mentah asli lagi untuk melatih model mentah.
        # Ini penting agar model 'mentah' tidak terpengaruh oleh pra-pemrosesan 'full_data'.
        df_raw_model_data = pd.read_csv(data_path) 
        df_raw_model_data.replace('?', np.nan, inplace=True) # Mengganti '?' dengan NaN
        for col in numerical_cols_input:
            df_raw_model_data[col] = pd.to_numeric(df_raw_model_data[col], errors='coerce') # Konversi numerik
        df_raw_model_data.dropna(inplace=True) # Menghapus baris dengan NaN

        # Memisahkan fitur (X_raw) dan target (y_raw) untuk model mentah
        X_raw = df_raw_model_data.drop("NObeyesdad", axis=1)
        y_raw = df_raw_model_data["NObeyesdad"]
        # Melatih LabelEncoder untuk target model mentah
        target_encoder_raw = LabelEncoder().fit(y_raw)
        y_raw_enc = target_encoder_raw.transform(y_raw)

        # Membuat salinan input pengguna untuk diproses oleh model mentah
        df_input_raw_model = df_input_original.copy() 

        # Mengkodekan fitur kategorikal pada data mentah dan input pengguna
        for col in categorical_cols_input: 
            le = LabelEncoder()
            # Melatih encoder pada kolom kategorikal dari data mentah
            if col in X_raw.columns:
                le.fit(X_raw[col])
                # Mengubah kolom kategorikal di X_raw menjadi numerik dan menyimpan kembali
                X_raw[col] = le.transform(X_raw[col]) 
            else:
                st.error(f"Kolom '{col}' tidak ditemukan di data mentah.")
                return # Menghentikan jika kolom penting tidak ada

            # Mengubah nilai input pengguna menggunakan encoder yang sama.
            val = str(df_input_raw_model[col][0])
            if val not in le.classes_:
                st.error(f"❌ Nilai '{val}' tidak valid untuk kolom '{col}' (Raw Model).")
                return # Menghentikan jika nilai input tidak valid

            df_input_raw_model[col] = le.transform([val])

        # Memastikan urutan fitur input pengguna cocok dengan urutan fitur data pelatihan mentah
        feature_order_raw = X_raw.columns.tolist()
        df_input_raw_model = df_input_raw_model[feature_order_raw]

        # Mendefinisikan model-model yang akan digunakan untuk skenario mentah
        models_raw = {
            "Logistic Regression (Raw)": LogisticRegression(max_iter=5000, random_state=42), 
            "Random Forest (Raw)": RandomForestClassifier(random_state=42),
            "KNN (Raw)": KNeighborsClassifier()
        }

        # Melatih dan memprediksi dengan setiap model mentah
        for name, model in models_raw.items():
            model.fit(X_raw, y_raw_enc) # Melatih model
            pred = model.predict(df_input_raw_model) # Melakukan prediksi
            st.write(f"**{name}**: {target_encoder_raw.inverse_transform(pred)[0]}") # Menampilkan hasil prediksi

        st.markdown("<br>", unsafe_allow_html=True) # Spasi


        # --- MODEL SETELAH PRA-PEMROSESAN ---
        # Bagian ini melatih dan memprediksi menggunakan model setelah pra-pemrosesan data lengkap (tanpa tuning).
        st.markdown("## 🔷 Hasil Prediksi Model Setelah Pra-pemrosesan")
        
        # Membuat salinan input pengguna untuk diproses oleh model pra-proses
        df_input_preprocessed_model = df_input_original.copy() 

        # Menerapkan encoding pada input pengguna menggunakan encoder yang sudah dilatih
        for col in categorical_cols_input: 
            le = label_encoders_preprocessed[col] # Menggunakan encoder dari preprocess_data() di awal
            val = str(df_input_preprocessed_model[col][0])
            if val not in le.classes_:
                st.error(f"❌ Nilai '{val}' tidak valid untuk kolom '{col}' (Preprocessed Model).")
                return # Menghentikan jika nilai input tidak valid
            df_input_preprocessed_model[col] = le.transform([val])

        # Menerapkan penskalaan pada input numerik pengguna menggunakan scaler yang sudah dilatih
        df_input_preprocessed_model[numerical_cols_input] = scaler_for_inference.transform(df_input_preprocessed_model[numerical_cols_input])

        # Memastikan urutan fitur input pengguna cocok dengan urutan fitur data pelatihan yang sudah diproses
        feature_order_preprocessed = X_train_full.columns.tolist() 
        df_input_preprocessed_model = df_input_preprocessed_model[feature_order_preprocessed]

        # Mendefinisikan model-model untuk skenario pra-pemrosesan
        models_pre = {
            "Logistic Regression (Preprocessed)": LogisticRegression(max_iter=5000, random_state=42), 
            "Random Forest (Preprocessed)": RandomForestClassifier(random_state=42),
            "KNN (Preprocessed)": KNeighborsClassifier()
        }

        # Melatih dan memprediksi dengan setiap model pra-pemrosesan
        for name, model in models_pre.items():
            model.fit(X_train_full, y_train_full) # Melatih model
            pred = model.predict(df_input_preprocessed_model) # Melakukan prediksi
            st.write(f"**{name}**: {target_encoder_preprocessed.inverse_transform(pred)[0]}") # Menampilkan hasil prediksi

        st.markdown("<br>", unsafe_allow_html=True) # Spasi


        # --- MODEL SETELAH TUNING ---
        # Bagian ini melatih dan memprediksi menggunakan model setelah hyperparameter tuning.
        st.markdown("## 🔷 Hasil Model Setelah Tuning")
        # Mendefinisikan grid parameter untuk proses tuning hyperparameter.
        param_grid = {
            'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']},
            'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
            'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
        }

        # Mendefinisikan model dasar sebelum tuning
        base_models = {
            'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42), 
            'Random Forest': RandomForestClassifier(random_state=42),
            'KNN': KNeighborsClassifier()
        }

        # Melakukan tuning dan prediksi dengan model terbaik untuk setiap algoritma
        for name in base_models:
            # GridSearchCV mencari kombinasi hyperparameter terbaik.
            grid = GridSearchCV(base_models[name], param_grid[name], cv=3, scoring='f1_weighted', n_jobs=-1)
            grid.fit(X_train_full, y_train_full) # Melatih GridSearchCV pada data pelatihan lengkap
            best_model = grid.best_estimator_ # Mendapatkan model terbaik dari hasil tuning
            
            pred = best_model.predict(df_input_preprocessed_model) # Melakukan prediksi
            st.write(f"**{name} (Tuned)**: {target_encoder_preprocessed.inverse_transform(pred)[0]}") # Menampilkan hasil prediksi

# =========================================================
# Mulai Aplikasi Streamlit
# =========================================================
# Baris ini menjalankan fungsi utama aplikasi Streamlit saat script dieksekusi.
run_text_classification()