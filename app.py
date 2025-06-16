import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Tingkat Obesitas", layout="wide")

# CSS Kustom untuk tampilan UI
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
        color: #333333;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5em;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# Memuat dataset untuk mendapatkan encoder dan scaler
data_path = 'ObesityDataSet.csv'  # Sesuaikan dengan path dataset Anda
df = pd.read_csv(data_path)
target_column = 'NObeyesdad'
features = df.drop(columns=[target_column])
target = df[target_column]

# Encoder untuk fitur kategorikal
categorical_cols = features.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le

# Encoder untuk target
target_encoder = LabelEncoder()
target_encoder.fit(target)

# Memuat model yang telah dilatih dari file .joblib
model_paths = {
    'Logistic Regression': 'best_logistic_regression_model.joblib',
    'Random Forest': 'best_random_forest_model.joblib',
    'SVM': 'best_svm_model.joblib'
}
models = {}
for name, path in model_paths.items():
    try:
        models[name] = joblib.load(path)
    except FileNotFoundError:
        st.error(f"File model {path} tidak ditemukan. Pastikan file ada di direktori yang sama dengan app.py.")
        st.stop()

# Memuat scaler
try:
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("File scaler.joblib tidak ditemukan. Pastikan file ada di direktori yang sama dengan app.py.")
    st.stop()

# UI Streamlit
st.title("Aplikasi Prediksi Tingkat Obesitas")

# Pilihan model
model_name = st.selectbox("Pilih Model untuk Prediksi", list(models.keys()))

# Form input pengguna
st.header("Masukkan Data Anda")
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    # Kolom 1: Informasi Pribadi
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=25)
        height = st.number_input("Tinggi Badan (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
        weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
    
    # Kolom 2: Kebiasaan Makan
    with col2:
        family_history = st.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])
        favc = st.selectbox("Sering Makanan Tinggi Kalori", ["yes", "no"])
        fcvc = st.number_input("Frekuensi Konsumsi Sayuran (1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        ncp = st.number_input("Jumlah Makanan Utama/Hari", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        caec = st.selectbox("Makanan Antar Waktu", ["no", "Sometimes", "Frequently", "Always"])
    
    # Kolom 3: Gaya Hidup
    with col3:
        smoke = st.selectbox("Merokok", ["yes", "no"])
        ch2o = st.number_input("Konsumsi Air (liter)", min_value=0.0, max_value=3.0, value=2.0, step=0.1)
        scc = st.selectbox("Pantau Kalori", ["yes", "no"])
        faf = st.number_input("Aktivitas Fisik (hari/minggu)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        tue = st.number_input("Waktu Elektronik (jam/hari)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
        calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportasi", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

    submit = st.form_submit_button("Prediksi")

# Proses prediksi
if submit:
    # Membuat DataFrame dari input
    input_data = {
        'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
        'family_history_with_overweight': family_history, 'FAVC': favc,
        'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke,
        'CH2O': ch2o, 'SCC': scc, 'FAF': faf, 'TUE': tue,
        'CALC': calc, 'MTRANS': mtrans
    }
    input_df = pd.DataFrame([input_data])

    # Encoding fitur kategorikal
    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Scaling fitur numerik
    numerical_cols = [col for col in features.columns if col not in categorical_cols]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Prediksi menggunakan model yang dipilih
    model = models[model_name]
    prediction = model.predict(input_df)
    result = target_encoder.inverse_transform(prediction)[0]

    # Menampilkan hasil
    st.success(f"Hasil Prediksi dengan {model_name}: **{result}**")

# Instruksi menjalankan aplikasi
st.markdown("""
### Cara Menjalankan Aplikasi:
1. Simpan kode ini sebagai `app.py`.
2. Pastikan file `ObesityDataSet.csv`, `scaler.joblib`, dan ketiga file model `.joblib` ada di direktori yang sama.
3. Install dependensi:
   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```
4. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
5. Buka browser di `http://localhost:8501`.
""")