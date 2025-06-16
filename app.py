import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Konfigurasi halaman
st.set_page_config(page_title="Kalkulator Obesitas", layout="wide")

# Gaya CSS kustom untuk antarmuka yang elegan
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #2d3436;
    }
    h1 {
        color: #0984e3;
        text-align: center;
        font-size: 2.8em;
        font-family: Arial, sans-serif;
    }
    .stButton > button {
        background-color: #0984e3;
        color: white;
        border-radius: 12px;
        padding: 15px 30px;
        font-size: 1.1em;
    }
    .stButton > button:hover {
        background-color: #0652dd;
    }
    .stTextInput, .stSelectbox, .stNumberInput {
        margin-bottom: 15px;
    }
    .sidebar .sidebar-content {
        background-color: #dfe6e9;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat dan memproses data
def initialize_data(file_path):
    try:
        data = pd.read_csv(file_path)
        target_col = 'NObeyesdad'
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Encoding kategori
        cat_columns = X.select_dtypes(include=['object']).columns
        encoders = {col: LabelEncoder().fit(X[col]) for col in cat_columns}
        target_enc = LabelEncoder().fit(y)
        return X, encoders, target_enc, cat_columns
    except Exception as e:
        st.error(f"Error memuat dataset: {e}")
        st.stop()

# Memuat data dan model
DATA_FILE = 'ObesityDataSet.csv'
X_data, cat_encoders, target_enc, cat_cols = initialize_data(DATA_FILE)

MODEL_FILES = {
    'Regresi Logistik': 'best_logistic_regression_model.joblib',
    'Hutan Acak': 'best_random_forest_model.joblib',
    'Mesin Vektor': 'best_svm_model.joblib'
}
loaded_models = {}
for model_name, file in MODEL_FILES.items():
    try:
        loaded_models[model_name] = joblib.load(file)
    except FileNotFoundError:
        st.error(f"Model {file} tidak ditemukan!")
        st.stop()

try:
    feature_scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Scaler tidak ditemukan!")
    st.stop()

# Antarmuka pengguna
st.title("✨ Kalkulator Tingkat Obesitas ✨")

# Sidebar untuk pemilihan model
with st.sidebar:
    st.markdown("### Pilih Algoritma")
    selected_model = st.selectbox("", list(loaded_models.keys()))

# Form input
st.markdown("### Data Pengguna")
with st.form(key='user_input'):
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### Profil")
        sex = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
        user_age = st.number_input("Umur", min_value=0, max_value=150, value=30)
        user_height = st.number_input("Tinggi (m)", min_value=0.5, max_value=2.5, value=1.65, step=0.01)
        user_weight = st.number_input("Berat (kg)", min_value=10.0, max_value=200.0, value=65.0, step=0.1)
        
        st.markdown("#### Pola Makan")
        family_obesity = st.selectbox("Riwayat Obesitas Keluarga", ["Ya", "Tidak"])
        high_cal_food = st.selectbox("Suka Makanan Berkalori Tinggi", ["Ya", "Tidak"])
        veg_freq = st.number_input("Frekuensi Sayuran (1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        meals_per_day = st.number_input("Jumlah Makan/Hari", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        snacking = st.selectbox("Ngemil", ["Tidak", "Kadang", "Sering", "Selalu"])

    with col_right:
        st.markdown("#### Aktivitas")
        smoking = st.selectbox("Merokok", ["Ya", "Tidak"])
        water_intake = st.number_input("Minum Air (liter)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        calorie_monitor = st.selectbox("Pantau Kalori", ["Ya", "Tidak"])
        exercise_freq = st.number_input("Olahraga (hari/minggu)", min_value=0.0, max_value=7.0, value=2.0, step=0.1)
        screen_time = st.number_input("Waktu Layar (jam)", min_value=0.0, max_value=24.0, value=3.0, step=0.1)
        alcohol = st.selectbox("Alkohol", ["Tidak", "Kadang", "Sering", "Selalu"])
        transport = st.selectbox("Transportasi", ["Mobil", "Motor", "Sepeda", "Transportasi Umum", "Jalan Kaki"])

    predict_button = st.form_submit_button("Hitung")

# Logika prediksi
if predict_button:
    if user_age < 0 or user_height <= 0 or user_weight <= 0:
        st.error("Masukkan data yang valid (nilai positif)!")
    else:
        input_dict = {
            'Gender': "Male" if sex == "Pria" else "Female",
            'Age': user_age, 'Height': user_height, 'Weight': user_weight,
            'family_history_with_overweight': "yes" if family_obesity == "Ya" else "no",
            'FAVC': "yes" if high_cal_food == "Ya" else "no",
            'FCVC': veg_freq, 'NCP': meals_per_day, 'CAEC': snacking.lower().capitalize(),
            'SMOKE': "yes" if smoking == "Ya" else "no", 'CH2O': water_intake,
            'SCC': "yes" if calorie_monitor == "Ya" else "no", 'FAF': exercise_freq,
            'TUE': screen_time, 'CALC': alcohol.lower().capitalize(), 'MTRANS': transport
        }
        input_frame = pd.DataFrame([input_dict])
        
        # Transformasi data
        for col in cat_cols:
            input_frame[col] = cat_encoders[col].transform(input_frame[col])
        num_cols = [col for col in X_data.columns if col not in cat_cols]
        input_frame[num_cols] = feature_scaler.transform(input_frame[num_cols])
        
        # Prediksi
        chosen_model = loaded_models[selected_model]
        pred = chosen_model.predict(input_frame)
        outcome = target_enc.inverse_transform(pred)[0]
        
        st.markdown(f"### Hasil: **{outcome}** (dengan {selected_model})")

# Petunjuk penggunaan
st.markdown("""
#### Petunjuk:
1. Simpan sebagai `app.py`.
2. Pastikan `ObesityDataSet.csv`, `scaler.joblib`, dan file model ada di folder yang sama.
3. Install: `pip install streamlit pandas scikit-learn joblib`.
4. Jalankan: `streamlit run app.py`.
5. Akses: `http://localhost:8501`.
""")