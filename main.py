# main_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Sistem Prediksi Obesitas", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# --- CSS Kustom untuk Tampilan UI Modern ---
st.markdown("""
<style>
    /* Theme utama dengan gradien modern */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Header utama dengan efek glassmorphism */
    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 3.5em;
        margin-bottom: 30px;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    h2 {
        color: #ffd700;
        font-size: 2.2em;
        margin-top: 40px;
        margin-bottom: 20px;
        padding: 15px 0;
        border-bottom: 3px solid #ffd700;
        font-weight: 600;
    }
    
    h3 {
        color: #87ceeb;
        font-size: 1.6em;
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: 500;
    }

    h4 {
        color: #fff8dc;
        font-size: 1.4em;
        margin-bottom: 15px;
        padding: 10px 15px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Form container dengan glassmorphism effect */
    .stForm {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        margin-bottom: 30px;
    }

    /* Input styling modern */
    .stSelectbox > label, .stNumberInput > label {
        font-size: 1.1em;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 10px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }

    .stSelectbox div[data-baseweb="select"], .stNumberInput div[data-baseweb="input"] {
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 12px 18px;
        color: #ffffff;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stSelectbox div[data-baseweb="select"]:hover, .stNumberInput div[data-baseweb="input"]:hover {
        border-color: #ffd700;
        background: rgba(255, 255, 255, 0.25);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stSelectbox div[data-baseweb="select"] input, .stNumberInput div[data-baseweb="input"] input {
        color: #ffffff !important; 
        font-weight: 500;
    }

    /* Tombol submit dengan animasi */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #ffd93d);
        color: #333333;
        border-radius: 15px;
        padding: 15px 40px;
        font-size: 1.3em;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: all 0.4s ease;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #ffd93d, #ff6b6b);
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.6);
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin: 25px 0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stDataFrame table {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    .stDataFrame th {
        background: rgba(255, 215, 0, 0.8);
        color: #333333;
        font-weight: bold;
        padding: 15px;
    }
    
    .stDataFrame td {
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        padding: 12px 15px;
    }

    /* Alert messages dengan styling modern */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 15px;
        padding: 20px;
        margin: 25px 0;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stSuccess {
        background: rgba(40, 167, 69, 0.8);
        color: #ffffff;
    }
    
    .stError {
        background: rgba(220, 53, 69, 0.8);
        color: #ffffff;
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.8);
        color: #333333;
    }

    .stInfo {
        background: rgba(23, 162, 184, 0.8);
        color: #ffffff;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
    }

    /* Kolom layout */
    .css-1offfwp { 
        padding: 0 2rem; 
    }

    /* Custom hasil prediksi */
    .prediction-result {
        background: rgba(255, 215, 0, 0.2);
        border-left: 5px solid #ffd700;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 10px;
        font-size: 1.1em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def preprocess_data(df, show=False):
    """
    Memproses data mentah untuk persiapan machine learning
    - Membersihkan missing values
    - Encoding variabel kategorikal
    - Normalisasi fitur numerik
    """
    df_processed = df.copy()
    df_processed.replace('?', np.nan, inplace=True)
    
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

    # Konversi ke tipe numerik
    for col in numerical_features:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
    # Hapus baris dengan nilai NaN
    df_processed.dropna(inplace=True) 

    # Pisahkan fitur dan target
    X = df_processed.drop("NObeyesdad", axis=1)
    y = df_processed["NObeyesdad"]

    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        if col in X.columns:
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    # Scale numerical features
    scaler = StandardScaler()
    if all(col in X.columns for col in numerical_features):
        X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return (X, X, y_encoded, y_encoded), label_encoders, target_encoder, scaler 


def run_obesity_prediction():
    st.title("🏥 Sistem Prediksi Tingkat Obesitas")
    st.markdown("### 📋 Masukkan Data Kesehatan untuk Analisis Prediksi")
    
    # Info box
    st.info("💡 Aplikasi ini menggunakan tiga algoritma machine learning: **Random Forest**, **Logistic Regression**, dan **Support Vector Machine (SVM)** untuk memberikan prediksi yang akurat.")

    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__),"ObesityDataSet.csv")

    try:
        df_full_dataset = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"❌ Error: File 'ObesityDataSet.csv' tidak ditemukan di {data_path}.")
        st.info("📁 Pastikan file dataset berada di direktori yang benar.")
        st.stop()

    # Preprocess data
    (X_train_processed, X_test_processed, y_train_processed, y_test_processed), encoders_dict, target_encoder_main, scaler_main = preprocess_data(df_full_dataset, show=False)

    # Define feature lists
    numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categoric_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

    # Input form dengan layout yang lebih baik
    with st.form("health_data_form"):
        st.markdown("#### 📊 Formulir Input Data Kesehatan")
        
        # Layout dengan 3 kolom
        col_left, col_center, col_right = st.columns(3)

        # Kolom Kiri: Data Demografis
        with col_left:
            st.markdown("#### 👤 Data Demografis")
            gender_input = st.selectbox("Jenis Kelamin", options=["Male", "Female"])
            age_input = st.number_input("Usia (tahun)", min_value=10, max_value=100, value=25)
            height_input = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.2, value=1.70, step=0.01)
            weight_input = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        
        # Kolom Tengah: Pola Makan
        with col_center:
            st.markdown("#### 🍽️ Pola Makan")
            family_history_input = st.selectbox("Riwayat Obesitas Keluarga", options=["yes", "no"])
            favc_input = st.selectbox("Konsumsi Makanan Berkalori Tinggi", options=["yes", "no"])
            fcvc_input = st.number_input("Konsumsi Sayuran (skala 1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            ncp_input = st.number_input("Frekuensi Makan Utama/Hari", min_value=1.0, max_value=4.0, value=3.0, step=0.5)
            caec_input = st.selectbox("Makan di Luar Jadwal", options=["no", "Sometimes", "Frequently", "Always"])

        # Kolom Kanan: Aktivitas & Kebiasaan
        with col_right:
            st.markdown("#### 🏃 Aktivitas & Kebiasaan")
            smoke_input = st.selectbox("Status Merokok", options=["yes", "no"])
            ch2o_input = st.number_input("Konsumsi Air/Hari (L)", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
            scc_input = st.selectbox("Memantau Kalori", options=["yes", "no"])
            faf_input = st.number_input("Aktivitas Fisik/Minggu", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
            tue_input = st.number_input("Penggunaan Teknologi/Hari (jam)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            calc_input = st.selectbox("Konsumsi Alkohol", options=["no", "Sometimes", "Frequently", "Always"])
            mtrans_input = st.selectbox("Transportasi Utama", options=[
                "Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

        st.markdown("---")
        form_submitted = st.form_submit_button("🔮 Prediksi Tingkat Obesitas", use_container_width=True)

    # Processing dan prediksi
    if form_submitted:
        # Compile input data
        user_input = {
            'Gender': [gender_input], 'Age': [age_input], 'Height': [height_input], 'Weight': [weight_input],
            'family_history_with_overweight': [family_history_input], 'FAVC': [favc_input],
            'FCVC': [fcvc_input], 'NCP': [ncp_input], 'CAEC': [caec_input], 'SMOKE': [smoke_input],
            'CH2O': [ch2o_input], 'SCC': [scc_input], 'FAF': [faf_input], 'TUE': [tue_input],
            'CALC': [calc_input], 'MTRANS': [mtrans_input]
        }
        df_user_input = pd.DataFrame(user_input)

        # Convert numeric columns
        for col in numeric_features:
            df_user_input[col] = pd.to_numeric(df_user_input[col])
        
        st.markdown("### 📈 Data Input yang Digunakan")
        st.dataframe(df_user_input, use_container_width=True)

        # --- Model 1: Raw Model (Baseline) ---
        st.markdown("## 🔸 Hasil Prediksi - Model Baseline")
        df_raw_dataset = pd.read_csv(data_path)
        df_raw_dataset.replace('?', np.nan, inplace=True)
        for col in numeric_features:
            df_raw_dataset[col] = pd.to_numeric(df_raw_dataset[col], errors='coerce')
        df_raw_dataset.dropna(inplace=True)

        X_raw = df_raw_dataset.drop("NObeyesdad", axis=1)
        y_raw = df_raw_dataset["NObeyesdad"]
        target_encoder_raw = LabelEncoder().fit(y_raw)
        y_raw_encoded = target_encoder_raw.transform(y_raw)

        df_input_raw = df_user_input.copy()

        # Encode categorical features for raw model
        for col in categoric_features:
            le = LabelEncoder()
            if col in X_raw.columns:
                le.fit(X_raw[col])
                X_raw[col] = le.transform(X_raw[col])
            else:
                st.error(f"Kolom '{col}' tidak ditemukan.")
                return

            val = str(df_input_raw[col][0])
            if val not in le.classes_:
                st.error(f"❌ Nilai '{val}' tidak valid untuk '{col}'.")
                return
            df_input_raw[col] = le.transform([val])

        feature_order_raw = X_raw.columns.tolist()
        df_input_raw = df_input_raw[feature_order_raw]

        # Raw models
        baseline_models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Support Vector Machine": SVC(random_state=42)
        }

        for model_name, model in baseline_models.items():
            model.fit(X_raw, y_raw_encoded)
            prediction = model.predict(df_input_raw)
            result = target_encoder_raw.inverse_transform(prediction)[0]
            st.markdown(f'<div class="prediction-result">🔹 <strong>{model_name}:</strong> {result}</div>', unsafe_allow_html=True)

        # --- Model 2: Preprocessed Model ---
        st.markdown("## 🔸 Hasil Prediksi - Model dengan Preprocessing")
        
        df_input_processed = df_user_input.copy()

        # Apply encoders
        for col in categoric_features:
            le = encoders_dict[col]
            val = str(df_input_processed[col][0])
            if val not in le.classes_:
                st.error(f"❌ Nilai '{val}' tidak valid untuk '{col}'.")
                return
            df_input_processed[col] = le.transform([val])

        # Apply scaling
        df_input_processed[numeric_features] = scaler_main.transform(df_input_processed[numeric_features])

        feature_order_processed = X_train_processed.columns.tolist()
        df_input_processed = df_input_processed[feature_order_processed]

        preprocessed_models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Support Vector Machine": SVC(random_state=42)
        }

        for model_name, model in preprocessed_models.items():
            model.fit(X_train_processed, y_train_processed)
            prediction = model.predict(df_input_processed)
            result = target_encoder_main.inverse_transform(prediction)[0]
            st.markdown(f'<div class="prediction-result">🔹 <strong>{model_name}:</strong> {result}</div>', unsafe_allow_html=True)

        # --- Model 3: Hyperparameter Tuned Models ---
        st.markdown("## 🔸 Hasil Prediksi - Model dengan Hyperparameter Tuning")
        
        # Parameter grids untuk tuning
        hyperparameter_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200], 
                'max_depth': [None, 10, 20], 
                'min_samples_split': [2, 5, 10]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100], 
                'penalty': ['l2'], 
                'solver': ['lbfgs', 'liblinear']
            },
            'Support Vector Machine': {
                'C': [0.1, 1, 10], 
                'kernel': ['rbf', 'linear'], 
                'gamma': ['scale', 'auto']
            }
        }

        base_models_tuning = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Support Vector Machine': SVC(random_state=42)
        }

        with st.spinner('🔄 Melakukan hyperparameter tuning...'):
            for model_name in base_models_tuning:
                grid_search = GridSearchCV(
                    base_models_tuning[model_name], 
                    hyperparameter_grids[model_name], 
                    cv=3, 
                    scoring='accuracy', 
                    n_jobs=-1
                )
                grid_search.fit(X_train_processed, y_train_processed)
                best_model = grid_search.best_estimator_
                
                prediction = best_model.predict(df_input_processed)
                result = target_encoder_main.inverse_transform(prediction)[0]
                st.markdown(f'<div class="prediction-result">🔹 <strong>{model_name} (Tuned):</strong> {result}</div>', unsafe_allow_html=True)

        # Informasi tambahan
        st.markdown("---")
        st.markdown("### 📚 Informasi Kategori Obesitas")
        obesity_info = {
            "Kategori": ["Insufficient Weight", "Normal Weight", "Overweight Level I", "Overweight Level II", "Obesity Type I", "Obesity Type II", "Obesity Type III"],
            "Deskripsi": ["Berat badan kurang", "Berat badan normal", "Kelebihan berat badan tingkat 1", "Kelebihan berat badan tingkat 2", "Obesitas tipe 1", "Obesitas tipe 2", "Obesitas tipe 3"]
        }
        st.dataframe(pd.DataFrame(obesity_info), use_container_width=True)

# Menjalankan aplikasi
if __name__ == "__main__":
    run_obesity_prediction()