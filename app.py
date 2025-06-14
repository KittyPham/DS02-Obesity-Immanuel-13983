import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Obesity Data Analysis", layout="wide")
st.title("Analisis dan Pemodelan Data Obesitas")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
section = st.sidebar.radio("Pilih Bagian", ["EDA", "Preprocessing Data", "Pemodelan & Evaluasi", "Hyperparameter Tuning"])

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('ObesityDataSet.csv')
    return df

df = load_data()

# Fungsi untuk menampilkan plot
def display_plot(fig):
    st.pyplot(fig)

# EDA
if section == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Head ObesityDataSet")
    st.write(df.head())

    st.subheader("Informasi Dataset")
    st.write(f'Jumlah baris: {df.shape[0]}, jumlah kolom: {df.shape[1]}')
    st.write(df.info())

    st.subheader("Deskripsi Statistik Fitur Numerik")
    st.write(df.describe())

    st.subheader("Missing Values per Kolom")
    st.write(df.isnull().sum().to_frame('missing_count'))

    st.subheader("Unique Values per Kolom")
    st.write(df.nunique().to_frame('unique_count'))

    st.subheader("Jumlah Baris Duplikat")
    dup_count = df.duplicated().sum()
    st.write(f'Jumlah baris duplikat: {dup_count}')

    st.subheader("Distribusi Kelas NObeyesdad")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='NObeyesdad', hue='NObeyesdad', order=df['NObeyesdad'].value_counts().index, palette='Set2', legend=False)
    plt.xticks(rotation=45)
    plt.title('Distribusi Kelas NObeyesdad')
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah')
    plt.tight_layout()
    display_plot(fig)

    st.subheader("Boxplot untuk Deteksi Outlier")
    num_cols = ['Age', 'Height', 'Weight']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    fig, ax = plt.subplots(figsize=(15, 5))
    df[num_cols].plot(kind='box', subplots=True, layout=(1, 3), ax=ax)
    plt.tight_layout()
    display_plot(fig)

    st.subheader("Kesimpulan EDA")
    st.markdown("""
    - Dataset memiliki 2111 baris dan 17 kolom.
    - Terdapat 14 kolom bertipe object dan sisanya numerik.
    - Ditemukan missing values, unique values, dan data duplikat.
    - Distribusi kelas target tidak seimbang.
    - Terdapat outlier pada kolom numerik.
    """)

# Preprocessing Data
elif section == "Preprocessing Data":
    st.header("Preprocessing Data")

    # Ganti '?' menjadi NaN
    df.replace('?', np.nan, inplace=True)

    # Konversi kolom numerik
    numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Imputasi missing values
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]]).ravel()

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            df[col] = imputer.fit_transform(df[[col]])

    # Hapus duplikat
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    st.subheader("Penghapusan Duplikat")
    st.write(f"Jumlah data sebelum hapus duplikat: {before}")
    st.write(f"Jumlah data setelah hapus duplikat: {after}")
    st.write(f"Jumlah data duplikat yang dihapus: {before - after}")

    # Penanganan outlier
    st.subheader("Boxplot Sebelum dan Sesudah Penanganan Outlier")
    num_cols = ['Age', 'Height', 'Weight']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    y_limits = {col: (df[col].min(), df[col].max()) for col in num_cols}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(num_cols):
        df[[col]].plot(kind='box', ax=axes[i])
        axes[i].set_title(f"{col} (sebelum)")
        axes[i].set_ylim(y_limits[col])
    plt.tight_layout()
    display_plot(fig)

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(num_cols):
        df[[col]].plot(kind='box', ax=axes[i])
        axes[i].set_title(f"{col} (setelah)")
        axes[i].set_ylim(y_limits[col])
    plt.tight_layout()
    display_plot(fig)

    # Encoding
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols.remove('NObeyesdad')
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])
    label_encoders['NObeyesdad'] = target_encoder

    st.subheader("Data Setelah Encoding")
    st.write(df.head())

    # Heatmap korelasi
    st.subheader("Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='magma', linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Heatmap Korelasi antar Fitur")
    plt.tight_layout()
    display_plot(fig)

    # Standarisasi dan SMOTE
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Distribusi Kelas Sebelum dan Sesudah SMOTE")
    fig, ax = plt.subplots()
    sns.countplot(x=y, order=pd.Series(y).value_counts().index)
    plt.title("Distribusi Kelas Sebelum SMOTE")
    plt.xticks(rotation=45)
    display_plot(fig)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    fig, ax = plt.subplots()
    sns.countplot(x=y_resampled, order=pd.Series(y_resampled).value_counts().index)
    plt.title("Distribusi Kelas Setelah SMOTE")
    plt.xticks(rotation=45)
    display_plot(fig)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    st.subheader("Pembagian Data")
    st.write(f"Jumlah Data Train: {X_train.shape}")
    st.write(f"Jumlah Data Test: {X_test.shape}")

    st.subheader("Kesimpulan Preprocessing")
    st.markdown("""
    - Missing values diatasi dengan modus untuk kategorikal dan median untuk numerik.
    - Data duplikat dihapus.
    - Outlier ditangani menggunakan metode IQR.
    - Fitur kategorikal diencode menggunakan Label Encoding.
    - Data dinormalisasi dengan StandardScaler.
    - Ketidakseimbangan kelas ditangani dengan SMOTE.
    - Data dibagi menjadi 80% train dan 20% test.
    """)

# Pemodelan & Evaluasi
elif section == "Pemodelan & Evaluasi":
    st.header("Pemodelan & Evaluasi")

    # Simpan data yang telah diproses
    if 'X_train' not in st.session_state:
        X = df.drop('NObeyesdad', axis=1)
        y = df['NObeyesdad']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['target_encoder'] = target_encoder

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    target_encoder = st.session_state['target_encoder']

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }

    results = {}
    for model_name, model in models.items():
        st.subheader(f"Hasil Evaluasi: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        results[model_name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "y_pred": y_pred
        }

        st.write(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                    xticklabels=target_encoder.classes_,
                    yticklabels=target_encoder.classes_)
        plt.title(f"Confusion Matrix – {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        display_plot(fig)

    results_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1 Score']]
    st.subheader("Tabel Hasil Evaluasi Model")
    st.write(results_df)

    st.subheader("Perbandingan Performa Model")
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.plot(kind='bar', ax=ax, color=sns.color_palette("coolwarm", n_colors=4))
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)
    plt.title("Perbandingan Performa Model (Tanpa Tuning)")
    plt.ylabel("Skor")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()
    display_plot(fig)

    st.subheader("Kesimpulan Pemodelan & Evaluasi")
    st.markdown("""
    - Pemodelan menggunakan Logistic Regression, Random Forest, dan SVM.
    - Random Forest memiliki performa terbaik, diikuti Logistic Regression dan SVM.
    """)

# Hyperparameter Tuning
elif section == "Hyperparameter Tuning":
    st.header("Hyperparameter Tuning")

    X_train = st.session_state.get('X_train')
    X_test = st.session_state.get('X_test')
    y_train = st.session_state.get('y_train')
    y_test = st.session_state.get('y_test')
    target_encoder = st.session_state.get('target_encoder')

    if X_train is None:
        st.warning("Silakan jalankan bagian Preprocessing Data dan Pemodelan terlebih dahulu.")
    else:
        param_grid = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }

        base_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42)
        }

        tuned_results = {}
        for name in base_models:
            st.subheader(f"Tuning Model: {name}")
            grid = GridSearchCV(
                base_models[name],
                param_grid[name],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            tuned_results[name] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "y_pred": y_pred
            }

            st.write(f"Best Params: {grid.best_params_}")
            st.write(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_encoder.classes_,
                        yticklabels=target_encoder.classes_)
            plt.title(f"Confusion Matrix (Setelah Tuning) - {name}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            display_plot(fig)

        results_df = pd.DataFrame(tuned_results).T[['Accuracy', 'Precision', 'Recall', 'F1 Score']]
        st.subheader("Tabel Hasil Evaluasi Model (Setelah Tuning)")
        st.write(results_df)

        st.subheader("Perbandingan Performa Model (Setelah Tuning)")
        fig, ax = plt.subplots(figsize=(12, 6))
        results_df.plot(kind='bar', ax=ax, color=sns.color_palette("coolwarm", n_colors=4))
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)
        plt.title("Perbandingan Performa Model (Setelah Hyperparameter Tuning)")
        plt.ylabel("Skor")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=0)
        plt.legend(loc='lower right')
        plt.grid(axis='y')
        plt.tight_layout()
        display_plot(fig)

        st.subheader("Kesimpulan Hyperparameter Tuning")
        st.markdown("""
        - Hyperparameter tuning meningkatkan performa ketiga model.
        - Logistic Regression meningkat signifikan (misalnya, dari 86% ke 93%).
        - Random Forest tetap menjadi model terbaik.
        - SVM juga menunjukkan peningkatan performa.
        """)
