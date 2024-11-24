from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging
import warnings
import time

# Matikan warning
warnings.simplefilter(action='ignore', category=Warning)
pd.options.mode.chained_assignment = None

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Variabel global untuk menyimpan model dan label encoder
model = None
label_encoders = {}
X_columns = []
y_column = ''

# Variabel global untuk menyimpan encoding manual
manual_encodings = {}

def debug_dataframe(df, stage=""):
    logging.debug(f"\n=== DataFrame Info ({stage}) ===")
    logging.debug(f"Shape: {df.shape}")
    logging.debug(f"Columns: {df.columns.tolist()}")
    logging.debug(f"Target unique values: {df[y_column].unique()}")
    logging.debug(f"Sample data:\n{df.head()}")
    logging.debug("="*50)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    
    # Simpan DataFrame ke variabel global
    globals()['last_uploaded_file'] = df
    
    columns = df.columns.tolist()
    return jsonify({'columns': columns})

@app.route('/process', methods=['POST'])
def process_data():
    global model, label_encoders, X_columns, y_column, scaler
    
    data = request.json
    X_columns = data['X_columns']
    y_column = data['y_column']
    
    if 'last_uploaded_file' not in globals():
        return jsonify({'error': 'No file uploaded'}), 400
        
    df = globals()['last_uploaded_file'].copy()
    
    # Simpan tipe data kolom
    column_types = {col: str(df[col].dtype) for col in X_columns}
    
    try:
        # 1. Handle missing values
        df = df.dropna()
        
        # 2. Encoding untuk fitur kategorikal
        for column in X_columns:
            if df[column].dtype == 'object':
                if column in manual_encodings and manual_encodings[column]:
                    df[column] = df[column].map(manual_encodings[column])
                else:
                    label_encoders[column] = LabelEncoder()
                    df[column] = label_encoders[column].fit_transform(df[column])
            df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # 3. Encoding untuk target
        status_map = {
            'tidak pernah': 0,
            'jarang': 3,
            'sering': 7,
            'selalu': 10
        }
        
        if df[y_column].dtype == 'object':
            df[y_column] = df[y_column].map(status_map)
        
        # 4. Kategorikan target untuk klasifikasi
        def categorize_score(score):
            if score <= 3:
                return 0  # Sehat
            elif score <= 7:
                return 1  # Berisiko Rendah
            else:
                return 2  # Berisiko Tinggi

        df['category'] = df[y_column].apply(categorize_score)
        
        # 5. Persiapkan data untuk model - Perbaiki warning DataFrame
        X = df[X_columns].copy()  # Tambahkan .copy()
        y = df['category'].copy()
        
        # Scaling langsung tanpa polynomial features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Parameter grid yang lebih sederhana
        param_grid = {
            'svm__C': [1, 10],  # Hanya 2 nilai
            'svm__gamma': ['scale', 0.1],  # Hanya 2 nilai
            'svm__kernel': ['rbf']  # Hanya 1 kernel
        }

        # Buat pipeline yang lebih sederhana
        pipeline = Pipeline([
            ('svm', SVC(probability=True, 
                       random_state=42,
                       class_weight='balanced'))  # Tetapkan class_weight
        ])

        # Kurangi cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Gunakan RandomizedSearchCV yang lebih cepat
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=5,  # Hanya coba 5 kombinasi
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,  # Gunakan semua CPU core
            verbose=1,
            random_state=42
        )

        # Split data dengan rasio yang lebih kecil untuk testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Latih model
        logging.info("Memulai pencarian parameter...")
        start_time = time.time()
        
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        
        end_time = time.time()
        logging.info(f"Pencarian parameter selesai dalam {end_time - start_time:.2f} detik")

        # Evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Jika akurasi rendah, gunakan ensemble
        if accuracy < 0.6:
            estimators = [
                ('svm', model),
                ('dt', DecisionTreeClassifier(
                    random_state=42, 
                    class_weight='balanced',
                    max_depth=5  # Batasi kedalaman untuk mencegah overfitting
                ))
            ]
            
            ensemble = VotingClassifier(
                estimators=estimators, 
                voting='soft',
                n_jobs=-1
            )
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model = ensemble
        
        # Classification report
        target_names = ['Sehat', 'Berisiko Rendah', 'Berisiko Tinggi']
        report = classification_report(
            y_test, 
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=1
        )
        
        # PCA untuk visualisasi
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot hasil
        plt.figure(figsize=(12, 8))
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        
        for i, label in enumerate(range(3)):
            mask = (y == label)
            if np.any(mask):
                plt.scatter(
                    X_pca[mask, 0], 
                    X_pca[mask, 1],
                    c=colors[i],
                    label=target_names[i],
                    alpha=0.7,
                    s=100
                )
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Visualisasi Data dengan PCA dan Klasifikasi')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Simpan plot
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'report': report,
            'accuracy': accuracy,
            'plot': plot_url,
            'column_types': column_types,
            'pca_components': {
                'PC1': X_pca[:, 0].tolist(),
                'PC2': X_pca[:, 1].tolist()
            },
            'explained_variance': {
                'PC1': float(pca.explained_variance_ratio_[0]),
                'PC2': float(pca.explained_variance_ratio_[1])
            },
            'class_distribution': pd.Series(y).value_counts().to_dict()
        })
        
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = {}
        
        # Encoding input data
        for column in X_columns:
            if column not in data:
                return jsonify({'error': f'Missing column: {column}'}), 400
                
            value = data[column]
            if column in manual_encodings:
                encoded_value = manual_encodings[column].get(value)
                if encoded_value is None:
                    return jsonify({'error': f'Invalid value for {column}: {value}'}), 400
                input_data[column] = float(encoded_value)
            else:
                input_data[column] = float(value)
        
        # Buat DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Normalisasi input
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(input_df)
        
        # Prediksi
        prediction = int(model.predict(X_scaled)[0])
        
        # Konversi ke status
        status_map = {
            0: "Sehat",
            1: "Berisiko Rendah",
            2: "Berisiko Berat"
        }
        
        color_map = {
            0: "green",
            1: "yellow",
            2: "red"
        }
        
        status = status_map[prediction]
        color = color_map[prediction]
        
        return jsonify({
            'prediction': status,
            'class': prediction,
            'color': color
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/check_categorical', methods=['POST'])
def check_categorical():
    data = request.json
    columns = data['columns']
    
    if 'last_uploaded_file' not in globals():
        return jsonify({'error': 'No file uploaded'}), 400
        
    df = globals()['last_uploaded_file']
    
    categorical_columns = {}
    for column in columns:
        if df[column].dtype == 'object':
            categorical_columns[column] = df[column].unique().tolist()
    
    return jsonify({
        'categorical_columns': list(categorical_columns.keys()),
        'unique_values': categorical_columns
    })

@app.route('/save_encoding', methods=['POST'])
def save_encoding():
    global manual_encodings
    manual_encodings = request.json
    return jsonify({'status': 'success'})

@app.route('/get_sample_predictions', methods=['POST'])
def get_sample_predictions():
    try:
        sample_size = 5
        predictions = []
        
        # Generate sample data
        for _ in range(sample_size):
            sample_data = {}
            for column in X_columns:
                if column in manual_encodings:
                    # Untuk kolom kategorikal
                    values = list(manual_encodings[column].keys())
                    sample_data[column] = np.random.choice(values)
                else:
                    # Untuk kolom numerik
                    sample_data[column] = np.random.randint(0, 11)
            
            # Transformasi data
            input_data = {}
            for column, value in sample_data.items():
                if column in manual_encodings:
                    input_data[column] = float(manual_encodings[column].get(value, 0))
                else:
                    input_data[column] = float(value)
            
            # Prediksi
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            # Tambahkan ke hasil
            predictions.append({
                'input_data': sample_data,
                'prediction': float(prediction)
            })
        
        return jsonify({'predictions': predictions})
        
    except Exception as e:
        print(f"Error generating sample predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)