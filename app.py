from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import requests
from datetime import datetime
from threading import Lock
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.decomposition import PCA

app = Flask(__name__)

# URL Firebase
FIREBASE_URL = 'https://svm-health-default-rtdb.firebaseio.com'

# Global variables
model = None
scaler = None
questions = None
training_data = None

# Tambahkan global lock
plot_lock = Lock()

def load_questions():
    global questions
    # Ambil data pertanyaan
    response = requests.get(f'{FIREBASE_URL}/questions.json')
    questions = response.json()
    return questions

def load_training_data():
    global model, scaler, training_data
    try:
        print("\n=== LOADING TRAINING DATA ===")
        
        response = requests.get(f'{FIREBASE_URL}/data_model.json')
        data_model = response.json()
        
        if not data_model or 'columns' not in data_model or 'data' not in data_model:
            raise Exception("Data model tidak lengkap")
        
        columns = data_model['columns']
        data = data_model['data']
        
        # Persiapkan data untuk training
        X = []  # fitur
        y = []  # label
        
        for record in data:
            features = []
            for i in range(len(record) - 1):
                # Convert value to float explicitly
                value = float(record[i]['value'])
                features.append(value)
            
            status = float(record[-1]['value'])
            if status not in [0, 1, 2]:
                continue
                
            X.append(features)
            y.append(status)

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Pipeline dengan feature scaling yang tepat
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, 
                       kernel='rbf',
                       C=10,
                       gamma='scale',
                       class_weight={0:1, 1:2, 2:3},
                       random_state=42))
        ])
        
        # Train pipeline
        pipeline.fit(X, y)
        
        # Simpan data training
        training_data = {
            'X': X,
            'y': y,
            'columns': columns[:-1],
            'pipeline': pipeline
        }
        
        return True
        
    except Exception as e:
        print(f"\n=== ERROR LOADING DATA ===")
        print(f"Error detail: {str(e)}")
        return False

    
def train_model():
    global model, scaler, training_data
    
    try:
        if training_data is None or len(training_data) == 0:
            raise Exception("Data training tidak tersedia")
        
        # Pisahkan fitur dan label
        X = training_data.iloc[:, :-1]  # Semua kolom kecuali yang terakhir
        y = training_data.iloc[:, -1]   # Kolom terakhir sebagai label
        
        # Scaling fitur
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = SVC(
            probability=True,
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Evaluasi model
        y_pred = model.predict(X_scaled)
        print("\nEvaluasi Model:")
        print("Accuracy:", accuracy_score(y, y_pred))
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        return True
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

def initialize_app():
    global model, scaler, questions, training_data
    try:
        load_questions()
    except Exception as e:
        print(f"Error initializing app: {str(e)}")
        return False

initialize_app()

@app.route('/')
def index():
    # Ambil data model dari Firebase
    response = requests.get(f'{FIREBASE_URL}/data_model.json')
    data_model = response.json()
    
    if not data_model or 'columns' not in data_model:
        print("Data model atau columns tidak ditemukan")
        columns = []
    else:
        columns = data_model['columns']
        print("Data columns dari Firebase:", columns)
        
        # Konversi dict ke list jika perlu
        if isinstance(columns, dict):
            # Pastikan urutan sesuai dengan indeks
            max_index = max(int(k) for k in columns.keys())
            columns = [columns.get(str(i)) for i in range(max_index + 1)]
            columns = [col for col in columns if col is not None]  # Hapus None values
            
            # Filter kolom - PERBAIKAN: tambahkan penggunaan_obat__obatan_terlarang ke whitelist
            columns = [col for col in columns[1:] if col['name'] not in ['nama', 'status_kesehatan']]
            
        print("Columns setelah diproses:", columns)
    
    return render_template('index.html', questions=columns, debug=True)

@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    try:
        print("\n=== PROCESSING SUBMISSION ===")
        
        if model is None or scaler is None:
            success = load_training_data()
            if not success:
                return jsonify({'error': 'Gagal memuat model'}), 500
        
        answers = request.json
        if not answers:
            return jsonify({'error': 'Data tidak ditemukan'}), 400

        # Convert input features to float array
        input_features = []
        for i in range(len(training_data['columns'])):
            value = float(answers.get(str(i), '0'))
            input_features.append(value)

        input_features = np.array([input_features], dtype=float)
        
        # Calculate risk score
        risk_score = np.mean(input_features)
        
        # Predict using pipeline
        prediction = training_data['pipeline'].predict(input_features)[0]
        probabilities = training_data['pipeline'].predict_proba(input_features)[0]
        
        # Adjust prediction based on risk score
        if risk_score > 3:
            prediction = 2.0
        elif risk_score > 2:
            prediction = 1.0
            
        status_map = {
            0: "Sehat",
            1: "Berisiko Ringan", 
            2: "Berisiko Berat"
        }

        recommendations = {
            "Sehat": """
                1. Pertahankan pola hidup sehat Anda
                2. Lakukan olahraga teratur minimal 30 menit sehari
                3. Jaga kualitas tidur 7-8 jam per hari
                4. Lakukan kegiatan yang menyenangkan dan relaksasi
                5. Pertahankan hubungan sosial yang positif
            """,
            
            "Berisiko Ringan": """
                1. Konsultasikan dengan psikolog/konselor
                2. Praktikkan teknik manajemen stres seperti meditasi
                3. Atur jadwal kerja dan istirahat yang seimbang
                4. Batasi penggunaan media sosial dan berita negatif
                5. Bergabung dengan grup dukungan atau komunitas positif
                6. Lakukan journaling untuk mencatat perasaan
            """,
            
            "Berisiko Berat": """
                1. Segera hubungi profesional kesehatan mental/psikiater
                2. Beritahu keluarga atau teman terdekat tentang kondisi Anda
                3. Hindari membuat keputusan besar saat ini
                4. Ikuti panduan pengobatan yang diberikan dokter
                5. Pastikan ada pendamping yang dapat dihubungi 24 jam
                6. Catat nomor layanan krisis: 119 (Setiap Provinsi)
                7. Bergabung dengan support group profesional
            """
        }

        predicted_status = status_map[int(prediction)]
        
        return jsonify({
            'status': predicted_status,
            'probabilities': {
                "Sehat": f"{probabilities[0]*100:.2f}%",
                "Berisiko Ringan": f"{probabilities[1]*100:.2f}%",
                "Berisiko Berat": f"{probabilities[2]*100:.2f}%"
            },
            'recommendation': recommendations[predicted_status]
        })

    except Exception as e:
        print(f"\n=== ERROR IN PREDICTION ===")
        print(f"Error detail: {str(e)}")
        return jsonify({'error': str(e)}), 500

    
application = app    

if __name__ == '__main__':
    app.run()