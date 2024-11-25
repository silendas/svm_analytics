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
            try:
                features = []
                # Ambil semua kolom kecuali status kesehatan
                for i in range(len(record)-1):
                    value = record[i]['value']
                    
                    # Konversi nilai
                    if isinstance(value, str):
                        if value.lower() == 'ya':
                            value = 1.0
                        elif value.lower() == 'tidak':
                            value = 0.0
                        else:
                            try:
                                value = float(value)
                            except ValueError:
                                print(f"Nilai tidak valid: {value}")
                                continue
                    else:
                        value = float(value)
                    
                    features.append(value)
                
                # Ambil status kesehatan
                status = float(record[-1]['value'])
                if status not in [0, 1, 2]:
                    continue
                    
                X.append(features)
                y.append(status)
                
            except (ValueError, TypeError, IndexError) as e:
                print(f"Gagal memproses record: {e}")
                continue

        X = np.array(X)
        y = np.array(y)
        
        print(f"\nShape data awal - X: {X.shape}, y: {y.shape}")
        
        # Pipeline preprocessing dan model dengan parameter yang lebih sesuai
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                probability=True,
                C=1.0,  # Nilai C yang lebih kecil untuk mengurangi overfitting
                gamma='auto',
                class_weight='balanced',
                random_state=42,
                max_iter=5000
            ))
        ])
        
        # Train model
        pipeline.fit(X, y)
        
        # Simpan model dan scaler
        model = pipeline.named_steps['svm']
        scaler = pipeline.named_steps['scaler']
        
        # Evaluasi model
        y_pred = pipeline.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"\nAkurasi model: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Simpan data training
        training_data = {
            'X': X,
            'y': y,
            'X_scaled': scaler.transform(X),
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
        
        # Reset model jika terjadi error sebelumnya
        global model, scaler, training_data
        if model is None or scaler is None:
            success = load_training_data()
            if not success:
                return jsonify({'error': 'Gagal memuat model'}), 500
        
        # Ambil dan validasi input
        answers = request.json
        if not answers:
            return jsonify({'error': 'Data tidak ditemukan'}), 400

        try:
            # Persiapkan input data
            input_features = []
            for i in range(len(training_data['columns'])):
                value = answers.get(str(i), '0')
                
                # Konversi nilai
                if isinstance(value, str):
                    if value.lower() == 'ya':
                        value = 1.0
                    elif value.lower() == 'tidak':
                        value = 0.0
                    else:
                        value = float(value)
                else:
                    value = float(value)
                    
                input_features.append(value)
            
            input_features = np.array([input_features])
            
            print(f"Input shape: {input_features.shape}")
            
            # Gunakan pipeline untuk prediksi
            prediction = training_data['pipeline'].predict(input_features)[0]
            probabilities = training_data['pipeline'].predict_proba(input_features)[0]
            
            print(f"Hasil prediksi: {prediction}")
            print(f"Probabilitas: {probabilities}")
            
            # Mapping status
            status_map = {
                0: "Sehat",
                1: "Berisiko Ringan",
                2: "Berisiko Berat"
            }
            
            predicted_status = status_map[int(prediction)]
            
            # Format probabilitas
            prob_status = {
                "Sehat": f"{probabilities[0]*100:.2f}%",
                "Berisiko Ringan": f"{probabilities[1]*100:.2f}%",
                "Berisiko Berat": f"{probabilities[2]*100:.2f}%"
            }

            # Rekomendasi
            recommendations = {
                "Sehat": "Pertahankan pola hidup sehat Anda. Tetap jaga keseimbangan aktivitas fisik dan mental.",
                "Berisiko Ringan": "Perhatikan kesehatan mental Anda. Disarankan untuk berkonsultasi dengan psikolog.",
                "Berisiko Berat": "Segera konsultasikan kondisi Anda dengan profesional kesehatan mental atau psikiater."
            }

            # Generate metrics
            metrics = generate_metrics()
            
            # Generate visualisasi
            plot_url = generate_visualization(input_features[0], prediction)
            
            if plot_url is None:
                return jsonify({
                    'status': predicted_status,
                    'probabilities': prob_status,
                    'recommendation': recommendations[predicted_status],
                    'metrics': metrics
                })

            # Cleanup plot setelah selesai
            plt.close('all')
            
            return jsonify({
                'status': predicted_status,
                'probabilities': prob_status,
                'recommendation': recommendations[predicted_status],
                'plot': plot_url,
                'metrics': metrics
            })

        except (ValueError, TypeError, KeyError) as e:
            print(f"\n=== ERROR PROCESSING INPUT ===")
            print(f"Error detail: {str(e)}")
            return jsonify({'error': f'Format input tidak valid: {str(e)}'}), 400

    except Exception as e:
        print("\n=== ERROR IN PREDICTION ===")
        print(f"Error detail: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return jsonify({'error': str(e)}), 500

def generate_visualization(input_point, prediction):
    try:
        with plot_lock:
            plt.clf()
            plt.close('all')
            
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
            
            # Ambil data training
            X_scaled = training_data['X_scaled']
            y = training_data['y']
            
            # Gunakan PCA untuk mereduksi dimensi menjadi 1D
            pca = PCA(n_components=1)
            X_pca = pca.fit_transform(X_scaled)
            
            # Transform input point menggunakan PCA yang sama
            input_pca = pca.transform(input_point.reshape(1, -1))
            
            # Scatter plot untuk setiap kategori
            class_names = ['Sehat', 'Berisiko Ringan', 'Berisiko Berat']
            colors = ['green', 'yellow', 'red']
            
            # Tambahkan jitter untuk menghindari tumpang tindih
            jitter = np.random.normal(0, 0.1, X_pca.shape[0])
            
            for i, (label, color) in enumerate(zip(class_names, colors)):
                mask = y == i
                plt.scatter(X_pca[mask, 0] + jitter[mask], 
                          np.ones(np.sum(mask)) * (i + 1), 
                          color=color, 
                          label=label, 
                          alpha=0.6,
                          s=100)
            
            # Plot titik input
            plt.scatter(input_pca[0], prediction + 1, 
                       color='blue', 
                       marker='*', 
                       s=300,
                       label='Input Anda',
                       zorder=5)
            
            plt.title('Visualisasi Status Kesehatan Mental', fontsize=14, pad=20)
            plt.xlabel('Nilai Fitur (PCA)', fontsize=12)
            plt.ylabel('Status Kesehatan Mental', fontsize=12)
            plt.yticks([1, 2, 3], class_names, fontsize=10)
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            
            plot_url = base64.b64encode(img.getvalue()).decode()
            
            plt.close(fig)
            plt.close('all')
            
            return plot_url
            
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        return None

def generate_metrics():
    try:
        # Gunakan data training yang sudah ada
        X_scaled = training_data['X_scaled']
        y = training_data['y']
        
        # Lakukan prediksi pada data training
        y_pred = model.predict(X_scaled)
        
        # Hitung metrik
        report = classification_report(y, y_pred, output_dict=True)
        accuracy = accuracy_score(y, y_pred)
        
        # Format output
        metrics = {
            'accuracy': accuracy * 100,  # Konversi ke persen
            'report': {
                'Sehat': {
                    'precision': report['0.0']['precision'],
                    'recall': report['0.0']['recall'],
                    'f1_score': report['0.0']['f1-score']
                },
                'Berisiko Ringan': {
                    'precision': report['1.0']['precision'],
                    'recall': report['1.0']['recall'],
                    'f1_score': report['1.0']['f1-score']
                },
                'Berisiko Berat': {
                    'precision': report['2.0']['precision'],
                    'recall': report['2.0']['recall'],
                    'f1_score': report['2.0']['f1-score']
                }
            }
        }
        
        print("\n=== METRICS ===")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        return metrics
        
    except Exception as e:
        print(f"Error generating metrics: {str(e)}")
        return {
            'accuracy': 0.0,
            'report': {
                'Sehat': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
                'Berisiko Ringan': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
                'Berisiko Berat': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            }
        }

@app.route('/manage')
def manage():
    return render_template('manage_menu.html')

@app.route('/manage/questions')
def manage_questions():
    if questions is None:
        load_questions()
    return render_template('manage_questions.html', questions=questions)

@app.route('/manage/data')
def manage_data():
    try:
        # Ambil data kolom dari Firebase
        response = requests.get(f'{FIREBASE_URL}/data_model/columns.json')
        columns_data = response.json()
        
        # Konversi list ke dictionary jika diperlukan
        if isinstance(columns_data, list):
            columns_data = {str(idx): col for idx, col in enumerate(columns_data)}
        elif columns_data is None:
            columns_data = {}
            
        # Ambil data records
        response = requests.get(f'{FIREBASE_URL}/data_model/data.json')
        records_data = response.json()
        
        # Konversi list ke dictionary jika diperlukan
        if isinstance(records_data, list):
            records_data = {str(idx): record for idx, record in enumerate(records_data)}
        elif records_data is None:
            records_data = {}
        
        return render_template('manage_data.html', 
                             columns_data=columns_data,
                             records_data=records_data)
                             
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return render_template('manage_data.html', 
                             columns_data={},
                             records_data={},
                             error="Gagal memuat data")

@app.route('/api/questions', methods=['GET', 'POST'])
def handle_questions():
    if request.method == 'POST':
        data = request.json
        if 'id' in data:  # update
            requests.patch(f'{FIREBASE_URL}/questions/{data["id"]}.json', json=data)
        else:  # create
            response = requests.get(f'{FIREBASE_URL}/questions.json')
            current_data = response.json() or {}
            new_id = str(len(current_data))
            requests.put(f'{FIREBASE_URL}/questions/{new_id}.json', json=data)
        return jsonify({'success': True})
    
    response = requests.get(f'{FIREBASE_URL}/questions.json')
    return jsonify(response.json())

@app.route('/api/questions/<id>', methods=['DELETE'])
def delete_question(id):
    requests.delete(f'{FIREBASE_URL}/questions/{id}.json')
    return jsonify({'success': True})

@app.route('/api/data_model/columns')
def get_columns():
    response = requests.get(f'{FIREBASE_URL}/data_model.json')
    data_model = response.json()
    
    if not data_model or 'columns' not in data_model:
        return jsonify([])
        
    columns = data_model['columns']
    
    # Konversi dict ke list jika perlu
    if isinstance(columns, dict):
        columns = [columns[str(i)] for i in range(len(columns))]
        
    return jsonify(columns)

@app.route('/api/data_model/data')
def get_data():
    response = requests.get(f'{FIREBASE_URL}/data_model.json')
    data_model = response.json()
    
    if not data_model or 'data' not in data_model:
        return jsonify({})
        
    return jsonify(data_model['data'])

@app.route('/api/data_model/data', methods=['POST'])
def add_data():
    new_data = request.json
    
    # Format data sesuai struktur yang diinginkan
    formatted_data = {
        "usia": {
            "name": "usia",
            "value": new_data.get('0')  # Ambil nilai usia
        },
        "jenis_kelamin": {
            "name": "jenis_kelamin",
            "value": new_data.get('1')  # Ambil nilai jenis kelamin
        },
        "status_kesehatan": {
            "name": "status_kesehatan",
            "value": new_data.get('2', 0)  # Ambil nilai status kesehatan, default 0
        }
    }
    
    # Generate ID baru
    new_id = str(int(datetime.now().timestamp() * 1000))
    
    # Update data di Firebase
    requests.patch(f'{FIREBASE_URL}/data_model/data.json', json={new_id: formatted_data})
    
    return jsonify({'success': True})
    
application = app    

if __name__ == '__main__':
    app.run()