from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import requests
from threading import Lock

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

def load_training_data():
    """Load and preprocess training data from Firebase with custom labels."""
    global training_data
    try:
        print("\n=== LOADING TRAINING DATA ===")
        response = requests.get(f'{FIREBASE_URL}/data_model.json')
        data_model = response.json()

        if not data_model or 'columns' not in data_model or 'data' not in data_model:
            raise ValueError("Invalid training data format")

        columns = data_model['columns']
        data = data_model['data']

        # Prepare features (X) and labels (y)
        X = []
        y = []

        for record in data:
            features = [float(r['value']) for r in record]
            label = float(record[8]['value'])  # Use value at index 8 as label
            X.append(features)
            y.append(label)

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

                # Split data into train and test sets for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Define parameter grid for GridSearchCV
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.1, 1, 10],
            'svm__kernel': ['rbf', 'linear']
        }

        # Define a pipeline with scaling and SVM model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, class_weight='balanced', random_state=42))
        ])

        # Perform GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2)
        grid_search.fit(X_train, y_train)

        # Evaluate model on test set
        y_pred = grid_search.best_estimator_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"Model Accuracy: {accuracy:.2f}%")
        print("Classification Report on Test Data:\n", classification_report(y_test, y_pred))

        training_data = {
            'X': X,
            'y': y,
            'columns': list(columns),
            'pipeline': grid_search.best_estimator_
        }

        print("Training data loaded successfully with custom labels.")
        print(f"Best Parameters: {grid_search.best_params_}")
        return True
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return False

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
    global training_data
    try:
        if training_data is None or 'pipeline' not in training_data:
            success = load_training_data()
            if not success:
                return jsonify({'error': 'Model not available'}), 500

        answers = request.json
        if not answers:
            return jsonify({'error': 'No data provided'}), 400

        # Prepare input features
        input_features = [float(answers.get(str(i), 0)) for i in range(len(training_data['columns']))]
        input_features = np.array([input_features], dtype=float)

        # Predict
        pipeline = training_data['pipeline']
        probabilities = pipeline.predict_proba(input_features)[0]
        prediction = pipeline.predict(input_features)[0]

        # Check if all classes are present
        class_count = probabilities.shape[0]
        if class_count < 3:
            probabilities = np.append(probabilities, [0] * (3 - class_count))

        status_map = {
            0: "Sehat",
            1: "Berisiko Ringan",
            2: "Berisiko Ringan",
            3: "Berisiko Berat"
        }
        recommendations = {
            "Sehat": "Tetap pertahankan pola hidup sehatmu saat ini dengan tidur yang cukup, olahraga rutin, dan makan makanan bergizi. Jaga keseimbangan antara aktivitas dan istirahat. Luangkan waktu untuk melakukan hobi dan bersosialisasi dengan orang-orang terdekat",
            
            "Berisiko Ringan": "Sebaiknya mulai berkonsultasi dengan psikolog profesional untuk mendapatkan panduan yang tepat. Cobalah teknik relaksasi sederhana seperti pernapasan dalam dan meditasi. Kurangi konsumsi kafein, jaga pola tidur, dan catat perubahan mood harianmu. Ceritakan perasaanmu pada orang yang kamu percaya",
            
            "Berisiko Berat": "Sangat disarankan untuk segera bertemu psikiater guna mendapatkan penanganan profesional yang sesuai. Pastikan ada pendampingan dari keluarga atau orang terdekat yang dapat dipercaya. Ikuti program terapi dan pengobatan yang direkomendasikan secara rutin. Bergabunglah dengan kelompok dukungan untuk berbagi pengalaman dan mendapat dukungan tambahan"
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
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

application = app

if __name__ == '__main__':
    app.run(debug=True)
