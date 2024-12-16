from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

        # Prepare features (X)
        X = []
        for record in data:
            features = [float(r['value']) for r in record[:-1]]  # Skip the original label
            X.append(features)

        X = np.array(X, dtype=float)

        # Define custom labels (y) based on feature values
        y = []
        for features in X:
            risk_score = np.mean(features)  # Example: use the average value of features as the criterion
            if risk_score <= 2:
                y.append(0)  # Sehat
            elif 2 < risk_score <= 2.5:
                y.append(1)  # Berisiko Tinggi
            else:
                y.append(2)  # Berisiko Berat

        y = np.array(y, dtype=int)

        # Check for missing classes and add synthetic samples if necessary
        missing_classes = set([0, 1, 2]) - set(y)
        if missing_classes:
            print(f"Adding synthetic samples for missing classes: {missing_classes}")
            synthetic_X = []
            synthetic_y = []
            for cls in missing_classes:
                # Add one synthetic sample for each missing class
                synthetic_X.append(np.mean(X, axis=0))  # Use the mean feature values as synthetic data
                synthetic_y.append(cls)
            X = np.vstack([X, np.array(synthetic_X)])
            y = np.hstack([y, np.array(synthetic_y)])

        # Define a pipeline with scaling and SVM model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                probability=True,
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight={0: 1, 1: 2, 2: 3},
                random_state=42
            ))
        ])

        # Train pipeline
        pipeline.fit(X, y)

        training_data = {
            'X': X,
            'y': y,
            'columns': columns[:-1],  # Ignore the last column (label) from columns
            'pipeline': pipeline
        }

        print("Training data loaded successfully with custom labels.")
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
            2: "Berisiko Berat"
        }
        recommendations = {
            "Sehat": "Pertahankan pola hidup sehat.",
            "Berisiko Ringan": "Konsultasikan dengan psikolog.",
            "Berisiko Berat": "Hubungi psikiater segera."
        }

        predicted_status = status_map[int(prediction)]

        return jsonify({
            'status': predicted_status,
            'probabilities': {
                "Sehat": f"{probabilities[0]*100:.2f}%",
                "Berisiko Ringan": f"{probabilities[1]*100:.2f}%",
                "Berisiko Berat": f"{probabilities[2]*100:.2f}%"  # Safe fallback
            },
            'recommendation': recommendations[predicted_status]
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

application = app

if __name__ == '__main__':
    app.run(debug=True)
