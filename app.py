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

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

warnings.simplefilter(action='ignore', category=Warning)
pd.options.mode.chained_assignment = None

model = None
label_encoders = {}
X_columns = []
y_column = ''

manual_encodings = {}

def debug_dataframe(df, stage=""):
    logging.debug("\n=== DataFrame Info ({}) ===".format(stage))
    logging.debug("Shape: {}".format(df.shape))
    logging.debug("Columns: {}".format(df.columns.tolist()))
    logging.debug("Target unique values: {}".format(df[y_column].unique()))
    logging.debug("Sample data:\n{}".format(df.head()))
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
    
    try:
        column_types = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                column_types[column] = 'categorical'
            elif np.issubdtype(df[column].dtype, np.number):
                if df[column].dtype in ['int64', 'int32']:
                    column_types[column] = 'integer'
                else:
                    column_types[column] = 'float'
            else:
                column_types[column] = 'unknown'

        df = df.dropna()
        
        unique_classes = df[y_column].nunique()
        if unique_classes < 2:
            return jsonify({
                'error': f'Kolom target {y_column} hanya memiliki {unique_classes} kelas unik. Dibutuhkan minimal 2 kelas untuk klasifikasi.'
            }), 400
        
        for column in X_columns:
            if df[column].dtype == 'object':
                if column in manual_encodings and manual_encodings[column]:
                    df[column] = df[column].map(manual_encodings[column])
                else:
                    label_encoders[column] = LabelEncoder()
                    df[column] = label_encoders[column].fit_transform(df[column])
            df[column] = pd.to_numeric(df[column], errors='coerce')
        
        status_map = {
            'tidak pernah': 0,
            'jarang': 3,
            'sering': 7,
            'selalu': 10
        }
        
        if df[y_column].dtype == 'object':
            if any(val.lower() in status_map for val in df[y_column].unique()):
                df[y_column] = df[y_column].str.lower().map(status_map)
            else:
                label_encoders[y_column] = LabelEncoder()
                df[y_column] = label_encoders[y_column].fit_transform(df[y_column])
        
        def categorize_score(score):
            if score <= 3:
                return 0  # Sehat
            elif score <= 7:
                return 1  # Berisiko Rendah
            else:
                return 2  # Berisiko Tinggi

        df['category'] = df[y_column].apply(categorize_score)
        
        unique_categories = df['category'].nunique()
        if unique_categories < 2:
            return jsonify({
                'error': f'Variabel target (Y) diproses hanya menghasilkan {unique_categories} kategori. ' +
                        'Pastikan variabel target memiliki variasi nilai yang cukup untuk menghasilkan ' +
                        'minimal 2 kategori (Sehat, Berisiko Rendah, atau Berisiko Tinggi). ' +
                        'Coba periksa distribusi nilai pada kolom target.'
            }), 400
        
        X = df[X_columns].copy()
        y = df['category'].copy()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        param_grid = {
            'svm__C': [1, 10],
            'svm__gamma': ['scale', 0.1],
            'svm__kernel': ['rbf']
        }

        pipeline = Pipeline([
            ('svm', SVC(probability=True, 
                       random_state=42,
                       class_weight='balanced'))
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=3,
            cv=cv,
            scoring='accuracy',
            n_jobs=None,
            verbose=0,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        logging.info("Memulai pencarian parameter...")
        start_time = time.time()
        
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        
        end_time = time.time()
        logging.info(f"Pencarian parameter selesai dalam {end_time - start_time:.2f} detik")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy < 0.6:
            estimators = [
                ('svm', model),
                ('dt', DecisionTreeClassifier(
                    random_state=42, 
                    class_weight='balanced',
                    max_depth=5
                ))
            ]
            
            ensemble = VotingClassifier(
                estimators=estimators, 
                voting='soft',
                n_jobs=None
            )
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model = ensemble
        
        target_names = ['Sehat', 'Berisiko Rendah', 'Berisiko Tinggi']
        report = classification_report(
            y_test, 
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=1
        )
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
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
        
        input_df = pd.DataFrame([input_data])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(input_df)
        
        prediction = int(model.predict(X_scaled)[0])
        
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
        
        for _ in range(sample_size):
            sample_data = {}
            for column in X_columns:
                if column in manual_encodings:
                    values = list(manual_encodings[column].keys())
                    sample_data[column] = np.random.choice(values)
                else:
                    sample_data[column] = np.random.randint(0, 11)
            
            input_data = {}
            for column, value in sample_data.items():
                if column in manual_encodings:
                    input_data[column] = float(manual_encodings[column].get(value, 0))
                else:
                    input_data[column] = float(value)
            
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            predictions.append({
                'input_data': sample_data,
                'prediction': float(prediction)
            })
        
        return jsonify({'predictions': predictions})
        
    except Exception as e:
        print(f"Error generating sample predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/check_valid_y_columns', methods=['POST'])
def check_valid_y_columns():
    if 'last_uploaded_file' not in globals():
        return jsonify({'error': 'No file uploaded'}), 400
        
    df = globals()['last_uploaded_file']
    valid_columns = []
    
    for column in df.columns:
        unique_values = df[column].nunique()
        if unique_values >= 3:
            valid_columns.append(column)
    
    return jsonify({
        'valid_columns': valid_columns
    })

application = app

if __name__ == '__main__':
    application.run()