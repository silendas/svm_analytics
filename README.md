# Aplikasi Analisis Kesehatan Mental dengan SVM

Aplikasi web untuk menganalisis kesehatan mental menggunakan Support Vector Machine (SVM). Aplikasi ini memungkinkan pengguna untuk mengupload dataset kesehatan mental, melakukan preprocessing data, melatih model SVM, dan membuat prediksi status kesehatan mental berdasarkan berbagai parameter input.

## Deskripsi
Aplikasi ini dikembangkan untuk membantu para profesional kesehatan mental dan peneliti dalam menganalisis data kesehatan mental secara efisien. Menggunakan algoritma Support Vector Machine (SVM) yang powerful untuk klasifikasi, aplikasi ini dapat mengkategorikan status kesehatan mental ke dalam tiga kategori:
- Sehat
- Berisiko Rendah  
- Berisiko Tinggi

## Instalasi
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Jalankan aplikasi: `python app.py`

## System Requirements
### Software Requirements
- Python 3.8 atau lebih tinggi

### Python Dependencies
- Flask==2.0.1
- pandas==1.3.0
- numpy==1.21.0
- scikit-learn==0.24.2
- matplotlib==3.4.2
- seaborn==0.11.1

## Fitur
- Upload dataset (.csv atau .xlsx)
- Pemilihan kolom input dan target
- Visualisasi hasil dengan PCA
- Prediksi data baru
- Encoding nilai kategorikal
- Optimasi model otomatis
- Visualisasi distribusi kelas
- Cross-validation
- Metrik evaluasi model