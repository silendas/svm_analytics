# Aplikasi Analisis Kesehatan Mental dengan SVM

Aplikasi web untuk menganalisis kesehatan mental menggunakan Support Vector Machine (SVM). Aplikasi ini memungkinkan pengguna untuk mengisi kuisioner kesehatan mental dan mendapatkan prediksi status kesehatan mental berdasarkan model SVM yang telah dilatih.

## Deskripsi
Aplikasi ini dikembangkan untuk membantu mendeteksi dini masalah kesehatan mental. Menggunakan algoritma Support Vector Machine (SVM), aplikasi ini dapat mengkategorikan status kesehatan mental ke dalam tiga kategori:
- Sehat
- Berisiko Ringan
- Berisiko Berat

## Fitur
- Kuisioner kesehatan mental interaktif
- Prediksi real-time dengan SVM
- Visualisasi hasil prediksi
- Probabilitas per kategori
- Rekomendasi berdasarkan hasil
- Metrik evaluasi model (akurasi, precision, recall, F1-score)
- Panel admin untuk manajemen:
  - Kelola pertanyaan kuisioner
  - Lihat dan kelola data responden
  - Monitoring performa model

## Instalasi
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Jalankan aplikasi: `python app.py`

## System Requirements
### Software Requirements
- Python 3.9
- Web browser modern

### Python Dependencies
- Flask==2.0.1
- Werkzeug==2.0.3
- gunicorn==20.1.0
- numpy==1.21.6
- pandas==1.3.5
- scikit-learn==1.0.2
- matplotlib==3.5.3
- requests==2.28.2