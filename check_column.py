# Import library yang diperlukan
import pandas as pd

# Baca file dataset (sesuaikan path file Anda)
df = pd.read_excel('datas.xlsx')  # Ganti 'dataset.csv' dengan nama file Anda

# Tampilkan nama-nama kolom
print("\nNama Kolom dalam Dataset:")
print(df.columns.tolist())

# Tampilkan informasi dataset
print("\nInformasi Dataset:")
print(df.info())

# Tampilkan 5 baris pertama data
print("\n5 Baris Pertama Data:")
print(df.head())

# Tampilkan statistik dasar dataset
print("\nStatistik Dasar Dataset:")
print(df.describe())

# Cek nilai unik dalam setiap kolom
print("\nNilai Unik dalam Setiap Kolom:")
for column in df.columns:
    print(f"\nKolom '{column}':")
    print(df[column].unique())
