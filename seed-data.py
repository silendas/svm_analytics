import pandas as pd
import firebase_admin
from firebase_admin import db
import os

# Initialize Firebase
cred = firebase_admin.credentials.Certificate('cred.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://svm-health-default-rtdb.firebaseio.com'
})

def clean_key(key):
    """Membersihkan string agar aman digunakan sebagai key di Firebase"""
    # Hapus karakter yang tidak diizinkan
    invalid_chars = ['$', '#', '[', ']', '/', '.', ' ']
    clean = str(key)
    for char in invalid_chars:
        clean = clean.replace(char, '_')
    return clean.lower()

def create_value_mappings(questions_data):
    """Membuat value mappings dari data questions"""
    value_mappings = {
        "columns": {}
    }
    
    # Ubah list menjadi dictionary dengan index sebagai key
    if isinstance(questions_data, list):
        questions_data = {str(idx): question for idx, question in enumerate(questions_data)}
    
    for idx, question in questions_data.items():
        question_name = clean_key(question["name"])
        
        if question.get('options'):  # Jika memiliki pilihan jawaban
            value_mappings["columns"][question_name] = {
                "type": "categorical",
                "options": {clean_key(opt["name"]): opt["value"] for opt in question["options"]}
            }
        elif question.get("type") == "number":  # Jika bertipe number dengan range
            value_mappings["columns"][question_name] = {
                "type": "range",
                "ranges": [
                    {"min": question.get("min", 0), 
                     "max": question.get("max", 10), 
                     "value": 1}
                ],
                "default": 0
            }
    
    return value_mappings

def seed_data():
    excel_file_path = "mental_health_predictions.xlsx"

    try:
        
        # Baca file Excel
        df = pd.read_excel(excel_file_path)
        
        # Ambil data questions
        questions_ref = db.reference("questions")
        questions_data = questions_ref.get()
        
        if not questions_data:
            raise Exception("Questions data tidak ditemukan")
            
        # Konversi list ke dictionary jika perlu
        if isinstance(questions_data, list):
            questions_dict = {str(idx): question for idx, question in enumerate(questions_data)}
        else:
            questions_dict = questions_data
            
        # Hapus tabel yang perlu di-reset
        tables_to_delete = ["data_model", "value_mappings"]  # Tidak menghapus questions
        for table in tables_to_delete:
            table_ref = db.reference(table)
            table_ref.delete()
            print(f"Tabel {table} berhasil dihapus")
        
        # Buat value mappings
        value_mappings = create_value_mappings(questions_dict)
        
        # Dapatkan nama-nama kolom
        columns = [clean_key(col) for col in df.columns.tolist()]
        
        # Buat dictionary untuk columns dengan tipe data
        columns_data = {}
        for idx, col_name in enumerate(columns):
            # Skip jika kolom adalah 'nama', 'nama_', atau 'Nama'
            if col_name.lower() in ['nama', 'nama_', 'nama']:
                continue
                
            # Cari question yang sesuai
            matching_question = next(
                (questions_dict[q_idx] for q_idx, q in questions_dict.items() 
                 if clean_key(q["name"]) == col_name), 
                None
            )
            
            if matching_question:
                column_info = {
                    "name": clean_key(matching_question["name"]),
                    "original_name": matching_question["name"],
                    "type": matching_question["type"],
                    "options": [
                        {"name": clean_key(opt["name"]), 
                         "original_name": opt["name"],
                         "value": opt["value"]} 
                        for opt in matching_question.get("options", [])
                    ],
                    "min": matching_question.get("min"),
                    "max": matching_question.get("max")
                }
                
            else:
                column_info = {
                    "name": col_name,
                    "original_name": df.columns[idx],
                    "type": "string"
                }
                
            columns_data[str(len(columns_data))] = column_info
        
        # Buat dictionary untuk data
        data_records = {}
        for idx, row in df.iterrows():
            record_data = {}
            new_col_idx = 0  # Inisialisasi indeks baru
            for col_idx, value in enumerate(row):
                if pd.notna(value):  # Skip nilai NaN
                    col_name = columns[col_idx]
                    
                    # Skip jika kolom adalah 'nama', 'nama_', atau 'Nama'
                    if col_name.lower() in ['nama', 'nama_', 'nama']:
                        continue
                        
                    col_info = columns_data[str(new_col_idx)]
                    
                    # Konversi nilai sesuai tipe
                    if col_info["type"] == "number":
                        processed_value = float(value)
                    else:
                        processed_value = clean_key(str(value))
                        
                        # Jika ada mapping, gunakan nilai yang sesuai
                        if col_name in value_mappings["columns"]:
                            mapping = value_mappings["columns"][col_name]
                            if mapping["type"] == "categorical":
                                processed_value = mapping["options"].get(processed_value, value)
                    
                    record_data[str(new_col_idx)] = {
                        "value": processed_value
                    }
                    new_col_idx += 1
            
            data_records[str(idx)] = record_data
        
        # Buat data model lengkap
        data_model = {
            "columns": columns_data,
            "data": data_records
        }

        # Simpan ke Firebase
        data_model_ref = db.reference("data_model")
        value_mappings_ref = db.reference("value_mappings")
        data_model_ref.set(data_model)
        value_mappings_ref.set(value_mappings)

        print("Data berhasil diimpor ke Firebase")
        print(f"Total kolom: {len(columns)}")
        print(f"Total baris data: {len(df)}")
        print("Value mappings berhasil dibuat")

    except Exception as e:
        print(f"Error: {e}")
        raise e

if __name__ == '__main__':
    seed_data()
