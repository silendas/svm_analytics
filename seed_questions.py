import firebase_admin
from firebase_admin import db

# Initialize Firebase
cred = firebase_admin.credentials.Certificate('cred.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://svm-health-default-rtdb.firebaseio.com'
})

def seed_questions():
    questions_ref = db.reference("questions")
    questions_ref.delete()
    
    questions = [
        {
            "name": "Usia (contoh: 22)",
            "type": "number",
            "min": 1,
            "max": 100,
            "options": None
        },
        {
            "name": "Jenis Kelamin",
            "type": "string",
            "options": [
                {"name": "Perempuan", "value": 0},
                {"name": "Laki-laki", "value": 1}
            ]
        },
        {
            "name": "Status Pekerjaan",
            "type": "string",
            "options": [
                {"name": "Pelajar/Mahasiswa", "value": 3},
                {"name": "Bekerja", "value": 1},
                {"name": "Tidak Bekerja", "value": 2}
            ]
        },
        {
            "name": "Pendidikan Terakhir",
            "type": "string",
            "options": [
                {"name": "SD", "value": 5},
                {"name": "SMP", "value": 4},
                {"name": "SMA/SMK/Sederajat", "value": 3},
                {"name": "D3", "value": 2},
                {"name": "S1", "value": 1},
                {"name": "Lainnya", "value": 6} # Value unik
            ]
        },
        {
            "name": "Berapa lama Anda menggunakan media sosial dalam sehari?",
            "type": "string",
            "options": [
                {"name": "Lebih dari 5 jam", "value": 2},
                {"name": "2-4 jam", "value": 1},
                {"name": "Kurang dari 2 jam", "value": 0}
            ]
        },
        {
            "name": "Seberapa sering Anda berinteraksi di media sosial dalam sehari seperti memposting foto atau video, memberi komentar, chatting, direct message dan interaksi lainnya?",
            "type": "string",
            "options": [
                {"name": "Tidak pernah", "value": 0},
                {"name": "Jarang", "value": 1},
                {"name": "Sering", "value": 2},
                {"name": "Selalu", "value": 3}
            ]
        },
        {
            "name": "Apakah Anda merasa bahwa penggunaan teknologi digital mempengaruhi kesehatan mental Anda?",
            "type": "string",
            "options": [
                {"name": "Tidak", "value": 0},
                {"name": "Ya", "value": 3}
            ]
        },
        {
            "name": "Menurut Anda seberapa berpengaruh konten yang Anda konsumsi terhadap pola pikir Anda? (exp: setelah menonton konten olahraga Anda jadi ingin olahraga ataupun konten lain)",
            "type": "string",
            "options": [
                {"name": "Tidak pernah", "value": 0},
                {"name": "Jarang", "value": 1},
                {"name": "Sering", "value": 2},
                {"name": "Selalu", "value": 3}
            ]
        },
        {
            "name": "Seberapa sering Anda merasa cemas atau tertekan setelah menggunakan teknologi digital?",
            "type": "string",
            "options": [
                {"name": "Tidak pernah", "value": 0},
                {"name": "Jarang", "value": 1},
                {"name": "Sering", "value": 2},
                {"name": "Selalu", "value": 3}
            ]
        },
        {
            "name": "Apakah Anda pernah merasa insecure setelah melihat konten di media sosial?",
            "type": "string",
            "options": [
                {"name": "Tidak pernah", "value": 0},
                {"name": "Jarang", "value": 1},
                {"name": "Sering", "value": 2},
                {"name": "Selalu", "value": 3}
            ]
        },
        {
            "name": "Apakah Anda punya tips dan triks untuk mengatasi perasaan insecure Anda?",
            "type": "string",
            "options": [
                {"name": "Tidak", "value": 3},
                {"name": "Ya", "value": 0}
            ]
        },
        {
            "name": "Seberapa sering Anda berolahraga dalam seminggu?",
            "type": "string",
            "options": [
                {"name": "Tidak pernah", "value": 3},
                {"name": "1-2 kali seminggu", "value": 3},
                {"name": "3-4 kali seminggu", "value": 2},
                {"name": "lebih dari 5 kali seminggu", "value": 1}
            ]
        },
        {
            "name": "Berapa lama rata-rata jumlah jam tidur Anda?",
            "type": "string",
            "options": [
                {"name": "Kurang dari 4 jam perhari", "value": 2},
                {"name": "5-8 jam perhari", "value": 0},
                {"name": "Lebih dari 9 jam", "value": 1}
            ]
        },
        {
            "name": "Pola makan sehat adalah Anda selalu teratur dalam waktu makan dan makanan Anda mengandung nutrisi yang seimbang  dengan porsi yang sesuai kebutuhan Anda. Apakah Anda menerapkan pola makan sehat tersebut?",
            "type": "string",
            "options": [
                {"name": "Tidak", "value": 3},
                {"name": "Jarang", "value": 2},
                {"name": "Sering", "value": 1},
                {"name": "Selalu", "value": 0}
            ]
        },
        {
            "name": "Apakah Anda merasa lebih nyaman berkomunikasi di platform digital?",
            "type": "string",
            "options": [
                {"name": "Tidak", "value": 1},
                {"name": "Nyaman keduanya", "value": 0},
                {"name": "Ya", "value": 2}
            ]
        },
        {
            "name": "Seberapa besar skala dukungan yang Anda terima? Contoh dukungan yang Anda terima bisa berupa: 1. Memiliki orang yang bisa menjadi teman berkeluh kesah. 2. Memiliki orang yang selalu bisa diandalkan dan membantu Anda. 3. Memiliki orang yang selalu mengharg",
            "type": "string",
            "options": [
                {"name": "1", "value": 1},
                {"name": "2", "value": 2},
                {"name": "3", "value": 3},
                {"name": "4", "value": 4},
                {"name": "5", "value": 5}
            ]
        },
        {
            "name": "Seberapa sering Anda berinteraksi di luar platform digital?",
            "type": "string",
            "options": [
                {"name": "Tidak pernah", "value": 3},
                {"name": "Jarang", "value": 2},
                {"name": "Sering", "value": 1},
                {"name": "Selalu", "value": 0}
            ]
        },
        {
            "name": "Apakah Anda pernah mengalami trauma?",
            "type": "string",
            "options": [
                {"name": "Tidak", "value": 0},
                {"name": "Ya", "value": 2}
            ]
        },
        {
            "name": "Apakah Anda pernah mencari bantuan profesional terkait kesehatan mental Anda?",
            "type": "string",
            "options": [
                {"name": "Tidak", "value": 0},
                {"name": "Ya", "value": 1}
            ]
        },
        {
            "name": "Jenis bantuan profesional apa yang pernah Anda terima?",
            "type": "string",
            "options": [
                {"name": "Tidak pernah", "value": 0},
                {"name": "Konsultasi", "value": 1},
                {"name": "Pengobatan", "value": 2},
                {"name": "Terapi", "value": 3}
            ]
        }
    ]

    for idx, question in enumerate(questions):
        questions_ref.child(str(idx)).set(question)

    print("Data questions berhasil diimpor ke Firebase")

if __name__ == '__main__':
    seed_questions()
