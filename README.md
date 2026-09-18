# Absensi Backend (absensi-be)

Sistem backend untuk absensi wajah (Face Recognition) menggunakan **FastAPI**, **YuNet** (deteksi wajah), dan **ArcFace** (ekstraksi vektor wajah). 

## 🛠 Prasyarat (Prerequisites)
Sebelum menjalankan proyek ini, pastikan Anda sudah menginstal:
- [Python 3.11.7](https://www.python.org/downloads/release/python-3117/) (Wajib untuk stabilitas library AI)
- [MySQL Server](https://dev.mysql.com/downloads/) (XAMPP / Laragon / Standalone)
- [Git](https://git-scm.com/)

## 🚀 Cara Instalasi (Setup)

### 1. Clone Repository
```bash
git clone https://github.com/Asenn4/absensi-be.git
cd absensi-be
```

### 2. Buat Virtual Environment (Sangat Direkomendasikan)
Gunakan virtual environment agar dependensi Python tidak bercampur dengan proyek lain.
```bash
# Membuat virtual environment (Pastikan menggunakan Python 3.11.7)
python -m venv .venv

# Mengaktifkan virtual environment (Windows)
.\.venv\Scripts\Activate.ps1
# Jika menggunakan CMD:
# .\.venv\Scripts\activate.bat
```

### 3. Instal Library (Dependencies)
Pastikan virtual environment sudah aktif (ada tulisan `(.venv)` di terminal), lalu jalankan:
```bash
pip install -r requirements.txt
```

### 4. Persiapan Model AI (Penting!)
Folder `model/` akan diabaikan oleh Git, jadi Anda harus mengunduh dan memasukkan model-model AI secara manual ke dalam folder tersebut dengan struktur seperti ini:
```text
absensi-be/
│
├── model/
│   ├── face_detection_yunet_2026may.onnx
│   └── buffalo_l/
│       └── w600k_r50.onnx
```
*(Hubungi developer utama atau cari link download untuk file `.onnx` di atas).*

### 5. Konfigurasi Database
1. Buka MySQL Anda (phpMyAdmin atau terminal).
2. Buat database baru sesuai dengan nama yang ada di dalam skrip (misal: `absensi_db`).
3. Konfigurasikan koneksi database di file kode atau file `.env` (pastikan username dan password MySQL Anda sudah sesuai).

## ▶️ Cara Menjalankan Aplikasi

Jika Anda ingin menjalankan server backend utama (FastAPI):
```bash
uvicorn main:app --reload
```
Atau jika server disetel di file lain seperti `main.py`:
```bash
python main.py
```

Jika Anda ingin menguji deteksi atau input manual, Anda bisa menggunakan:
```bash
python test_absensi.py
python register_user.py
```

## 📝 Catatan Penting
- Jangan pernah memodifikasi langsung file model `.onnx` atau menghapusnya.
- Pastikan kamera (Webcam) tidak sedang digunakan oleh aplikasi lain jika ada kode yang mengakses kamera secara langsung.
