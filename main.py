from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

# Import class AI yang sudah Anda buat tadi
from face_logic import FaceEngine

# Inisialisasi Aplikasi FastAPI
app = FastAPI(title="Jetson Attend API")

# Konfigurasi CORS (PENTING!)
# Ini mengizinkan Next.js (port 3000) untuk berkomunikasi dengan Python (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Saat production ganti dengan URL Next.js (misal: http://localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nyalakan Mesin AI saat server pertama kali hidup
print("Menyiapkan Server AI...")
engine = FaceEngine()

@app.get("/")
def home():
    return {"message": "API Jetson Attend Backend Aktif dan Siap Menerima Foto! 🚀"}

@app.post("/api/extract")
async def extract_face(file: UploadFile = File(...)):
    # 1. Simpan foto yang dikirim dari browser ke file sementara
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Serahkan foto tersebut ke face_logic.py
    vektor, pesan = engine.get_face_vector(temp_file)
    
    # 3. Hapus foto sementaranya agar hardisk tidak penuh
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    # 4. Kembalikan hasilnya ke Next.js dalam format JSON
    if vektor is None:
        return {"status": "error", "message": pesan}
        
    return {
        "status": "success",
        "message": "DNA Wajah berhasil diekstrak",
        "vector_length": len(vektor),
        "vector_data": vektor
    }