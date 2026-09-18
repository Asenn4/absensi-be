import os
import json
import shutil
from datetime import datetime
from typing import List, Optional

import numpy as np
import mysql.connector
from mysql.connector import Error
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "db_absensi",
}

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "yunet"
DISTANCE_THRESHOLD = 0.20

VALID_ROLES = ("admin", "user")
VALID_STATUS = ("present", "late", "alpha")

TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI(title="Sistem Absensi Pengenalan Wajah")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    """Membuka koneksi baru ke database MySQL."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Gagal koneksi ke database: {e}")


def save_temp_file(file: UploadFile) -> str:
    """Menyimpan file upload ke folder temp_images dan mengembalikan path-nya."""
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    temp_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}{ext}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path


def remove_temp_file(path: Optional[str]):
    """Menghapus file sementara jika ada, baik proses berhasil maupun gagal."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def extract_embedding(image_path: str) -> List[float]:
    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
    )
    embedding = result[0]["embedding"]
    return embedding


def cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = np.dot(a, b) / (norm_a * norm_b)
    return float(1 - similarity)

@app.post("/register")
async def register(
    nama: str = Form(...),
    nim: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
    file: UploadFile = File(...),
):
    temp_path = None
    conn = None
    try:
        role = role.lower().strip()
        if role not in VALID_ROLES:
            raise HTTPException(
                status_code=400,
                detail=f"Role tidak valid. Gunakan salah satu dari: {', '.join(VALID_ROLES)}.",
            )

        temp_path = save_temp_file(file)

        try:
            embedding = extract_embedding(temp_path)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Wajah tidak terdeteksi pada foto yang diunggah. "
                       "Gunakan foto dengan wajah yang jelas dan pencahayaan cukup.",
            )

        embedding_json = json.dumps(embedding)

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM user WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            raise HTTPException(status_code=400, detail="Email sudah terdaftar.")

        query = """
            INSERT INTO user (nama, nim, email, password, role, face_embed)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (nama, nim, email, password, role, embedding_json))
        conn.commit()
        new_id = cursor.lastrowid
        cursor.close()

        return {
            "status": "success",
            "message": "Registrasi berhasil.",
            "data": {
                "id": new_id,
                "nama": nama,
                "nim": nim,
                "email": email,
                "role": role,
            },
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan database: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada server: {e}")
    finally:
        # File sementara selalu dihapus, baik sukses maupun gagal
        remove_temp_file(temp_path)
        if conn is not None and conn.is_connected():
            conn.close()

@app.post("/login")
async def login(
    email: str = Form(...),
    password: str = Form(...),
):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, nama, nim, email, role, password, create_at FROM user WHERE email = %s",
            (email,),
        )
        user = cursor.fetchone()
        cursor.close()

        if not user or user["password"] != password:
            raise HTTPException(status_code=401, detail="Email atau password salah.")

        user.pop("password", None)
        if user.get("create_at"):
            user["create_at"] = user["create_at"].strftime("%Y-%m-%d %H:%M:%S")

        return {
            "status": "success",
            "message": "Login berhasil.",
            "data": user,
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan database: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada server: {e}")
    finally:
        if conn is not None and conn.is_connected():
            conn.close()


@app.post("/absen")
async def absen(
    file: UploadFile = File(...),
    device_loc: Optional[str] = Form(None),
):
    
    temp_path = None
    conn = None
    try:
        temp_path = save_temp_file(file)

        try:
            input_embedding = extract_embedding(temp_path)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Wajah tidak terdeteksi pada foto. Coba lagi dengan posisi wajah "
                       "yang lebih jelas dan pencahayaan yang cukup.",
            )

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, nama, face_embed FROM user")
        users = cursor.fetchall()
        cursor.close()

        if not users:
            raise HTTPException(status_code=404, detail="Belum ada data user terdaftar.")

        best_match = None
        best_distance = float("inf")

        for user in users:
            if not user.get("face_embed"):
                continue
            try:
                stored_embedding = json.loads(user["face_embed"])
            except (TypeError, json.JSONDecodeError):
                continue

            distance = cosine_distance(input_embedding, stored_embedding)
            if distance < best_distance:
                best_distance = distance
                best_match = user

        if best_match is None or best_distance >= DISTANCE_THRESHOLD:
            raise HTTPException(
                status_code=404,
                detail="Wajah tidak dikenali. Pastikan Anda sudah terdaftar sebelumnya.",
            )

        confidence_score = round((1 - best_distance) * 100, 2)

        cursor = conn.cursor()
        query = """
            INSERT INTO absen (user_id, device_loc, status, confidence_score)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (best_match["id"], device_loc, "present", confidence_score))
        conn.commit()
        id_absen = cursor.lastrowid

        cursor.execute("SELECT scan_time FROM absen WHERE id = %s", (id_absen,))
        row = cursor.fetchone()
        scan_time = row[0] if row else datetime.now()
        cursor.close()

        return {
            "status": "success",
            "message": f"Absen berhasil untuk {best_match['nama']}.",
            "data": {
                "id_absen": id_absen,
                "user_id": best_match["id"],
                "nama": best_match["nama"],
                "scan_time": scan_time.strftime("%Y-%m-%d %H:%M:%S"),
                "device_loc": device_loc,
                "status": "present",
                "confidence_score": confidence_score,
                "distance": round(float(best_distance), 4),
            },
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan database: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada server: {e}")
    finally:
        # File sementara selalu dihapus, baik proses berhasil maupun gagal
        remove_temp_file(temp_path)
        if conn is not None and conn.is_connected():
            conn.close()


@app.get("/")
async def root():
    return {"message": "Sistem Absensi Pengenalan Wajah - API aktif."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)