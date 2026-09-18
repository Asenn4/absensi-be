import cv2
import pymysql
import json
import uuid
import numpy as np
import os
import time
from face_logic import FaceEngine

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = ""
DB_NAME = "absensi"

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    print("=== SISTEM ABSENSI (TESTING) ===")
    
    # 1. Load data semua pengguna dari database
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT id, nama, face_embed FROM user WHERE face_embed IS NOT NULL")
        users = cursor.fetchall()
        print(f"[DB] Berhasil memuat {len(users)} pengguna dari database.")
    except Exception as e:
        print(f"❌ Error Database: {e}")
        return

    print("\n[AI] Menyiapkan Kamera dan Model Wajah...")
    engine = FaceEngine()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera!")
        return

    print("\n---> CARA PENGGUNAAN <---")
    print("Tekan 'a' : Untuk Absen (sistem akan mengecek wajahmu)")
    print("Tekan 'q' : Untuk Keluar\n")
    
    photo_path = "temp_absen.jpg"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil gambar.")
            break
            
        # Tambahkan teks panduan di layar
        cv2.putText(frame, "Tekan 'a' untuk Absen | 'q' untuk Keluar", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Kamera Absensi", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            print("\n[Proses] Memeriksa wajah...")
            cv2.imwrite(photo_path, frame)
            
            # Ekstrak wajah dari foto
            vektor_absen, pesan = engine.get_face_vector(photo_path)
            
            if os.path.exists(photo_path):
                os.remove(photo_path)
                
            if vektor_absen is None:
                print(f"❌ Wajah tidak jelas/tidak ditemukan: {pesan}")
                continue
                
            # Mencari kecocokan dengan data di database
            best_match = None
            best_score = -1
            
            for user in users:
                try:
                    vektor_db = json.loads(user['face_embed'])
                    score = cosine_similarity(vektor_absen, vektor_db)
                    
                    if score > best_score:
                        best_score = score
                        best_match = user
                except Exception as e:
                    print(f"Error membandingkan data user {user['nama']}: {e}")
                    
            # Ambang batas kemiripan (threshold), biasanya 0.4 - 0.6 untuk ArcFace
            THRESHOLD = 0.4
            
            if best_match and best_score >= THRESHOLD:
                print(f"✅ DIKENALI: {best_match['nama']} (Akurasi: {best_score:.2f})")
                
                # Simpan ke tabel absen
                try:
                    absen_id = str(uuid.uuid4())
                    sql_absen = """
                    INSERT INTO absen (id, user_id, scan_time, device_loc, status, confidence_score)
                    VALUES (%s, %s, NOW(), %s, %s, %s)
                    """
                    # Status bisa dinamis (Hadir/Terlambat), untuk sekarang kita default Hadir
                    val = (absen_id, best_match['id'], "Kamera Utama (Test)", "Hadir", float(best_score))
                    cursor.execute(sql_absen, val)
                    connection.commit()
                    print(f"📝 Berhasil mencatat kehadiran {best_match['nama']} di database!")
                except Exception as e:
                    print(f"❌ Error mencatat absen ke DB: {e}")
                    
            else:
                print(f"❌ TIDAK DIKENALI. (Skor kemiripan tertinggi: {best_score:.2f})")
                
        elif key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if 'connection' in locals() and connection.open:
        cursor.close()
        connection.close()
    print("Kamera ditutup.")

if __name__ == "__main__":
    main()
