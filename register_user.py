import cv2
import pymysql
import uuid
import json
import os
from face_logic import FaceEngine

# Konfigurasi Database (sesuaikan jika ada password di MySQL-mu)
DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = ""
DB_NAME = "absensi"

def main():
    print("=== PENDAFTARAN PENGGUNA BARU ===")
    nama = input("Masukkan Nama: ")
    email = input("Masukkan Email: ")
    password = input("Masukkan Password: ")
    role = input("Masukkan Role (admin/user) [tekan enter untuk default 'user']: ")
    if not role:
        role = "user"
        
    print("\n[AI] Menyiapkan Kamera dan Model Wajah...")
    engine = FaceEngine()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera!")
        return

    print("\n---> Tekan 's' pada jendela kamera untuk MENGAMBIL FOTO, atau 'q' untuk BATAL <---")
    photo_path = "temp_register.jpg"
    
    face_captured = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil gambar dari kamera.")
            break
            
        cv2.imshow("Pendaftaran - Tekan 's' untuk jepret", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(photo_path, frame)
            face_captured = True
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if not face_captured:
        print("Pendaftaran dibatalkan.")
        return
        
    print("\n[AI] Mengekstrak DNA wajah...")
    vektor, pesan = engine.get_face_vector(photo_path)
    
    # Hapus foto sementara
    if os.path.exists(photo_path):
        os.remove(photo_path)
        
    if vektor is None:
        print(f"❌ Gagal ekstrak wajah: {pesan}")
        return
        
    print("✅ Wajah berhasil diekstrak! Menyimpan data ke database...")
    
    # Ubah array vektor menjadi format string JSON agar bisa disimpan di kolom Text MySQL
    vektor_str = json.dumps(vektor)
    user_id = str(uuid.uuid4())
    
    # Simpan ke DB MySQL
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cursor = connection.cursor()
        
        sql = """
        INSERT INTO user (id, nama, email, password, role, face_embed, create_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """
        val = (user_id, nama, email, password, role, vektor_str)
        cursor.execute(sql, val)
        connection.commit()
        
        print(f"\n🎉 BERHASIL! Pengguna '{nama}' berhasil didaftarkan ke database.")
        
    except Exception as e:
        print(f"❌ Error Database: {e}")
        print("Pastikan XAMPP MySQL nyala dan nama database 'pt_absensi' sudah benar.")
    finally:
        if 'connection' in locals() and connection.open:
            cursor.close()
            connection.close()

if __name__ == "__main__":
    main()
