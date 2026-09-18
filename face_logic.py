import cv2
import numpy as np
import onnxruntime as ort
import os

class FaceEngine:
    def __init__(self):
        # Menggunakan struktur folder Anda
        yunet_path = "model/face_detection_yunet_2026may.onnx"
        arcface_path = "model/buffalo_l/w600k_r50.onnx"

        print("🔄 Memuat model AI...")
        
        # 1. Setup YuNet (Face Detection)
        self.detector = cv2.FaceDetectorYN.create(
            model=yunet_path,
            config="",
            input_size=(320, 320), # Akan diubah dinamis di bawah
            score_threshold=0.8    # Minimal yakin 80% itu wajah
        )

        # 2. Setup ArcFace (Face Recognition)
        self.session = ort.InferenceSession(arcface_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        print("✅ Model AI siap digunakan!")

    def get_face_vector(self, image_path):
        # Baca gambar dari file
        img = cv2.imread(image_path)
        if img is None:
            return None, "❌ Gambar tidak ditemukan."

        # ==========================================
        # STANDARISASI RESOLUSI SEBELUM MASUK YUNET
        # ==========================================
        max_size = 640
        h, w = img.shape[:2]
        
        # Jika sisi terpanjang gambar lebih dari 640, perkecil proporsional
        if max(h, w) > max_size:
            scale = max_size / float(max(h, w))
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # Simpan hasil resize ke dalam folder untuk dilihat
        cv2.imwrite("hasil_resize.jpg", img)
        # ==========================================

        height, width, _ = img.shape
        self.detector.setInputSize((width, height))

        # Proses Deteksi Wajah
        _, faces = self.detector.detect(img)
        if faces is None:
            return None, "❌ Tidak ada wajah yang terdeteksi."

        # Ambil wajah pertama yang ditemukan
        box = faces[0][:4].astype(int)
        x, y, w, h = box
        
        # Cegah error jika wajah berada di ujung/luar frame
        x, y = max(0, x), max(0, y)
        face_crop = img[y:y+h, x:x+w]

        if face_crop.size == 0:
            return None, "❌ Gagal memotong wajah."

        # Pre-processing untuk ArcFace (Wajib 112x112 piksel)
        face_resized = cv2.resize(face_crop, (112, 112))
        face_blob = cv2.dnn.blobFromImage(
            face_resized, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )

        # Proses Ekstraksi Vektor Wajah
        embeddings = self.session.run(None, {self.input_name: face_blob})[0]
        
        # Jadikan array 1D (512 angka)
        vector_512 = embeddings.flatten().tolist()
        return vector_512, "Sukses"

# ==========================================
# BLOK TESTING
# ==========================================
if __name__ == "__main__":
    engine = FaceEngine()
    
    test_image = "tes.jpg"
    
    if os.path.exists(test_image):
        vektor, pesan = engine.get_face_vector(test_image)
        if vektor:
            print(f"🎉 BERHASIL! Mendapatkan {len(vektor)} angka vektor.")
            print(f"Preview 5 angka pertama: {vektor[:5]}")
        else:
            print(pesan)
    else:
        print(f"⚠️ Tolong siapkan file '{test_image}' di dalam folder absensi-be untuk melakukan testing.")