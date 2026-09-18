import cv2
import time
import psutil


# ==========================================
# KONFIGURASI
# ==========================================

CAMERA_INDEX = 0

YUNET_MODEL = "model/face_detection_yunet_2026may.onnx"

# Resolusi yang akan diuji
RESOLUTIONS = [
    (320, 240),
    (640, 480),
    (800, 600),
    (960, 540),
    (1024, 576),
    (1024, 768),
    (1280, 720),
    (1280, 960),
    (1600, 900),
    (1600, 1200),
    (1920, 1080),
    (1920, 1440),
    (2388, 3184),
    (3184, 2388),
]


# ============================================================
# FUNGSI MEMBUKA KAMERA
# ============================================================

def open_camera(width, height):
    """
    Membuka kamera baru untuk setiap resolusi.
    Kamera ditutup kembali setelah pengujian selesai.
    """

    # CAP_MSMF adalah backend kamera Windows
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("ERROR: Kamera tidak dapat dibuka.")
        return None

    # Meminta kamera menggunakan resolusi tertentu
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Beri waktu kamera menerapkan konfigurasi
    time.sleep(1)

    return cap


# ============================================================
# FUNGSI TEST SATU RESOLUSI
# ============================================================

def test_resolution(detector, width, height):
    print("\n" + "=" * 65)
    print(f"Testing resolusi: {width} x {height}")
    print("=" * 65)

    cap = None

    try:
        # Buka kamera baru
        cap = open_camera(width, height)

        if cap is None:
            return False, None

        # Ambil resolusi aktual kamera
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Resolusi yang diminta : {width} x {height}")
        print(f"Resolusi aktual kamera: {actual_width} x {actual_height}")

        # Buang beberapa frame awal
        # Hal ini membantu menghindari frame buffer yang belum stabil
        for _ in range(5):
            ret, _ = cap.read()

            if not ret:
                print("PERINGATAN: Gagal membaca frame awal.")

        # Ambil frame utama
        ret, frame = cap.read()

        if not ret or frame is None:
            print("HASIL: GAGAL membaca frame dari kamera.")
            return False, None

        # Pastikan frame memiliki format yang valid
        print(f"Shape frame           : {frame.shape}")
        print(f"Dtype                 : {frame.dtype}")
        print(f"Contiguous            : {frame.flags['C_CONTIGUOUS']}")

        # YuNet membutuhkan gambar 3 channel BGR
        if len(frame.shape) != 3:
            print("HASIL: GAGAL - frame bukan gambar 3 dimensi.")
            return False, None

        if frame.shape[2] != 3:
            print("HASIL: GAGAL - frame bukan BGR 3 channel.")
            return False, None

        # Salin frame agar memory layout aman untuk OpenCV
        frame = frame.copy()

        frame_height, frame_width = frame.shape[:2]

        # ====================================================
        # INFERENCE YUNET
        # ====================================================

        try:
            # setInputSize menggunakan format (width, height)
            detector.setInputSize((frame_width, frame_height))

            # Mulai menghitung waktu inference
            start_time = time.perf_counter()

            _, faces = detector.detect(frame)

            end_time = time.perf_counter()

            elapsed = end_time - start_time

            fps = 1 / elapsed if elapsed > 0 else 0

            # Cek penggunaan RAM proses Python
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            jumlah_wajah = 0 if faces is None else len(faces)

            print("Deteksi YuNet        : BERHASIL")
            print(f"Jumlah wajah         : {jumlah_wajah}")
            print(f"Waktu deteksi       : {elapsed:.4f} detik")
            print(f"FPS estimasi         : {fps:.2f}")
            print(f"RAM proses           : {memory_mb:.2f} MB")

            result = {
                "requested_width": width,
                "requested_height": height,
                "actual_width": actual_width,
                "actual_height": actual_height,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "elapsed": elapsed,
                "fps": fps,
                "memory_mb": memory_mb,
                "faces": jumlah_wajah,
            }

            return True, result

        except cv2.error as e:
            print("HASIL: GAGAL inference YuNet.")
            print(f"Error OpenCV: {e}")

            return False, None

        except Exception as e:
            print("HASIL: GAGAL inference YuNet.")
            print(f"Error: {e}")

            return False, None

    except cv2.error as e:
        print("HASIL: GAGAL membaca kamera atau frame.")
        print(f"Error OpenCV: {e}")

        return False, None

    except Exception as e:
        print("HASIL: GAGAL pada proses pengujian.")
        print(f"Error: {e}")

        return False, None

    finally:
        # Selalu tutup kamera setelah satu resolusi selesai diuji
        if cap is not None:
            cap.release()

        # Bersihkan resource OpenCV
        cv2.destroyAllWindows()

        # Beri waktu backend MSMF membersihkan stream kamera
        time.sleep(1)


# ============================================================
# FUNGSI UTAMA
# ============================================================

def main():

    print("=" * 65)
    print("TEST MAKSIMAL RESOLUSI KAMERA + YUNET")
    print("=" * 65)

    # ========================================================
    # LOAD MODEL YUNET
    # ========================================================

    print("\nMemuat model YuNet...")

    try:
        detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL,
            "",
            (320, 320),
            0.9,
            0.3,
            5000
        )

        print("Model YuNet berhasil dimuat.")

    except Exception as e:
        print("ERROR: Gagal memuat model YuNet.")
        print(e)
        return

    # ========================================================
    # PENGUJIAN SEMUA RESOLUSI
    # ========================================================

    results = []

    for width, height in RESOLUTIONS:

        success, result = test_resolution(
            detector,
            width,
            height
        )

        if success and result is not None:
            results.append(result)

    # ========================================================
    # RINGKASAN HASIL
    # ========================================================

    print("\n\n")
    print("=" * 110)
    print("RINGKASAN HASIL PENGUJIAN")
    print("=" * 110)

    if not results:
        print("Tidak ada resolusi yang berhasil diuji.")
        return

    print(
        f"{'Diminta':<18}"
        f"{'Aktual':<18}"
        f"{'Frame':<18}"
        f"{'Waktu(s)':<12}"
        f"{'FPS':<10}"
        f"{'RAM(MB)':<12}"
        f"{'Wajah':<8}"
    )

    print("-" * 110)

    for result in results:

        requested = (
            f"{result['requested_width']}x"
            f"{result['requested_height']}"
        )

        actual = (
            f"{result['actual_width']}x"
            f"{result['actual_height']}"
        )

        frame = (
            f"{result['frame_width']}x"
            f"{result['frame_height']}"
        )

        print(
            f"{requested:<18}"
            f"{actual:<18}"
            f"{frame:<18}"
            f"{result['elapsed']:<12.4f}"
            f"{result['fps']:<10.2f}"
            f"{result['memory_mb']:<12.2f}"
            f"{result['faces']:<8}"
        )

    # ========================================================
    # CARI RESOLUSI TERTINGGI YANG BERHASIL
    # ========================================================

    highest = max(
        results,
        key=lambda x: x["frame_width"] * x["frame_height"]
    )

    print("\n" + "=" * 65)
    print("HASIL AKHIR")
    print("=" * 65)

    print(
        f"Resolusi tertinggi berhasil: "
        f"{highest['frame_width']} x {highest['frame_height']}"
    )

    total_pixels = (
        highest["frame_width"] *
        highest["frame_height"]
    )

    print(f"Total pixel              : {total_pixels:,}")
    print(f"FPS estimasi             : {highest['fps']:.2f}")
    print(f"RAM proses               : {highest['memory_mb']:.2f} MB")

    print("\nCatatan:")
    print("- Resolusi aktual kamera digunakan sebagai acuan utama.")
    print("- Resolusi yang diminta belum tentu didukung kamera.")
    print("- YuNet berhasil berarti frame berhasil dibaca dan diproses.")
    print("- FPS tinggi lebih penting daripada resolusi tinggi untuk real-time.")


# ============================================================
# PROGRAM DIMULAI
# ============================================================

if __name__ == "__main__":
    main()