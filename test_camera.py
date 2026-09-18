import cv2

def test_camera():
    print("Membuka kamera... Tekan 'q' pada keyboard untuk keluar.")
    
    # Membuka koneksi ke kamera utama (biasanya index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera!")
        print("Coba ganti angka 0 pada cv2.VideoCapture(0) menjadi 1 atau 2 jika kamu memiliki lebih dari 1 kamera.")
        return

    while True:
        # Membaca frame dari kamera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Gagal mendapatkan gambar dari kamera.")
            break
            
        # Menampilkan gambar dalam window
        cv2.imshow('Test Kamera (Tekan "q" untuk menutup)', frame)
        
        # Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Membersihkan resource setelah selesai
    cap.release()
    cv2.destroyAllWindows()
    print("Kamera ditutup.")

if __name__ == "__main__":
    test_camera()
