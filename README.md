# 🌌 NEBULA HANDS

Sistem partikel 3D interaktif yang dikendalikan **gestur tangan** lewat webcam, dibuat dengan **Python, OpenCV, MediaPipe, dan NumPy**.

Layar terbagi dua: kamera dengan kerangka tangan di kiri, kanvas partikel di kanan. Setiap tangan yang terdeteksi mendapat bentuknya sendiri.

## ✨ Gestur

| Gestur | Efek |
|---|---|
| ✋ 5 jari | Jaringan node memenuhi layar (tangan menyibak node) |
| ☝️ 1 jari | Otak 3D dengan neuron yang menyala |
| ✌️ 2 jari | Teks 3D (default: "I LOVE YOU") |
| 3 jari | Hati 3D berdetak |
| 4 jari (jempol dilipat) | Planet bercincin |
| 🤘 Metal | Kubus 3D berputar |
| ✊ Kepalan, lalu buka | Bola energi mengisi daya, lalu meledak |
| 🤏 Cubit | Lubang hitam |
| 🫰 Finger heart | Hujan hati kecil di seluruh layar |
| 🫶 Dua tangan membentuk hati | Hati besar |

**Multi-tangan:** dua tangan 3 jari menghasilkan 2 hati, tangan kiri 1 jari dan tangan kanan metal menghasilkan otak dan kubus, dan seterusnya. Pengecualian: 5 jari dan finger heart selalu menjadi efek layar penuh.

## 🚀 Instalasi

Butuh Python 3.9–3.12 dan webcam.

```bash
git clone https://github.com/<username>/nebula-hands.git
cd nebula-hands
pip install -r requirements.txt
python nebula_hands.py
```

Saat pertama dijalankan, model `hand_landmarker.task` (±8 MB) diunduh otomatis.

### Opsi

```bash
python nebula_hands.py --text "HALO DUNIA"   # ganti teks gestur 2 jari
python nebula_hands.py --particles 4000      # lebih ringan untuk laptop lambat
python nebula_hands.py --hands 3             # deteksi hingga 3 tangan (maks 4)
python nebula_hands.py --width 640 --height 480
python nebula_hands.py --layout overlay      # partikel di atas gambar kamera
```

## ⌨️ Tombol

| Tombol | Fungsi |
|---|---|
| `Q` / `Esc` | Keluar |
| `L` | Tampilan terbagi / gabung |
| `S` | Kerangka tangan on/off |
| `C` | Ganti palet warna |
| `G` | Glow on/off |
| `[` `]` | Kecerahan |
| `F` | Layar penuh |
| `P` | Screenshot |
| `H` | Bantuan on/off |

## 🛠️ Mengatasi Masalah

- **`CERTIFICATE_VERIFY_FAILED` di macOS**: jalankan `open "/Applications/Python 3.12/Install Certificates.command"`, atau unduh [model ini](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) lewat browser dan simpan di folder project.
- **Layar hitam / kamera gagal dibuka (macOS)**: buka *System Settings → Privacy & Security → Camera*, aktifkan Terminal, lalu jalankan ulang. Kalau punya lebih dari satu kamera, coba `--camera 1`.
- **Terasa lambat**: kurangi partikel dengan `--particles 4000`.

## 🧠 Cara Kerja

1. **MediaPipe Hand Landmarker** mendeteksi 21 titik pada setiap tangan.
2. Gestur dikenali dari posisi relatif ujung jari terhadap telapak, lalu distabilkan beberapa frame.
3. Setiap bentuk (otak, kubus, planet, hati, teks) berupa titik 3D yang dirotasi dan diproyeksikan dengan perspektif.
4. Partikel bergerak menuju targetnya dengan fisika pegas dan redaman, lalu dirender sebagai titik tajam dengan glow tipis.
