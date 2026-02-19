Traffic Violation & Smart Traffic Light System
Repositori ini berisi implementasi sistem Computer Vision untuk mendeteksi pelanggaran lalu lintas (khususnya penggunaan helm) dan mensimulasikan manajemen lampu lalu lintas cerdas berdasarkan kepadatan kendaraan. Proyek ini menggunakan model deteksi objek YOLOv11 yang telah di training ulang dan custom untuk mengidentifikasi kendaraan, orang, dan penggunaan helm secara real-time dari sebuah gambar.

Fitur Utama
Deteksi Kendaraan Multi-Kelas: Mampu mengidentifikasi dan membedakan antara motor, mobil, bus, dan truk.

Deteksi Pelanggaran Helm: Secara otomatis mengidentifikasi pengendara motor dan memeriksa apakah mereka menggunakan helm atau tidak.

Asosiasi Pengendara-Motor: Menggunakan Intersection over Union (IoU) untuk secara cerdas menghubungkan orang dengan motor terdekat untuk analisis helm.

Simulasi Durasi Lampu Lalu Lintas: Menghitung jumlah total kendaraan dalam frame untuk merekomendasikan durasi lampu hijau yang dinamis, dengan batas minimum dan maksimum.

Visualisasi yang Jelas: Memberikan output visual dengan kotak pembatas (bounding box) dan label yang jelas:

Pelanggar (Tanpa Helm): Ditandai dengan warna merah.

Pengendara Aman (Pakai Helm): Ditandai dengan warna hijau.

Kendaraan Lain: Ditandai dengan warna biru dan oranye.

Informasi Lampu Lalu Lintas: Menampilkan jumlah kendaraan dan durasi lampu di pojok kanan atas.

Teknologi yang Digunakan
Python 3.12

Ultralytics YOLOv11: Sebagai engine utama untuk deteksi objek.

OpenCV: Untuk pemrosesan gambar, seperti membaca, menulis, dan menggambar pada gambar.

NumPy: Untuk operasi numerik yang efisien, terutama dalam kalkulasi IoU.

Matplotlib: Untuk menampilkan hasil akhir di dalam notebook.
