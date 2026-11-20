# UTS Pembelajaran Mesin - Kelompok 10

Regresi durasi pengiriman, klasifikasi fraud, dan segmentasi pelanggan dari data dosen.

| Anggota | NIM | Kelas | GitHub |
| --- | --- | --- | --- |
| Ghavind Azzarya | 101032300012 | TK-47-03 | [GID0317](https://github.com/GID0317) |
| Muammar Yaldi Fadhil | 101032300019 | TK-47-03 | [Lianaulya](https://github.com/Lianaulya) |

## Berkas pengumpulan

| Berkas | Isi |
| --- | --- |
| [laporan.pdf](laporan.pdf) | Pembahasan semua subsoal Tugas 1-3 dalam 16 halaman. |
| [kode/](kode/README.md) | Implementasi Python, tiga notebook demo, dependensi, dan cara menjalankan. |
| [prediksi_regresi.csv](prediksi_regresi.csv) | 60.000 prediksi, kolom `id_paket,p50,p90`. |
| [prediksi_klasifikasi.csv](prediksi_klasifikasi.csv) | 106.674 prediksi, kolom `id_transaksi,prob,tandai`. |
| [cluster.csv](cluster.csv) | 120.000 label, kolom `id_pelanggan,cluster`. |
| [kontribusi.md](kontribusi.md) | Bukti PR, cakupan kontribusi, penggunaan AI, dan status persetujuan. |

## Reproduksi

Gunakan Python 3.12. Ekstrak lima berkas `.csv.gz` dari rilis dosen ke `data_uts_ml/`, lalu jalankan dari root repositori.

```sh
python -m pip install -r kode/requirements-regresi.txt
python kode/jalankan_semua.py
```

Runner menghasilkan kembali tiga CSV prediksi dan seluruh JSON/CSV bukti evaluasi di `hasil/`. Seed, versi pustaka, dan hash SHA-256 input dicatat dalam `metrics.json` masing-masing tugas. Pembahasan ringkas per tugas tersedia di [regresi](hasil/regresi/ringkasan.md), [klasifikasi](hasil/klasifikasi/ringkasan.md), dan [clustering](hasil/clustering/ringkasan.md).

Pada pemeriksaan 19 November 2026, ketiga skrip selesai dari satu perintah. Seluruh 17 berkas keluaran yang dihasilkan cocok dengan keluaran dalam repositori setelah normalisasi akhir baris Git. Waktu pada mesin pemeriksaan sekitar 250 detik.

## Notebook untuk presentasi

Buka [notebook regresi](kode/01_regresi.ipynb), [klasifikasi fraud](kode/02_klasifikasi.ipynb), atau [segmentasi pelanggan](kode/03_clustering.ipynb). Setiap notebook memuat tujuan, contoh data, alasan metode, perhitungan ulang, tabel, grafik, dan batas kesimpulan. Notebook memanggil skrip Python yang sama dengan runner.

```sh
python -m pip install -r kode/requirements-notebook.txt
python -m jupyterlab
```

Jalankan dari root repo, buka notebook di folder `kode/`, lalu pilih **Run All Cells**. Data tetap diperlukan untuk menjalankan ulang. Output contoh sudah tersimpan sehingga tabel dan grafik juga dapat dibaca langsung di GitHub. Petunjuk demo ada di [panduan kode](kode/README.md#demo-ujian-lisan-dengan-notebook).

Pada 20 November 2026, ketiga notebook dijalankan dari sel pertama sampai terakhir tanpa error. Seluruh 17 JSON/CSV hasilnya cocok dengan keluaran repo setelah normalisasi akhir baris. Pengujian memakai kernel Python lokal; alur Google Colab belum diuji.

## Tata tulis laporan

Laporan memakai sampul, daftar isi, lima bab, daftar pustaka, serta penomoran tabel dan gambar per bab. Acuan tata tulis adalah [template resmi tugas akhir FTE Telkom University](https://see.telkomuniversity.ac.id/wp-content/uploads/2022/01/Format-Buku-Tugas-Akhir.docx): kertas A4, Times New Roman 12 untuk isi, spasi 1,5, rata kiri-kanan, margin kiri 4 cm dan sisi lain 3 cm. Ukuran teks tabel dan keterangan gambar disesuaikan agar terbaca. Template tersebut menjadi acuan format; ketentuan tugas UTS tetap mengatur isi dan batas maksimal 18 halaman. Versi ini berjumlah 16 halaman termasuk sampul dan daftar isi.
