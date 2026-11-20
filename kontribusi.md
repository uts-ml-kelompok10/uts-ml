# Kontribusi Kelompok 10

## Anggota dan bukti pekerjaan

| Anggota | NIM dan kelas | Akun | Cakupan berkas yang diajukan | Bukti GitHub |
| --- | --- | --- | --- | --- |
| Ghavind Azzarya | 101032300012, TK-47-03 | GID0317 | Tugas 3, meliputi kode clustering, perbandingan metode, stabilitas ARI, sensitivitas fitur, profil segmen, analisis anomali, dan `cluster.csv`. Penyusunan laporan gabungan dan pemeriksaan reproduksi dilakukan dalam sesi kerja dengan bantuan Codex. | [PR #1](https://github.com/uts-ml-kelompok10/uts-ml/pull/1), commit `4b204b9`. |
| Muammar Yaldi Fadhil | 101032300019, TK-47-03 | Lianaulya | Pengajuan dan integrasi berkas Tugas 1 dan 2, meliputi kode regresi dan klasifikasi, runner gabungan, dependensi, bukti evaluasi, serta kedua CSV prediksi. | [PR #2](https://github.com/uts-ml-kelompok10/uts-ml/pull/2), commit `11ef4a6`. |

PR #1 dan #2 telah digabung ke `main` melalui akun masing-masing anggota. Tabel mencatat cakupan pengajuan yang dapat ditelusuri dari GitHub. Kode, pembahasan, dan laporan disiapkan dengan bantuan AI sebagaimana dijelaskan berikut.

## Penggunaan AI

OpenAI Codex/ChatGPT membantu menyusun dan memeriksa implementasi Python, audit ketersediaan fitur, validasi waktu, percobaan model, kalibrasi, aturan biaya, analisis clustering, pembahasan hasil, penyusunan PDF, notebook presentasi, serta pemeriksaan struktur pengumpulan. AI juga membantu latihan penjelasan untuk ujian lisan.

Hasil numerik laporan menggunakan keluaran kode pada data dosen. Repo publik UTS tahun sebelumnya ditinjau untuk melihat susunan penyajian dan jenis pembahasan. Dataset, angka hasil, dan konfigurasi eksperimennya berbeda dari tugas ini. Rujukan metode dalam laporan mengarah ke dokumentasi pustaka yang digunakan.

## Pemeriksaan keluaran

Pemeriksaan teknis dalam sesi kerja tanggal 19 November 2026 mencakup hal berikut.

- Menjalankan `python kode/jalankan_semua.py` dengan data resmi dari awal hingga selesai.
- Mencocokkan 17 JSON/CSV keluaran regenerasi dengan berkas repo setelah normalisasi akhir baris Git.
- Memeriksa header, urutan dan keunikan ID, jumlah baris, peluang [0,1], keputusan biner, label cluster integer, serta ketentuan P90 minimal P50.
- Mencocokkan isi laporan dengan metrik, tabel audit, profil segmen, dan seluruh butir Tugas 1-3.
- Merender dan memeriksa seluruh halaman PDF agar tabel, grafik, nomor halaman, dan teks terbaca.

Angka validasi Agustus dibedakan dari prediksi September-Oktober. Jeda kematangan label fraud 14 hari dicatat sebagai asumsi. Skor pada label uji dan segmen referensi dosen yang tersembunyi belum tersedia.

Pada 20 November 2026, format laporan disesuaikan dengan acuan tata tulis FTE Telkom University, mencakup sampul, daftar isi, bab, tipografi, margin, dan penomoran tabel/gambar. Audit bahasa dengan skill humanizer mencakup seluruh 16 subbab dan daftar pustaka. Istilah diperjelas dan kalimat dirapikan dengan mempertahankan angka, konfigurasi, rujukan, dan batas kesimpulan. Keterangan kelas dihapus dari sampul dan jenjang ditulis S-1. PDF akhir berjumlah 17 halaman termasuk sampul dan daftar isi; seluruh halaman dirender dan diperiksa. Tiga notebook demo telah dijalankan dari awal hingga akhir dengan kernel Python lokal. Seluruh 17 JSON/CSV hasilnya cocok dengan keluaran repo setelah normalisasi akhir baris Git. Implementasi eksperimen pada skrip `.py` tetap dipakai oleh notebook.

## Persetujuan anggota

Naskah laporan gabungan beserta dokumen kontribusi ini disiapkan untuk review kedua anggota. Saat naskah dibuat, persetujuan terhadap versi laporan gabungan belum tercatat. Catatan persetujuan diisi dari review atau komentar nyata pada PR laporan setelah setiap anggota membaca isinya.

| Anggota | Status persetujuan laporan dan kontribusi |
| --- | --- |
| Ghavind Azzarya | Menunggu review naskah laporan gabungan. |
| Muammar Yaldi Fadhil | Menunggu review naskah laporan gabungan. |

Kedua anggota bertanggung jawab memahami dan mampu mereproduksi seluruh tugas sesuai ketentuan ujian lisan.
