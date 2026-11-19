# Bahan laporan UTS — Klasifikasi fraud

Tujuannya memperkirakan probabilitas fraud saat transaksi terjadi, lalu memilih transaksi yang layak ditinjau. Biaya dalam soal terdiri dari Rp25.000 per transaksi yang ditandai dan nominal transaksi fraud yang terlewat. Semua angka di bawah berasal dari rilis dosen dan dapat dihasilkan kembali dengan `python kode/klasifikasi.py`.

## Label, ketersediaan fitur, dan pergeseran pola

Data berlabel memuat 403.077 transaksi Januari–Agustus 2025, termasuk 2.622 fraud terkonfirmasi (0,65%). Data uji resmi berisi 106.674 transaksi September–Oktober tanpa label. Target adalah konfirmasi yang sudah tercatat saat snapshot 31 Agustus, sehingga angka 0 juga dapat mencakup fraud yang belum terkonfirmasi. Tidak tersedia tanggal konfirmasi; jeda 14 hari adalah asumsi untuk mengurangi risiko label terbaru belum lengkap. Label yang terlalu baru tidak diubah menjadi 0.

Flag review manual mencatat proses setelah transaksi. Dari 6.898 riwayat yang pernah ditinjau, 2.095 (30,37%) terkonfirmasi fraud, dibanding 527 dari 396.179 transaksi yang tidak ditinjau (0,13%). Dengan split dan konfigurasi model sama, penambahan flag ini menaikkan AP awal Agustus secara semu dari **0,4065** ke **0,6498**. Flag tersebut dikeluarkan dari prediksi akhir. Fitur internal `v01–v50` serta profil refund dan volume merchant juga dikeluarkan karena waktu pembentukannya tidak dijamin oleh kamus. Fitur riwayat bawaan yang dikatakan tersedia sebelum transaksi dipertahankan; dua kolom riwayat kosong pada 102.621 baris latih diperlakukan sebagai missing.

Proporsi kasus terkonfirmasi turun dari 0,65% pada Juli menjadi 0,46% pada Agustus. Di antara kasus terkonfirmasi, porsi perangkat baru turun dari 44,57% ke 28,98%, dan IP negara berbeda dari 9,68% ke 1,63%. Perubahan ini dapat mencerminkan pergeseran modus, kelengkapan label, atau keduanya. Porsi perangkat baru pada **seluruh transaksi** juga dicatat terpisah (Juli 4,30%, Agustus 4,25%, September 4,18%, Oktober 3,96%); pola pada kasus terkonfirmasi tidak bisa disimpulkan untuk test yang tidak berlabel. Jendela latih 90 hari dibandingkan dengan riwayat penuh. Riwayat penuh dipilih karena AP pengembangan sedikit lebih baik, sementara kalibrator dan ambang dipasang ulang dari cohort terbaru yang cukup berjarak dari snapshot.

## Riwayat, validasi, dan pemilihan model

Untuk setiap pengguna dan merchant, fitur yang dibentuk mencakup jumlah dan rerata transaksi sebelumnya, menit sejak transaksi sebelumnya, jumlah dan nominal transaksi 1 dan 7 hari ke belakang, dan rasio nominal sekarang terhadap rerata historis. Jendela memakai `[waktu sekarang − jendela, waktu sekarang)`: transaksi pada detik yang sama tidak saling terlihat. Riwayat tidak memakai label, melanjutkan transaksi yang sudah terjadi dari train ke test, dan tidak mengirim ID pengguna atau merchant sebagai fitur model. Pada test terdapat 11.458 pengguna baru (15.490 baris) dan 152 merchant baru (771 baris) relatif terhadap seluruh train. Riwayat awal kosong/kosong statistik ditangani oleh LightGBM; aktivitas kemudian dapat memakai kejadian sebelumnya yang sudah berlalu. Ablation pada awal Agustus menunjukkan AP **0,3776** tanpa fitur riwayat buatan dan **0,4065** saat ditambahkan; fitur riwayat bawaan tetap tersedia di kedua versi.

| Strategi LightGBM | AP rata-rata Mei–Juni: riwayat penuh | AP rata-rata Mei–Juni: 90 hari |
| --- | ---: | ---: |
| Tanpa bobot | 0,7921 | 0,7912 |
| Bobot positif 8× | 0,7890 | 0,7841 |
| Undersampling negatif (pertahankan 30%) | **0,7923** | 0,7769 |

Baseline probabilitas konstan memberi AP 0,0063. Split maju Mei dan Juni memakai fit sebelum awal bulan dikurangi 14 hari, kemudian menilai bagian awal bulan yang mencapai setidaknya 14 hari dari bulan berikutnya. Ukuran evaluasi dan prevalensi alami tidak berubah oleh sampling. Undersampling riwayat penuh dipilih pada AP tertinggi, tetapi selisih 0,00025 dari model tanpa bobot tidak cukup untuk klaim keunggulan yang kuat. Strategi ketiga selain pembobotan dan resampling adalah mengubah ambang keputusan berdasar biaya; ini mengubah flag dan biaya, tidak mengubah AP.

Simulasi Agustus memfit model pada 274.647 baris sebelum 17 Juni (jumlah sebelum sampling). Kalibrasi sigmoid memakai 14.769 baris awal Juli (109 positif), lalu ambang dipilih pada 14.483 baris Juli berikutnya sebelum 18 Juli (112 positif). Agustus tidak dipakai untuk memilih strategi, melatih kalibrator, atau menyesuaikan ambang evaluasi Agustus. Kalibrasi menggeser rata-rata probabilitas awal Agustus dari 0,94% ke 0,64% (proporsi positif tercatat 0,66%); Brier membaik dari 0,00517 menjadi 0,00508, tetapi log loss memburuk dari 0,02528 menjadi 0,02631. Jadi perbaikan kalibrasi tidak seragam untuk semua metrik. AP tetap sama karena sigmoid terpasang mempertahankan urutan skor.

| Usia minimum cohort evaluasi dari snapshot | Transaksi | Positif tercatat | AP |
| --- | ---: | ---: | ---: |
| 7 hari | 40.963 | 234 | 0,3459 |
| 14 hari | 29.236 | 192 | 0,4065 |
| 21 hari | 17.437 | 137 | 0,4849 |

Sensitivitas di atas memakai model, kalibrator, dan aturan yang sama pada subset tanggal berbeda. Ini tidak memisahkan efek kematangan label dari perbedaan komposisi tanggal. AP seluruh Agustus adalah 0,2912; pada 23.668 transaksi akhir Agustus AP 0,1013 dan hanya 53 fraud terkonfirmasi. Angka akhir Agustus dilaporkan sebagai pembanding yang lebih terpengaruh asumsi kelengkapan label, bukan sebagai pengukuran pasti seluruh fraud.

## Keputusan berdasarkan biaya

Untuk probabilitas terkalibrasi `p`, kerugian harapan bila transaksi dibiarkan adalah `p × nominal`. Aturan teoritis menandai jika nilai itu di atas Rp25.000. Grid faktor ambang dipilih pada bagian Juli terpisah dengan meminimalkan biaya yang ditentukan soal. Faktor yang dipilih 0,5 setara ambang Rp12.500 untuk **membuat keputusan**. Setiap transaksi yang ditandai tetap dihitung Rp25.000 dalam **evaluasi biaya**.

| Kebijakan pada awal Agustus (29.236 transaksi) | Ditandai | Fraud terdeteksi | Fraud terlewat | Total biaya simulasi |
| --- | ---: | ---: | ---: | ---: |
| Tidak menandai apa pun | 0 | 0 | 192 | Rp202.602.900 |
| Menandai semua | 29.236 | 192 | 0 | Rp730.900.000 |
| Ambang teoritis 1× | 324 | 114 | 78 | Rp74.924.300 |
| Ambang terpilih 0,5× | 464 | 127 | 65 | **Rp72.651.400** |

Aturan terpilih mencapai recall 66,15% dan precision 27,37% pada label yang tersedia; biaya terdiri dari Rp61.051.400 nominal yang terlewat dan Rp11.600.000 pemeriksaan. Penurunan biaya terhadap tanpa penandaan sekitar 64,14%, dengan asumsi soal bahwa fraud yang ditandai berhasil tertahan. Ambang terpilih tidak selalu mengungguli ambang teoritis pada setiap cohort; pada akhir Agustus biaya tercatatnya justru lebih tinggi (Rp28,00 juta vs Rp26,19 juta).

## Letak kesalahan dan hasil akhir

- **Kanal:** QRIS melewatkan 45 dari 68 fraud terkonfirmasi (recall 33,82%), dengan Rp57.625.500 nominal terlewat. Ini sekitar 94,4% dari seluruh nominal terlewat pada cohort awal Agustus.
- **Kategori merchant:** `jasa_lainnya` melewatkan 20 dari 29 kasus (Rp31.207.700); `toko_online` melewatkan 18 dari 34 kasus (Rp14.828.700). Angka kanal dan kategori tidak dijumlahkan karena dapat merujuk transaksi yang sama.
- **Pola fitur:** aturan deskriptif yang saling eksklusif memeriksa IP luar negara, perangkat baru, gagal login, lalu kartu tidak hadir. Kelompok perangkat baru menangkap 51 dari 58 kasus, sedangkan kelompok tanpa empat penanda ini melewatkan 52 dari 95 kasus. Aturan ini membantu membaca kesalahan, bukan label modus fraud yang diketahui. Contoh IP berbeda hanya memiliki satu positif, sehingga tidak cukup untuk kesimpulan kuat.

Untuk prediksi resmi, fit diperbarui dengan 327.091 baris sebelum 18 Juli (sebelum sampling). Kalibrasi dan pemilihan ambang bergeser ke dua rentang awal Agustus; faktor akhir tetap 0,5. CSV berisi **106.674 transaksi** dan menandai **2.918 transaksi**. Biaya pemeriksaan saja pada test adalah Rp72.950.000 bila seluruh flag ditindaklanjuti. AP, recall, nominal fraud terlewat, dan total biaya resmi masih belum diketahui karena label test disembunyikan.
