# Bahan laporan UTS — Segmentasi pelanggan

Tujuan bagian ini adalah menemukan kelompok perilaku belanja yang cukup stabil dan dapat dipakai untuk merancang eksperimen pemasaran. Data resmi berisi 120.000 pelanggan dan 60 kolom, tanpa label. ID unik, tidak kosong, dan hanya dipakai untuk mencocokkan hasil. Seluruh angka berasal dari data dosen. Kode dan tabel dapat direproduksi melalui `python kode/clustering.py`.

## Audit dan persiapan fitur

- **Nilai kosong:** 6.736 rata-rata nilai keranjang kosong, semuanya pada pelanggan dengan nol pesanan dalam 90 hari. Nilai keranjang belum terdefinisi; model mengisi median dari sampel fit sambil mempertahankan frekuensi nol. Profil keranjang dan preferensi kategori/waktu dihitung hanya pada pelanggan yang pernah memesan dalam jendela tersebut.
- **Konsistensi:** 656 rasio konversi di luar [0,1] diperlakukan sebagai missing, lalu diisi median. Ada 101 total belanja 30 hari yang melebihi total belanja 90 hari; dua total tersebut tidak dipakai sebagai fitur. Belanja 90 hari tetap ditampilkan di profil sebagai statistik deskriptif dari data asli. Kamus menyebut `n01–n10`, tetapi file berisi `n01–n15`; semuanya anonim dan diuji sebagai satu blok sensitivitas.
- **Distribusi dan outlier:** median belanja 90 hari Rp1.136.500, persentil 99 Rp26.695.608, dan maksimum Rp261.836.800. Model memakai `log1p` pada recency, jumlah pesanan, keranjang, sesi, dan durasi sesi, kemudian pembatasan kuantil 0,5%–99,5% serta penskalaan median/IQR pada fitur numerik. Deteksi anomali memakai data yang belum dibatasi kuantil, setelah transformasi dan scaling, agar penyimpangan ekstrem masih terlihat.
- **Komposisi:** 16 proporsi kategori dan empat proporsi waktu dinormalisasi terpisah agar tiap blok berjumlah satu, lalu diambil akar kuadrat. Jarak Euclidean pada blok akar sebanding dengan jarak Hellinger; nol dapat dipertahankan. Ada 73.187 baris yang memiliki setidaknya satu nol pada blok kategori dan 146 pada blok waktu. Penyimpangan jumlah dari 1 paling besar 0,0005 untuk kategori dan 0,0001 untuk waktu. Bagi pelanggan tanpa pesanan, model memakai profil komposisi median dari pelanggan aktif pada sampel fit agar proporsi tersebut tidak dianggap preferensi yang teramati.
- **Kategori dan bobot:** kota tier dan perangkat di-one-hot dengan bobot rendah. Tiga blok numerik—pembelian, promosi/pembayaran, aktivitas—diberi bobot berturut-turut 1,00; 0,70; dan 0,65, lalu setiap kolom dibagi akar jumlah fitur bloknya. Komposisi kategori dan waktu diberi bobot 0,48 dan 0,28 setelah transformasi akar. Indikator kota tier dan perangkat diberi faktor 0,16/√2. Bobot memprioritaskan perilaku yang dapat diterjemahkan menjadi aksi, sambil tetap mempertimbangkan minat dan konteks.

Belanja 30 dan 90 hari berkorelasi tinggi (Spearman 0,972); frekuensi 30 dan 90 hari juga berkorelasi (0,868). Jendela pendek dan total belanja tidak dimasukkan agar aktivitas yang sama tidak berkali-kali menentukan jarak. ID, besaran demografis, umur keanggotaan, dan kolom aktivitas tambahan dikesampingkan dari model utama agar representasi fokus pada penggunaan untuk pemasaran. Uji sensitivitas untuk KMeans k=3 menunjukkan bahwa melepas blok minat dan konteks tetap menghasilkan partisi hampir sama (ARI 0,988), sementara menambahkan 15 fitur CRM anonim mengubahnya tajam (ARI 0,0006). Silhouette versi CRM memang lebih tinggi (0,576), tetapi dihitung di ruang fitur yang berbeda, sehingga tidak membuktikan segmentasinya lebih berguna. Dasar untuk tidak memakai CRM adalah makna kolom yang tidak jelas, pengaruh ekstrem pada partisi, dan tujuan interpretasi.

## Metode, jumlah cluster, dan kestabilan

Perbandingan memakai KMeans, Gaussian Mixture dengan kovarians diagonal, dan hierarki Ward untuk k=2–8. Semua model dinilai pada **2.600 pelanggan pembanding yang sama**, disisihkan dari sampel fit. KMeans/GMM memakai 9.000 titik fit; Ward memakai 3.500 untuk membatasi biaya memori, lalu pelanggan pembanding dialokasikan ke centroid terdekat. Hasil Ward berlaku untuk prosedur sampel dan alokasi ini. BIC dicatat untuk GMM dalam ruang fitur dan sampel yang sama; nilai paling rendah ada pada k=8, tetapi salah satu komponen hanya mencakup sekitar 1,65% sampel pembanding.

Tiga pengulangan menggunakan seed 101–103 dan subsampel 80% (7.200 pelanggan atau 2.800 untuk Ward), dengan preprocessing dan model dilatih ulang. ARI dihitung untuk ketiga pasangan pengulangan pada pelanggan pembanding yang sama. Silhouette mengukur pemisahan, sedangkan ARI mengukur apakah pembagian bertahan ketika sampel fit berubah. Model yang punya segmen di bawah 3% atau di atas 80% pada pembanding tidak dipilih untuk menjaga kegunaan kampanye. Di antara yang lolos, digunakan skor `silhouette − 0,25 × (1 − ARI rata-rata)`.

| Konfigurasi | Silhouette | ARI rata-rata | Skor pemilihan |
| --- | ---: | ---: | ---: |
| KMeans, k=2 | 0,1739 | 0,6260 | 0,0804 |
| KMeans, k=3 | 0,1688 | **0,9738** | **0,1622** |
| KMeans, k=4 | 0,1715 | 0,9555 | 0,1604 |
| GMM diagonal, k=2 (terbaik GMM) | 0,1397 | 0,9774 | 0,1340 |
| Ward, k=2 (terbaik Ward) | 0,1645 | 0,9271 | 0,1462 |

KMeans 3 dipilih karena relatif stabil dengan lebih sedikit segmen daripada k=4. Selisih keduanya kecil dan silhouette 0,1688 menunjukkan segmen masih tumpang tindih. ARI terendah antarpengulangan untuk k=3 adalah 0,9713. Ini adalah kestabilan internal, bukan skor ARI terhadap label referensi dosen. Model terpilih kemudian dilatih ulang dengan seluruh pelanggan.

## Profil segmen dan usulan aksi

| Label | Segmen | Pelanggan | Profil terukur | Aksi pemasaran yang diuji |
| --- | --- | ---: | --- | --- |
| 0 | Pembeli rutin bernilai tinggi | 39.679 (33,07%) | Median 13 pesanan dan Rp4.576.200 belanja selama 90 hari; median porsi voucher 28,2%. | Uji pengingat pembelian ulang kategori makanan/minuman di pagi hari, dibanding kelompok kontrol tanpa pengingat. |
| 1 | Pemburu promo malam | 53.357 (44,46%) | Median 7 pesanan dan Rp796.100 belanja; median porsi voucher 72,2%. Malam adalah waktu belanja terbesar secara rata-rata pada pelanggan aktif (34,89%). | Uji voucher malam untuk kategori fashion dengan batas margin, lalu ukur tambahan pembelian dan laba dibanding kontrol. |
| 2 | Pelanggan mulai pasif | 26.483 (22,07%) | Median 48,9 hari sejak transaksi terakhir, 2 pesanan, dan Rp273.300 belanja; 22,47% belum memesan dalam 90 hari. | Uji satu tawaran reaktivasi dengan jeda kirim, lalu ukur pesanan tambahan dalam 30 hari dibanding kontrol. |
| -1 | Aktivitas tidak biasa | 481 (0,40%) | Median 231,72 sesi/minggu, durasi sesi 0,75 menit, dan rasio konversi valid 0,7%. | Periksa kualitas pencatatan sesi sebelum menyertakan kelompok ini dalam kampanye. |

Usulan di atas adalah hipotesis eksperimen; tabel belum membuktikan kampanye akan meningkatkan penjualan. Profil yang lebih rinci ada di `profil_segmen.csv`.

Anomali ditentukan dari dua sinyal sekaligus: jarak terhadap centroid melebihi persentil 98,5 **di kelompoknya**, serta skor Isolation Forest masuk ekor tertinggi 2,5% secara global. Isolation Forest dilatih pada 25.000 pelanggan yang disampel dengan seed tetap. Ambang jarak 97,5/98,5/99 menghasilkan masing-masing 677/481/364 anomali. Label `-1` menandai observasi yang tidak cocok dengan kelompok, bukan tuduhan fraud. CSV akhir mempunyai satu ID dan satu label integer untuk tiap pelanggan, mengikuti urutan sumber.
