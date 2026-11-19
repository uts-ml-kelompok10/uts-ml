# Bahan laporan UTS — Regresi durasi pengiriman

Model memperkirakan durasi dalam jam ketika kurir menjemput paket. P50 adalah median prediksi, sedangkan P90 adalah batas atas dengan target cakupan 90%. Semua hasil berasal dari data rilis dosen. Skrip `python kode/regresi.py` menghasilkan angka, tabel segmen, dan CSV prediksi dari berkas mentah.

## Audit data dan waktu ketersediaan

Data meliputi 250.000 paket latih (Januari–Agustus 2025) dan 60.000 paket uji (September–Oktober 2025), tanpa ID kosong, ganda, atau beririsan. Hanya kolom target yang tidak ada pada data uji. Berat kosong pada 13.897 paket latih dan 3.265 paket uji. Jarak nol ada pada 703 paket latih dan 177 paket uji; kami perlakukan sebagai jarak yang belum andal, menambahkan penanda dan mempertahankan baris. Berat kosong juga diberi penanda. P99 durasi latih adalah 224,70 jam, tetapi maksimum mencapai 25.452 jam. Durasi ekstrem tetap dinilai pada studi retrospektif; pengecualian fit mengikuti kapan label tersedia, bukan besarnya target saja.

Pemilihan fitur mengikuti kamus data. Fitur internal `f01–f50` dianggap tersedia saat pickup karena waktu perhitungannya dinyatakan dalam kamus. ID dan `jumlah_scan` tidak dipakai pada model akhir. Kota tujuan memiliki 405 level pada data berlabel, dan 2.342 paket uji memakai satu mitra yang tidak muncul pada data berlabel. Ridge menggunakan one-hot dengan pengelompokan kategori jarang yang dipelajari saat fit. LightGBM memakai kategori native dengan regularisasi kelompok; kategori di luar kosakata fit diperlakukan sebagai missing.

Median `jumlah_scan` turun dari 8 di data latih menjadi 1 di data uji, dan korelasi Spearman scan dengan durasi di data latih adalah 0,769. Ketersediaan jumlah scan saat pickup tidak terjamin. Pada set Agustus yang sama, model dengan scan tercatat terlihat lebih baik daripada model tanpa scan (pinball P50 9,40 vs 10,83; P90 9,95 vs 11,42). Ketika kolom scan pada set Agustus diganti dengan nilai konstan 1, cakupan P90 model dengan scan jatuh menjadi 45,38%. Ini uji sensitivitas terhadap satu skenario, bukan estimasi skor test atau bukti semua scan tercatat terlambat.

## Validasi dan model

Validasi maju memilih model pada Mei dan Juni dengan melatih hanya pada paket sebelum awal periode evaluasi yang diperkirakan sudah selesai saat itu. Ketersediaan label diperkirakan sebagai tanggal dan jam pickup + durasi; jam pickup hanya memiliki resolusi satu jam. Ada 2.804 label pada file latih yang menurut hitungan ini baru tersedia pada atau setelah 1 September, meski kamus menyebut data ditarik 31 Agustus. Untuk simulasi cutoff, label seperti ini tidak dipakai sebelum waktu selesai. Kedua fold tuning hanya menilai label yang tersedia sebelum 1 Juli; akibatnya, sebagian paket Juni yang sangat lambat tidak masuk tuning. Evaluasi Agustus memakai seluruh label Agustus **secara retrospektif**, termasuk 2.600 yang baru tersedia sejak September; label tersebut tidak digunakan untuk fit atau kalibrasi pada cutoff Agustus.

| Metode | Pinball P50 (jam) | Pinball P90 (jam) |
| --- | ---: | ---: |
| Baseline kuantil global | 12,76 | 8,61 |
| Ridge log, alpha 1 | 6,95 | 4,34 |
| Ridge log, alpha 25 | 6,96 | 4,35 |
| LightGBM kuantil, 31 daun dan 160 pohon | **6,57** | 4,14 |
| LightGBM kuantil, 63 daun dan 240 pohon | 6,62 | 4,14 |

Tabel menampilkan rerata pinball dua fold waktu. Nilai lebih rendah lebih baik. Ridge memodelkan log durasi dan menambahkan kuantil residual latih untuk estimasi P50/P90. LightGBM melatih dua model dengan loss kuantil, masing-masing untuk P50 dan P90. Konfigurasi 31 daun dipilih dari rerata jumlah kedua loss sebelum menyentuh evaluasi Agustus. Dua split acak dengan ukuran latih dan validasi yang sama menghasilkan rerata **7,14/4,97 jam**, dibanding **6,57/4,14 jam** pada split waktu. Split acak tidak lebih baik pada sampel ini. Karena distribusi tanggal dan kasusnya berbeda, selisih ini bukan ukuran murni pengaruh kebocoran waktu; simulasi waktu dipilih karena lebih sesuai penggunaan.

## Kalibrasi dan hasil Agustus

Model uji Agustus dilatih pada 178.548 label yang tersedia sebelum Juli. Koreksi P90 dihitung pada 22.519 paket pickup 1–20 Juli yang selesai sebelum Agustus, dari kuantil ke-90 residual bertanda aktual dikurangi P90 awal, dengan penyesuaian ukuran sampel. Koreksi +4,62 jam membuat cakupan pada holdout Agustus naik dari 88,24% menjadi **91,15%**. Model akhir mengganti periode fit dan kalibrasi maju satu bulan, memakai 212.489 label fit dan 22.719 label kalibrasi; koreksi P90 adalah −0,39 jam. Label yang belum tersedia tidak diisikan nol. Seleksi label selesai dan perubahan pola antarbulan membatasi jaminan coverage conformal; angka Agustus merupakan hasil empiris dan cakupan September–Oktober belum diketahui karena label test disembunyikan.

| Agustus, 34.168 paket | Pinball P50 (jam) | Pinball P90 (jam) | Cakupan P90 |
| --- | ---: | ---: | ---: |
| Baseline | 17,16 | 15,97 | 89,41% |
| LightGBM sebelum koreksi | 10,83 | 11,42 | 88,24% |
| LightGBM setelah koreksi | **10,83** | **11,41** | **91,15%** |

Rata-rata galat absolut P50 adalah 21,66 jam dan bias P50 −12,42 jam. Ini berarti prediksi median cenderung lebih rendah dari durasi aktual secara rata-rata; bias mean ini juga dipengaruhi ekor durasi yang panjang. Cakupan keseluruhan menyembunyikan perbedaan antarkelompok:

- **Layanan:** pinball P50 layanan hemat 14,28 jam (11.349 paket), dibanding 9,76 jam pada reguler (14.335 paket).
- **Pulau tujuan:** Sulawesi memiliki pinball P50 26,71 jam dan P90 34,66 jam (2.181 paket), sehingga prediksi untuk tujuan ini perlu diperiksa lebih dekat.
- **Mitra:** K07 hanya mencakup 70,45% (2.677 paket) setelah kalibrasi global. Koreksi per segmen layak diuji pada periode terpisah sebelum dipakai; hasil Agustus tidak dipakai balik untuk mengklaim perbaikan.

Partial dependence memakai 500 paket Agustus yang sama dan mengganti satu fitur pada satu waktu. Rerata P50 naik dari 47,87 jam pada jarak 165,8 km ke 73,21 jam pada 2.208,1 km. Pada grid berat 0,40–6,23 kg rerata P50 sekitar 65,76–65,77 jam. Ini perilaku model pada grid, bukan efek kausal; jarak yang diubah mungkin tidak cocok dengan kombinasi kota paket tersebut.

`prediksi_regresi.csv` berisi 60.000 baris dengan kolom `id_paket,p50,p90`. Nilai finite dan P90 ≥ P50; 11 crossing mentah diperbaiki sebelum ekspor. Rincian sumber perhitungan: `metrics.json`, `tuning_waktu.csv`, `error_per_segmen.csv`, dan `partial_dependence.csv` di folder ini.
