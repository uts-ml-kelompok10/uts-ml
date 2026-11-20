# UTS Pembelajaran Mesin: reproduksi tiga tugas

## Demo ujian lisan dengan notebook

| Tugas | Notebook | Fokus penjelasan |
| --- | --- | --- |
| 1 | [01_regresi.ipynb](01_regresi.ipynb) | P50/P90, validasi waktu, dan perubahan cakupan sesudah kalibrasi. |
| 2 | [02_klasifikasi.ipynb](02_klasifikasi.ipynb) | Peluang fraud, keputusan review, dan perhitungan biaya. |
| 3 | [03_clustering.ipynb](03_clustering.ipynb) | Alasan tiga cluster, profil pelanggan, stabilitas, dan anomali. |

Notebook juga menjalankan Python. Penyajiannya menggabungkan penjelasan, sel kode, tabel, dan grafik. Implementasi model tetap berada di skrip `.py` per tugas; sel eksperimen memanggil skrip tersebut agar perhitungan demo dan pengumpulan tetap sama.

Gunakan Python 3.12 dan letakkan lima berkas data dosen di `data_uts_ml/`. Dari root repo:

```powershell
python -m pip install -r kode/requirements-notebook.txt
python -m jupyterlab
```

Buka salah satu notebook dari folder `kode/` dan pilih kernel Python tempat dependensi dipasang. Gunakan **Restart Kernel and Run All Cells** untuk mengulang dari awal. Jika folder data berbeda, set variabel lingkungan `UTS_DATA_DIR` ke lokasi data sebelum menjalankan JupyterLab. Hasil run baru disimpan di `.cache/demo-notebook/<tugas>/`; tabel dan grafik setelah sel eksperimen membaca hasil run tersebut.

Ketiga notebook telah dijalankan penuh pada 20 November 2026. Di mesin pemeriksaan, sel eksperimen regresi memerlukan sekitar 3,5 menit, klasifikasi 50 detik, dan clustering 20 detik. Waktu bergantung mesin. Output tersimpan dapat dibaca di GitHub sebelum menjalankan ulang. Pengujian dilakukan dengan kernel lokal; penggunaan Colab memerlukan repo lengkap, dependensi, dan data, serta belum diuji dalam pemeriksaan ini.

Untuk demo lima menit, siapkan lingkungan dan jalankan ulang sebelum ujian. Saat presentasi, ikuti urutan berikut, serta siap menjalankan sel yang diminta penguji.

1. Jelaskan satu baris data mewakili apa dan keputusan yang ingin dibantu.
2. Tunjukkan metode serta asumsi penting: waktu fitur, jeda label 14 hari, atau praproses clustering sesuai tugas.
3. Tunjukkan sel eksperimen dan hasil yang dihasilkan, lalu bacakan satu tabel atau grafik utama.
4. Jelaskan batas hasil: cakupan berlaku pada cohort evaluasi; biaya resmi menunggu label; ARI clustering mengukur kestabilan internal dan manfaat kampanye masih perlu diuji.

Untuk pertanyaan detail implementasi, buka skrip `.py` yang bersangkutan. Pembahasan seluruh subsoal ada di [laporan](../laporan.pdf).

## Tugas 1 — Regresi

Tugas 1 memperkirakan durasi pengiriman sejak paket dijemput hingga diterima. P50 adalah estimasi median durasi; P90 adalah batas atas yang ditargetkan mencakup sekitar 90% durasi. Kedua angka dalam jam.

Letakkan `regresi_train.csv.gz` dan `regresi_test.csv.gz` dari rilis dosen di `data_uts_ml/`, setingkat dengan folder `kode/`. Data mentah dan ZIP tidak ikut Git. Gunakan Python 3.12 dan pasang dependensi sekali. Dari root proyek UTS:

```powershell
python -m pip install -r kode/requirements-regresi.txt
python kode/regresi.py
```

Perintah terakhir mengulang audit, pemilihan model, kalibrasi, evaluasi, dan penulisan prediksi. Seed 42, satu thread, serta versi library dan SHA-256 input dicatat pada `hasil/regresi/metrics.json`. Jika dataset ditempatkan di folder lain, gunakan `--data <folder>`. Gunakan `--output <folder>` untuk menyimpan hasil di lokasi berbeda.

| Keluaran | Isi |
| --- | --- |
| `prediksi_regresi.csv` | Satu baris per paket uji, dengan kolom `id_paket,p50,p90` dan P90 ≥ P50. |
| `hasil/regresi/metrics.json` | Audit input, hasil pemilihan model, perbandingan validasi waktu dan acak, kalibrasi, serta ablation `jumlah_scan`. |
| `hasil/regresi/tuning_waktu.csv` | Metrik masing-masing konfigurasi pada dua fold waktu. |
| `hasil/regresi/error_per_segmen.csv` | Pinball loss, cakupan, MAE dan bias pada Agustus menurut layanan, pulau tujuan dan mitra kurir. Bias = rerata prediksi P50 dikurangi durasi sebenarnya. |
| `hasil/regresi/partial_dependence.csv` | Rerata P50 pada grid jarak dan berat. Perbandingan memakai 500 paket Agustus yang sama. |

## Ringkasan metode

- Fitur dipilih berdasarkan kamus data dan waktu pickup. ID paket dan jumlah scan dikeluarkan. Fitur internal `f01–f50` dipakai berdasarkan keterangan bahwa fitur itu dihitung saat pickup. Ablation scan hanya untuk mengukur risiko pemakaian informasi setelah pickup.
- Pemilihan model memakai fold Mei dan Juni, dengan training dari bulan sebelumnya. Model dipilih dari rerata jumlah pinball loss P50 dan P90; Agustus tidak digunakan untuk memilih model. Dua split acak dengan ukuran yang sama menunjukkan efek mengabaikan urutan waktu.
- Tanggal pickup, jam pickup, dan durasi memberi perkiraan kapan label regresi tersedia. Training setiap fold hanya memakai paket yang diperkirakan sudah selesai sebelum cutoff. Label yang belum tersedia dikeluarkan, bukan diubah menjadi durasi nol.
- Simulasi Agustus melatih model sebelum Juli dan menghitung koreksi P90 dari hari 1–20 Juli yang labelnya sudah tersedia sebelum Agustus. Prediksi September–Oktober memakai aturan yang sama, bergeser satu bulan. Jika distribusinya berubah, cakupan 90% tidak terjamin.
- P50 dan P90 diurutkan dengan menaikkan P90 bila perlu, bukan dengan menukar dua quantile. Semua metrik diukur dalam jam, kecuali cakupan yang merupakan proporsi.

Dokumentasi metode: [LightGBM: kategori dan nilai kosong](https://lightgbm.readthedocs.io/en/v4.6.0/Advanced-Topics.html), [parameter LightGBM](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html), dan [Ridge di scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

## Tugas 2 — Klasifikasi fraud

Letakkan `klasifikasi_train.csv.gz` dan `klasifikasi_test.csv.gz` dari dosen di folder `data_uts_ml/`. Pakai dependensi yang sama dengan Tugas 1, lalu jalankan:

```powershell
python kode/klasifikasi.py
```

Perintah itu menjalankan audit, fitur riwayat, pemilihan model, kalibrasi, evaluasi biaya, dan ekspor. Opsi `--data` dan `--output` berfungsi seperti pada skrip regresi. Seed dan versi library disimpan bersama hash data input.

| Keluaran | Isi |
| --- | --- |
| `prediksi_klasifikasi.csv` | Satu baris per transaksi uji, dengan kolom `id_transaksi,prob,tandai`. |
| `hasil/klasifikasi/metrics.json` | Audit label, hasil model, kalibrasi, biaya, dan ablation. |
| `hasil/klasifikasi/tuning_waktu.csv` | Average precision tiap strategi pada dua fold maju. |
| `hasil/klasifikasi/audit_bulanan.csv` | Pergeseran fitur dan proporsi fraud terkonfirmasi jika label tersedia. |
| `hasil/klasifikasi/sensitivitas_label.csv` | Evaluasi ulang skor Agustus pada cohort dengan jarak minimum 7, 14, dan 21 hari ke snapshot; model tetap sama. |
| `hasil/klasifikasi/error_per_segmen.csv` | Recall, kegagalan deteksi, dan biaya per kanal, kategori merchant, serta pola fitur. |

Kamus data tidak menyediakan waktu konfirmasi label, sehingga jeda 14 hari adalah asumsi, bukan jadwal konfirmasi yang diketahui. Fitur riwayat pengguna dan merchant dihitung hanya dari transaksi pada detik yang lebih awal. ID entitas dipakai sebagai kunci agregasi saja. Flag peninjauan, fitur anonim, dan agregat refund/volume merchant yang tidak memiliki jaminan waktu tidak masuk model akhir.

LightGBM dibandingkan tanpa bobot, dengan bobot kelas positif, dan dengan pengurangan sampel negatif. Masing-masing memakai riwayat penuh atau 90 hari terakhir, dipilih dari average precision Mei–Juni. Setelah itu, periode terpisah digunakan untuk [kalibrasi sigmoid](https://scikit-learn.org/stable/modules/calibration.html) dan pemilihan faktor ambang biaya. Flag dibuat jika `prob × nominal > 25.000 × faktor`. Perbandingan biaya pada Agustus memakai label terkonfirmasi yang tersedia; biaya aktual September–Oktober tidak diketahui.

## Tugas 3 — Segmentasi pelanggan

Letakkan `clustering.csv.gz` di `data_uts_ml/`. Gunakan dependensi yang sama lalu jalankan:

```powershell
python kode/clustering.py
```

Skrip membandingkan KMeans, Gaussian Mixture dengan kovarians diagonal, dan Ward untuk k=2–8. Ketiganya diuji pada 2.600 pelanggan pembanding yang sama. Ward dilatih pada sampel yang lebih kecil karena kebutuhan memori, lalu mengalokasikan pelanggan pembanding ke centroid terdekat. Silhouette dan ARI antarsubsampel dipakai bersama untuk memilih model. ARI ini mengukur kestabilan internal; label referensi dosen tidak tersedia. Profil dan aksi segmen dijelaskan di [bahan laporan](../hasil/clustering/ringkasan.md).

`cluster.csv` berisi `id_pelanggan,cluster` dengan satu baris per pelanggan; `-1` adalah anomali. Tabel pemilihan metode, ARI per pasangan seed, sensitivitas fitur, profil dan SHA-256 input disimpan di `hasil/clustering/`.

Untuk mereproduksi angka dari **ketiga bagian** dari root proyek UTS setelah instalasi dependensi:

```powershell
python kode/jalankan_semua.py
```

Dokumentasi metode: [clustering scikit-learn](https://scikit-learn.org/stable/modules/clustering.html), [Gaussian Mixture](https://scikit-learn.org/stable/modules/mixture.html), dan [Ward linkage di SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).
