# UTS Pembelajaran Mesin — Kelompok 10

## Tugas 3 — Segmentasi pelanggan

Ekstrak `clustering.csv.gz` dari rilis dosen ke folder `data_uts_ml/` di root repositori. Gunakan Python 3.12. Dari root repositori, pasang dependensi lalu jalankan:

```sh
python -m pip install -r kode/requirements-clustering.txt
python kode/clustering.py
```

Skrip membandingkan KMeans, Gaussian Mixture, dan Ward untuk k=2–8, mengukur kestabilan dengan ARI antarsubsampel, lalu melatih model terpilih pada seluruh data. Hasilnya tersimpan di `cluster.csv` (kolom `id_pelanggan,cluster`; `-1` menandai anomali) dan `hasil/clustering/`. Seed, versi library, dan SHA-256 input tercatat di `hasil/clustering/metrics.json`.

Lihat [pembahasan Tugas 3](../hasil/clustering/ringkasan.md) untuk pemilihan fitur, perbandingan model, profil segmen, dan usulan aksi pemasaran.

Jika data ada di folder lain, pakai `--data <folder>`. Opsi `--output <folder>` menyimpan hasil di lokasi lain.
