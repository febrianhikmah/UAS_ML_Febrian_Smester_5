# Machine Learning UAS - Klasifikasi Batu Empedu dan Klastering Risiko Bencana

Project ini merupakan aplikasi Streamlit untuk UAS Machine Learning yang berisi dua studi kasus utama:

1. Klasifikasi risiko penyakit batu empedu berdasarkan data kesehatan pasien.
2. Klastering wilayah/negara berdasarkan indikator risiko bencana lingkungan.

Aplikasi ini menyediakan halaman penjelasan dataset, dashboard visualisasi, proses training dan evaluasi model, penjelasan cara kerja model terbaik, serta halaman prediksi interaktif.

## Identitas

- Nama: Febrian Hikmah Nur Rohim
- NIM: B2D023016
- Program Studi: S1 Sains Data
- Universitas: Universitas Muhammadiyah Semarang
- Dosen Pengampu: Bapak Saeful Amri, S.Kom., M.Kom

## Fitur Aplikasi

- About Dataset: penjelasan sumber dan karakteristik dataset kesehatan dan lingkungan.
- Dashboard: visualisasi data kesehatan dan lingkungan menggunakan grafik interaktif.
- Machine Learning: training, evaluasi, dan perbandingan beberapa model.
- Cara Kerja Model Terbaik: penjelasan tahapan Logistic Regression dan K-Means.
- Prediction App: prediksi risiko batu empedu dan prediksi klaster risiko wilayah.
- Contact Me: informasi kontak pembuat aplikasi.

## Studi Kasus

### 1. Klasifikasi Penyakit Batu Empedu

Dataset kesehatan digunakan untuk memprediksi status penyakit batu empedu. Target yang digunakan adalah `Y_Gallstone Status`, sedangkan fitur yang digunakan meliputi:

- Usia
- Jenis kelamin
- Comorbidity
- Body Mass Index (BMI)
- Visceral Fat Rating (VFR)
- Glucose
- Total Cholesterol
- Vitamin D

Beberapa model klasifikasi yang dibandingkan:

- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- XGBoost, jika library tersedia

Model utama yang digunakan pada aplikasi prediksi adalah Logistic Regression.

### 2. Klastering Risiko Bencana

Dataset lingkungan digunakan untuk mengelompokkan wilayah berdasarkan indikator risiko bencana. Fitur utama yang digunakan meliputi:

- WRI
- Exposure
- Vulnerability
- Susceptibility
- Lack of Coping Capabilities
- Lack of Adaptive Capacities

Metode klastering yang tersedia:

- K-Means
- Fuzzy C-Means
- DBSCAN
- Hierarchical Clustering
- Grid Based, sebagai pendekatan pembanding

Model utama yang digunakan pada aplikasi prediksi klaster adalah K-Means.

## Struktur Project

```text
.
|-- utama.py
|-- tentang_data.py
|-- visulisasi_uas.py
|-- ML_uas.py
|-- tahapan_model.py
|-- prediksi.py
|-- tentang_gua.py
|-- data uas ML kesehatan.xlsx
|-- data uas ML Lingkungan.xlsx
|-- logistic_regression_gallstone.pkl
|-- kmeans_clustering_model.pkl
|-- requirements.txt
`-- README.md
```

Keterangan file:

- `utama.py`: file utama untuk menjalankan aplikasi Streamlit.
- `tentang_data.py`: halaman penjelasan dataset.
- `visulisasi_uas.py`: dashboard visualisasi data.
- `ML_uas.py`: proses training, evaluasi, dan perbandingan model.
- `tahapan_model.py`: penjelasan tahapan model terbaik.
- `prediksi.py`: halaman prediksi interaktif.
- `tentang_gua.py`: halaman kontak.
- `*.xlsx`: dataset yang digunakan.
- `*.pkl`: model machine learning yang sudah disimpan.

## Instalasi

Pastikan Python sudah terinstall. Disarankan menggunakan virtual environment.

```bash
python -m venv venv
```

Aktifkan virtual environment.

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan Aplikasi

Jalankan perintah berikut dari folder project:

```bash
streamlit run utama.py
```

Setelah itu, aplikasi akan terbuka di browser melalui alamat lokal Streamlit, biasanya:

```text
http://localhost:8501
```

## Dataset

Dataset kesehatan:

- Sumber: UCI Machine Learning Repository
- Topik: Gallstone disease
- Digunakan untuk klasifikasi status penyakit batu empedu.

Dataset lingkungan:

- Sumber: Kaggle
- Topik: Global Disaster Risk Index
- Digunakan untuk klastering wilayah berdasarkan risiko bencana.

## Output Aplikasi

Output pada aplikasi meliputi:

- Visualisasi distribusi data dan hubungan antar variabel.
- Evaluasi model klasifikasi menggunakan accuracy, precision, recall, F1-score, dan ROC-AUC.
- Evaluasi model klastering menggunakan silhouette score, Davies-Bouldin index, dan Calinski-Harabasz index.
- Prediksi risiko batu empedu berdasarkan input pasien.
- Prediksi klaster risiko bencana berdasarkan input indikator wilayah.
- Rekomendasi kesehatan dan rekomendasi kebijakan wilayah secara edukatif.

## Catatan

Aplikasi ini dibuat untuk kebutuhan pembelajaran dan tugas akademik. Hasil prediksi kesehatan bersifat edukatif dan tidak menggantikan diagnosis atau konsultasi dokter.

## Kontak

- Email: febriannurrohim12@gmail.com
- LinkedIn: https://www.linkedin.com/in/febrian-nurrohim-0249602b5/
- GitHub: https://github.com/febrianhikmah
- Instagram: https://www.instagram.com/rohimm.02/
