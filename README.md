# CapCut Review Sentiment Analysis 🎬

#### Analisis Komparatif Naive Bayes vs. Support Vector Machine untuk Klasifikasi Sentimen Ulasan Google Play Store, dengan Penanganan Class Imbalance menggunakan SMOTE dan TF-IDF

![alt text](?raw=true)
![alt text](?raw=true)

## 📌 Ringkasan Proyek

Proyek ini membangun pipeline analisis sentimen dua tahap: Python digunakan untuk mengambil data ulasan mentah dari Google Play Store (via google-play-scraper), dan R digunakan untuk keseluruhan proses analisis — preprocessing teks, pelabelan, ekstraksi fitur, hingga klasifikasi sentimen (Positif / Negatif / Netral) dari ulasan pengguna aplikasi CapCut. Dua algoritma klasifikasi — Multinomial Naive Bayes dan Support Vector Machine (Kernel Linear) — dibandingkan performanya setelah data diproses melalui pembobotan TF-IDF dan penyeimbangan kelas menggunakan SMOTE.

Proyek ini relevan sebagai studi kasus text mining pada data ulasan berbahasa Indonesia yang tidak baku (bahasa gaul, singkatan, typo), sekaligus mendemonstrasikan penanganan masalah class imbalance yang umum terjadi pada dataset ulasan aplikasi dunia nyata.

## 🎯 Latar Belakang

CapCut menjadi salah satu aplikasi edit video paling banyak diunduh secara global, dengan lonjakan penggunaan signifikan seiring tren konten video pendek di TikTok, Instagram Reels, dan YouTube Shorts. Volume ulasan yang masuk setiap hari mencapai ribuan, dan data ini merupakan sumber insight yang sangat berharga bagi tim pengembang untuk memahami kepuasan pengguna dan memprioritaskan perbaikan fitur.

Masalahnya, ulasan pengguna sulit dianalisis secara manual:
- Volumenya terlalu besar untuk dibaca satu per satu.
- Bahasanya tidak baku — penuh singkatan, istilah teknis editing, dan typo.
- Distribusi sentimen sering timpang: ulasan netral cenderung jauh lebih sedikit dibanding positif/negatif, sehingga model klasifikasi standar cenderung mengabaikannya.

Proyek ini menjawab tantangan tersebut dengan pipeline otomatis yang tidak hanya mengklasifikasikan sentimen, tetapi juga secara eksplisit menangani ketidakseimbangan data melalui SMOTE — pendekatan yang jarang dieksplorasi secara mendalam pada studi kasus ulasan CapCut.

## 🔄 Alur Kerja (Data Pipeline)

![alt text](?raw=true)

### Tahapan detail:

#### 1. Data Scraping (Python)
   Ulasan mentah diambil dari Google Play Store menggunakan library google-play-scraper, dengan app_id = com.lemon.lvoverseas (ID resmi aplikasi CapCut di Play Store). Atribut yang diambil meliputi userName, score, content, dan metadata lain, lalu disimpan sebagai .csv.
#### 2. Text Preprocessing (R)
Case folding → cleaning (URL, emoji, tanda baca) → normalisasi kata gaul/singkatan → tokenizing → stopword removal → stemming (kamus Katadasar).
#### 3. Pelabelan Hibrida 
Kombinasi rating bintang dan skor leksikon domain-spesifik (kamus custom untuk istilah seperti lag, bug, ngelek, estetik) untuk menentukan label akhir Positif / Negatif / Netral.
#### 4. TF-IDF Weighting
Representasi teks bersih diubah menjadi matriks numerik berbobot.
#### 5. Data Splitting & SMOTE 
Data dibagi 80% latih / 20% uji, lalu SMOTE (k=5) diterapkan pada data latih untuk menyeimbangkan kelas minoritas (Netral).
#### 6. Model Training
Naive Bayes dan SVM (Kernel Linear) dilatih menggunakan library e1071 dan caret.
#### 7. Evaluation
Performa dievaluasi menggunakan Confusion Matrix (Accuracy, Precision, Recall, F1-Score, Kappa Index).

## 📊 Hasil Performa

Hasil evaluasi model pada data uji (pasca-SMOTE, total 8.931 baris):

| Algoritma | Akurasi |	Kappa Index	| Status |
|:---------:|:-------:|:-----------:|:------:|
| Multinomial Naive Bayes |	85.60%	| 0.7840	| Layak |
| SVM (Kernel Linear) |	91.26% |	0.8689	| Unggul |

#### Insight utama:

- SVM secara konsisten mengungguli Naive Bayes, sejalan dengan kemampuannya memetakan batas keputusan pada ruang fitur TF-IDF berdimensi tinggi.
- Penerapan SMOTE meningkatkan kemampuan kedua model dalam mengenali kelas Netral tanpa indikasi overfitting.

'📁 Detail lengkap metrik evaluasi (Precision/Recall per kelas, confusion matrix) tersedia di `results/evaluation_report.md`.'



