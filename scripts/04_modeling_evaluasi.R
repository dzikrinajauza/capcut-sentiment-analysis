# =================================================================
# TAHAP 4: SPLIT DATA & MODELING (MULTINOMIAL NAIVE BAYES VS SVM)
# Penelitian: Analisis Sentimen CapCut 2026 - Dzikrina Jauza Hasna
# =================================================================

# -----------------------------------------------------------------
# 1. LOAD LIBRARY YANG DIBUTUHKAN
# -----------------------------------------------------------------
library(tidyverse)
library(caret)       # Library utama untuk split data & confusion matrix
library(e1071)       # Library untuk algoritma SVM
library(naivebayes)  # Library khusus untuk Multinomial Naive Bayes yang valid
library(ggplot2)


# -----------------------------------------------------------------
# 2. LOAD DATA SEIMBANG (HASIL SMOTE TAHAP 3)
# -----------------------------------------------------------------
data_modeling <- readRDS("Data_Capcut_Balanced_SMOTE.rds")
data_modeling$label <- as.factor(data_modeling$label)

# -----------------------------------------------------------------
# 3. SPLIT DATA (TRAINING 80% & TESTING 20%)
# -----------------------------------------------------------------
set.seed(123) 
index_train <- createDataPartition(data_modeling$label, p = 0.8, list = FALSE)

train_data <- data_modeling[index_train, ]
test_data  <- data_modeling[-index_train, ]

# Memisahkan fitur kata (X) dan label target (Y) khusus untuk Multinomial NB
X_train <- as.matrix(train_data %>% select(-label))
Y_train <- train_data$label
X_test  <- as.matrix(test_data %>% select(-label))
Y_test  <- test_data$label

cat("Jumlah Data Training:", nrow(train_data), "\n")
cat("Jumlah Data Testing :", nrow(test_data), "\n\n")

# -----------------------------------------------------------------
# 4. ALGORITMA 1: MULTINOMIAL NAIVE BAYES MODELING
# -----------------------------------------------------------------
print("Sedang melatih model Multinomial Naive Bayes...")
# laplace = 1 digunakan untuk smoothing agar tidak ada probabilitas bernilai 0
model_nb <- multinomial_naive_bayes(x = X_train, y = Y_train, laplace = 1)

# Prediksi data testing menggunakan Multinomial Naive Bayes
pred_nb <- predict(model_nb, newdata = X_test)

cat("\n=========================================\n")
cat("   HASIL EVALUASI MULTINOMIAL NAIVE BAYES ")
cat("\n=========================================\n")
print(confusionMatrix(pred_nb, Y_test))

# -----------------------------------------------------------------
# 5. ALGORITMA 2: SUPPORT VECTOR MACHINE (SVM) MODELING
# -----------------------------------------------------------------
print("Sedang melatih model Support Vector Machine (SVM)...")
model_svm <- svm(label ~ ., data = train_data, kernel = "linear")

# Prediksi data testing menggunakan SVM
pred_svm <- predict(model_svm, test_data)

cat("\n=========================================\n")
cat("           HASIL EVALUASI SVM            ")
cat("\n=========================================\n")
print(confusionMatrix(pred_svm, test_data$label))

# -----------------------------------------------------------------
# 6. VISUALISASI SEBARAN DATA SVM DENGAN PCA (2 DIMENSI)
# -----------------------------------------------------------------

# 1. Memeras ratusan kolom kata menjadi 2 sumbu utama (PC1 dan PC2)
print("Membersihkan kolom bervarians nol sebelum PCA...")

# Deteksi kolom mana saja yang variansnya 0 (isinya sama semua)
zero_var_cols <- which(apply(X_train, 2, var) == 0)

# Buat salinan X_train khusus untuk grafik PCA agar X_train asli tidak rusak
if(length(zero_var_cols) > 0) {
  X_train_pca <- X_train[, -zero_var_cols]
  cat("Ada", length(zero_var_cols), "kolom kata yang dibuang untuk grafik PCA.\n")
} else {
  X_train_pca <- X_train
}

# Jalankan ulang PCA menggunakan data yang sudah bersih
pca_result <- prcomp(X_train_pca, scale. = TRUE)

# 2. Membuat dataframe baru khusus untuk grafik
data_grafik <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Sentimen = Y_train
)

# 3. Menggambar plot (Scatter Plot) dengan Zoom-in (Filter Outlier Visual)
# Saya memperkirakan batas PC1: 0 s/d 60 dan PC2: -60 s/d 60 berdasarkan gambar Anda.
grafik_svm_zoom <- ggplot(data_grafik, aes(x = PC1, y = PC2, color = Sentimen)) +
  geom_point(alpha = 0.6, size = 2) +
  stat_ellipse(linewidth = 1) + 
  scale_color_manual(values = c("Positif" = "blue", "Netral" = "gray", "Negatif" = "red")) +
  
  # KUNCI ZOOM: Tentukan batas sumbu yang akan ditampilkan
  # Sesuaikan angka ini jika perlu
  scale_x_continuous(limits = c(0, 40)) + # Menampilkan PC1 dari 0 s/d 60
  scale_y_continuous(limits = c(-40, 40)) + # Menampilkan PC2 dari -60 s/d 60
  
  theme_minimal() +
  labs(title = "Visualisasi Persebaran Sentimen CapCut (Zoomed-in PCA 2D)",
       subtitle = "Data telah difilter secara visual (limits) untuk detail kelompok utama",
       x = "Komponen Utama 1 (PC1)",
       y = "Komponen Utama 2 (PC2)")

# Perintah wajib untuk menampilkan grafik ke tab Plots
print(grafik_svm_zoom)