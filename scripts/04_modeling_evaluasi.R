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