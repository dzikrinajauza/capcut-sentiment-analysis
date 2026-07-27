# =================================================================
# TAHAP 3: PENYEIMBANGAN DATA MENGGUNAKAN SMOTE (Versi Kepatuhan R 4.5+)
# Penelitian: Analisis Sentimen CapCut 2026 - Dzikrina Jauza Hasna
# =================================================================

# -----------------------------------------------------------------
# 1. LOAD LIBRARY YANG DIBUTUHKAN
# -----------------------------------------------------------------
library(tidyverse)
library(tidytext)
library(tm)
library(recipes) # Paket persiapan data modern
library(themis)  # Paket alternatif untuk eksekusi SMOTE

# -----------------------------------------------------------------
# 2. LOAD DATA & RE-CONSTRUCT MATRIKS TF-IDF
# -----------------------------------------------------------------
data_clean <- read.csv("../Hasil_Preprocessing_dan_Labeling_Capcut_2026.csv", stringsAsFactors = FALSE)
data_clean <- data_clean %>% mutate(id = as.character(id))

ulasan_words <- data_clean %>%
  select(id, content_clean, label) %>%
  filter(content_clean != "") %>%
  unnest_tokens(word, content_clean) %>%
  count(id, label, word, sort = TRUE) %>%
  ungroup()

ulasan_tfidf <- ulasan_words %>%
  bind_tf_idf(word, id, n)

dtm_tfidf <- ulasan_tfidf %>%
  cast_dtm(id, word, tf_idf)

# -----------------------------------------------------------------
# 3. KONVERSI STRUKTUR DATA UNTUK FORMAT RECIPES
# -----------------------------------------------------------------
dtm_matrix <- as.matrix(dtm_tfidf)
df_tfidf_matrix <- as.data.frame(dtm_matrix)

labels <- data_clean %>%
  filter(id %in% rownames(df_tfidf_matrix)) %>%
  arrange(match(id, rownames(df_tfidf_matrix))) %>%
  select(label)

# Satukan matriks kata dengan kolom target label
df_for_smote <- cbind(df_tfidf_matrix, label = as.factor(labels$label))

cat("\n--- DISTRIBUSI KELAS SEBELUM SMOTE ---\n")
print(table(df_for_smote$label))

# -----------------------------------------------------------------
# 4. EKSEKUSI TAHAPAN SMOTE DENGAN THEMIS
# -----------------------------------------------------------------
print("Komputer sedang menghitung data sintetis SMOTE, mohon ditunggu sejenak...")

# Membuat resep formula SMOTE
smote_recipe <- recipe(label ~ ., data = df_for_smote) %>%
  step_smote(label, over_ratio = 1, neighbors = 5)

# Eksekusi kalkulasi algoritma tetangga terdekat
smote_prep <- prep(smote_recipe)
data_balanced <- juice(smote_prep)

cat("\n--- DISTRIBUSI KELAS SETELAH SMOTE ---\n")
print(table(data_balanced$label))

# -----------------------------------------------------------------
# 5. AMANKAN DATA BALANCED KE MEMORI LOKAL
# -----------------------------------------------------------------
saveRDS(data_balanced, "Data_Capcut_Balanced_SMOTE.rds")
print("Proses SMOTE Selesai! Data seimbang berhasil disimpan dalam format .rds")