# -----------------------------------------------------------------
# PERSIAPAN: LOAD LIBRARY
# -----------------------------------------------------------------
library(tidyverse) # mencari dan mengganti teks, merapikan data
library(tidytext)  # untuk tokenisasi teks
library(katadasaR) # kamus kata dasar bahasa Indonesia
library(tm)        # text mining

# -----------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------
path_data <- "G:/My Drive/Dokumen/UT - tugas/UT TUTON/Semester 6/Metodologi Penelitian/Penelitian/Analisis Sentimen Pengguna aplikasi/Scrapping 5000 data penilaian apk capcut 2026/ulasan_capcut.csv"
df <- read.csv(path_data, stringsAsFactors = FALSE)

# melihat kata yang paling sering muncul di data
df %>% 
  unnest_tokens(word, content) %>% 
  count(word, sort = TRUE) %>% 
  head(100) # Memunculkan 100 kata teratas yang paling populer

# -----------------------------------------------------------------
# 2. PROSES PREPROCESSING DATA
# -----------------------------------------------------------------

# A. Fungsi pembantu untuk Normalisasi Kata
normalisasi_kata <- function(teks) {
  # Formula baku str_replace_all : str_replace_all(1. Corong datanya, 2. "Kata yang dicari", 3. "Kata pengganti")
  # Formula baku kata yang dicari : \\b + kata_pertama + \\b + | + \\b + kata_kedua + \\b
  teks <- str_replace_all(teks, "\\bapk\\b|\\bapkinis\\b|\\bapki\\b", "aplikasi")
  teks <- str_replace_all(teks, "\\bpakee\\b|\\bpake\\b", "pakai")
  teks <- str_replace_all(teks, "\\bbet\\b|\\bbgt\\b", "banget") # Tambah bgt
  teks <- str_replace_all(teks, "\\bga\\b|\\bgak\\b|\\bgk\\b", "tidak") # Tambah gk & ga
  teks <- str_replace_all(teks, "\\bngelek\\b|\\blag\\b", "lambat")
  teks <- str_replace_all(teks, "\\bdonlod\\b|\\bdownload\\b", "unduh")
  teks <- str_replace_all(teks, "\\btrus\\b|\\bterus\\b", "kemudian")
  teks <- str_replace_all(teks, "\\bkarne\\b|\\bkarna\\b", "karena") # Tambah karna
  teks <- str_replace_all(teks, "\\bgw\\b", "saya") # Tambah gw
  teks <- str_replace_all(teks, "\\byg\\b", "yang") # Tambah yg
  teks <- str_replace_all(teks, "\\baja\\b", "saja") # Tambah aja
  teks <- str_replace_all(teks, "\\budah\\b", "sudah") # Tambah udah
  teks <- str_replace_all(teks, "\\bvidio\\b", "video") # Tambah vidio
  teks <- str_replace_all(teks, "\\bbadusss\\b|\\bbaguss\\b|\\bbagusss\\b", "bagus")
  return(teks)
}

# B. Fungsi pembantu untuk Stemming Bahasa Indonesia
#    Potong kalimatnya di setiap spasi → Buang imbuhan di tiap katanya → Lem kembali dengan spasi menjadi kalimat utuh
stemming_indonesia <- function(teks) {
  kata <- unlist(strsplit(as.character(teks), " "))
  
  # Tugasnya sapply = mengambil deretan kata yang sudah dipotong tadi, lalu satu per satu dimasukkan ke dalam rumus katadasaR
  kata_dasar <- sapply(kata, katadasaR) # Mesin pengulang loop
  
  # Tugasnya paste = adalah lem perekat. Kebalikan dari strsplit
  paste(kata_dasar, collapse = " ")
}

# Vektorisasi fungsi stemming agar bisa berjalan di dalam mutate
stemming_vectorized <- Vectorize(stemming_indonesia)

# Eksekusi Preprocessing Pipeline
df_cleaned <- df %>%
  mutate(
    id = row_number(),
    # A. CLEANING: Hapus URL, angka, tanda baca, dan spasi berlebih
    content_clean = str_replace_all(content, "http\\S+\\s*", ""),
    content_clean = str_replace_all(content_clean, "[0-9]+", ""),
    content_clean = str_replace_all(content_clean, "[[:punct:]]", " "),
    content_clean = str_replace_all(content_clean, "\\s+", " "),
    content_clean = str_trim(content_clean),
    
    # B. CASE FOLDING: Mengubah ke huruf kecil
    content_clean = tolower(content_clean),
    
    # C. NORMALISASI: Mengubah kata tidak baku
    content_clean = normalisasi_kata(content_clean),
    
    # F. STEMMING: Mengubah ke kata dasar
    content_clean = stemming_vectorized(content_clean)
  )

# -----------------------------------------------------------------
# 3. KAMUS LEKSIKON KHUSUS CAPCUT
# -----------------------------------------------------------------
capcut_lexicon <- data.frame(
  word = c(
    # Positif
    "seru", "simpel", "bagus", "mantap", "manfaat", "mudah", "gampang", "oke", "ok", 
    "kreatif", "keren", "puas", "bantu", "estetik", "lancar", "rekomendasi",
    # Negatif
    "kikir", "bayar", "tagihan", "rugi", "iklan", "lambat", "premium", 
    "jelek", "kecewa", "bug", "error", "idiot", "buruk", "sedot", "mahal", 
    "kesal", "parah", "sulit", "lemot", "crash"
  ),
  lex_score = c(
    rep(1, 16),  # Skor 1 untuk semua kata positif
    rep(-1, 20)  # Skor -1 untuk semua kata negatif
  )
)

# -----------------------------------------------------------------
# 4. PROSES TOKENISASI & STOPWORD REMOVAL
# -----------------------------------------------------------------
# Definisikan koleksi kata hubung umum Indonesia secara manual
daftar_stopword_manual <- c(
  "yang", "dan", "di", "dari", "untuk", "dengan", "ke", "sebagai", "itu", "atau",
  "dalam", "bisa", "ini", "ada", "akan", "adalah", "pada", "juga", "saya", "kami",
  "kamu", "dia", "mereka", "ia", "anda", "kita", "sudah", "telah", "bukan", "tidak",
  "namun", "tetapi", "oleh", "karena", "sehingga", "jika", "maka", "melainkan",
  "bahwa", "saja", "banyak", "sedikit", "sangat", "lebih", "paling", "kurang",
  "dapat", "harus", "boleh", "mampu", "perlu", "kembali", "bahkan", "pas", "kalo",
  "sampai", "tersebut", "suatu", "sebuah", "tentang", "mengenai", "secara",
  "begitu", "begitupun", "sambil", "selama", "sebelum", "sesudah", "setelah",
  "ketika", "sementara", "serta", "pun", "pula", "terlalu", "melakukan", "sih",
  "ih", "kok", "oh", "si", "nya", "tuh", "an", "lah", "ya", "aja", "udah"
)

stopword_id <- tibble(word = daftar_stopword_manual)

# Pipeline kalkulasi skor leksikon
analysis <- df_cleaned %>%
  # D. TOKENIZING: Memecah ulasan bersih menjadi token kata
  unnest_tokens(word, content_clean) %>% 
  # E. STOPWORD REMOVAL: Menyaring kata berdasarkan daftar manual
  anti_join(stopword_id, by = "word") %>%
  # Menggabungkan dengan kamus leksikon khusus CapCut
  inner_join(capcut_lexicon, by = "word") %>%
  group_by(id) %>%
  summarise(total_lex_score = sum(lex_score))

# -----------------------------------------------------------------
# 5. PELABELAN HIBRIDA (Gabungan Rating + Teks Hasil Preprocessing)
# -----------------------------------------------------------------
df_final <- df_cleaned %>%
  left_join(analysis, by = "id") %>%
  mutate(total_lex_score = replace_na(total_lex_score, 0)) %>%
  mutate(label = case_when(
    score >= 4 | total_lex_score > 0 ~ "Positif",
    score <= 2 | total_lex_score < 0 ~ "Negatif",
    TRUE ~ "Netral"
  ))

# -----------------------------------------------------------------
# 6. MENAMPILKAN DISTRIBUSI HASIL & SIMPAN
# -----------------------------------------------------------------
print(table(df_final$label))

# Simpan Hasil yang sudah bersih dan sudah berlabel
write.csv(df_final, "Hasil_Preprocessing_dan_Labeling_Capcut_2026.csv", row.names = FALSE)