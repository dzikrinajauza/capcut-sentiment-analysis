import pandas as pd
from google_play_scraper import reviews, Sort

app_id = 'com.lemon.lvoverseas'

def get_reviews(app_id, lang='id', count=5000, sort=Sort.NEWEST, filter_score_with=None, filter_device_with=None, continuation_token=None):
    try:
        result, continuation_token = reviews(
            app_id,
            lang=lang,
            country='id',
            sort=sort,
            count=count,
            filter_score_with=filter_score_with,
            filter_device_with=filter_device_with,
            continuation_token=continuation_token
        )
        return result, continuation_token
    except Exception as e:
        print("Error:", e)
        return None, None

# Memanggil fungsi
print(f"Mulai mengambil ulasan untuk aplikasi: {app_id}...")
reviews_list, continuation_token = get_reviews(app_id)

if reviews_list is not None:
    print("Jumlah ulasan yang didapat:", len(reviews_list))
    
    if len(reviews_list) > 0:
        # 1. Masukkan hasil scraping ke dalam DataFrame
        df_busa = pd.DataFrame(reviews_list)

        # 2. Simpan ke dalam file CSV
        csv_filename = "com.lemon.lvoverseas.csv"
        df_busa.to_csv(csv_filename, index=False)

        # 3. Tampilkan pesan sukses
        print(f"\nData berhasil disimpan ke '{csv_filename}' dengan jumlah {len(df_busa)} baris.")

        # 4. Tampilkan 5 data teratas (Gunakan print agar muncul di terminal)
        print("\n5 Data Teratas:")
        print(df_busa[['userName', 'score', 'content']].head()) # Menampilkan kolom tertentu agar lebih rapi di terminal

else:
    print("Tidak dapat mengambil ulasan.")
