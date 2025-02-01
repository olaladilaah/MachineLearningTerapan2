# MachineLearningTerapan2
Submission Machine Learning Terapan : Proyek Akhir Kirim Submission dan Review

# Laporan Machine Learning Terapan - Adilah Widiasti - olaladilah 
Sistem Rekomendasi: 
Memberikan Rekomendasi Film Berdasarkan Genre yang Tersedia
Dataset yang digunakan pada proyek ini : https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset

## Project Overview
Proyek ini akan membahas sebuah perusahaan yang bergerak di sektor perfilman dan bertujuan untuk meningkatkan jumlah pengunjung pada platform streaming film mereka. Untuk itu, perusahaan akan mengembangkan sistem rekomendasi dengan menerapkan pendekatan Machine Learning, yang akan menyarankan film berdasarkan genre untuk para penggunanya. Sistem rekomendasi dirancang untuk memberikan pilihan yang relevan sesuai dengan preferensi pengguna. Hal ini menjadi penting, karena tidak semua pengguna dapat langsung menentukan pilihan mereka saat pertama kali menggunakan aplikasi atau layanan baru.

### Latar belakang  
Industri perfilman terus berkembang pesat, dengan banyaknya film baru yang dirilis setiap tahunnya. Di era digital saat ini, platform streaming menjadi salah satu cara utama bagi masyarakat untuk mengakses film. Namun, dengan begitu banyaknya pilihan film yang tersedia, pengguna sering kali merasa kesulitan untuk memilih film yang sesuai dengan minat mereka. Hal ini dapat menyebabkan pengalaman menonton yang kurang memuaskan, bahkan meningkatkan kemungkinan pengguna untuk meninggalkan platform.

Untuk mengatasi masalah tersebut, banyak platform streaming mulai mengimplementasikan sistem rekomendasi untuk membantu pengguna menemukan film yang sesuai dengan preferensi mereka. Salah satu pendekatan yang umum digunakan adalah *content-based filtering*, yaitu sistem yang merekomendasikan film berdasarkan kesamaan genre atau karakteristik lain yang ada pada film yang sudah pernah ditonton atau disukai pengguna sebelumnya.

Pentingnya sistem rekomendasi ini tidak hanya membantu pengguna untuk menemukan film yang mereka sukai, tetapi juga meningkatkan tingkat kepuasan dan keterlibatan pengguna pada platform streaming. Oleh karena itu, pada proyek ini akan dikembangkan sebuah sistem rekomendasi film yang memberikan saran film berdasarkan genre yang tersedia di platform, sehingga pengguna dapat lebih mudah menemukan film yang sesuai dengan minat mereka.

## Bussiness Understanding 

### Problem Statements  
Berdasarkan penjelasan di atas, beberapa permasalahan yang perlu diperhatikan adalah sebagai berikut:
1. Model *Machine Learning* mana yang paling tepat untuk mengatasi masalah ini?
2. Apa metode yang dapat digunakan untuk mengevaluasi apakah hasil rekomendasi dari model *Machine Learning* tersebut sudah efektif?

### Goals
Untuk menjawab pertanyaan tersebut, penjelasannya adalah sebagai berikut:
1. Model yang tepat untuk mengatasi masalah ini adalah model berbasis konten, yang dikenal dengan istilah *Content-Based Filtering*.
2. Melakukan penilaian menggunakan metrik untuk mengevaluasi kinerja model *Machine Learning* tersebut.

## Data Understanding
1. Jumlah Data
    a.  Dataset Film (movies.csv)
        - Jumlah Baris: 10,000 (misalnya, jika dataset memiliki 10,000 film)
        - Jumlah Kolom: 3 (misalnya, movieId, title, genres)
    b.  Dataset Rating (ratings.csv)
        - Jumlah Baris: 100,000 (misalnya, jika dataset memiliki 100,000 rating)
        - Jumlah Kolom: 3 (misalnya, userId, movieId, rating)
2. Kondisi Data
    a.  Missing Values:
        - Dataset film (film_data): Setelah pembersihan, tidak ada nilai yang hilang pada kolom movieId, title, dan genres.
        - Dataset rating (rating_data): Terdapat beberapa nilai yang hilang pada kolom rating, tetapi setelah penggabungan dengan dataset film, data yang hilang dihapus.
    b.  Duplikat:
        - Dataset film: Tidak ada duplikat setelah proses pembersihan berdasarkan movieId.
        - Dataset rating: Mungkin terdapat duplikat pada rating untuk film yang sama oleh pengguna yang sama, tetapi tidak dihapus dalam konteks ini karena setiap rating dianggap unik.
    c. Outlier:
        - Dataset rating: Analisis histogram menunjukkan distribusi rating yang normal, tetapi perlu dilakukan analisis lebih lanjut untuk mengidentifikasi outlier secara spesifik. Misalnya, rating di luar rentang 1-5 dapat dianggap sebagai outlier.
3. Tautan Sumber Data
    - Dataset dapat diakses melalui tautan berikut: Kaggle Movie Recommender Dataset
4. Uraian Seluruh Fitur pada Data
    a. Dataset Film (movies.csv):
        - movieId: ID unik untuk setiap film.
        - title: Judul film.
        - genres: Genre film yang dapat terdiri dari beberapa genre yang dipisahkan dengan koma.
    b. Dataset Rating (ratings.csv):
        - userId: ID unik untuk setiap pengguna.
        - movieId: ID film yang dinilai (merujuk pada movieId di dataset film).
        - rating: Nilai yang diberikan oleh pengguna untuk film tersebut, biasanya dalam rentang 1-5.

## Data Preparation
Persiapan data adalah langkah untuk menyiapkan data sebelum pembuatan model Pembelajaran Mesin. Berikut adalah tahapan yang dilakukan:  
1.  Menggabungkan Dataset dan Menangani Nilai Hilang
    a.  Menggabungkan Dataset:
        Dataset film (film_data) dan dataset rating (rating_data) digabungkan berdasarkan movieId menggunakan metode left join
    b.  Melihat Data yang Hilang:
        Memeriksa jumlah nilai yang hilang dalam dataset gabungan.
    c.  Menghapus Data yang Hilang:
        Menghapus baris yang memiliki nilai hilang untuk memastikan data bersih.
    d.  Mengurutkan Data:
        Mengurutkan data berdasarkan movieId untuk memudahkan analisis lebih lanjut.
    e.  Melihat Jumlah Film dalam Data Bersih:
        Memeriksa jumlah film yang tersisa setelah pembersihan.
2.  Menghapus Data Duplikat
    - Menghapus data duplikat berdasarkan movieId untuk memastikan setiap film hanya muncul sekali.
3.  Mengonversi Data Series Menjadi List
    - Mengonversi kolom movieId, title, dan genres menjadi list untuk memudahkan pembuatan DataFrame baru.
4.  Membuat DataFrame Baru
    - Membuat DataFrame baru yang berisi ID film, judul film, dan genre.
5.  Ekstraksi Fitur dengan TF-IDF
    a.  Inisialisasi TfidfVectorizer:
        - Menginisialisasi objek TfidfVectorizer untuk menghitung TF-IDF dari genre film.
    b.  Fit dan Transformasi Data Genre:
        - Melakukan perhitungan IDF pada data genre dan mengubahnya menjadi matriks TF-IDF.
    c.  Melihat Ukuran Matriks TF-IDF:
        - Memeriksa ukuran matriks TF-IDF yang dihasilkan.
    d.  Mengubah Vektor TF-IDF Menjadi Matriks:
        - Mengubah matriks TF-IDF menjadi bentuk yang lebih mudah dibaca.
    e. Membuat DataFrame untuk Melihat Matriks TF-IDF:
        - Membuat DataFrame untuk menampilkan matriks TF-IDF dengan judul film sebagai indeks.

## Modeling 
Pada bagian ini, akan membahas pembangunan sistem rekomendasi film menggunakan metode Content-Based Filtering dengan pendekatan cosine similarity. Sistem ini dirancang untuk memberikan rekomendasi film berdasarkan kesamaan genre antara film yang telah ada.  
Proses selanjutnya adalah membuat model adapun tahap-tahapnya diantaranya sebagai berikut:  
1.  Menghitung Similaritas Cosine
    Setelah melakukan ekstraksi fitur menggunakan TF-IDF, langkah selanjutnya adalah menghitung similaritas antar film. Similaritas cosine digunakan untuk mengukur seberapa mirip dua film berdasarkan genre mereka. Proses ini dilakukan dengan menggunakan fungsi cosine_similarity dari library sklearn. Hasil dari perhitungan ini adalah matriks similaritas yang menunjukkan seberapa mirip setiap film dengan film lainnya. Matriks ini disimpan dalam DataFrame untuk kemudahan akses dan analisis lebih lanjut.
2.  Fungsi Rekomendasi Film
    Setelah mendapatkan matriks similaritas, perlu membuat fungsi yang dapat memberikan rekomendasi film berdasarkan judul film yang diberikan. Fungsi recommend_movies akan mengambil judul film sebagai input dan mengembalikan daftar film yang paling mirip.    
    Fungsi ini menggunakan argpartition untuk menemukan indeks film dengan similaritas tertinggi, kemudian mengembalikan DataFrame yang berisi judul film dan genre dari film-film yang direkomendasikan.
3.  Hasil Rekomendasi
    Untuk menguji fungsi rekomendasi, akan menggunakan judul film tertentu, misalnya "Piper (2016)". Dengan memanggil fungsi recommend_movies, dapat melihat film-film yang direkomendasikan berdasarkan genre film tersebut.
    Hasil dari fungsi ini akan menampilkan daftar film yang mirip dengan "Piper (2016)", berdasarkan genre yang sama. Ini memberikan pengguna pilihan film yang relevan untuk ditonton.
4.  Evaluasi
    Berdasarkan hasil rekomendasi yang dihasilkan, dapat mengevaluasi sistem dengan melihat relevansi film yang direkomendasikan. Dalam contoh ini, jika semua 10 film yang direkomendasikan relevan dengan film yang diuji, maka dapat menyimpulkan bahwa sistem rekomendasi ini memiliki Precision yang tinggi, yaitu 10/10 atau 100%.

### Result  
Setelah model selesai dibangun, jalankan model untuk menampilkan hasil rekomendasi. Sebagai contoh, akan menguji model menggunakan judul film *Piper (2016)*.

|    |   id   | movie_title  |   genre   |
|----|--------|--------------|-----------|
|2565| 160718 | Piper (2016) | Animation |

**Tabel 2. Informasi film yang diuji**
Dari Tabel 2, dapat dilihat bahwa *Piper (2016)* merupakan film dengan genre Animation. Selanjutnya, mari lihat rekomendasi film yang memiliki genre yang sama dengan film tersebut.

|   | movie_title  |   genre   |
|---|--------------|-----------|
| 0 | A Plasticine Crow (1981) | Animation |
| 1 | The Red Turtle (2016) | Animation |
| 2 | The Monkey King (1964) | Animation |
| 3 | Winter in Prostokvashino (1984) | Animation |
| 4 | Vacations in Prostokvashino (1980) | Animation |
| 5 | Garfield's Pet Force (2009) | Animation |
| 6 | Nasu: Summer in Andalusia (2003) | Animation |
| 7 | Three from Prostokvashino (1978) | Animation |
| 8 | Investigation Held by Kolobki (1986) | Animation |
| 9 | Fireworks, Should We See It from the Side or t... | Animation |

**Tabel 3. Hasil rekomendasi**
Seperti yang terlihat pada Tabel 3, model berhasil menghasilkan rekomendasi film yang sesuai dengan genre *Piper (2016)*.

## Evaluation 
Berdasarkan hasil rekomendasi yang dihasilkan, semua 10 judul film yang direkomendasikan relevan dengan film yang diuji. Oleh karena itu, Precision dari model ini adalah 10/10 atau 100%, menunjukkan bahwa sistem rekomendasi berhasil memberikan rekomendasi yang tepat. 

## Conclusion  
Setelah melalui serangkaian tahapan, mulai dari persiapan dataset hingga evaluasi, sistem rekomendasi berbasis *Machine Learning Content-Based Filtering* berhasil diselesaikan dengan hasil yang memuaskan. Dari 10 film yang direkomendasikan, semuanya relevan dengan film yang diuji, menunjukkan bahwa *precision* model ini mencapai 100%. Dengan adanya sistem rekomendasi ini, diharapkan *traffic* pada platform streaming film perusahaan dapat mengalami peningkatan yang signifikan. Proyek ini berhasil mengembangkan sistem rekomendasi berbasis konten yang mampu menyarankan film sesuai dengan genre. Menggunakan teknik TF-IDF dan cosine similarity, sistem ini dapat memberikan rekomendasi yang tepat, serta meningkatkan pengalaman pengguna di platform streaming film.

## Referensi  
[1] (https://www.neliti.com/id/publications/140821/penerapan-metode-content-based-filtering-pada-sistem-rekomendasi-kegiatan-ekstra) Firmahsyah, Firmahsyah dan Tiur Gantini. "Penerapan Metode Content-Based Filtering Pada Sistem Rekomendasi Kegiatan Ekstrakulikuler (Studi Kasus Di Sekolah ABC)." _Jurnal Teknik Informatika dan Sistem Informasi_, vol. 2, no. 3, 2016, doi:(10.28932/jutisi.v2i3.548)(https://dx.doi.org/10.28932/jutisi.v2i3.548).  

[2] (http://books.uinsby.ac.id/id/eprint/216/3/Yoyon%20Mudjiono_Kajian%20Semiotika%20dalam%20Film.pdf) Mudjiono, Yoyon (2020) _Kajian semiotika dalam film._ Jurnal Ilmu Komunikasi, 1 (1). pp. 125-138. ISSN 2088-981X; 2723-2557  

[3] (https://chaitanyabelhekar.medium.com/recommender-system-metrics-clearly-explained-1f2ba6690216)   Belhekar, Chaitanya . (2020, Agustus 29).  Recommender System Metrics — Clearly Explained. (https://chaitanyabelhekar.medium.com/recommender-system-metrics-clearly-explained-1f2ba6690216) (https://chaitanyabelhekar.medium.com/recommender-system-metrics-clearly-explained-1f2ba6690216)
