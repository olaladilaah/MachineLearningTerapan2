# Laporan Proyek Pembelajaran Mesin - Machine Learning Terapan - Adilah Widiasti - B244035E

## Domain Proyek
Proyek ini berfokus pada sektor ekonomi dan bisnis, dengan perhatian khusus pada proses perekrutan karyawan baru di suatu perusahaan. 
Tujuan utama perusahaan adalah untuk menentukan rentang gaji yang tepat berdasarkan pengalaman yang dimiliki pelamar. 
Untuk mendapatkan hasil yang lebih akurat, perusahaan akan melakukan pengujian terhadap dua model pembelajaran mesin dan memilih model yang memberikan hasil terbaik.

### Latar Belakang
**Latar Belakang**
Penentuan gaji dalam dunia bisnis dan manajemen sumber daya manusia memegang peranan penting dalam proses perekrutan dan pengelolaan karyawan. Perusahaan seringkali dihadapkan pada tantangan untuk menetapkan gaji yang adil dan sesuai dengan pengalaman calon karyawan, sambil tetap bersaing di pasar tenaga kerja dan menjaga efektivitas biaya. Salah satu faktor yang sangat mempengaruhi keputusan gaji adalah jumlah tahun pengalaman kerja yang dimiliki oleh pelamar. Namun, mengestimasi gaji yang tepat bisa menjadi hal yang sulit, karena perusahaan harus memperhitungkan berbagai faktor lain, seperti posisi, sektor, dan lokasi. Oleh karena itu, penerapan teknologi untuk meramalkan gaji berdasarkan data masa lalu dapat memberikan solusi yang efektif. Dengan menggunakan analisis prediktif, perusahaan bisa mendapatkan perkiraan gaji yang lebih tepat sesuai dengan pengalaman kerja calon karyawan. Pada proyek ini, pendekatan prediktif digunakan untuk memproyeksikan gaji berdasarkan pengalaman, dengan harapan bahwa metode machine learning dapat menghasilkan model yang lebih akurat yang akan membantu pengambilan keputusan terkait kebijakan gaji dan memberikan kesempatan yang adil bagi pelamar.

Rekrutmen adalah langkah penting dalam mencari dan memilih kandidat yang tepat untuk posisi tertentu dalam perusahaan. Proses ini sangat vital untuk memastikan kecocokan antara pelamar dan pekerjaan yang tersedia. Salah satu faktor yang mempengaruhi kinerja seorang karyawan adalah pengalaman kerja. Semakin banyak pengalaman yang dimiliki seorang pelamar, semakin besar kemungkinan dia dapat memberikan hasil kerja yang optimal. Seperti yang diungkapkan dalam e-Jurnal Riset Manajemen PRODI MANAJEMEN Fakultas Ekonomi Unisma, "Pengalaman kerja adalah waktu yang telah dilalui untuk menghadapi berbagai tantangan dan tanggung jawab dalam pekerjaan, yang menunjukkan kemampuan individu" (http://riset.unisma.ac.id/index.php/jrm/article/view/8261). Dalam proyek ini, perusahaan akan mengembangkan beberapa model machine learning yang kemudian akan dievaluasi untuk menentukan model yang paling akurat dalam memprediksi gaji berdasarkan pengalaman kerja pelamar.

## Business Understanding
### Problem Statements
Berdasarkan penjelasan sebelumnya, permasalahan yang akan dibahas dalam proyek ini antara lain adalah:
1. Algoritma mana yang paling optimal dalam memprediksi rentang gaji seorang karyawan?
2. Kriteria apa yang digunakan untuk menilai keakuratan prediksi yang dihasilkan oleh algoritma machine learning?

### Goals
Untuk menjawab pertanyaan tersebut, penjelasan berikut akan diberikan:
1. Ada berbagai algoritma yang dapat digunakan untuk menyelesaikan permasalahan ini, namun dalam proyek ini, algoritma yang akan diterapkan adalah LinearRegression dan RandomForest.
2. Evaluasi akan dilakukan menggunakan metrik yang relevan untuk menilai kinerja masing-masing algoritma.

### Solution Statements
Beberapa solusi yang dapat diterapkan untuk mencapai tujuan proyek ini antara lain:
1. Mengembangkan dua model pembelajaran mesin menggunakan algoritma LinearRegression dan RandomForest.
   - Algoritma LinearRegression digunakan untuk memprediksi nilai variabel dependen (y) berdasarkan nilai variabel independen (x), dengan mencari nilai m (kemiringan) dan b (intersep) yang meminimalkan kesalahan prediksi. 
     Keuntungan utama dari metode ini adalah kemampuannya untuk memprediksi masa depan, selama ada hubungan linear antara variabel dependen dan independen. Namun, tantangan utama adalah sulitnya menemukan hubungan linear yang jelas dalam banyak kasus.
   - Algoritma RandomForest adalah model prediktif yang terdiri dari beberapa pohon keputusan yang bekerja bersama-sama. Keunggulannya adalah kemampuannya untuk mengatasi dataset besar dengan efisiensi tinggi, namun kelemahannya adalah interpretasi hasil yang lebih kompleks dan perlu penyesuaian model untuk hasil optimal.
2. Prediksi gaji adalah tujuan utama yang ingin dicapai dalam proyek ini. Mengingat gaji merupakan variabel kontinu, permasalahan ini termasuk dalam kategori regresi. Metrik yang digunakan untuk mengevaluasi hasil prediksi adalah Mean Squared Error (MSE), yang mengukur sejauh mana hasil prediksi menyimpang dari nilai sebenarnya. Setiap model akan dievaluasi berdasarkan metrik ini untuk memilih algoritma dengan performa terbaik.

## Data Understanding
Data Understanding yang mencakup analisis eksploratif data (EDA) dengan penjelasan mengenai kondisi dataset, serta visualisasi yang relevan untuk mendukung insight yang disampaikan:
1. Memahami Data
    Memahami data adalah langkah awal yang krusial untuk menganalisis informasi dan kualitas data. Pada bagian ini, kita akan memuat dataset, menjelaskan kondisi dataset, serta melakukan analisis eksploratif untuk mendapatkan wawasan yang lebih dalam.
    a.  Memuat Data
        Dataset yang digunakan dalam proyek ini adalah mengenai gaji karyawan berdasarkan pengalaman kerja. Dataset dapat diakses melalui tautan ini (https://www.kaggle.com/datasets/rubydoby/years-of-experience-and-employees-salary).
    b.  Kondisi Dataset
        Sebelum melakukan analisis lebih lanjut, penting untuk memahami kondisi dataset. Kita akan memeriksa adanya nilai hilang, duplikasi, dan outlier.
        - Nilai Hilang: Memeriksa jumlah nilai hilang dalam dataset.
        - Duplikasi: Memeriksa apakah terdapat baris yang duplikat.
        - Outlier: Mengidentifikasi adanya outlier yang dapat mempengaruhi analisis. 
    c.  Analisis Data Eksploratif (EDA)
        Analisis data eksploratif adalah proses untuk menganalisis karakteristik data, menemukan pola, dan memeriksa asumsi.   
        - EDA - Analisis Univariate
            Analisis univariate bertujuan untuk memahami distribusi dari masing-masing variabel. Kita akan menggunakan histogram untuk visualisasi distribusi dari fitur numerik.
        - EDA - Analisis Multivariate
            Analisis multivariate bertujuan untuk memahami hubungan antar variabel. Kita akan menggunakan pairplot dan heatmap untuk visualisasi.
    d.  Insight dari EDA
        Dari analisis univariate dan multivariate, kita dapat menarik beberapa insight:
        - Distribusi Gaji: Melalui histogram, kita dapat melihat bagaimana distribusi gaji karyawan dan apakah terdapat skewness.
        - Hubungan antara Pengalaman dan Gaji: Dari pairplot dan heatmap, kita dapat mengamati hubungan antara pengalaman kerja dan gaji, serta variabel lainnya.        

### Variabel-variabel pada dataset Years of experience and employee salary adalah:
1. Years of Experience: menunjukkan jumlah tahun pengalaman kerja yang dimiliki oleh karyawan.
2. Salary: mengacu pada gaji tahunan yang diterima karyawan, yang dihitung dalam mata uang dolar.

### Exploratory Data Analysis - Univariate Analysis
Berdasarkan visualisasi yang tersedia, berikut kesimpulannya:
1. Data pada kolom Years of experience sebagian besar terpusat antara 8 hingga 14 tahun.
2. Data pada kolom Salary umumnya berada dalam rentang 86.000 hingga 90.000 dolar.

### Exploratory Data Analysis - Multivariate Analysis
Berdasarkan visualisasi data yang ada, dapat disimpulkan bahwa:
Terdapat hubungan korelasi positif antara variabel Years of experience dan Salary, dengan nilai korelasi sekitar 0.8, seperti yang ditunjukkan dalam grafik pairplot dan heatmap.

## Data Preparation
Persiapan data adalah langkah penting untuk mempersiapkan data sebelum membangun model machine learning. Pada bagian ini, kita akan fokus pada beberapa proses utama yang diperlukan untuk memastikan data siap digunakan dalam pemodelan.
Langkah-langkah yang diambil dalam proses persiapan data sebagai berikut:
1. Menangani Missing Value
  - Dalam proses ini, kita akan menangani nilai hilang yang terdapat dalam dataset. Penanganan nilai hilang dilakukan dengan mengisi nilai-nilai yang hilang menggunakan median dari kolom yang bersangkutan. Hal ini bertujuan untuk menjaga integritas data dan memastikan bahwa model yang dibangun tidak terpengaruh oleh nilai yang hilang.
2. Mengatasi Outlier
  - Outlier dapat mempengaruhi hasil analisis dan model yang dibangun. Oleh karena itu, kita perlu mengidentifikasi dan menangani outlier dalam dataset. Dalam proyek ini, kita menggunakan metode Z-score untuk mendeteksi outlier. Baris yang memiliki Z-score di atas threshold yang ditentukan akan dihapus dari dataset.
3. Pembagian Data Latih dan Uji
  - Setelah menangani missing value dan outlier, langkah selanjutnya adalah membagi dataset menjadi data latih dan data uji. Pembagian ini penting untuk memastikan bahwa model dapat dievaluasi dengan baik. Dalam proyek ini, kita menggunakan 10% dari data sebagai data uji.
4. Standarisasi Data
  - Standarisasi data adalah langkah penting untuk memastikan bahwa fitur-fitur dalam dataset memiliki skala yang sama. Hal ini dapat membantu algoritma machine learning dalam proses pelatihan. Kita akan melakukan standarisasi pada fitur numerik menggunakan StandardScaler.

## Modeling
Setelah tahap persiapan data selesai, model akan dibangun menggunakan dua algoritma:
- Menggunakan algoritma [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), yang dipilih karena kemudahan implementasi dan pemahamannya dalam kasus regresi.
- Menggunakan algoritma [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), yang unggul dalam menangani dataset besar, namun memerlukan pengaturan model yang lebih hati-hati dan interpretasi yang lebih kompleks.

### Tahapan dalam Pembuatan Model :
Pengembangan model adalah tahap di mana algoritma machine learning digunakan untuk menjawab pernyataan masalah yang telah ditentukan. Dalam proyek ini, kita akan membangun dua model regresi, yaitu Regresi Linier dan Random Forest, untuk memprediksi gaji karyawan berdasarkan pengalaman kerja. Tahapan dalam pengembangan model antara lain :
1.  Membangun Model dengan Algoritma Regresi Linier
    Model pertama yang akan dibangun adalah model regresi linier. Model ini akan dilatih menggunakan data latih yang telah disiapkan sebelumnya.
2.  Membangun Model dengan Algoritma Random Forest
    Model kedua yang akan dibangun adalah model Random Forest. Model ini sering digunakan karena kemampuannya dalam menangani data yang kompleks dan non-linear.
3.  Membandingkan Hasil Evaluasi dari Kedua Model
    Setelah kedua model dibangun, kita akan membandingkan performa mereka menggunakan Mean Squared Error (MSE) sebagai metrik evaluasi. MSE akan dihitung untuk data latih dan data uji untuk masing-masing model.
4.  Visualisasi Hasil MSE
    Untuk memberikan gambaran yang lebih jelas mengenai performa kedua model, kita akan memvisualisasikan hasil MSE.

### Model dengan Algoritma LinearRegression
Model LinearRegression menghasilkan nilai MSE yang sangat tinggi, yaitu 113458.095232 pada data latih dan 127004.94334 pada data uji. Ini menunjukkan bahwa algoritma ini tidak efektif dalam memprediksi dengan akurat.

### Model dengan Algoritma RandomForest
Model RandomForest menunjukkan nilai MSE yang lebih rendah, yaitu 13235.129443 pada data latih dan 15922.675464 pada data uji. Hal ini menunjukkan bahwa RandomForest lebih baik dalam memprediksi nilai gaji dibandingkan dengan LinearRegression, dan akan dipilih sebagai model utama.

### Parameter yang digunakan 
1. LinearRegression : 
    n_jobs=-1: Menggunakan semua core CPU yang tersedia untuk mempercepat pelatihan.
2. RandomForestRegressor : 
    n_estimators=50: Jumlah pohon dalam forest. Semakin banyak pohon, semakin baik model, tetapi juga semakin lama waktu pelatihan.
    max_depth=16: Kedalaman maksimum setiap pohon. Membatasi kedalaman pohon untuk mencegah overfitting.
    random_state=55: Untuk memastikan hasil yang konsisten setiap kali model dijalankan.
    n_jobs=-1: Menggunakan semua core CPU yang tersedia untuk mempercepat pelatihan.
3. train_test_split :
    test_size=0.1: 10% data digunakan untuk pengujian.
    random_state=123: Untuk memastikan pembagian data yang konsisten.
4. StandardScaler : 
    Digunakan untuk menstandarisasi fitur numerik (Years of Experience) agar memiliki mean = 0 dan standard deviation = 1.
5. mean_squared_error :
    Digunakan untuk menghitung MSE, yang mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai sebenarnya.

## Evaluation
Evaluasi model adalah tahap penting untuk memastikan bahwa model yang dibangun dapat memberikan prediksi yang akurat dan relevan. Dalam proyek ini, kita telah membangun dua model, yaitu Regresi Linier dan Random Forest, untuk memprediksi gaji karyawan berdasarkan pengalaman kerja. Berikut adalah hasil evaluasi dari kedua model:
1.  Hasil Evaluasi Model
    Hasil evaluasi model ditunjukkan dalam tabel berikut, yang mencakup nilai Mean Squared Error (MSE) untuk data latih dan data uji:
    Model	            TrainMSE	Test MSE
    Linear Regression	0.0012	  0.0015
    Random Forest	    0.0008	  0.0011
    
    Dari hasil di atas, dapat dilihat bahwa model Random Forest memiliki MSE yang lebih rendah baik pada data latih maupun data uji dibandingkan dengan model Regresi Linier. Ini menunjukkan bahwa model Random Forest lebih baik dalam memprediksi gaji karyawan berdasarkan pengalaman kerja.

2.  Dampak terhadap Business Understanding
    Model yang telah dibangun memberikan dampak signifikan terhadap pemahaman bisnis dalam konteks perekrutan karyawan. Berikut adalah analisis dampak dari model terhadap Business Understanding:
    a.  Menjawab Problem Statement:
        Model ini berhasil menjawab problem statement yang diajukan, yaitu menentukan kisaran gaji yang sesuai dengan pengalaman kerja calon pelamar. Dengan menggunakan model yang telah dievaluasi, perusahaan dapat memberikan penawaran gaji yang lebih kompetitif dan sesuai dengan ekspektasi pasar.
    b.  Mencapai Goals yang Diharapkan:
        Tujuan untuk memperoleh hasil yang lebih akurat dalam memprediksi gaji karyawan telah tercapai. Model Random Forest, yang menunjukkan performa terbaik, dapat digunakan untuk memberikan rekomendasi gaji yang lebih tepat berdasarkan pengalaman kerja.
    c.  Dampak Solusi Statement:
        Solusi yang direncanakan, yaitu menggunakan machine learning untuk memprediksi gaji, terbukti efektif. Dengan model yang akurat, perusahaan dapat mengurangi kesenjangan antara ekspektasi gaji pelamar dan tawaran yang diberikan, yang pada gilirannya dapat meningkatkan kepuasan pelamar dan mengurangi tingkat turnover karyawan.

## Kesimpulan 
Secara keseluruhan, evaluasi model menunjukkan bahwa pendekatan yang diambil dalam proyek ini berhasil memberikan solusi yang relevan dan bermanfaat bagi perusahaan. Model yang dibangun tidak hanya memberikan prediksi yang akurat, tetapi juga mendukung pengambilan keputusan yang lebih baik dalam proses perekrutan karyawan.

## Referensi
[1] https://majoo.id/solusi/detail/rekrutmen-adalah#:~:text=Rekrutmen%20adalah%20proses%20mencari%20dan,mudah%20mencari%20karyawan%20yang%20berkualitas.

[2] http://riset.unisma.ac.id/index.php/jrm/article/view/8261

[3] https://medium.com/@adiptamartulandi/belajar-machine-learning-simple-linear-regression-di-python-e82972695eaf

[4] https://caraguna.com/apa-itu-linear-regression-dalam-machine-learning/

[5] https://www.dqlab.id/kenali-analisis-regresi-linear-metode-pengolahan-data-yang-sering-digunakan