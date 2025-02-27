# -*- coding: utf-8 -*-
"""
Analisis Prediktif - Proyek Gaji Karyawan Berdasarkan Pengalaman Kerja

Machine Learning Terapan - Adilah Widiasti - B244035E

Dataset yang digunakan dalam proyek ini :  
https://www.kaggle.com/datasets/rubydoby/years-of-experience-and-employees-salary

Deskripsi Proyek :
Proyek ini berfokus pada topik ekonomi dan bisnis, dengan penekanan pada proses perekrutan karyawan baru di perusahaan. Di sini, perusahaan berusaha menentukan kisaran gaji yang sesuai dengan pengalaman kerja calon pelamar. Untuk memperoleh hasil yang lebih akurat, perusahaan berencana untuk menguji dua model machine learning dan memilih model yang memberikan prediksi paling akurat.
"""
# Menginstal library Kaggle
#!pip install kaggle

# Membuat direktori untuk Kaggle
#!rm -rf ~/.kaggle && mkdir ~/.kaggle/

# Memindahkan berkas kaggle.json ke direktori yang sesuai
#!mv kaggle.json ~/.kaggle/kaggle.json

# Mengubah izin berkas
#!chmod 600 ~/.kaggle/kaggle.json

# Mengunduh dataset
#!kaggle datasets download -d rubydoby/years-of-experience-and-employees-salary

# Mengekstrak berkas zip
#!unzip /content/years-of-experience-and-employees-salary.zip

#1. Mengimpor Library yang Diperlukan

# Mengimpor library untuk pemrosesan data dan visualisasi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mengimpor library untuk persiapan data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Mengimpor model machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# 2. Memahami Data
#Memahami data adalah langkah awal untuk menganalisis informasi dan kualitas data.

# 2.1 Memuat Data
#Memuat dataset agar lebih mudah dipahami. Dataset yang digunakan dalam proyek ini: https://www.kaggle.com/datasets/rubydoby/years-of-experience-and-employees-salary

# Membaca dataset
data_gaji = pd.read_csv('/content/employee_salaries.csv')
data_gaji

# 2.2 Analisis Data Eksploratif (EDA)
#Analisis data eksploratif adalah proses untuk menganalisis karakteristik data, menemukan pola, dan memeriksa asumsi.

# 2.2.1 EDA - Deskripsi Variabel
# Menampilkan informasi dataset
data_gaji.info()

# Menampilkan deskripsi statistik
data_gaji.describe()

# 2.2.2 EDA - Menangani Nilai Hilang dan Outlier

# Memeriksa nilai hilang dalam dataset
print("Jumlah nilai hilang sebelum imputasi:")
print(data_gaji.isna().sum())

# Mengisi nilai hilang dengan median (untuk kolom numerik)
data_gaji.fillna(data_gaji.median(), inplace=True)

# Memeriksa nilai hilang setelah imputasi
print("Jumlah nilai hilang setelah imputasi:")
print(data_gaji.isna().sum())

# Visualisasi untuk mendeteksi outlier
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=data_gaji['Years of Experience'])
plt.title('Boxplot Years of Experience')

plt.subplot(1, 2, 2)
sns.boxplot(x=data_gaji['Salary'])
plt.title('Boxplot Salary')
plt.show()

# Menangani outlier menggunakan metode Z-score
from scipy.stats import zscore

# Menghitung Z-score untuk kolom numerik
z_scores = np.abs(zscore(data_gaji[['Years of Experience', 'Salary']]))

# Menentukan threshold untuk Z-score
threshold = 3

# Menghapus baris yang memiliki Z-score di atas threshold
data_gaji = data_gaji[(z_scores < threshold).all(axis=1)]

# Memeriksa ukuran dataset setelah menghapus outlier
print("Ukuran dataset setelah menghapus outlier:")
print(data_gaji.shape)

# 2.2.3 EDA - Analisis Univariate

# Visualisasi histogram untuk fitur numerik
data_gaji.hist(bins=50, figsize=(20, 15))
plt.show()

# 2.2.4 EDA - Analisis Multivariate

# Mengamati hubungan antar fitur dengan pairplot
sns.pairplot(data_gaji, diag_kind='kde')

# Mengamati korelasi antar fitur dengan heatmap
plt.figure(figsize=(10, 8))
matriks_korelasi = data_gaji.corr().round(2)
sns.heatmap(data=matriks_korelasi, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Matriks Korelasi Fitur Numerik", size=20)

# 3. Persiapan Data 
#Persiapan data adalah langkah untuk mempersiapkan data sebelum membangun model machine learning.

# 3.1 Pembagian Data Latih dan Uji

# Memisahkan dataset menjadi data latih dan data uji
X = data_gaji.drop(["Salary"], axis=1)
y = data_gaji["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

print(f'Total sampel dalam dataset: {len(X)}')
print(f'Total sampel dalam data latih: {len(X_train)}')
print(f'Total sampel dalam data uji: {len(X_test)}')

# 3.2 Standarisasi Data

# Melakukan standarisasi pada data latih
fitur_numerik = ['Years of Experience']
scaler = StandardScaler()
scaler.fit(X_train[fitur_numerik])
X_train[fitur_numerik] = scaler.transform(X_train[fitur_numerik])

# Menampilkan statistik data yang telah distandarisasi
X_train[fitur_numerik].describe().round(4)

# 4. Pengembangan Model
#Pengembangan model adalah tahap di mana algoritma machine learning digunakan untuk menjawab pernyataan masalah.

### 4.1 Mempersiapkan Dataframe untuk Analisis Model

# Membuat dataframe untuk menyimpan hasil MSE
hasil_model = pd.DataFrame(index=['train_mse', 'test_mse'], columns=['LinearRegression'])

# 4.2 Membangun Model dengan Algoritma Regresi Linier

# Membuat dan melatih model regresi linier
model_regresi = LinearRegression(n_jobs=-1)
model_regresi.fit(X_train, y_train)
hasil_model.loc['train_mse', 'LinearRegression'] = mean_squared_error(y_pred=model_regresi.predict(X_train), y_true=y_train)

# 5. Evaluasi Model
#Evaluasi model adalah tahap untuk memastikan model dapat membuat prediksi yang akurat.

# Melakukan standarisasi pada data uji
X_test[fitur_numerik] = scaler.transform(X_test[fitur_numerik])

# 5.1 Evaluasi Model Menggunakan Metrik MSE

# Membuat dataframe untuk menyimpan nilai MSE
mse_hasil = pd.DataFrame(columns=['train', 'test'], index=['LinearRegression'])

# Menghitung MSE untuk model regresi linier
mse_hasil.loc['LinearRegression', 'train'] = mean_squared_error(y_true=y_train, y_pred=model_regresi.predict(X_train)) / 1e3
mse_hasil.loc['LinearRegression', 'test'] = mean_squared_error(y_true=y_test, y_pred=model_regresi.predict(X_test)) / 1e3

# Menampilkan hasil MSE
mse_hasil

# Visualisasi hasil MSE
fig, ax = plt.subplots()
mse_hasil.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

# 5.2 Melakukan Pengujian Model

# Melakukan pengujian terhadap model
data_prediksi = X_test.iloc[2:3].copy()
hasil_prediksi = {'y_true': y_test[2:3]}
hasil_prediksi['prediksi_LinearRegression'] = model_regresi.predict(data_prediksi).round(1)

pd.DataFrame(hasil_prediksi)

# 6. Peningkatan Model
#Karena hasil prediksi dari model regresi linier kurang akurat, kita akan membandingkannya dengan algoritma Random Forest.

#6.1 Membangun Model dengan Algoritma Random Forest
# Membuat model prediksi menggunakan algoritma Random Forest
model_rf = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
model_rf.fit(X_train, y_train)

# Menyimpan hasil MSE untuk model Random Forest
hasil_model.loc['train_mse', 'RandomForest'] = mean_squared_error(y_pred=model_rf.predict(X_train), y_true=y_train)

# 6.2 Membandingkan Hasil Evaluasi dari Kedua Model Menggunakan Metrik MSE

# Membuat dataframe untuk menyimpan nilai MSE dari kedua model
mse_hasil = pd.DataFrame(columns=['train', 'test'], index=['LinearRegression', 'RandomForest'])

# Menghitung MSE untuk kedua model
model_dict = {'LinearRegression': model_regresi, 'RandomForest': model_rf}
for nama_model, model in model_dict.items():
    mse_hasil.loc[nama_model, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train)) / 1e3
    mse_hasil.loc[nama_model, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test)) / 1e3

# Menampilkan hasil MSE untuk kedua model
mse_hasil

# Visualisasi hasil MSE
fig, ax = plt.subplots()
mse_hasil.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

# 6.3 Melakukan Pengujian dari Kedua Model

# Melakukan pengujian terhadap kedua model
data_prediksi = X_test.iloc[2:3].copy()
hasil_prediksi = {'y_true': y_test[2:3]}
for nama_model, model in model_dict.items():
    hasil_prediksi['prediksi_' + nama_model] = model.predict(data_prediksi).round(1)

pd.DataFrame(hasil_prediksi)
