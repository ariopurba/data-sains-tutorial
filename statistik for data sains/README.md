## Bagian ini akan Mempelajari Mengenai Topik Berikut:
### Outline

1. [Statistik Deskriptif](#1-descriptive-statistics-statistik-deskriptif)
2. [Probabilitas](#2-probability-probabilitas)
3. [Distribusi Probabilitas](#3-probability-distributions-distribusi-probabilitas)
4. [Kasus Nyata: Analisis Durasi Menonton Video](#kasus-nyata-analisis-durasi-menonton-video)
5. [Langkah-langkah Analisis Statistik Deskriptif](#langkah-langkah-analisis-statistik-deskriptif)
    - [Mean (Rata-rata)](#1-mean-rata-rata)
    - [Median (Nilai Tengah)](#2-median-nilai-tengah)
    - [Modus (Nilai yang paling sering muncul)](#3-modus-nilai-yang-paling-sering-muncul)
    - [Range (Rentang)](#4-range-rentang)
    - [Variance (S²)](#5-variance-s²)
    - [Standard Deviation (S)](#6-standard-deviation-s)
6. [Kapan Ini Digunakan di Data Sains?](#kapan-ini-digunakan-di-data-sains)
7. [Contoh Probabilitas](#contoh-1-probabilitas-user-menonton-lebih-dari-50-menit)
8. [Distribusi Normal dan Z-Score](#distribusi-normal-dan-z-score)
9. [Distribusi Normal & Probabilitas](#distribusi-normal--probabilitas)
10. [Kesimpulan Pemakaian dalam Data Sains](#kesimpulan-pemakaian-dalam-data-sains)

---
Langsung menuju ke bagian yang diinginkan dengan klik judul pada outline di atas.

# Statistik Dalam Data Sains
## **1. Descriptive Statistics (Statistik Deskriptif)**

> Digunakan untuk meringkas dan mendeskripsikan data.

**Main Point:**

-   **Ukuran Pemusatan**: Mean (rata-rata), Median, Modus
    
-   **Ukuran Penyebaran**: Range, Varians, Standar Deviasi
    
-   **Visualisasi**: Histogram, Boxplot, Bar chart
    

----------

## **2. Probability (Probabilitas)**

> Ilmu menghitung kemungkinan suatu peristiwa.

**Main Point:**

-   **Ruang Sampel & Kejadian**
    
-   **Aturan Probabilitas**: AND (∩), OR (∪), komplemen
    
-   **Conditional Probability**: P(A|B) = P(A ∩ B)/P(B)
    
-   **Independen vs Dependen**
    

----------

## **3. Probability Distributions (Distribusi Probabilitas)**

> Fungsi yang memetakan nilai ke kemungkinan kemunculan.

**Main Point:**

-   **Distribusi Diskrit**: Binomial, Poisson
    
-   **Distribusi Kontinu**: Normal, Exponential
    
-   **Properties**: mean, variance, bentuk kurva



Baik! Kita mulai dengan **statistik deskriptif** menggunakan contoh nyata dari dunia data sains.

----------

## **Kasus Nyata: Analisis Durasi Menonton Video**

Misalnya kamu adalah data scientist di perusahaan edukasi (misalnya seperti Udemy atau Coursera). Kamu ingin menganalisis **berapa lama rata-rata user menonton video kursus dalam sehari** untuk memahami apakah pengguna cukup engage atau tidak.

### **Dataset Sederhana (dalam menit)**:

Kamu punya data berikut dari 10 user:

```plaintext
[40, 50, 60, 30, 45, 90, 55, 60, 20, 65]

```

----------

## **Langkah-langkah Analisis Statistik Deskriptif**

----------

### **1. Mean (Rata-rata)**

Formula:

$Mean=∑xin\text{Mean} = \frac{\sum x_i}{n}$

Perhitungan:

$Mean=40+50+60+30+45+90+55+60+20+6510=51510=51.5 menit\text{Mean} = \frac{40 + 50 + 60 + 30 + 45 + 90 + 55 + 60 + 20 + 65}{10} = \frac{515}{10} = 51.5 \text{ menit}$

----------

### **2. Median (Nilai Tengah)**

Langkah:

1.  Urutkan data: [20, 30, 40, 45, 50, 55, 60, 60, 65, 90]
    
2.  Karena jumlah data genap (10), ambil dua nilai tengah dan rata-rata:
    

$Median=50+552=52.5 menit\text{Median} = \frac{50 + 55}{2} = 52.5 \text{ menit}$

----------

### **3. Modus (Nilai yang paling sering muncul)**

Dari data: [40, 50, 60, 30, 45, 90, 55, 60, 20, 65]  
Yang muncul 2x adalah **60**.

$Modus=60 menit\text{Modus} = 60 \text{ menit}$

----------

### **4. Range (Rentang)**

$Range=Max−Min=90−20=70 menit\text{Range} = \text{Max} - \text{Min} = 90 - 20 = 70 \text{ menit}$

----------

### **5. Variance (S²)**

Formula:

$S2=1n∑i=1n(xi−xˉ)2S^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$

Langkah:

1.  Hitung deviasi tiap titik data dari rata-rata (51.5), lalu kuadratkan:
    

```plaintext
(40 - 51.5)² = 132.25  
(50 - 51.5)² = 2.25  
(60 - 51.5)² = 72.25  
(30 - 51.5)² = 462.25  
(45 - 51.5)² = 42.25  
(90 - 51.5)² = 1,482.25  
(55 - 51.5)² = 12.25  
(60 - 51.5)² = 72.25  
(20 - 51.5)² = 992.25  
(65 - 51.5)² = 182.25  

```

2.  Total = 3,452.5
    
3.  Variance = 3,452.5 / 10 = **345.25**
    

----------

### **6. Standard Deviation (S)**

$S=S2=345.25≈18.57 menitS = \sqrt{S^2} = \sqrt{345.25} ≈ 18.57 \text{ menit}$

----------

## **Kapan Ini Digunakan di Data Sains?**

-   **Mean, Median, Modus** → untuk memahami data sentral
    
-   **Range, SD, Varians** → untuk melihat seberapa tersebar atau bervariasinya perilaku pengguna
    
-   **Use-case**: Segmentasi pengguna, deteksi outlier, preprocessing sebelum machine learning
    

----------

# Probabilitas

## **1. PROBABILITAS (Probability)**

### Tujuan:

Mengukur kemungkinan suatu kejadian. Dalam data sains, probabilitas digunakan saat:

-   Prediksi kelas (dalam klasifikasi)
    
-   Deteksi anomali
    
-   Perhitungan model Bayesian
    
-   Inferensi statistik
    

----------

### **Contoh 1: Probabilitas user menonton lebih dari 50 menit**

**Langkah:**

-   Total data: 10 user
    
-   User yang menonton >50 menit: 60, 90, 55, 60, 65 → total 5 user
    

**Formula:**

$P(X>50)=jumlah kejadian sesuaijumlah total kejadian=510=0.5P(X > 50) = \frac{\text{jumlah kejadian sesuai}}{\text{jumlah total kejadian}} = \frac{5}{10} = 0.5P(X>50)=jumlah total kejadianjumlah kejadian sesuai​=105​=0.5$

📌 _Dalam data sains_: probabilitas ini bisa digunakan untuk mengukur **engagement rate** atau probabilitas seorang user masuk kategori aktif.

----------

### **Contoh 2: Probabilitas user menonton tepat 60 menit**

Data 60 muncul **2 kali**.

$P(X=60)=210=0.2P(X = 60) = \frac{2}{10} = 0.2P(X=60)=102​=0.2$

📌 _Dipakai dalam:_ menghitung frekuensi spesifik, misalnya dalam **deteksi pola waktu akses atau lama sesi pengguna**.

----------

## **2. DISTRIBUSI PROBABILITAS (Probability Distribution)**

### Tujuan:

Menjelaskan bagaimana probabilitas tersebar untuk setiap nilai kemungkinan dari sebuah variabel acak.

----------

### **Jenis Distribusi:**

-   **Diskrit**: Binomial, Poisson
    
-   **Kontinu**: Normal, Exponential
    

----------

### **Kita fokus ke Distribusi Normal**

Sebab data durasi sering dianggap **berdistribusi normal** di dunia nyata.

----------

## **Distribusi Normal dan Z-Score**

Untuk melihat seberapa jauh data dari rata-rata dalam satuan standar deviasi.

**Rumus Z-Score:**

$Z=X−μσZ = \frac{X - \mu}{\sigma}Z=σX−μ​$

-   X = nilai individu
    
-   μ = mean (51.5)
    
-   σ = standar deviasi (18.57)
    

----------

### **Contoh: Seberapa jauh 90 menit dari rata-rata?**

$Z=90−51.518.57≈38.518.57≈2.07Z = \frac{90 - 51.5}{18.57} ≈ \frac{38.5}{18.57} ≈ 2.07Z=18.5790−51.5​≈18.5738.5​≈2.07$

Artinya: nilai 90 berada **sekitar 2.07 standar deviasi di atas rata-rata**.

📌 _Dalam data sains_: Ini penting untuk:

-   Deteksi outlier (nilai yang jauh dari rata-rata)
    
-   P-Value dalam inferensi statistik
    
-   Menentukan probabilitas di distribusi kontinu
    

----------

### **Distribusi Normal & Probabilitas**

Dengan distribusi normal standar (Z), kamu bisa cari **area di bawah kurva** (menggunakan tabel Z) untuk menghitung:

Contoh:

> "Berapa kemungkinan user menonton **kurang dari 90 menit**?"

Jika Z = 2.07 → dari tabel Z, P(Z < 2.07) ≈ 0.9812  
Jadi,

$P(X<90)≈0.9812P(X < 90) ≈ 0.9812P(X<90)≈0.9812$

Artinya: **98.12% user berada di bawah 90 menit.**

----------

### **Kesimpulan Pemakaian dalam Data Sains:**

| Proses                          | Penjelasan                                    | Aplikasi                                          |
| ------------------------------- | --------------------------------------------- | ------------------------------------------------- |
| **Probabilitas dasar**          | Hitung kemungkinan suatu nilai atau range     | Prediksi user behavior, klasifikasi               |
| **Distribusi Probabilitas**     | Model bentuk sebaran data                     | Outlier detection, simulasi data                  |
| **Distribusi Normal & Z-Score** | Standarisasi data untuk statistik inferensial | P-Value, confidence interval, preprocessing model |
