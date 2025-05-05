# Daftar Isi

1.  **Pengantar NLP**
    *   Proses NLP: Teks → Preprocessing → Embedding → Model → Output
    *   Contoh

2.  **Preprocessing di NLP**
    *   Teks Contoh (Data Sederhana)
    *   Tahapan Preprocessing
        *   Tokenisasi
        *   Vocabulary (Kamus Kata Unik)
        *   Bag-of-Words (BoW)
        *   TF-IDF (Term Frequency-Inverse Document Frequency)
            *   Formula
            *   Langkah 1: Hitung TF
            *   Langkah 2: Hitung IDF
            *   Langkah 3: Hitung TF-IDF
            *   TF-IDF Matrix
        *   Word Embedding (Konsep Manual)
            *   Contoh Representasi Kalimat dengan Rata-rata Vektor
    *   Ringkasan Tahapan

3.  **Pemodelan / Machine Learning untuk Tugas NLP**

4.  **Contoh Kasus: Text Classification (Spam vs Non-Spam) dengan Naive Bayes**
    *   Data Sederhana (Dataset Train)
    *   STEP 1: Preprocessing (Tokenisasi + Normalisasi)
    *   STEP 2: Bangun Vocabulary (kamus kata)
    *   STEP 3: Representasi dengan Bag of Words (BoW)
    *   STEP 4: Hitung Probabilitas dengan Naive Bayes
        *   Rumus
        *   Hitung P(C) (probabilitas kelas)
        *   Hitung Likelihood P(w|C) (dengan Laplace smoothing)
            *   Kelas Spam
            *   Kelas Ham
    *   STEP 5: Prediksi Kalimat Baru
    *   Hasil

Di Tulisan ini saya akan menjabarkan material singkat mengenai NLP (natural language processing).  
mari kita mulai.

secara garis besar proses dari NLP adalah sebagai berikut:**  
Teks → Preprocessing → Embedding → Masuk ke Model → Output / Prediksi**

contoh :

Teks:  `"saya suka nonton film ini"`

**Preprocessing**:  
→  `"suka nonton film"`

**Embedding**:  
→  `[0.4, 0.2, 0.6]`  (representasi vektor)

**Model Sentiment**:  
→ Masukkan ke model → prediksi  `"positif"`

okey mari kita masuk ke bagian preprocesing di NLP.

**TEKS CONTOH (DATA SEDERHANA):**

Dokumen 1: “saya suka belajar AI”  
Dokumen 2: “saya belajar machine learning”

1: Tokenisasi

D1: [“saya”, “suka”, “belajar”, “AI”]  
D2: [“saya”, “belajar”, “machine”, “learning”]

2. Vocabulary (Kamus Kata Unik)

Gabungan dari semua kata di kedua dokumen:

[“saya”, “suka”, “belajar”, “AI”, “machine”, “learning”]`

3. Bag-of-Words (BoW)

Representasi setiap dokumen dalam bentuk vektor berdasarkan frekuensi kata.

![](https://miro.medium.com/v2/resize:fit:700/1*z8piAH4AvgS6ls6TpK8KrQ.png)

4. TF-IDF (Term Frequency-Inverse Document Frequency)

Formula:

-   TF (term frequency) = (jumlah kemunculan kata di dokumen) / (jumlah total kata di dokumen)
-   IDF (inverse document frequency) = log(N / df),  
    di mana: N = jumlah dokumen, df = jumlah dokumen yang mengandung kata tersebut

Langkah 1: Hitung TF

Contoh TF untuk  `"saya"`  di Dokumen 1:

-   `TF("saya", D1) = 1 / 4 = 0.25`
-   `TF("belajar", D2) = 1 / 4 = 0.25`

(semua kata muncul 1 kali dalam dokumen berisi 4 kata)

Langkah 2: Hitung IDF

df("saya")     = 2 → IDF = log(2 / 2) = 0  
df("suka")     = 1 → IDF = log(2 / 1) = 0.301  
df("belajar")  = 2 → IDF = log(2 / 2) = 0  
df("AI")       = 1 → IDF = log(2 / 1) = 0.301  
df("machine")  = 1 → IDF = log(2 / 1) = 0.301  
df("learning") = 1 → IDF = log(2 / 1) = 0.301

Langkah 3: Hitung TF-IDF

Contoh  `"suka"`  di D1:

-   `TF = 1/4 = 0.25`, IDF = 0.301  
    →  `TF-IDF = 0.25 * 0.301 = 0.07525`

Contoh  `"saya"`  di D1:

-   `TF = 0.25`, IDF = 0 →  `TF-IDF = 0`

TF-IDF Matrix:

![](https://miro.medium.com/v2/resize:fit:700/1*X7nh-_pAMuhng7NS7rcjiA.png)

5. Word Embedding (Konsep Manual)

Misalkan kamu buat embedding secara manual dengan dimensi 2:

![](https://miro.medium.com/v2/resize:fit:700/1*h9qXvew5tdHzyzc7UzhNgQ.png)

Kalimat  `"saya suka belajar AI"`  bisa direpresentasikan sebagai:

-   Rata-rata vektor dari semua kata

```python
import numpy as np  
  
v_saya = np.array([0.1, 0.3])  
v_suka = np.array([0.4, 0.2])  
v_belajar = np.array([0.3, 0.5])  
v_ai = np.array([0.7, 0.9])
  
  
kalimat = np.mean([v_saya, v_suka, v_belajar, v_ai], axis=0)  
print(kalimat)  # output: vektor representasi kalimat
```

Ringkasan:

-   Tokenisasi → pisahkan kata
-   BoW → hitung frekuensi
-   TF-IDF → beri bobot penting
-   Embedding → ubah kata jadi vektor makna

Okey setelah tahap  **teks sudah dalam bentuk angka (vektor)**  dan bisa dimasukkan ke dalam model.

Pemodelan / Machine Learning Untuk tugas NLP:

![](https://miro.medium.com/v2/resize:fit:700/1*fXs_tFwjg5fyuHYu35vjog.png)

Sekian materi singkat mengenai langkah-langkah dalam NLP. Tulisan berikutnya akan mengenai contoh kasus sederhana Text Klasifikasi dengan Naive bayes untuk masalah Spam / Non-Spam .


----------

Sekarang mari kita bahas contoh **Text Classification: Spam vs Non-Spam** secara **detail**, dari teks mentah → preprocessing → representasi angka → perhitungan manual → klasifikasi. Kita akan pakai **algoritma sederhana Naive Bayes**, agar bisa dihitung dengan manual.

Contoh Data sederhana:  
dataset train:

![](https://cdn-images-1.medium.com/max/800/1*7yq9vLHlUJNmR2A9dbMlEA.png)

STEP 1: Preprocessing (Tokenisasi + Normalisasi)

Buat list kata dari semua kalimat:

-   “beli sekarang” → `['beli', 'sekarang']`
-   “diskon besar” → `['diskon', 'besar']`
-   “saya suka belajar” → `['saya', 'suka', 'belajar']`
-   “belajar sekarang di rumah” → `['belajar', 'sekarang', 'di', 'rumah']`

STEP 2: Bangun Vocabulary (kamus kata)

Gabungkan semua kata dari dataset:

['beli', 'sekarang', 'diskon', 'besar', 'saya', 'suka', 'belajar', 'di', 'rumah']

Total 9 kata unik → ini jadi dimensi dari representasi kita.

----------

STEP 3: Representasi dengan Bag of Words (BoW)

Misal: Kalimat `"saya suka belajar"` jadi vektor:

[beli=0, sekarang=0, diskon=0, besar=0, saya=1, suka=1, belajar=1, di=0, rumah=0]  
→ [0, 0, 0, 0, 1, 1, 1, 0, 0]

----------

STEP 4: Hitung Probabilitas dengan Naive Bayes

Rumus:

Untuk teks `x`, cari kelas `C` dengan probabilitas tertinggi:

P(C|x) ∝ P(C) × P(w1|C) × P(w2|C) × ...

Hitung P(C) (probabilitas kelas):

-   P(Spam) = 2 / 4 = 0.5
-   P(Ham) = 2 / 4 = 0.5

Hitung Likelihood P(w|C)

(dengan Laplace smoothing)

Kita hitung jumlah kemunculan setiap kata di kelas masing-masing:

Kelas Spam:

Kalimat:

-   “beli sekarang” → beli, sekarang
-   “diskon besar” → diskon, besar

Gabungan kata Spam = `['beli', 'sekarang', 'diskon', 'besar']`

![](https://cdn-images-1.medium.com/max/800/1*qOsATdS8HipAW7j-PH8n5g.png)

Total kata = 4  
Ukuran vocab = 9  
→ Pakai **Laplace smoothing**

P(w|Spam) = (jumlah_kata + 1) / (total_kata + vocab_size)

Contoh:

P('beli' | Spam) = (1 + 1) / (4 + 9) = 2 / 13  
P('belajar' | Spam) = (0 + 1) / (4 + 9) = 1 / 13

Kelas Ham:

Gabungan kata Ham = `['saya', 'suka', 'belajar', 'belajar', 'sekarang', 'di', 'rumah']`

![](https://cdn-images-1.medium.com/max/800/1*Lb8S3PxXF2NzWZRSoCyULA.png)

Total kata = 7  
Ukuran vocab = 9

Contoh:

P('belajar' | Ham) = (2 + 1) / (7 + 9) = 3 / 16  
P('beli' | Ham) = (0 + 1) / (7 + 9) = 1 / 16

STEP 5: Prediksi Kalimat Baru

Misal input baru:

“sekarang belajar”  
→ token: ['sekarang', 'belajar']

Hitung:

P(Spam | x):

P(Spam) = 0.5    
P(‘sekarang’|Spam) = 2/13    
P(‘belajar’|Spam) = 1/13

P = 0.5 × (2/13) × (1/13) = 0.5 × 2/169 = 1/169 ≈ 0.0059

P(Ham | x):

P(Ham) = 0.5    
P(‘sekarang’|Ham) = 2/16    
P(‘belajar’|Ham) = 3/16

P = 0.5 × (2/16) × (3/16) = 0.5 × 6/256 = 3/256 ≈ 0.0117

----------

Hasil:

Karena **P(Ham | x) > P(Spam | x)** → maka kalimat `"sekarang belajar"` diklasifikasikan sebagai **Ham** (bukan spam).