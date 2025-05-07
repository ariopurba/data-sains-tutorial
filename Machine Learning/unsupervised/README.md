## Outline
- [Unsupervised Learning](#unsupervised-learning)
  - [1. Definisi Unsupervised Learning](#1-definisi-unsupervised-learning)
  - [2. Algoritma K-Means](#2-algoritma-k-means)
  - [3. Contoh Perhitungan Manual](#3-contoh-perhitungan-manual)
  - [4. Langkah Kerja Algoritma K-Means](#4-langkah-kerja-algoritma-k-means)
  - [5. Visualisasi dan Analogi](#5-visualisasi-dan-analogi)
  - [6. Kelebihan dan Kekurangan](#6-kelebihan-dan-kekurangan)
  - [7. Implementasi Kode (Python)](#7-implementasi-kode-python)
  - [8. Contoh Aplikasi Nyata](#8-contoh-aplikasi-nyata)
- [Elbow Method](#elbow-method)
  - [1. Konsep Elbow Method](#1-konsep-elbow-method)
  - [2. Rumus Inersia (Within-Cluster Sum of Squares/WCSS)](#2-rumus-inersia-within-cluster-sum-of-squareswcss)
  - [3. Contoh Perhitungan Manual](#3-contoh-perhitungan-manual-1)
  - [4. Interpretasi Elbow Method](#4-interpretasi-elbow-method)
  - [5. Implementasi Python dengan Scikit-Learn](#5-implementasi-python-dengan-scikit-learn)
  - [6. Kelebihan dan Kekurangan](#6-kelebihan-dan-kekurangan-1)
  - [7. Contoh Aplikasi Nyata](#7-contoh-aplikasi-nyata-1)
  - [8. Tips Praktis](#8-tips-praktis)

## Unsupervised Learning
---

### **1. Definisi Unsupervised Learning**  
**Konsep**:  
Teknik ML yang menemukan pola tersembunyi dalam data **tanpa label/target**. Bertujuan untuk:  
- **Clustering**: Mengelompokkan data berdasarkan kemiripan (contoh: segmentasi pelanggan).  
- **Dimensionality Reduction**: Mengurangi fitur dengan mempertahankan informasi penting (contoh: PCA).  

**Ciri Khas**:  
- Tidak ada guidance dari label (berbeda dengan supervised learning).  
- Bergantung pada pengukuran kesamaan (similarity) atau jarak (distance).  

---

### **2. Algoritma K-Means**  
#### **Tujuan**:  
Mengelompokkan data ke dalam $k$ cluster dengan meminimalkan variansi dalam cluster.  

#### **Rumus Matematis**:  
1. **Fungsi Objektif (Inertia)**:  
$
J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$  
- $C_i$: Cluster ke-$i$.  
- $\mu_i$: Pusat cluster (centroid) ke- $i$.  
- $||x - \mu_i||$: Jarak Euclidean antara titik $x$ dan centroid $\mu_i$.  

2. **Jarak Euclidean**:  
$
d(x, \mu_i) = \sqrt{\sum_{j=1}^{n} (x_j - \mu_{ij})^2}
$  

---

### **3. Contoh Perhitungan Manual**  
**Dataset Sederhana (2D)**:  
| Titik | Koordinat (X,Y) |  
|-------|----------------|  
| A     | (1, 1)         |  
| B     | (1, 2)         |  
| C     | (10, 10)       |  
| D     | (10, 11)       |  

**Langkah 1: Inisialisasi Centroid** (Misal $k=2$):  
- Centroid awal: $\mu_1 = (1, 1)$, $\mu_2 = (10, 10)$.  

**Langkah 2: Hitung Jarak dan Kelompokkan**  
- Hitung jarak setiap titik ke centroid:  
  - $d(A, \mu_1) = 0$, $d(A, \mu_2) = \sqrt{(1-10)^2 + (1-10)^2} \approx 12.73$.  
  - **Kesimpulan**: A masuk **Cluster 1**.  
  - Lakukan hal yang sama untuk titik B, C, D.  

**Hasil Pengelompokan Iterasi 1**:  
- Cluster 1: A, B  
- Cluster 2: C, D  

**Langkah 3: Update Centroid**  
- Cluster 1: $(1, 1.5)$
- Cluster 2: $(10, 10.5)$

**Langkah 4: Ulangi hingga Konvergen**  
Pada iterasi berikutnya, cluster tidak berubah lagi → **Algoritma berhenti**.  

---

### **4. Langkah Kerja Algoritma K-Means**  
1. **Pilih jumlah cluster $k$** (bisa menggunakan Elbow Method).  
2. **Inisialisasi centroid** secara acak atau heuristic.  
3. **Hitung jarak** semua titik data ke centroid.  
4. **Kelompokkan titik** ke cluster terdekat.  
5. **Update centroid** dengan rata-rata titik dalam cluster.  
6. **Ulangi** langkah 3-5 hingga centroid stabil atau maksimum iterasi tercapai.  

```python
# Pseudocode K-Means
1. Initialize k centroids randomly
2. While not converged:
3.   Assign each point to the nearest centroid
4.   Update centroids as mean of points in each cluster
```

---

### **5. Visualisasi dan Analogi**  
**Visualisasi**:  
![K-Means Animation](https://miro.medium.com/v2/resize:fit:1400/1*KrcZK0xYgTa4qFrVtdfJHg.gif)  
*Proses iterasi K-Means (Sumber: Towards Data Science)*  

**Analogi**:  
Bayangkan Anda ingin mengelompokkan **100 orang** di lapangan berdasarkan posisi mereka:  
1. Pilih 3 titik acak sebagai "kapten" (centroid).  
2. Setiap orang memilih kapten terdekat.  
3. Kapten pindah ke tengah kelompoknya.  
4. Ulangi hingga tidak ada yang pindah kelompok.  

---

### **6. Kelebihan dan Kekurangan**  
✅ **Kelebihan**:  
- Cepat dan efisien untuk data besar.  
- Mudah diimplementasikan.  

❌ **Kekurangan**:  
- Sensitif terhadap inisialisasi centroid awal.  
- Harus tentukan $k$ sebelumnya.  
- Tidak bekerja baik untuk cluster non-spherical atau berbeda ukuran.  

**Solusi**:  
- Gunakan **K-Means++** untuk inisialisasi centroid lebih baik.  
- Evaluasi dengan **Silhouette Score** atau **Elbow Method**.  

---

### **7. Implementasi Kode (Python)**  
```python
from sklearn.cluster import KMeans
import numpy as np

# Data contoh
X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])

# K-Means dengan k=2
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

print("Centroid akhir:", kmeans.cluster_centers_)
print("Label cluster:", kmeans.labels_)
```

**Output**:  
```
Centroid akhir: [[ 1.   1.5] [10.  10.5]]
Label cluster: [0 0 1 1]
```

---

### **8. Contoh Aplikasi Nyata**  
- **Segmentasi Pelanggan**: Kelompokkan pelanggan berdasarkan perilaku pembelian.  
- **Kompresi Gambar**: Kurangi warna dengan mengelompokkan pixel yang mirip.  
- **Anomali Deteksi**: Identifikasi outlier sebagai cluster kecil.  


---


---
### **1. Konsep Elbow Method**
**Definisi**:  
Teknik untuk menentukan jumlah cluster ($k$) optimal dengan memplot **nilai inersia** (variasi dalam cluster) terhadap berbagai nilai $k$. Titik di mana penurunan inersia mulai melambat ("siku") dipilih sebagai $k$ optimal.

**Analog Sederhana**:  
Bayangkan Anda meregangkan karet gelang. Awalnya mudah (penurunan inersia besar), tapi setelah titik tertentu, usaha lebih besar hanya menghasilkan peregangan kecil (penurunan inersia minimal). Titik peralihan ini adalah "siku".

---

### **2. Rumus Inersia (Within-Cluster Sum of Squares/WCSS)**
$$
\text{Inersia} = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$$
- $C_i$: Cluster ke-$i$.
- $\mu_i$: Centroid cluster ke-$i$.
- $||x - \mu_i||$: Jarak Euclidean titik ke centroid.

---

### **3. Contoh Perhitungan Manual**
**Dataset Sederhana (2D)**:  
| Titik | Koordinat (X,Y) |
|-------|----------------|
| A     | (1, 1)         |
| B     | (1, 2)         |
| C     | (10, 10)       |
| D     | (10, 11)       |

**Langkah 1: Hitung Inersia untuk Berbagai $k$**  
- **Kasus $k=1$**:
  - Centroid: $\mu = (\frac{1+1+10+10}{4}, \frac{1+2+10+11}{4}) = (5.5, 6)$
  - Inersia:  
    $$
    (1-5.5)^2 + (1-6)^2 + (1-5.5)^2 + (2-6)^2 + \dots \approx 120.5
    $$

- **Kasus $k=2$**:
  - Centroid optimal: $\mu_1 = (1, 1.5)$, $\mu_2 = (10, 10.5)$
  - Inersia:  
    $$
    (1-1)^2 + (1-1.5)^2 + (1-1)^2 + (2-1.5)^2 + \dots = 1.0
    $$

- **Kasus $k=3$**:
  - Inersia turun sedikit (misal: 0.6).

**Langkah 2: Plot Nilai Inersia**  
| $k$ | Inersia |
|-----|---------|
| 1   | 120.5   |
| 2   | 1.0     |
| 3   | 0.6     |
| 4   | 0.0     |

---

### **4. Interpretasi Elbow Method**
- **$k=1$ ke $k=2$**: Penurunan inersia drastis (120.5 → 1.0).  
- **$k=2$ ke $k=3$**: Penurunan kecil (1.0 → 0.6).  
- **Kesimpulan**: $k=2$ adalah pilihan optimal karena penambahan cluster tidak memberikan manfaat signifikan.

---

### **5. Implementasi Python dengan Scikit-Learn**
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Data contoh
X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])

# Hitung inersia untuk k=1 hingga k=4
inertias = []
for k in range(1, 5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(range(1, 5), inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```


### **6. Kelebihan dan Kekurangan**
✅ **Kelebihan**:  
- Mudah diimplementasikan.  
- Visual intuitif.  

❌ **Kekurangan**:  
- Tidak selalu jelas titik "siku"-nya (misal: kurva halus).  
- Tidak cocok untuk data dengan cluster density berbeda.  

**Solusi Alternatif**:  
- **Silhouette Score**: Mengukur seberapa mirip titik dengan clusternya sendiri vs cluster lain.  
- **Gap Statistic**: Bandingkan inersia data aktual dengan data referensi acak.  

---

### **7. Contoh Aplikasi Nyata**
1. **Segmentasi Pelanggan**:  
   - Pilih $k$ optimal untuk mengelompokkan pelanggan berdasarkan pembelian.  
2. **Analisis Citra**:  
   - Kompresi warna dengan mengelompokkan pixel (pilih $k$ warna dominan).  

---

### **8. Tips Praktis**
- **Range $k$**: Coba nilai $k$ dari 1 hingga $\sqrt{n}$ ($n$ = jumlah sampel).  
- **Scaling Data**: Selalu normalisasi data (misal: StandardScaler) sebelum K-Means.  
- **Validasi Silang**: Gunakan metode lain sebagai pembanding.  

Dengan Elbow Method, Anda bisa menghindari pemilihan $k$ yang sembarang dan meningkatkan kualitas clustering! 🎯

---
