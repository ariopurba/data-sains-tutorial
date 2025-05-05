## Model Evaluasi
---

### **1. Konsep Dasar Evaluasi Model**
Evaluasi model bertujuan untuk mengukur seberapa baik model memprediksi target. Metrik yang digunakan tergantung pada jenis masalah:
- **Klasifikasi**: Akurasi, Presisi, Recall, F1-Score, ROC-AUC.
- **Regresi**: MSE, RMSE, MAE, R-squared.

---

### **2. Contoh Data Sederhana (Klasifikasi)**
Misalkan kita memiliki model yang memprediksi **apakah pasien terkena diabetes (1) atau tidak (0)**. Berikut hasil prediksi vs aktual dari 10 sampel:

| Pasien | Aktual (y) | Prediksi (ŷ) |
|--------|------------|--------------|
| 1      | 0          | 0            |
| 2      | 0          | 1            |
| 3      | 1          | 1            |
| 4      | 0          | 0            |
| 5      | 1          | 0            |
| 6      | 1          | 1            |
| 7      | 0          | 0            |
| 8      | 1          | 1            |
| 9      | 0          | 1            |
| 10     | 1          | 0            |

---

### **3. Confusion Matrix**
Langkah pertama adalah membuat **confusion matrix**:

|                | Prediksi 0 | Prediksi 1 |
|----------------|------------|------------|
| **Aktual 0**   | TN = 3     | FP = 2     |
| **Aktual 1**   | FN = 2     | TP = 3     |

- **True Positive (TP)**: Prediksi 1 dan benar (contoh: Pasien 3,6,8).
- **False Positive (FP)**: Prediksi 1 tetapi salah (Pasien 2,9).
- **True Negative (TN)**: Prediksi 0 dan benar (Pasien 1,4,7).
- **False Negative (FN)**: Prediksi 0 tetapi salah (Pasien 5,10).

---

### **4. Hitung Metrik Klasifikasi**
#### **a. Accuracy (Akurasi)**
Persentase prediksi yang benar secara keseluruhan.  
$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{3 + 3}{10} = 0.6 \ (60\%)
$

#### **b. Precision (Presisi)**
Proporsi prediksi positif yang benar.  
$
\text{Precision} = \frac{TP}{TP + FP} = \frac{3}{3 + 2} = 0.6 \ (60\%)
$

#### **c. Recall (Sensitivity/True Positive Rate)**
Proporsi kasus positif yang terdeteksi.  
$
\text{Recall} = \frac{TP}{TP + FN} = \frac{3}{3 + 2} = 0.6 \ (60\%)
$

#### **d. F1-Score**
Rata-rata harmonik Presisi dan Recall.  
$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.6 \times 0.6}{1.2} = 0.6 \ (60\%)
$

#### **e. Specificity (True Negative Rate)**
Proporsi kasus negatif yang terdeteksi.  
$
\text{Specificity} = \frac{TN}{TN + FP} = \frac{3}{3 + 2} = 0.6 \ (60\%)
$

---

### **5. Contoh Perhitungan untuk Regresi**
Misalkan kita punya data prediksi harga rumah:

| Aktual (y) | Prediksi (ŷ) |
|------------|--------------|
| 100        | 90           |
| 200        | 210          |
| 150        | 140          |
| 300        | 320          |

#### **a. Mean Absolute Error (MAE)**
$
\text{MAE} = \frac{1}{n} \sum |y_i - ŷ_i| = \frac{|100-90| + |200-210| + |150-140| + |300-320|}{4} = \frac{50}{4} = 12.5
$

#### **b. Mean Squared Error (MSE)**
$
\text{MSE} = \frac{1}{n} \sum (y_i - ŷ_i)^2 = \frac{10^2 + 10^2 + 10^2 + 20^2}{4} = \frac{700}{4} = 175
$

#### **c. Root Mean Squared Error (RMSE)**
$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{175} \approx 13.23
$

#### **d. R-squared (R²)**
$
R^2 = 1 - \frac{\sum (y_i - ŷ_i)^2}{\sum (y_i - \bar{y})^2}, \quad \bar{y} = \frac{100+200+150+300}{4} = 187.5
$
$
R^2 = 1 - \frac{700}{(100-187.5)^2 + (200-187.5)^2 + (150-187.5)^2 + (300-187.5)^2} = 1 - \frac{700}{21875} \approx 0.968
$

---

### **6. Interpretasi Metrik**
- **Akurasi 60%**: Model benar 6 dari 10 kasus.  
- **Presisi 60%**: Dari 5 prediksi positif, 3 benar.  
- **Recall 60%**: Dari 5 kasus aktual positif, 3 terdeteksi.  
- **MAE 12.5**: Prediksi meleset rata-rata ±12.5 unit.  
- **R² 0.968**: 96.8% variasi data dijelaskan oleh model (sangat baik).  

---

### **7. Kode Python untuk Evaluasi**
#### **Klasifikasi**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_actual = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]

print("Accuracy:", accuracy_score(y_actual, y_pred))
print("Precision:", precision_score(y_actual, y_pred))
print("Recall:", recall_score(y_actual, y_pred))
print("F1-Score:", f1_score(y_actual, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_actual, y_pred))
```

#### **Regresi**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_actual = [100, 200, 150, 300]
y_pred = [90, 210, 140, 320]

print("MAE:", mean_absolute_error(y_actual, y_pred))
print("MSE:", mean_squared_error(y_actual, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_actual, y_pred)))
print("R²:", r2_score(y_actual, y_pred))
```

---

### **8. Visualisasi Metrik**
#### **Confusion Matrix**
```python
import seaborn as sns
sns.heatmap(confusion_matrix(y_actual, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

#### **ROC Curve (Klasifikasi)**
```python
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_actual, y_pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

---

### **9. Kesimpulan**
- **Untuk Klasifikasi**: Fokus pada Precision jika ingin minimalkan FP (contoh: spam detection), Recall jika ingin minimalkan FN (contoh: diagnosis penyakit).  
- **Untuk Regresi**: Gunakan RMSE untuk memberi penekanan pada error besar, MAE untuk interpretasi langsung.  
- **Pilih Metrik Sesuai Tujuan Bisnis**: Tidak ada metrik terbaik universal!  


---

## Train-test split, Cross Validation

### **1. Konsep Dasar**

#### **Train-Test Split**

-   **Tujuan**: Membagi dataset menjadi dua bagian:
    
    -   **Training set**: Untuk melatih model.
        
    -   **Test set**: Untuk menguji performa model pada data yang belum pernah dilihat.
        
-   **Rasio Umum**: 70-30% atau 80-20%.
    

#### **Cross-Validation (Validasi Silang)**

bantu ubah markdown ini dengan menambah '\$' di awal dan akhir pada bagian rumus:"

### **1. Contoh Data Klasifikasi Sederhana**

Kita gunakan dataset **"Apakah Bermain Tenis?"** dengan 14 sampel:

| Outlook  | Temperature | Humidity | Windy | Play? |
| -------- | ----------- | -------- | ----- | ----- |
| Sunny    | Hot         | High     | No    | No    |
| Sunny    | Hot         | High     | Yes   | No    |
| Overcast | Hot         | High     | No    | Yes   |
| Rainy    | Mild        | High     | No    | Yes   |
| Rainy    | Cool        | Normal   | No    | Yes   |
| Rainy    | Cool        | Normal   | Yes   | No    |
| Overcast | Cool        | Normal   | Yes   | Yes   |
| Sunny    | Mild        | High     | No    | No    |
| Sunny    | Cool        | Normal   | No    | Yes   |
| Rainy    | Mild        | Normal   | No    | Yes   |
| Sunny    | Mild        | Normal   | Yes   | Yes   |
| Overcast | Mild        | High     | Yes   | Yes   |
| Overcast | Hot         | Normal   | No    | Yes   |
| Rainy    | Mild        | High     | Yes   | No    |

**Target**: Kolom "Play?" (Yes/No).

---

### **2. Train-Test Split untuk Klasifikasi**

#### **Langkah 1: Bagi Data (70% Train, 30% Test)**

* **Training Set (10 sampel)**:

  ```python
  X_train = [
      ['Sunny', 'Hot', 'High', 'No'],
      ['Sunny', 'Hot', 'High', 'Yes'],
      ['Overcast', 'Hot', 'High', 'No'],
      ['Rainy', 'Mild', 'High', 'No'],
      ['Rainy', 'Cool', 'Normal', 'No'],
      ['Rainy', 'Cool', 'Normal', 'Yes'],
      ['Overcast', 'Cool', 'Normal', 'Yes'],
      ['Sunny', 'Mild', 'High', 'No'],
      ['Sunny', 'Cool', 'Normal', 'No'],
      ['Rainy', 'Mild', 'Normal', 'No']
  ]
  y_train = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
  ```

* **Test Set (4 sampel)**:

  ```python
  X_test = [
      ['Sunny', 'Mild', 'Normal', 'Yes'],
      ['Overcast', 'Mild', 'High', 'Yes'],
      ['Overcast', 'Hot', 'Normal', 'No'],
      ['Rainy', 'Mild', 'High', 'Yes']
  ]
  y_test = ['Yes', 'Yes', 'Yes', 'No']
  ```

#### **Langkah 2: Latih Model (Contoh: Decision Tree)**

Kita akan menggunakan **entropy** sebagai kriteria split.
**Perhitungan manual untuk root node**:

1. Hitung entropy target di training set:

   * 6 "Yes", 4 "No" → $Entropy(S) = -\frac{6}{10}\log_2(\frac{6}{10}) - \frac{4}{10}\log_2(\frac{4}{10}) = 0.971$.

2. Hitung information gain untuk fitur "Outlook":

   * **Sunny**: 3 "No", 2 "Yes" → Entropy = 0.971.
   * **Overcast**: 3 "Yes" → Entropy = 0.
   * **Rainy**: 4 "Yes", 1 "No" → Entropy = 0.722.
   * $IG = 0.971 - (\frac{5}{10} \times 0.971 + \frac{3}{10} \times 0 + \frac{4}{10} \times 0.722) = 0.246$.

Fitur dengan IG tertinggi dipilih sebagai root node.

#### **Langkah 3: Evaluasi pada Test Set**

Prediksi untuk test set:

* Model memprediksi: \['Yes', 'Yes', 'Yes', 'No']
* Aktual: \['Yes', 'Yes', 'Yes', 'No']

**Confusion Matrix**:

|                | Prediksi No | Prediksi Yes |
| -------------- | ----------- | ------------ |
| **Aktual No**  | 1 (TN)      | 0 (FP)       |
| **Aktual Yes** | 0 (FN)      | 3 (TP)       |

**Akurasi**: $\frac{TP + TN}{Total} = \frac{3 + 1}{4} = 100\%$ *(Contoh ini sederhana, akurasi bisa lebih rendah di kasus nyata)*.

---

### **3. Cross-Validation (Stratified k-Fold)**

#### **Langkah 1: Bagi Data menjadi 5 Fold**

Karena klasifikasi, gunakan **stratified sampling** untuk menjaga distribusi kelas di setiap fold:

| Fold | Data Indeks (Play? = No/Yes)            |
| ---- | --------------------------------------- |
| 1    | \[0, 1] (No, No) vs \[2, 3] (Yes, Yes)  |
| 2    | \[5, 13] (No, No) vs \[4, 6] (Yes, Yes) |
| 3    | \[7] (No) vs \[8, 9] (Yes, Yes)         |
| 4    | - vs \[10, 11] (Yes, Yes)               |
| 5    | - vs \[12] (Yes)                        |

#### **Langkah 2: Latih dan Evaluasi per Fold**

**Contoh Iterasi 1**:

* **Train**: Fold 2-5 (12 sampel).
* **Test**: Fold 1 (2 "No", 2 "Yes").
* Hitung akurasi (misal: 75%).

**Iterasi 2**:

* **Train**: Fold 1,3-5.
* **Test**: Fold 2.
* Hitung akurasi.

**Akurasi Rata-Rata**:

$$
\text{Akurasi CV} = \frac{75\% + 80\% + \dots}{5}
$$

---

### **4. Implementasi Python**

#### **Train-Test Split**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Data
X = df.drop('Play?', axis=1).values
y = df['Play?'].values

# Split (stratify untuk jaga distribusi kelas)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Model
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

#### **Stratified k-Fold Cross-Validation**

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 5-Fold CV dengan stratifikasi
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    DecisionTreeClassifier(criterion='entropy'),
    X, y, cv=skf, scoring='accuracy'
)

print("Akurasi per Fold:", scores)
print("Akurasi Rata-rata:", scores.mean())
```

---

### **5. Hasil dan Interpretasi**

* **Train-Test Split**:

  * Akurasi 100% (karena dataset kecil dan sederhana).
  * Confusion Matrix menunjukkan tidak ada error.

* **Cross-Validation**:

  * Memberikan estimasi performa yang lebih stabil.
  * Contoh output:

    ```
    Akurasi per Fold: [0.8, 0.8, 0.6, 1.0, 0.8]
    Akurasi Rata-rata: 0.8
    ```

---

### **6. Visualisasi Cross-Validation**

![Stratified k-Fold](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_003.png)
*Setiap fold menjaga proporsi kelas (merah: "No", biru: "Yes")*.

---

### **7. Mengapa Stratifikasi Penting?**

* Jika kelas tidak seimbang (misal: 90% "Yes", 10% "No"), pembagian acak bisa membuat fold tanpa kelas minoritas.
* Stratifikasi memastikan setiap fold mewakili distribusi asli.

---

### **8. Perbandingan Metode**

| Metode               | Kelebihan                       | Kekurangan                     |
| -------------------- | ------------------------------- | ------------------------------ |
| **Train-Test Split** | Cepat, baik untuk dataset besar | Varians tinggi jika data kecil |
| **Cross-Validation** | Estimasi lebih stabil           | Komputasi lebih berat          |

---

### **9. Tips Praktis**

1. Untuk dataset kecil (<1000 sampel), gunakan Cross-Validation.
2. Untuk dataset besar, Train-Test Split sudah cukup.
3. Selalu gunakan **stratify=y** dalam `train_test_split` untuk klasifikasi.
4. Pilih **k=5 atau k=10** untuk k-Fold.
