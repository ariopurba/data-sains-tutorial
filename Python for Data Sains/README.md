# Python untuk Data Science

## Daftar Isi
1. [Pengenalan Python untuk Data Science](#1-pengenalan-python-untuk-data-science)
2. [Struktur Data Python](#2-struktur-data-python)
3. [Library Data Science](#3-library-data-science)
4. [Data Wrangling](#4-data-wrangling)
5. [Analisis Data Eksploratif](#5-analisis-data-eksploratif)
6. [Machine Learning Dasar](#6-machine-learning-dasar)
7. [Proyek Praktis](#7-proyek-praktis)

## 1. Pengenalan Python untuk Data Science

### 1.1 Mengapa Python?
- Sintaks yang mudah dibaca dan dipelajari
- Ekosistem data science yang kaya (NumPy, Pandas, Scikit-learn)
- Komunitas yang besar dan dukungan yang ekstensif
- Gratis dan open source

### 1.2 Setup Lingkungan Kerja
```python
# Cek versi Python dan library utama
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
```

### 1.3 Dasar Python
```python
# Variabel dan tipe data
nama = "Data Scientist"    # string
usia = 25                  # integer
nilai = 98.5              # float
is_active = True          # boolean

# List dan operasinya
skills = ['Python', 'SQL', 'Statistics']
skills.append('Machine Learning')
print(skills)

# Dictionary
profile = {
    'nama': 'John Doe',
    'skill': skills,
    'experience': 2
}
```

## 2. Struktur Data Python

### 2.1 List dan Tuple
```python
# List - mutable
numbers = [1, 2, 3, 4, 5]
numbers[0] = 10  # Bisa diubah

# Tuple - immutable
coordinates = (10.5, 20.7)  # Tidak bisa diubah
```

### 2.2 Dictionary dan Set
```python
# Dictionary
student = {
    'name': 'Alice',
    'scores': [85, 90, 88],
    'active': True
}

# Set - unique values
unique_numbers = {1, 2, 2, 3, 3, 4}  # Hasil: {1, 2, 3, 4}
```

## 3. Library Data Science

### 3.1 NumPy Basics
```python
import numpy as np

# Array creation
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Operations
print(arr * 2)  # Element-wise multiplication
print(matrix.sum())  # Sum all elements
print(matrix.shape)  # Get dimensions
```

### 3.2 Pandas Fundamentals
```python
import pandas as pd

# Create DataFrame
data = {
    'Name': ['John', 'Anna', 'Peter'],
    'Age': [28, 22, 35],
    'City': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)

# Basic operations
print(df.head())
print(df.describe())
print(df['Age'].mean())
```

### 3.3 Visualization with Matplotlib
```python
import matplotlib.pyplot as plt

# Simple line plot
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Simple Line Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
```

## 4. Data Wrangling

### 4.1 Handling Missing Data
```python
# Missing data operations
df['Age'].fillna(df['Age'].mean(), inplace=True)
df.dropna(subset=['Name'], inplace=True)
```

### 4.2 Data Cleaning
```python
# Remove duplicates
df.drop_duplicates(inplace=True)

# Data type conversion
df['Age'] = df['Age'].astype(int)
```

[...existing content...]

## 5. Analisis Data Eksploratif (EDA)
### Statistik Deskriptif
```python
import pandas as pd
import seaborn as sns

# Load dataset sampel
df = pd.read_csv('sample_data.csv')

# Ringkasan statistik
print(df.describe())
print(df.info())

# Cek missing values
print(df.isnull().sum())
```

### Visualisasi untuk EDA
```python
# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='nilai', bins=30)
plt.title('Distribusi Nilai')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

**Latihan EDA:**
1. Analisis dataset iris atau titanic
2. Buat minimal 3 jenis visualisasi berbeda
3. Interpretasi hasil analisis

## 6. Machine Learning dengan Scikit-learn
### Persiapan Data
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Model Dasar
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluasi
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

**Latihan Machine Learning:**
1. Implementasi model regresi linear
2. Buat model klasifikasi sederhana
3. Evaluasi dan improve model

## 7. Proyek Praktis
### Analisis Dataset COVID-19
```python
# Import dan persiapan data
covid_df = pd.read_csv('covid_data.csv')

# Cleaning
covid_df['date'] = pd.to_datetime(covid_df['date'])
covid_df = covid_df.fillna(0)

# Analisis trend
plt.figure(figsize=(12, 6))
plt.plot(covid_df['date'], covid_df['cases'])
plt.title('Trend Kasus COVID-19')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Kasus')
plt.xticks(rotation=45)
plt.show()
```

### Tips Pengerjaan Proyek
1. Mulai dengan pertanyaan yang jelas
2. Dokumentasikan setiap langkah analisis
3. Visualisasikan temuan penting
4. Buat kesimpulan yang actionable

## Referensi dan Sumber Belajar
- Python Documentation: [python.org](https://python.org)
- Pandas Documentation: [pandas.pydata.org](https://pandas.pydata.org)
- Scikit-learn Tutorial: [scikit-learn.org](https://scikit-learn.org)
- Kaggle Datasets: [kaggle.com/datasets](https://kaggle.com/datasets)

## Kesimpulan
- Praktik adalah kunci dalam data science
- Mulai dengan dataset sederhana
- Tingkatkan kompleksitas secara bertahap
- Bergabung dengan komunitas data science