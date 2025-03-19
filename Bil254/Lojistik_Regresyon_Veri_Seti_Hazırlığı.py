# Gerekli kütüphaneleri içe aktarıyoruz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Veri setini bilgisayardan yüklüyoruz
dataset_path = r"C:\Users\Lenovo\Desktop\archive\diabetes.csv"
df = pd.read_csv(dataset_path)

# Veri setinin ilk 5 satırına göz atalım
print(df.head())

# Eksik değer kontrolü (0 olan biyolojik olarak mümkün olmayan sütunlar)
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nSıfır değer içeren sütunlar:\n", (df[zero_columns] == 0).sum())

# Bu sütunlardaki 0 değerlerini medyanla dolduralım
for column in zero_columns:
    median_value = df[column].median()
    df[column] = df[column].replace(0, median_value)

# Ölçeklendirme işlemi
scaler = StandardScaler()
feature_columns = df.columns.drop('Outcome')  # Target sütunu hariç tüm özellikler
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Bağımlı ve bağımsız değişkenlerin ayrılması
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Veriyi eğitim ve test olarak bölüyoruz (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Lojistik Regresyon modelini kurup eğitiyoruz
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapıyoruz
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Model performansını ölçüyoruz
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Sonuçları yazdırıyoruz
print("\nModel Performans Metrikleri:")
print(f"Doğruluk (Accuracy): {accuracy:.4f}")
print(f"Özgüllük (Precision): {precision:.4f}")
print(f"Hassasiyet (Recall): {recall:.4f}")
print(f"F1 Skoru: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Karışıklık matrisi görselleştirmesi
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi')
plt.show()
