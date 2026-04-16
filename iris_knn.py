import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─── Veri Yükleme ──────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
sinif_isimleri = iris.target_names  # ['setosa', 'versicolor', 'virginica']
 
# ─── Eğitim / Test Bölme (%80 / %20) ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# ─── Ölçekleme (KNN için gerekli, NB için opsiyonel) ──────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
 
# ═══════════════════════════════════════════════════════════════════════════════
# 1) K-EN YAKIN KOMŞU (KNN)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  K-EN YAKIN KOMŞU (KNN)")
print("=" * 60)
 
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_train_s, y_train)
y_pred_knn = knn.predict(X_test_s)
 
print(f"\nDoğruluk (Test): {accuracy_score(y_test, y_pred_knn):.4f}")
 
# En iyi k değerini bul
k_skorlari = []
for k in range(1, 21):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    skor = cross_val_score(knn_k, X_train_s, y_train, cv=5, scoring="accuracy")
    k_skorlari.append((k, skor.mean(), skor.std()))
 
en_iyi_k, en_iyi_skor, _ = max(k_skorlari, key=lambda x: x[1])
print(f"En iyi k (çapraz doğrulama): k={en_iyi_k}, CV doğruluk={en_iyi_skor:.4f}")
 
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_knn, target_names=sinif_isimleri))
 
print("Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred_knn))
 