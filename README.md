# Müşteri Kaybı (Churn) Tahmin Modeli

Bu proje, Northwind veritabanındaki müşteri verilerini kullanarak, müşterilerin kayıp (churn) durumunu tahmin eden bir derin öğrenme modeli geliştirmeyi amaçlamaktadır.

## Proje Yapısı

```
.
├── src/
│   ├── data_loader.py      # Veritabanından veri çekme işlemleri
│   ├── feature_engineering.py  # Özellik mühendisliği işlemleri
│   └── model.py           # Derin öğrenme modeli
├── requirements.txt       # Proje bağımlılıkları
└── README.md             # Proje dokümantasyonu
```

## Özellikler

- RFM (Recency, Frequency, Monetary) analizi
- Temporal özellikler (mevsimsellik analizi)
- Sınıf dengesizliği yönetimi (SMOTE)
- Özellik ölçeklendirme
- Derin öğrenme modeli
- Model performans değerlendirme ve görselleştirme

## Kurulum

1. Gerekli bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

2. Veritabanı bağlantısını yapılandırın (data_loader.py içinde)

## Kullanım

1. Veri hazırlama ve özellik mühendisliği:

```python
from src.feature_engineering import FeatureEngineering, FeatureConfig

feature_engineering = FeatureEngineering(FeatureConfig())
X, y = feature_engineering.prepare_data()
```

2. Model eğitimi ve değerlendirme:

```python
from src.model import ChurnPredictor

model = ChurnPredictor(input_dim=X.shape[1])
history = model.train(X_train, y_train, X_val, y_val)
metrics = model.evaluate(X_test, y_test)
```

## Model Mimarisi

- Giriş katmanı: 64 nöron, ReLU aktivasyon
- Gizli katman: 32 nöron, ReLU aktivasyon
- Çıkış katmanı: 1 nöron, Sigmoid aktivasyon
- Batch Normalization ve Dropout katmanları
- Early Stopping ile aşırı öğrenme kontrolü

## Değerlendirme Metrikleri

- Doğruluk (Accuracy)
- AUC-ROC
- Hassasiyet (Precision)
- Duyarlılık (Recall)
- Karmaşıklık Matrisi (Confusion Matrix)

## Geliştirme

1. Feature Engineering:

   - Yeni özellikler ekleme
   - Özellik seçimi
   - Özellik ölçeklendirme

2. Model Geliştirme:

   - Hiperparametre optimizasyonu
   - Model mimarisi değişiklikleri
   - Ensemble yöntemleri

3. Değerlendirme:
   - Cross-validation
   - Model karşılaştırmaları
   - A/B testleri

Katkıda Bulunanlar

- Gamze Kevser Temür
- İlayda Akyüz
- Rabia Gülizar Tuncer
- Buse Erarslan
