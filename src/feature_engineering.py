# feature_engineering.py dosyanızda

from data_loader import DataLoader
import pandas as pd
import numpy as np

# DataLoader sınıfını başlatın
data_loader = DataLoader()

# Veritabanından verileri çekelim
customers_df = data_loader.fetch_customers()
orders_df = data_loader.fetch_orders()
order_details_df = data_loader.fetch_order_details()

# Feature Engineering işlemi
def feature_engineering(customers_df, orders_df, order_details_df):
    # 1. Siparişlerin son tarihine göre 6 ay içinde tekrar sipariş verip vermediğini belirlemek
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    orders_df['last_order_date'] = orders_df.groupby('customer_id')['order_date'].transform('max')
    
    # 2. Sipariş sayısı
    order_count = orders_df.groupby('customer_id').size().reset_index(name='order_count')
    
    # 3. Toplam harcama (siparişlerin toplam tutarı)
    order_details_df['total_order_value'] = order_details_df['quantity'] * order_details_df['unit_price']
    total_spending = order_details_df.groupby('order_id')['total_order_value'].sum().reset_index()
    
    # Sipariş ve müşteri bilgilerini birleştirelim
    order_summary = orders_df.merge(order_count, on='customer_id', how='left')
    order_summary = order_summary.merge(total_spending, on='order_id', how='left')
    
    # 4. Ortalama sipariş büyüklüğü
    order_summary['average_order_value'] = order_summary['total_order_value'] / order_summary['order_count']
    
    # 5. Son 6 ayda tekrar sipariş verip vermediği bilgisini oluşturma
    today = pd.to_datetime("1998-05-06")
    order_summary['months_since_last_order'] = (today - order_summary['last_order_date']).dt.days // 30
    order_summary['Churn'] = order_summary['months_since_last_order'].apply(lambda x: 1 if x >= 6 else 0)
    
    # 6. Müşteri bilgilerini ekleyelim (örneğin yaş, cinsiyet, vb.)
    customers_df = customers_df[['customer_id', 'company_name', 'contact_name', 'country']]
    full_data = order_summary.merge(customers_df, on='customer_id', how='left')
    
    # 7. Feature'lar: Toplam harcama, Sipariş sayısı, Ortalama sipariş büyüklüğü, vb.
    features = full_data[['customer_id', 'total_order_value', 'order_count', 'average_order_value', 'Churn']]
    
    # X (özellikler) ve y (hedef değişken)
    X = features.drop(columns=['Churn'])
    y = features['Churn']
    
    return X, y

# Feature engineering fonksiyonunu çalıştırıp veriyi işleyelim
X, y = feature_engineering(customers_df, orders_df, order_details_df)

# İşlenen verinin ilk birkaç satırını inceleyelim
print(X.head())
print(y[10:20])
