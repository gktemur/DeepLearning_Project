import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .env dosyasını yükleyin
load_dotenv()

# .env dosyasından PostgreSQL bağlantı bilgilerini al
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

# PostgreSQL bağlantı dizesi oluştur
DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# SQLAlchemy Engine oluştur
engine = create_engine(DATABASE_URL)

# Bağlantıyı test etmek için fonksiyon
def test_connection():
    try:
        with engine.connect() as connection:
            print("✅ Veritabanına başarıyla bağlanıldı!")
            return True
    except Exception as e:
        print(f"❌ Veritabanına bağlanırken hata oluştu: {e}")
        return False

# DataLoader sınıfı
class DataLoader:
    def __init__(self):
        self.engine = engine

    def fetch_customers(self):
        try:
            query = "SELECT * FROM customers;"
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            print(f"[customers] veri alırken hata: {e}")
            return pd.DataFrame()

    def fetch_orders(self):
        try:
            query = "SELECT * FROM orders;"
            df = pd.read_sql_query(query, self.engine)
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            return df
        except Exception as e:
            print(f"[orders] veri alırken hata: {e}")
            return pd.DataFrame()

    def fetch_order_details(self):
        try:
            query = "SELECT * FROM order_details;"
            df = pd.read_sql_query(query, self.engine)
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
            return df
        except Exception as e:
            print(f"[order_details] veri alırken hata: {e}")
            return pd.DataFrame()
