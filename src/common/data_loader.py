import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
from typing import Optional

# Ensure base path is set for relative imports or .env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(os.getenv('LOG_FILE', 'logs/app.log'))
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.getenv('LOG_FILE', 'logs/app.log'),
    filemode='a'
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)

class DataLoader:
    """Data loader class for fetching data from PostgreSQL database using SQLAlchemy"""
    
    def __init__(self):
        """Initialize database connection parameters"""
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")

        if not all([db_user, db_password, db_host, db_port, db_name]):
            raise EnvironmentError("One or more required DB environment variables are missing.")
        
        self.db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self.engine = create_engine(self.db_url)
    
    def fetch_customers(self) -> pd.DataFrame:
        query = text("SELECT customer_id, company_name, contact_name, country FROM customers")
        return self._fetch(query, "customer")

    def fetch_orders(self) -> pd.DataFrame:
        query = text("SELECT order_id, customer_id, order_date, required_date, shipped_date FROM orders")
        return self._fetch(query, "order")

    def fetch_order_details(self) -> pd.DataFrame:
        query = text("SELECT order_id, product_id, unit_price, quantity, discount FROM order_details")
        return self._fetch(query, "order details")

    def _fetch(self, query: text, name: str) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error fetching {name} data: {str(e)}")
            raise

if __name__ == "__main__":
    loader = DataLoader()
    try:
        print("\nCustomers:")
        print(loader.fetch_customers().head())

        print("\nOrders:")
        print(loader.fetch_orders().head())

        print("\nOrder Details:")
        print(loader.fetch_order_details().head())
    except Exception as e:
        print(f"Error: {str(e)}")
