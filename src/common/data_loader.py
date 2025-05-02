import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
from typing import Optional

# Load environment variables
load_dotenv()

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(os.getenv('LOG_FILE', 'logs/app.log'))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.getenv('LOG_FILE', 'logs/app.log'),
    filemode='a'  # append mode
)

# Add console handler for immediate feedback
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
        self.db_url = f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'postgres')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'northwind')}"
        self.engine = create_engine(self.db_url)
    
    def fetch_customers(self) -> pd.DataFrame:
        """Fetch customer data"""
        query = text("""
        SELECT 
            customer_id,
            company_name,
            contact_name,
            country
        FROM customers
        """)
        
        try:
            with self.engine.connect() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error fetching customer data: {str(e)}")
            raise
    
    def fetch_orders(self) -> pd.DataFrame:
        """Fetch order data"""
        query = text("""
        SELECT 
            order_id,
            customer_id,
            order_date,
            required_date,
            shipped_date
        FROM orders
        """)
        
        try:
            with self.engine.connect() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error fetching order data: {str(e)}")
            raise
    
    def fetch_order_details(self) -> pd.DataFrame:
        """Fetch order details data"""
        query = text("""
        SELECT 
            order_id,
            product_id,
            unit_price,
            quantity,
            discount
        FROM order_details
        """)
        
        try:
            with self.engine.connect() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error fetching order details data: {str(e)}")
            raise

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    try:
        customers = loader.fetch_customers()
        orders = loader.fetch_orders()
        order_details = loader.fetch_order_details()
        
        print("\nSample data from each table:")
        print("\nCustomers:")
        print(customers.head())
        print("\nOrders:")
        print(orders.head())
        print("\nOrder Details:")
        print(order_details.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")
