import os
import pandas as pd
import psycopg2
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
    """Data loader class for fetching data from the database"""
    
    def __init__(self):
        """Initialize database connection parameters"""
        self.db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'GYK2Northwind'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '12345')
        }
    
    def _get_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Create database connection"""
        try:
            conn = psycopg2.connect(**self.db_params)
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            return None
    
    def fetch_customers(self) -> pd.DataFrame:
        """Fetch customer data from database"""
        try:
            conn = self._get_connection()
            if conn is None:
                raise Exception("Could not establish database connection")
            
            query = """
                SELECT 
                    customer_id,
                    company_name,
                    contact_name,
                    contact_title,
                    address,
                    city,
                    region,
                    postal_code,
                    country,
                    phone,
                    fax
                FROM customers
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Successfully fetched {len(df)} customer records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching customer data: {str(e)}")
            raise
    
    def fetch_orders(self) -> pd.DataFrame:
        """Fetch order data from database"""
        try:
            conn = self._get_connection()
            if conn is None:
                raise Exception("Could not establish database connection")
            
            query = """
                SELECT 
                    order_id,
                    customer_id,
                    employee_id,
                    order_date,
                    required_date,
                    shipped_date,
                    ship_via,
                    freight,
                    ship_name,
                    ship_address,
                    ship_city,
                    ship_region,
                    ship_postal_code,
                    ship_country
                FROM orders
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Successfully fetched {len(df)} order records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching order data: {str(e)}")
            raise
    
    def fetch_order_details(self) -> pd.DataFrame:
        """Fetch order details data from database"""
        try:
            conn = self._get_connection()
            if conn is None:
                raise Exception("Could not establish database connection")
            
            query = """
                SELECT 
                    order_id,
                    product_id,
                    unit_price,
                    quantity,
                    discount
                FROM order_details
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Successfully fetched {len(df)} order detail records")
            return df
            
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
