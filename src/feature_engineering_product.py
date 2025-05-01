import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple
import os
from sqlalchemy import create_engine, text
from src.data_loader import DataLoader

class ProductFeatureEngineering:
    """Feature engineering for product purchase prediction"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.smote = SMOTE(sampling_strategy=0.7, k_neighbors=3)
        self.category_columns = None
        self.data_loader = DataLoader()
        
    def _load_data(self) -> pd.DataFrame:
        """Load and merge required tables using DataLoader"""
        # Fetch data from database
        orders = self.data_loader.fetch_orders()
        order_details = self.data_loader.fetch_order_details()
        
        # SQL query for products and categories
        query = """
        SELECT 
            p.product_id,
            p.product_name,
            c.category_id,
            c.category_name
        FROM products p
        JOIN categories c ON p.category_id = c.category_id
        """
        
        # Execute query for products and categories
        with self.data_loader.engine.connect() as conn:
            products_categories = pd.read_sql(text(query), conn)
        
        # Merge all tables
        df = order_details.merge(orders, on='order_id')
        df = df.merge(products_categories, on='product_id')
        
        return df
    
    def _calculate_category_spending(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate total spending per customer per category"""
        # Calculate total spending per order detail
        df['TotalSpent'] = df['quantity'] * df['unit_price'] * (1 - df['discount'])
        
        # Group by customer and category
        category_spending = df.groupby(['customer_id', 'category_name'])['TotalSpent'].sum().unstack()
        
        # Fill NaN with 0
        category_spending = category_spending.fillna(0)
        
        return category_spending
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for multiple products"""
        # Get last order date for each customer
        last_orders = df.groupby('customer_id')['order_date'].max()
        
        # Calculate days since last order
        last_orders = pd.to_datetime(last_orders)
        days_since_last = (pd.Timestamp.now() - last_orders).dt.days
        
        # Create target matrix for 3 products
        target = pd.DataFrame(index=last_orders.index)
        
        # Product 1: Recent purchasers (last 90 days)
        # Ensure at least 30% of customers are marked as 1
        recent_threshold = days_since_last.quantile(0.3)
        target['product_1'] = (days_since_last <= recent_threshold).astype(int)
        
        # Product 2: High-value customers (top 30% spenders)
        customer_spending = df.groupby('customer_id')['TotalSpent'].sum()
        # Ensure at least 30% of customers are marked as 1
        spending_threshold = customer_spending.quantile(0.7)
        target['product_2'] = (customer_spending > spending_threshold).astype(int)
        
        # Product 3: Frequent buyers (top 30% order count)
        customer_orders = df.groupby('customer_id')['order_id'].nunique()
        # Ensure at least 30% of customers are marked as 1
        order_threshold = customer_orders.quantile(0.7)
        target['product_3'] = (customer_orders > order_threshold).astype(int)
        
        # Print class distribution for each product
        print("\nClass distribution for each product:")
        for product in target.columns:
            class_counts = target[product].value_counts()
            print(f"\n{product}:")
            print(f"Class 0: {class_counts.get(0, 0)} samples")
            print(f"Class 1: {class_counts.get(1, 0)} samples")
            print(f"Ratio: {class_counts.get(1, 0) / len(target):.2%}")
            
            # Ensure each class has at least 2 samples
            if class_counts.get(0, 0) < 2 or class_counts.get(1, 0) < 2:
                print(f"Warning: {product} has too few samples in one class. Adjusting threshold...")
                if class_counts.get(0, 0) < 2:
                    # If class 0 has too few samples, increase the threshold
                    if product == 'product_1':
                        recent_threshold = days_since_last.quantile(0.4)
                        target['product_1'] = (days_since_last <= recent_threshold).astype(int)
                    elif product == 'product_2':
                        spending_threshold = customer_spending.quantile(0.6)
                        target['product_2'] = (customer_spending > spending_threshold).astype(int)
                    else:
                        order_threshold = customer_orders.quantile(0.6)
                        target['product_3'] = (customer_orders > order_threshold).astype(int)
                else:
                    # If class 1 has too few samples, decrease the threshold
                    if product == 'product_1':
                        recent_threshold = days_since_last.quantile(0.2)
                        target['product_1'] = (days_since_last <= recent_threshold).astype(int)
                    elif product == 'product_2':
                        spending_threshold = customer_spending.quantile(0.8)
                        target['product_2'] = (customer_spending > spending_threshold).astype(int)
                    else:
                        order_threshold = customer_orders.quantile(0.8)
                        target['product_3'] = (customer_orders > order_threshold).astype(int)
                
                # Print updated distribution
                class_counts = target[product].value_counts()
                print(f"Updated distribution for {product}:")
                print(f"Class 0: {class_counts.get(0, 0)} samples")
                print(f"Class 1: {class_counts.get(1, 0)} samples")
                print(f"Ratio: {class_counts.get(1, 0) / len(target):.2%}")
        
        return target
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using RobustScaler"""
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle class imbalance using SMOTE for each product separately"""
        X_resampled_list = []
        y_resampled_dict = {}

        # Determine the maximum sample size after all resamplings
        max_len = 0
        for product in y.columns:
            _, y_resampled = self.smote.fit_resample(X, y[product])
            max_len = max(max_len, len(y_resampled))

        for product in y.columns:
            y_product = y[product]
            X_resampled, y_resampled = self.smote.fit_resample(X, y_product)

            # If necessary, oversample again to reach max_len
            if len(X_resampled) < max_len:
                additional_idx = np.random.choice(len(X_resampled), max_len - len(X_resampled), replace=True)
                X_resampled = pd.concat([X_resampled, X_resampled.iloc[additional_idx]]).reset_index(drop=True)
                y_resampled = pd.concat([pd.Series(y_resampled), pd.Series(np.array(y_resampled)[additional_idx])]).reset_index(drop=True)
            else:
                X_resampled = X_resampled.reset_index(drop=True)
                y_resampled = pd.Series(y_resampled).reset_index(drop=True)

            X_resampled_list.append(X_resampled)
            y_resampled_dict[product] = y_resampled

        # Use one consistent X for all targets (e.g., from the first resampling)
        X_final = X_resampled_list[0]
        y_final = pd.DataFrame(y_resampled_dict)

        return X_final, y_final
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare data for model training"""
        # Load and process data
        df = self._load_data()
        category_spending = self._calculate_category_spending(df)
        target = self._create_target_variable(df)
        
        # Store category columns for later use
        self.category_columns = category_spending.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            category_spending,
            target,
            test_size=0.2,
            random_state=42,
            stratify=target['product_1']  # Stratify by first product
        )
        
        # Handle class imbalance
        X_train_resampled, y_train_resampled = self._handle_class_imbalance(X_train, y_train)
        
        # Scale features
        X_train_scaled = self._scale_features(X_train_resampled)
        X_test_scaled = self._scale_features(X_test)
        
        # Print data quality information
        print("\nData Quality Information:")
        print(f"Total customers: {len(category_spending)}")
        print(f"Number of categories: {len(self.category_columns)}")
        print(f"Training set size: {len(X_train_scaled)}")
        print(f"Test set size: {len(X_test_scaled)}")
        print("\nClass distribution in training set:")
        for product in y_train_resampled.columns:
            print(f"{product}: {y_train_resampled[product].value_counts().to_dict()}")
        print("\nClass distribution in test set:")
        for product in y_test.columns:
            print(f"{product}: {y_test[product].value_counts().to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train_resampled, y_test 