import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict
import os
from sqlalchemy import create_engine, text
from src.data_loader import DataLoader

class ProductFeatureEngineering:
    """Feature engineering for product purchase prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(sampling_strategy=0.7, k_neighbors=3)
        self.category_columns = [
            "Electronics",
            "Clothing",
            "Books",
            "Home",
            "Sports",
            "Beauty",
            "Food",
            "Toys"
        ]
        self.target_products = [
            "SmartWatch_Pro",  # Yeni akıllı saat
            "SportRunner_X",   # Yeni spor ayakkabı
            "KitchenMaster_AI" # Yeni mutfak robotu
        ]
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
        """Create target variables for new products based on customer purchase history"""
        # Her ürün için hedef değişken oluştur
        targets = pd.DataFrame(index=df.index)
        
        # SmartWatch_Pro için hedef
        # Son 6 ayda Electronics kategorisinde yüksek harcama yapanlar
        targets['SmartWatch_Pro'] = (df['Electronics'] > df['Electronics'].quantile(0.7)).astype(int)
        
        # SportRunner_X için hedef
        # Son 6 ayda Sports ve Clothing kategorilerinde yüksek harcama yapanlar
        sports_clothing_avg = (df['Sports'] + df['Clothing']) / 2
        targets['SportRunner_X'] = (sports_clothing_avg > sports_clothing_avg.quantile(0.7)).astype(int)
        
        # KitchenMaster_AI için hedef
        # Son 6 ayda Home ve Food kategorilerinde yüksek harcama yapanlar
        home_food_avg = (df['Home'] + df['Food']) / 2
        targets['KitchenMaster_AI'] = (home_food_avg > home_food_avg.quantile(0.7)).astype(int)
        
        return targets
    
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
        """Prepare features and targets for model training"""
        # Örnek veri oluştur
        np.random.seed(42)
        n_samples = 1000
        
        # Kategori bazlı harcamalar
        data = {
            category: np.random.gamma(shape=2, scale=100, size=n_samples)
            for category in self.category_columns
        }
        df = pd.DataFrame(data)
        
        # Hedef değişkenleri oluştur
        targets = self._create_target_variable(df)
        
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            df, targets, test_size=0.2, random_state=42
        )
        
        # Özellikleri ölçeklendir
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (
            pd.DataFrame(X_train_scaled, columns=self.category_columns, index=X_train.index),
            pd.DataFrame(X_test_scaled, columns=self.category_columns, index=X_test.index),
            y_train,
            y_test
        )
    
    def prepare_customer_features(self, customer_features: Dict[str, float]) -> np.ndarray:
        """Prepare customer features for prediction"""
        try:
            # Eksik kategorileri kontrol et
            missing_categories = [cat for cat in self.category_columns if cat not in customer_features]
            if missing_categories:
                print(f"Warning: Missing categories: {missing_categories}")
                print("Using 0 for missing categories")
            
            # Özellikleri doldur
            features = np.zeros(len(self.category_columns))
            for i, category in enumerate(self.category_columns):
                if category in customer_features:
                    features[i] = customer_features[category]
            
            # Tek örnek için boyut düzenleme
            features = features.reshape(1, -1)
            
            # Scaler'ın fit edilip edilmediğini kontrol et
            if not hasattr(self.scaler, 'scale_'):
                raise ValueError("Scaler not fitted. Call prepare_data() first.")
            
            # Özellikleri ölçeklendir
            features_scaled = self.scaler.transform(features)
            
            return features_scaled
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            raise 