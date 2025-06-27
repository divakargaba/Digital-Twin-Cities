import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

class SDOHDataProcessor:
    def __init__(self, csv_path):
        """
        Initialize the SDOH data processor.
        
        Args:
            csv_path (str): Path to the SDOH CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self.processed_df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, sample_size=None):
        """
        Load the SDOH data with optional sampling for development.
        
        Args:
            sample_size (int, optional): Number of rows to sample for faster processing
        """
        print("Loading SDOH data...")
        if sample_size:
            # Sample data for faster development
            self.df = pd.read_csv(self.csv_path, nrows=sample_size)
            print(f"Loaded {len(self.df)} sample rows")
        else:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.df)} total rows")
            
        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
    def explore_data(self):
        """
        Perform basic data exploration and statistics.
        """
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
            
        print("\n=== DATA EXPLORATION ===")
        print(f"Total records: {len(self.df):,}")
        print(f"Unique states: {self.df['StateAbbr'].nunique()}")
        print(f"Unique counties: {self.df['CountyName'].nunique()}")
        print(f"Unique census tracts: {self.df['LocationID'].nunique()}")
        
        print("\n=== SDOH MEASURES ===")
        measure_counts = self.df['Measure'].value_counts()
        for measure, count in measure_counts.items():
            print(f"{measure}: {count:,} records")
            
        print("\n=== DATA QUALITY ===")
        print(f"Missing values per column:")
        missing_data = self.df.isnull().sum()
        for col, missing in missing_data[missing_data > 0].items():
            print(f"  {col}: {missing:,} ({missing/len(self.df)*100:.1f}%)")
            
        print("\n=== NUMERICAL STATISTICS ===")
        numeric_cols = ['Data_Value', 'MOE', 'TotalPopulation']
        print(self.df[numeric_cols].describe())
        
    def pivot_data(self):
        """
        Pivot the data from long format to wide format for ML.
        Each census tract becomes a row with SDOH measures as columns.
        """
        print("Pivoting data to wide format...")
        
        # Pivot the measures to columns
        pivot_df = self.df.pivot_table(
            index=['LocationID', 'StateAbbr', 'CountyName', 'TotalPopulation', 'Geolocation'],
            columns='Measure',
            values='Data_Value',
            aggfunc='first'  # Take first value if duplicates
        ).reset_index()
        
        # Flatten column names
        pivot_df.columns.name = None
        
        # Rename columns to be ML-friendly
        column_mapping = {
            'Crowding among housing units': 'crowding_pct',
            'Housing cost burden among households': 'housing_cost_burden_pct',
            'No broadband internet subscription among households': 'no_broadband_pct',
            'No high school diploma among adults aged 25 years or older': 'no_hs_diploma_pct',
            'Persons aged 65 years or older': 'seniors_pct',
            'Persons living below 150% of the poverty level': 'poverty_150_pct',
            'Persons of racial or ethnic minority status': 'minority_pct',
            'Single-parent households': 'single_parent_pct',
            'Unemployment among people 16 years and older in the labor force': 'unemployment_pct'
        }
        
        pivot_df = pivot_df.rename(columns=column_mapping)
        
        # Extract coordinates from geolocation
        pivot_df['latitude'] = pivot_df['Geolocation'].str.extract(r'POINT \(([^ ]+) ([^)]+)\)')[1].astype(float)
        pivot_df['longitude'] = pivot_df['Geolocation'].str.extract(r'POINT \(([^ ]+) ([^)]+)\)')[0].astype(float)
        
        # Drop original geolocation column
        pivot_df = pivot_df.drop('Geolocation', axis=1)
        
        self.processed_df = pivot_df
        print(f"Pivoted data shape: {pivot_df.shape}")
        print(f"Features: {list(pivot_df.columns)}")
        
        return pivot_df
        
    def handle_missing_values(self, strategy='drop'):
        """
        Handle missing values in the processed data.
        
        Args:
            strategy (str): 'drop' to remove rows with missing values, 
                          'impute' to fill with median values
        """
        if self.processed_df is None:
            print("No processed data. Call pivot_data() first.")
            return
            
        print(f"Handling missing values with strategy: {strategy}")
        
        if strategy == 'drop':
            initial_rows = len(self.processed_df)
            self.processed_df = self.processed_df.dropna()
            final_rows = len(self.processed_df)
            print(f"Dropped {initial_rows - final_rows} rows with missing values")
            
        elif strategy == 'impute':
            # Impute numerical columns with median
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.processed_df[col].isnull().any():
                    median_val = self.processed_df[col].median()
                    self.processed_df[col].fillna(median_val, inplace=True)
                    print(f"Imputed {col} with median: {median_val:.2f}")
                    
    def encode_categorical_features(self):
        """
        Encode categorical features for ML models.
        """
        if self.processed_df is None:
            print("No processed data. Call pivot_data() first.")
            return
            
        print("Encoding categorical features...")
        
        categorical_cols = ['StateAbbr', 'CountyName']
        
        for col in categorical_cols:
            if col in self.processed_df.columns:
                le = LabelEncoder()
                self.processed_df[f'{col}_encoded'] = le.fit_transform(self.processed_df[col])
                self.label_encoders[col] = le
                print(f"Encoded {col} -> {col}_encoded")
                
    def create_target_variable(self, target_type='health_index'):
        """
        Create a target variable for ML prediction.
        
        Args:
            target_type (str): Type of target to create
                - 'health_index': Composite health index
                - 'vulnerability_score': Social vulnerability score
        """
        if self.processed_df is None:
            print("No processed data. Call pivot_data() first.")
            return
            
        print(f"Creating target variable: {target_type}")
        
        if target_type == 'health_index':
            # Create a composite health index (lower is better)
            # Higher values of negative factors contribute to worse health
            negative_factors = [
                'crowding_pct', 'housing_cost_burden_pct', 'no_broadband_pct',
                'no_hs_diploma_pct', 'poverty_150_pct', 'unemployment_pct'
            ]
            
            # Higher values of positive factors contribute to better health
            positive_factors = ['seniors_pct']  # More seniors might indicate better health outcomes
            
            # Calculate health index (weighted average of factors)
            health_index = 0
            
            # Add negative factors (higher = worse health)
            for factor in negative_factors:
                if factor in self.processed_df.columns:
                    health_index += self.processed_df[factor] * 0.15  # Weight each factor
                    
            # Subtract positive factors (higher = better health)
            for factor in positive_factors:
                if factor in self.processed_df.columns:
                    health_index -= self.processed_df[factor] * 0.1
                    
            self.processed_df['health_index'] = health_index
            
        elif target_type == 'vulnerability_score':
            # Create social vulnerability score
            vulnerability_factors = [
                'crowding_pct', 'housing_cost_burden_pct', 'no_broadband_pct',
                'no_hs_diploma_pct', 'poverty_150_pct', 'unemployment_pct',
                'minority_pct', 'single_parent_pct'
            ]
            
            vulnerability_score = 0
            for factor in vulnerability_factors:
                if factor in self.processed_df.columns:
                    vulnerability_score += self.processed_df[factor] * 0.125  # Equal weights
                    
            self.processed_df['vulnerability_score'] = vulnerability_score
            
        print(f"Created target variable: {target_type}")
        
    def prepare_ml_data(self, target_col='health_index', test_size=0.2, random_state=42):
        """
        Prepare data for machine learning models.
        
        Args:
            target_col (str): Name of the target column
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        if self.processed_df is None:
            print("No processed data. Call pivot_data() first.")
            return None, None, None, None, None
            
        # Select features for ML (exclude identifiers and target)
        exclude_cols = ['LocationID', 'StateAbbr', 'CountyName', 'latitude', 'longitude']
        if target_col in self.processed_df.columns:
            exclude_cols.append(target_col)
            
        feature_cols = [col for col in self.processed_df.columns if col not in exclude_cols]
        
        X = self.processed_df[feature_cols]
        y = self.processed_df[target_col] if target_col in self.processed_df.columns else None
        
        # Split the data
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"Test set: {X_test.shape[0]} samples")
            print(f"Features: {feature_cols}")
            
            return X_train, X_test, y_train, y_test, feature_cols
        else:
            print(f"Target column '{target_col}' not found in data")
            return X, None, None, None, feature_cols
            
    def get_sample_data(self, state=None, n_samples=1000):
        """
        Get a sample of the processed data for quick testing.
        
        Args:
            state (str, optional): Filter by state abbreviation
            n_samples (int): Number of samples to return
            
        Returns:
            pd.DataFrame: Sample of processed data
        """
        if self.processed_df is None:
            print("No processed data. Call pivot_data() first.")
            return None
            
        sample_df = self.processed_df.copy()
        
        if state:
            sample_df = sample_df[sample_df['StateAbbr'] == state]
            
        if len(sample_df) > n_samples:
            sample_df = sample_df.sample(n=n_samples, random_state=42)
            
        return sample_df 