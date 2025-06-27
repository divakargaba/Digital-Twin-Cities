import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_processor import SDOHDataProcessor
from models.ann_predictor import ANNPredictor
from models.random_forest_predictor import RandomForestPredictor
from models.xgboost_predictor import XGBoostPredictor
from models.health_predictor import HealthPredictor
import config
import warnings
warnings.filterwarnings('ignore')

class SDOHMLPipeline:
    def __init__(self, csv_path):
        """
        Initialize the SDOH ML pipeline.
        
        Args:
            csv_path (str): Path to the SDOH CSV file
        """
        self.csv_path = csv_path
        self.data_processor = SDOHDataProcessor(csv_path)
        self.models = {}
        self.results = {}
        
    def load_and_process_data(self, sample_size=None, target_type='health_index'):
        """
        Load and process the SDOH data for ML.
        
        Args:
            sample_size (int, optional): Number of rows to sample
            target_type (str): Type of target variable to create
        """
        print("=== LOADING AND PROCESSING DATA ===")
        
        # Load data
        self.data_processor.load_data(sample_size=sample_size)
        
        # Explore data
        self.data_processor.explore_data()
        
        # Pivot data to wide format
        self.data_processor.pivot_data()
        
        # Handle missing values
        self.data_processor.handle_missing_values(strategy='drop')
        
        # Encode categorical features
        self.data_processor.encode_categorical_features()
        
        # Create target variable
        self.data_processor.create_target_variable(target_type=target_type)
        
        print("Data processing completed!")
        
    def prepare_models(self):
        """
        Initialize all ML models with optimized parameters.
        """
        print("\n=== PREPARING MODELS ===")
        
        # ANN Model
        ann_params = {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'max_iter': 500,
            'random_state': 42
        }
        self.models['ann'] = ANNPredictor(ann_params)
        
        # Random Forest Model
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        self.models['random_forest'] = RandomForestPredictor(rf_params)
        
        # XGBoost Model
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.models['xgboost'] = XGBoostPredictor(xgb_params)
        
        # Health Predictor (Random Forest variant)
        self.models['health_predictor'] = HealthPredictor(model_type='random_forest')
        
        print(f"Initialized {len(self.models)} models")
        
    def train_all_models(self, target_col='health_index'):
        """
        Train all models on the processed data.
        
        Args:
            target_col (str): Name of the target column
        """
        print("\n=== TRAINING MODELS ===")
        
        # Prepare data for ML
        X_train, X_test, y_train, y_test, feature_names = self.data_processor.prepare_ml_data(
            target_col=target_col
        )
        
        if X_train is None:
            print("Failed to prepare ML data")
            return
            
        self.feature_names = feature_names
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name.upper()}...")
            
            try:
                if hasattr(model, 'train'):
                    if name == 'ann':
                        model.train(X_train, y_train, X_test, y_test)
                    else:
                        model.train(X_train, y_train)
                        
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    self.results[name] = {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'y_test': y_test,
                        'y_pred': y_pred
                    }
                    
                    print(f"  MSE: {mse:.4f}")
                    print(f"  RMSE: {rmse:.4f}")
                    print(f"  MAE: {mae:.4f}")
                    print(f"  R²: {r2:.4f}")
                    
            except Exception as e:
                print(f"  Error training {name}: {e}")
                
    def compare_models(self):
        """
        Compare performance of all trained models.
        """
        if not self.results:
            print("No model results available. Train models first.")
            return
            
        print("\n=== MODEL COMPARISON ===")
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name.upper(),
                'MSE': results['mse'],
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'R²': results['r2']
            })
            
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model = comparison_df.iloc[0]['Model']
        print(f"\nBest performing model: {best_model}")
        
        return comparison_df
        
    def feature_importance_analysis(self, model_name='random_forest'):
        """
        Analyze feature importance for the specified model.
        
        Args:
            model_name (str): Name of the model to analyze
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
            
        model = self.models[model_name]
        
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n=== FEATURE IMPORTANCE ({model_name.upper()}) ===")
            print(importance_df.head(10).to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            top_features = importance_df.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 10 Feature Importances - {model_name.upper()}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance_df
            
    def create_predictions_dashboard(self, state_filter=None):
        """
        Create a comprehensive predictions dashboard.
        
        Args:
            state_filter (str, optional): Filter data by state
        """
        print("\n=== CREATING PREDICTIONS DASHBOARD ===")
        
        # Get sample data for predictions
        sample_data = self.data_processor.get_sample_data(state=state_filter, n_samples=1000)
        
        if sample_data is None:
            print("No sample data available")
            return
            
        # Prepare features for prediction
        exclude_cols = ['LocationID', 'StateAbbr', 'CountyName', 'latitude', 'longitude']
        if 'health_index' in sample_data.columns:
            exclude_cols.append('health_index')
        if 'vulnerability_score' in sample_data.columns:
            exclude_cols.append('vulnerability_score')
            
        feature_cols = [col for col in sample_data.columns if col not in exclude_cols]
        X_pred = sample_data[feature_cols]
        
        # Make predictions with all models
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_pred)
                predictions[name] = pred
                print(f"Generated predictions for {name}")
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                
        # Create predictions DataFrame
        pred_df = sample_data[['LocationID', 'StateAbbr', 'CountyName', 'latitude', 'longitude']].copy()
        
        for name, pred in predictions.items():
            pred_df[f'{name}_prediction'] = pred
            
        # Add actual values if available
        if 'health_index' in sample_data.columns:
            pred_df['actual_health_index'] = sample_data['health_index']
            
        if 'vulnerability_score' in sample_data.columns:
            pred_df['actual_vulnerability_score'] = sample_data['vulnerability_score']
            
        # Save predictions
        pred_df.to_csv('model_predictions.csv', index=False)
        print(f"Saved predictions to model_predictions.csv")
        
        return pred_df
        
    def generate_insights_report(self):
        """
        Generate a comprehensive insights report.
        """
        print("\n=== GENERATING INSIGHTS REPORT ===")
        
        # Model comparison
        comparison_df = self.compare_models()
        
        # Feature importance for best model
        best_model = comparison_df.iloc[0]['Model'].lower()
        importance_df = self.feature_importance_analysis(best_model)
        
        # Create insights report
        report = f"""
# SDOH Machine Learning Insights Report

## Model Performance Summary
{comparison_df.to_string(index=False)}

## Key Findings

### Best Performing Model
- **Model**: {comparison_df.iloc[0]['Model']}
- **R² Score**: {comparison_df.iloc[0]['R²']:.4f}
- **RMSE**: {comparison_df.iloc[0]['RMSE']:.4f}

### Top Predictive Factors
"""
        
        if importance_df is not None:
            top_features = importance_df.head(5)
            for _, row in top_features.iterrows():
                report += f"- **{row['feature']}**: {row['importance']:.4f}\n"
                
        report += f"""
## Recommendations

1. **Focus on High-Impact Factors**: The top predictive factors should be prioritized in intervention strategies.

2. **Model Deployment**: The {comparison_df.iloc[0]['Model']} model shows the best performance and should be used for predictions.

3. **Data Quality**: Consider collecting additional data on the most important features to improve model accuracy.

4. **Geographic Analysis**: Use the spatial data to identify high-risk areas and target interventions accordingly.
"""
        
        # Save report
        with open('sdoh_insights_report.md', 'w') as f:
            f.write(report)
            
        print("Insights report saved to sdoh_insights_report.md")
        
        return report

def main():
    """
    Main function to run the complete ML pipeline.
    """
    # Initialize pipeline
    csv_path = "Non-Medical_Factor_Measures_for_Census_Tract__ACS_2017-2021_20250626.csv"
    pipeline = SDOHMLPipeline(csv_path)
    
    # Load and process data (start with sample for faster processing)
    pipeline.load_and_process_data(sample_size=10000, target_type='health_index')
    
    # Prepare models
    pipeline.prepare_models()
    
    # Train all models
    pipeline.train_all_models(target_col='health_index')
    
    # Compare models
    pipeline.compare_models()
    
    # Generate insights
    pipeline.generate_insights_report()
    
    # Create predictions dashboard
    pipeline.create_predictions_dashboard(state_filter='CA')  # Focus on California
    
    print("\n=== PIPELINE COMPLETED ===")

if __name__ == "__main__":
    main() 