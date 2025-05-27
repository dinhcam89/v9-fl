
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class PerformanceTableGenerator:
    """
    Generates comprehensive performance analysis tables for federated learning sessions.
    """
    
    def __init__(self, client_id: str, wallet_address: str = None):
        self.client_id = client_id
        self.wallet_address = wallet_address
        self.session_data = {
            'validation_performance': [],
            'test_performance': [],
            'global_evaluation_performance': [],
            'ensemble_weights': [],
            'client_info': {
                'client_id': client_id,
                'wallet_address': wallet_address,
                'session_start': None,
                'session_end': None
            },
            'dataset_info': {
                'total_samples': 0,
                'total_features': 0,
                'training_samples': 0,
                'validation_samples': 0,
                'test_samples': 0,
                'fraud_cases_train': 0,
                'normal_cases_train': 0,
                'fraud_cases_val': 0,
                'normal_cases_val': 0,
                'fraud_cases_test': 0,
                'normal_cases_test': 0,
                'fraud_percentage_train': 0.0,
                'fraud_percentage_val': 0.0,
                'fraud_percentage_test': 0.0,
                'dataset_file': None,
                'preprocessing_applied': [],
                'feature_columns': []
            }
        }
        
    def record_validation_performance(self, round_num: int, phase: str, metrics: Dict[str, Any], confusion_matrix: Dict[str, int]):
        """Record validation set performance data"""
        record = {
            'round': round_num,
            'phase': phase,
            'loss': metrics.get('loss', 0.0),
            'accuracy': metrics.get('accuracy', 0.0) * 100 if metrics.get('accuracy', 0.0) <= 1.0 else metrics.get('accuracy', 0.0),
            'precision': metrics.get('precision', 0.0) * 100 if metrics.get('precision', 0.0) <= 1.0 else metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0) * 100 if metrics.get('recall', 0.0) <= 1.0 else metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0) * 100 if metrics.get('f1_score', 0.0) <= 1.0 else metrics.get('f1_score', 0.0),
            'tp': confusion_matrix.get('TP', 0),
            'tn': confusion_matrix.get('TN', 0),
            'fp': confusion_matrix.get('FP', 0),
            'fn': confusion_matrix.get('FN', 0)
        }
        self.session_data['validation_performance'].append(record)
        
    def record_test_performance(self, round_num: int, phase: str, metrics: Dict[str, Any], confusion_matrix: Dict[str, int]):
        """Record test set performance data"""
        record = {
            'round': round_num,
            'phase': phase,
            'loss': metrics.get('loss', 0.0),
            'accuracy': metrics.get('accuracy', 0.0) * 100 if metrics.get('accuracy', 0.0) <= 1.0 else metrics.get('accuracy', 0.0),
            'precision': metrics.get('precision', 0.0) * 100 if metrics.get('precision', 0.0) <= 1.0 else metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0) * 100 if metrics.get('recall', 0.0) <= 1.0 else metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0) * 100 if metrics.get('f1_score', 0.0) <= 1.0 else metrics.get('f1_score', 0.0),
            'tp': confusion_matrix.get('TP', 0),
            'tn': confusion_matrix.get('TN', 0),
            'fp': confusion_matrix.get('FP', 0),
            'fn': confusion_matrix.get('FN', 0)
        }
        self.session_data['test_performance'].append(record)
        
    def record_global_evaluation(self, round_num: int, metrics: Dict[str, Any], confusion_matrix: Dict[str, int], 
                               score: float = 0.0, rank: int = 0, reward: float = 0.0):
        """Record global model evaluation performance"""
        record = {
            'round': round_num,
            'loss': metrics.get('loss', 0.0),
            'accuracy': metrics.get('accuracy', 0.0) * 100 if metrics.get('accuracy', 0.0) <= 1.0 else metrics.get('accuracy', 0.0),
            'precision': metrics.get('precision', 0.0) * 100 if metrics.get('precision', 0.0) <= 1.0 else metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0) * 100 if metrics.get('recall', 0.0) <= 1.0 else metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0) * 100 if metrics.get('f1_score', 0.0) <= 1.0 else metrics.get('f1_score', 0.0),
            'tp': confusion_matrix.get('TP', 0),
            'tn': confusion_matrix.get('TN', 0),
            'fp': confusion_matrix.get('FP', 0),
            'fn': confusion_matrix.get('FN', 0),
            'score': score,
            'rank': rank,
            'reward': reward
        }
        self.session_data['global_evaluation_performance'].append(record)
        
    def record_ensemble_weights(self, round_num: int, phase: str, weights: List[float], 
                              model_names: List[str] = None):
        """Record ensemble weight evolution"""
        if model_names is None:
            model_names = ['LR', 'SVC', 'RF', 'KNN', 'CatBoost', 'LightGBM', 'XGBoost']
            
        record = {
            'round': round_num,
            'phase': phase
        }
        
        # Add individual model weights
        for i, name in enumerate(model_names):
            if i < len(weights):
                record[name.lower()] = weights[i]
            else:
                record[name.lower()] = 0.0
                
    def record_dataset_info(self, total_samples: int, total_features: int, 
                           training_samples: int, validation_samples: int, test_samples: int,
                           fraud_cases_train: int, fraud_cases_val: int, fraud_cases_test: int,
                           dataset_file: str = None, preprocessing_applied: List[str] = None,
                           feature_columns: List[str] = None):
        """Record comprehensive dataset information"""
        self.session_data['dataset_info'].update({
            'total_samples': total_samples,
            'total_features': total_features,
            'training_samples': training_samples,
            'validation_samples': validation_samples,
            'test_samples': test_samples,
            'fraud_cases_train': fraud_cases_train,
            'normal_cases_train': training_samples - fraud_cases_train,
            'fraud_cases_val': fraud_cases_val,
            'normal_cases_val': validation_samples - fraud_cases_val,
            'fraud_cases_test': fraud_cases_test,
            'normal_cases_test': test_samples - fraud_cases_test,
            'fraud_percentage_train': (fraud_cases_train / training_samples * 100) if training_samples > 0 else 0.0,
            'fraud_percentage_val': (fraud_cases_val / validation_samples * 100) if validation_samples > 0 else 0.0,
            'fraud_percentage_test': (fraud_cases_test / test_samples * 100) if test_samples > 0 else 0.0,
            'dataset_file': dataset_file,
            'preprocessing_applied': preprocessing_applied or [],
            'feature_columns': feature_columns or []
        })
        
    def generate_validation_table(self) -> pd.DataFrame:
        """Generate validation performance table"""
        if not self.session_data['validation_performance']:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.session_data['validation_performance'])
        
        # Format columns for better display
        df['accuracy'] = df['accuracy'].round(2).astype(str) + '%'
        df['precision'] = df['precision'].round(2).astype(str) + '%'
        df['recall'] = df['recall'].round(2).astype(str) + '%'
        df['f1_score'] = df['f1_score'].round(2).astype(str) + '%'
        df['loss'] = df['loss'].round(4)
        
        return df[['round', 'phase', 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'tp', 'tn', 'fp', 'fn']]
    
    def generate_test_table(self) -> pd.DataFrame:
        """Generate test performance table"""
        if not self.session_data['test_performance']:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.session_data['test_performance'])
        
        # Format columns for better display
        df['accuracy'] = df['accuracy'].round(2).astype(str) + '%'
        df['precision'] = df['precision'].round(2).astype(str) + '%'
        df['recall'] = df['recall'].round(2).astype(str) + '%'
        df['f1_score'] = df['f1_score'].round(2).astype(str) + '%'
        df['loss'] = df['loss'].round(4)
        
        return df[['round', 'phase', 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'tp', 'tn', 'fp', 'fn']]
    
    def generate_global_evaluation_table(self) -> pd.DataFrame:
        """Generate global evaluation performance table"""
        if not self.session_data['global_evaluation_performance']:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.session_data['global_evaluation_performance'])
        
        # Format columns for better display
        df['accuracy'] = df['accuracy'].round(2).astype(str) + '%'
        df['precision'] = df['precision'].round(2).astype(str) + '%'
        df['recall'] = df['recall'].round(2).astype(str) + '%'
        df['f1_score'] = df['f1_score'].round(2).astype(str) + '%'
        df['loss'] = df['loss'].round(4)
        df['reward'] = df['reward'].round(6)
        
        return df[['round', 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'tp', 'tn', 'fp', 'fn', 'score', 'rank', 'reward']]
    
    def generate_ensemble_weights_table(self) -> pd.DataFrame:
        """Generate ensemble weights evolution table"""
        if not self.session_data['ensemble_weights']:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.session_data['ensemble_weights'])
        
        # Round weights to 4 decimal places
        weight_columns = [col for col in df.columns if col not in ['round', 'phase']]
        for col in weight_columns:
            df[col] = df[col].round(4)
            
        return df
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive text summary report"""
        report = []
        report.append(f"# Federated Learning Performance Analysis Report")
        report.append(f"Client ID: {self.client_id}")
        report.append(f"Wallet Address: {self.wallet_address}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*100)
        
        # Dataset Information Section
        report.append("\n## Dataset Information")
        report.append("-" * 50)
        dataset_info = self.session_data['dataset_info']
        
        if dataset_info.get('dataset_file'):
            report.append(f"Dataset File: {dataset_info['dataset_file']}")
        
        report.append(f"Total Original Samples: {dataset_info.get('total_samples', 0):,}")
        report.append(f"Total Features: {dataset_info.get('total_features', 0):,}")
        
        # Data Split Information
        report.append(f"\n### Data Split Distribution")
        report.append(f"├── Training Set:   {dataset_info.get('training_samples', 0):,} samples")
        report.append(f"├── Validation Set: {dataset_info.get('validation_samples', 0):,} samples")
        report.append(f"└── Test Set:       {dataset_info.get('test_samples', 0):,} samples")
        
        # Fraud Distribution by Set
        report.append(f"\n### Fraud Case Distribution")
        
        # Training Set
        train_fraud = dataset_info.get('fraud_cases_train', 0)
        train_normal = dataset_info.get('normal_cases_train', 0)
        train_total = train_fraud + train_normal
        train_fraud_pct = dataset_info.get('fraud_percentage_train', 0.0)
        
        report.append(f"**Training Set ({train_total:,} samples):**")
        report.append(f"  ├── Fraud Cases:  {train_fraud:,} ({train_fraud_pct:.2f}%)")
        report.append(f"  └── Normal Cases: {train_normal:,} ({100-train_fraud_pct:.2f}%)")
        
        # Validation Set
        val_fraud = dataset_info.get('fraud_cases_val', 0)
        val_normal = dataset_info.get('normal_cases_val', 0)
        val_total = val_fraud + val_normal
        val_fraud_pct = dataset_info.get('fraud_percentage_val', 0.0)
        
        report.append(f"\n**Validation Set ({val_total:,} samples):**")
        report.append(f"  ├── Fraud Cases:  {val_fraud:,} ({val_fraud_pct:.2f}%)")
        report.append(f"  └── Normal Cases: {val_normal:,} ({100-val_fraud_pct:.2f}%)")
        
        # Test Set
        test_fraud = dataset_info.get('fraud_cases_test', 0)
        test_normal = dataset_info.get('normal_cases_test', 0)
        test_total = test_fraud + test_normal
        test_fraud_pct = dataset_info.get('fraud_percentage_test', 0.0)
        
        report.append(f"\n**Test Set ({test_total:,} samples):**")
        report.append(f"  ├── Fraud Cases:  {test_fraud:,} ({test_fraud_pct:.2f}%)")
        report.append(f"  └── Normal Cases: {test_normal:,} ({100-test_fraud_pct:.2f}%)")
        
        # Preprocessing Information
        preprocessing = dataset_info.get('preprocessing_applied', [])
        if preprocessing:
            report.append(f"\n### Preprocessing Applied")
            for i, process in enumerate(preprocessing, 1):
                report.append(f"  {i}. {process}")
        
        # Feature Information
        feature_columns = dataset_info.get('feature_columns', [])
        if feature_columns:
            report.append(f"\n### Feature Columns ({len(feature_columns)} total)")
            if len(feature_columns) <= 10:
                # Show all if 10 or fewer
                for i, feature in enumerate(feature_columns, 1):
                    report.append(f"  {i:2d}. {feature}")
            else:
                # Show first 5 and last 5 if more than 10
                for i, feature in enumerate(feature_columns[:5], 1):
                    report.append(f"  {i:2d}. {feature}")
                report.append(f"  ... ({len(feature_columns)-10} more features)")
                for i, feature in enumerate(feature_columns[-5:], len(feature_columns)-4):
                    report.append(f"  {i:2d}. {feature}")
        
        # Session Information
        report.append(f"\n## Session Information")
        report.append("-" * 50)
        client_info = self.session_data['client_info']
        if client_info.get('session_start'):
            start_time = datetime.fromisoformat(client_info['session_start'].replace('Z', '+00:00'))
            report.append(f"Session Start: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        if client_info.get('session_end'):
            end_time = datetime.fromisoformat(client_info['session_end'].replace('Z', '+00:00'))
            report.append(f"Session End:   {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            if client_info.get('session_start'):
                duration = end_time - start_time
                hours, remainder = divmod(duration.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                report.append(f"Duration:      {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        total_rounds = len(set(p['round'] for p in self.session_data['test_performance']))
        report.append(f"Total Rounds:  {total_rounds}")
        
        # Validation Performance Summary
        if self.session_data['validation_performance']:
            report.append("\n## Validation Set Performance Summary")
            report.append("-" * 50)
            val_df = self.generate_validation_table()
            report.append(val_df.to_string(index=False))
            
        # Test Performance Summary  
        if self.session_data['test_performance']:
            report.append("\n## Test Set Performance Summary")
            report.append("-" * 50)
            test_df = self.generate_test_table()
            report.append(test_df.to_string(index=False))
            
        # Global Evaluation Summary
        if self.session_data['global_evaluation_performance']:
            report.append("\n## Global Model Evaluation Summary")
            report.append("-" * 50)
            global_df = self.generate_global_evaluation_table()
            report.append(global_df.to_string(index=False))
            
        # Ensemble Weights Evolution
        if self.session_data['ensemble_weights']:
            report.append("\n## Ensemble Weights Evolution")
            report.append("-" * 50)
            weights_df = self.generate_ensemble_weights_table()
            report.append(weights_df.to_string(index=False))
            
        # Performance insights
        report.append(self._generate_insights())
        
        return "\n".join(report)
    
    def _generate_insights(self) -> str:
        """Generate performance insights"""
        insights = ["\n## Performance Insights"]
        
        # Analyze test performance trends
        if len(self.session_data['test_performance']) > 1:
            test_data = self.session_data['test_performance']
            initial_recall = test_data[0]['recall']
            final_recall = test_data[-1]['recall']
            
            if final_recall > initial_recall:
                insights.append(f"✅ Recall improved from {initial_recall:.1f}% to {final_recall:.1f}% (+{final_recall-initial_recall:.1f}%)")
            elif final_recall < initial_recall:
                insights.append(f"⚠️  Recall decreased from {initial_recall:.1f}% to {final_recall:.1f}% ({final_recall-initial_recall:.1f}%)")
            else:
                insights.append(f"➡️  Recall remained stable at {final_recall:.1f}%")
                
        # Analyze global evaluation trends
        if len(self.session_data['global_evaluation_performance']) > 1:
            global_data = self.session_data['global_evaluation_performance']
            avg_score = sum(d['score'] for d in global_data) / len(global_data)
            best_round = max(global_data, key=lambda x: x['score'])
            insights.append(f"🏆 Best performance in Round {best_round['round']} with score {best_round['score']:.0f}")
            insights.append(f"📊 Average contribution score: {avg_score:.0f}")
            
        # Analyze ensemble evolution
        if len(self.session_data['ensemble_weights']) > 1:
            weights_data = self.session_data['ensemble_weights']
            first_weights = weights_data[0]
            last_weights = weights_data[-1]
            
            # Find most improved model
            max_change = 0
            best_model = ""
            for model in ['lr', 'svc', 'rf', 'knn', 'catboost', 'lightgbm', 'xgboost']:
                if model in first_weights and model in last_weights:
                    change = last_weights[model] - first_weights[model]
                    if abs(change) > max_change:
                        max_change = abs(change)
                        best_model = model.upper()
                        
            if best_model:
                insights.append(f"🔄 {best_model} showed the largest weight change ({max_change:.3f})")
                
        return "\n".join(insights)
    
    def save_session_data(self, output_dir: str = "performance_analysis"):
        """Save all session data and tables to files with organized folder structure"""
        # Create organized folder structure: client_id/timestamp/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        client_folder = os.path.join(output_dir, self.client_id, timestamp)
        os.makedirs(client_folder, exist_ok=True)
        
        base_filename = f"{self.client_id}_{timestamp}"
        
        # Save raw session data as JSON
        json_file = os.path.join(client_folder, f"{base_filename}_session_data.json")
        with open(json_file, 'w') as f:
            json.dump(self.session_data, f, indent=2, default=str)
            
        # Save tables as CSV
        tables = {
            'validation': self.generate_validation_table(),
            'test': self.generate_test_table(), 
            'global_evaluation': self.generate_global_evaluation_table(),
            'ensemble_weights': self.generate_ensemble_weights_table()
        }
        
        for table_name, df in tables.items():
            if not df.empty:
                csv_file = os.path.join(client_folder, f"{base_filename}_{table_name}.csv")
                df.to_csv(csv_file, index=False)
                
        # Save summary report
        report_file = os.path.join(client_folder, f"{base_filename}_summary_report.txt")
        with open(report_file, 'w') as f:
            f.write(self.generate_summary_report())
            
        # Create a session summary file for quick reference
        session_summary_file = os.path.join(client_folder, "session_summary.json")
        session_summary = {
            'client_id': self.client_id,
            'wallet_address': self.wallet_address,
            'timestamp': timestamp,
            'session_start': self.session_data['client_info'].get('session_start'),
            'session_end': self.session_data['client_info'].get('session_end'),
            'total_rounds': len(set(p['round'] for p in self.session_data['test_performance'])) if self.session_data['test_performance'] else 0,
            'dataset_info': self.session_data['dataset_info'],
            'files_generated': [
                f"{base_filename}_session_data.json",
                f"{base_filename}_validation.csv" if not tables['validation'].empty else None,
                f"{base_filename}_test.csv" if not tables['test'].empty else None,
                f"{base_filename}_global_evaluation.csv" if not tables['global_evaluation'].empty else None,
                f"{base_filename}_ensemble_weights.csv" if not tables['ensemble_weights'].empty else None,
                f"{base_filename}_summary_report.txt"
            ]
        }
        
        # Remove None values from files_generated
        session_summary['files_generated'] = [f for f in session_summary['files_generated'] if f is not None]
        
        with open(session_summary_file, 'w') as f:
            json.dump(session_summary, f, indent=2, default=str)
            
        logger.info(f"Performance analysis saved to {client_folder}")
        return client_folder

    def print_session_summary(self):
        """Print a formatted session summary to console"""
        print("\n" + "="*100)
        print(f"🎯 FEDERATED LEARNING SESSION COMPLETE - {self.client_id}")
        print("="*100)
        
        print(f"\n📊 VALIDATION SET PERFORMANCE")
        print("-" * 50)
        val_table = self.generate_validation_table()
        if not val_table.empty:
            print(val_table.to_string(index=False))
        else:
            print("No validation data recorded")
            
        print(f"\n🧪 TEST SET PERFORMANCE") 
        print("-" * 50)
        test_table = self.generate_test_table()
        if not test_table.empty:
            print(test_table.to_string(index=False))
        else:
            print("No test data recorded")
            
        print(f"\n🌍 GLOBAL MODEL EVALUATION")
        print("-" * 50)
        global_table = self.generate_global_evaluation_table()
        if not global_table.empty:
            print(global_table.to_string(index=False))
        else:
            print("No global evaluation data recorded")
            
        print(f"\n🔄 ENSEMBLE WEIGHTS EVOLUTION")
        print("-" * 50)
        weights_table = self.generate_ensemble_weights_table()
        if not weights_table.empty:
            print(weights_table.to_string(index=False))
        else:
            print("No ensemble weights recorded")
            
        print(self._generate_insights())
        print("="*100)