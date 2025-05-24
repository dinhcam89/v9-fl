"""
Enhanced Federated Learning Client with GA-Stacking Ensemble optimization.
Extends the base client with local ensemble optimization capabilities.
Uses scikit-learn models exclusively (no PyTorch).
"""

import os
import json
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split

import random
import threading
from web3.exceptions import TransactionNotFound
from hexbytes import HexBytes
import pandas as pd

import faulthandler
faulthandler.enable()

import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, log_loss
)

import numpy as np
from sklearn.preprocessing import StandardScaler

from ipfs_connector import IPFSConnector
from blockchain_connector import BlockchainConnector

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import GA-Stacking modules
from ga_stacking.pipeline import GAStackingPipeline
from ga_stacking.base_models import BASE_MODELS, META_MODELS
from ga_stacking.utils import split_and_scale, evaluate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("FL-Client-Ensemble")


class GAStackingClient(fl.client.NumPyClient):
    """Federated Learning client with GA-Stacking ensemble optimization using scikit-learn."""
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ensemble_size: int = 5,
        ipfs_connector: Optional[IPFSConnector] = None,
        blockchain_connector: Optional[BlockchainConnector] = None,
        wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        client_id: str = None,
        ga_generations: int = 3,
        ga_population_size: int = 8
    ):
        """
        Initialize GA-Stacking client with scikit-learn models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            ensemble_size: Number of models in the ensemble
            ipfs_connector: IPFS connector
            blockchain_connector: Blockchain connector
            wallet_address: Client's wallet address
            private_key: Client's private key
            client_id: Client identifier
            ga_generations: Number of GA generations to run
            ga_population_size: Size of GA population
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.ensemble_size = ensemble_size
        self.ipfs = ipfs_connector or IPFSConnector()
        self.blockchain = blockchain_connector
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.client_id = client_id or f"client-{os.getpid()}"
        self.ga_generations = ga_generations
        self.ga_population_size = ga_population_size
        
        # Initialize GA-Stacking pipeline
        self.ga_pipeline = GAStackingPipeline(
            base_models=BASE_MODELS,
            meta_models=META_MODELS,
            generations=ga_generations,
            pop_size=ga_population_size,
            verbose=True
        )
        
        # Metrics storage
        self.metrics_history = []
        
        # Ensemble state
        self.ensemble_state = None
        self.ga_results = None
        
        logger.info(f"Initialized {self.client_id} with GA-Stacking pipeline")
        logger.info(f"IPFS node: {self.ipfs.ipfs_api_url}")
        
        # Verify blockchain authentication if available
        if self.blockchain and self.wallet_address:
            try:
                is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
                if is_authorized:
                    logger.info(f"Client {self.wallet_address} is authorized on the blockchain ✅")
                else:
                    logger.warning(f"Client {self.wallet_address} is NOT authorized on the blockchain ❌")
                    logger.warning("The server may reject this client's contributions")
            except Exception as e:
                logger.error(f"Failed to verify client authorization: {e}")
    
    # Rest of the GAStackingClient class remains unchanged...
    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays."""
        if not hasattr(self, 'ensemble_state') or self.ensemble_state is None:
            # If ensemble not trained yet, return empty parameters
            return [np.array([])]
        
        # Serialize ensemble state to JSON and convert to ndarray
        ensemble_bytes = json.dumps(self.ensemble_state).encode('utf-8')
        return [np.frombuffer(ensemble_bytes, dtype=np.uint8)]
    
    def set_parameters_from_ensemble_state(self, ensemble_state):
        """Set ensemble state from a dictionary."""
        self.ensemble_state = ensemble_state
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set parameters from a list of NumPy arrays.
        
        Args:
            parameters: List of parameter arrays
        """
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            # Try to interpret as serialized ensemble state
            try:
                ensemble_bytes = parameters[0].tobytes()
                ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
                self.set_parameters_from_ensemble_state(ensemble_state)
                logger.info("Successfully loaded ensemble state from parameters")
                return
            except Exception as e:
                logger.error(f"Failed to deserialize ensemble state: {e}")
        
        logger.warning("Failed to set parameters - invalid format")
    
    
    def _print_evaluation_report(self, dataset_name, y_true, y_pred, loss=None, round_num=None):
        """
        Print a comprehensive evaluation report with metrics table and confusion matrix.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset (e.g., "Validation", "Test")
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted probabilities or labels
        loss : float, optional
            Loss value to include in report
        round_num : int, optional
            Round number for context
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
        import pandas as pd
        
        # Convert probabilities to binary predictions if needed
        if y_pred.max() <= 1.0 and y_pred.min() >= 0.0 and len(np.unique(y_pred)) > 2:
            # This looks like probabilities
            binary_pred = (y_pred > 0.5).astype(int)
            # Calculate loss from probabilities
            if loss is None:
                clipped_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                loss = log_loss(y_true, clipped_pred)
        else:
            # This looks like binary predictions
            binary_pred = y_pred.astype(int)
            if loss is None:
                loss = 0.0  # Can't calculate meaningful loss from binary predictions
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, binary_pred)
        precision = precision_score(y_true, binary_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, binary_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, binary_pred, pos_label=1, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, binary_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge case where only one class is present
            if len(np.unique(y_true)) == 1 and len(np.unique(binary_pred)) == 1:
                if y_true[0] == 0 and binary_pred[0] == 0:
                    tn, fp, fn, tp = len(y_true), 0, 0, 0
                elif y_true[0] == 1 and binary_pred[0] == 1:
                    tn, fp, fn, tp = 0, 0, 0, len(y_true)
                else:
                    tn, fp, fn, tp = 0, len(y_true), 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
        # Create header
        header = f"\n{'='*60}"
        if round_num:
            header += f"\n  ROUND {round_num} - {dataset_name.upper()} SET EVALUATION REPORT"
        else:
            header += f"\n  {dataset_name.upper()} SET EVALUATION REPORT"
        header += f"\n{'='*60}"
        
        # Create metrics table
        metrics_data = {
            'Metric': ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [f"{loss:.4f}", f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"],
            'Percentage': ['-', f"{accuracy*100:.2f}%", f"{precision*100:.2f}%", f"{recall*100:.2f}%", f"{f1*100:.2f}%"]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create confusion matrix table
        cm_data = {
            '': ['Predicted: 0 (Normal)', 'Predicted: 1 (Fraud)'],
            'Actual: 0 (Normal)': [f'{tn}', f'{fp}'],
            'Actual: 1 (Fraud)': [f'{fn}', f'{tp}']
        }
        
        cm_df = pd.DataFrame(cm_data)
        
        # Print the report
        print(header)
        print(f"\nDataset: {dataset_name}")
        print(f"Total Samples: {len(y_true)}")
        print(f"Fraud Cases: {np.sum(y_true)} ({np.sum(y_true)/len(y_true)*100:.2f}%)")
        print(f"Normal Cases: {len(y_true) - np.sum(y_true)} ({(len(y_true) - np.sum(y_true))/len(y_true)*100:.2f}%)")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(metrics_df.to_string(index=False))
        
        print(f"\n🔍 CONFUSION MATRIX:")
        print(cm_df.to_string(index=False))
        
        print(f"\n📈 DETAILED BREAKDOWN:")
        print(f"  • True Positives (TP):  {tp:4d} - Correctly identified fraud cases")
        print(f"  • True Negatives (TN):  {tn:4d} - Correctly identified normal cases") 
        print(f"  • False Positives (FP): {fp:4d} - Normal cases incorrectly flagged as fraud")
        print(f"  • False Negatives (FN): {fn:4d} - Fraud cases missed (most critical!)")
        
        # Add interpretation
        print(f"\n💡 INTERPRETATION:")
        if recall >= 0.9:
            print(f"  ✅ EXCELLENT recall ({recall:.1%}) - Catching most fraud cases!")
        elif recall >= 0.7:
            print(f"  ⚠️  GOOD recall ({recall:.1%}) - Missing some fraud cases")
        else:
            print(f"  ❌ LOW recall ({recall:.1%}) - Missing many fraud cases!")
            
        if precision >= 0.7:
            print(f"  ✅ GOOD precision ({precision:.1%}) - Low false alarm rate")
        elif precision >= 0.5:
            print(f"  ⚠️  MODERATE precision ({precision:.1%}) - Some false alarms")
        else:
            print(f"  ❌ LOW precision ({precision:.1%}) - High false alarm rate")
        
        print(f"{'='*60}\n")
        
        return {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1_score': f1,
            'loss': loss,
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        }

    def fit(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train the model using GA-Stacking in Round 1 and fine-tune in subsequent rounds.
        """
        # Check if client is authorized (if blockchain is available)
        if self.blockchain and self.wallet_address:
            try:
                is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
                if not is_authorized:
                    logger.warning(f"Client {self.wallet_address} is not authorized to participate")
                    # Return empty parameters with auth failure flag
                    return parameters_to_ndarrays(parameters), 0, {"error": "client_not_authorized"}
            except Exception as e:
                logger.error(f"Failed to check client authorization: {e}")
                # Continue anyway, server will check again

        # Get config parameters
        server_round = config.get("server_round", 0)
        is_first_round = server_round == 1
        global_model_ipfs_hash = config.get("global_model_ipfs_hash", None)
        
        # Prepare data
        X_train, y_train = self.X_train, self.y_train
        X_val, y_val = self.X_val, self.y_val
        initial_weights = None
        
        # ROUND 1: Train base models + GA-Stacking from scratch
        if is_first_round:
            logger.info("Round 1: Training base models with GA-Stacking")
            try:
                # Initialize the GA pipeline with config parameters
                self.ga_pipeline = GAStackingPipeline(
                    pop_size=10,
                    generations=5,
                    cv_folds=5,
                    verbose=True
                )
                
                # Generate random initial weights if none provided
                if initial_weights is None:
                    # Get the number of base models
                    num_models = len(self.ga_pipeline.base_models)
                    logger.info(f"Generating equal initial weights for {num_models} base models")
                    
                    # Set all weights to 1/num_models
                    initial_weights = np.full(num_models, 1 / num_models)
                    
                    logger.info(f"Generated equal initial weights: {initial_weights}")
                
                # Train the GA-Stacking pipeline with initial weights
                self.ga_results = self.ga_pipeline.train(
                    X_train, y_train, X_val, y_val, 
                    init_weights=initial_weights
                )
                
                # Save ensemble state
                self.ensemble_state = self.ga_pipeline.get_ensemble_state()
                
                model_dir = os.path.join("models", f"client_{self.client_id}")
                # Save base models for future rounds
                self.ga_pipeline.save_base_models(model_dir)
                
                logger.info(f"Completed GA-Stacking with {len(self.ensemble_state.get('model_names', []))} base models")
                
                # ROUND 1 EVALUATION REPORTING
                print(f"\n🎯 ROUND 1 TRAINING COMPLETED - DETAILED EVALUATION")
                
                # 1. Validation Set Evaluation with Full Report (including loss)
                print(f"\n📊 EVALUATING ON VALIDATION SET...")
                y_pred_val = self.ga_pipeline.predict(X_val)
                
                # Print comprehensive validation set report
                val_report = self._print_evaluation_report(
                    "Validation", self.y_val, y_pred_val, round_num=server_round
                )
                
                # 2. Test Set Evaluation with Full Report
                print(f"\n🧪 EVALUATING ON TEST SET...")
                y_pred_test = self.ga_pipeline.predict(self.X_test)
                
                # Print comprehensive test set report
                test_report = self._print_evaluation_report(
                    "Test", self.y_test, y_pred_test, round_num=server_round
                )
                
            except Exception as e:
                logger.error(f"Error in GA-Stacking: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # ROUNDS 2+: Fine-tune with aggregated weights
        else:
            logger.info(f"Round {server_round}: Fine-tuning with aggregated weights")
            try:
                if global_model_ipfs_hash:
                    # Log the hash we're trying to retrieve
                    logger.info(f"Retrieving aggregated weights from IPFS hash: {global_model_ipfs_hash}")
                    
                    # Retrieve aggregated ensemble weights from IPFS
                    try:
                        global_model_data = self.ipfs.get_json(global_model_ipfs_hash)
                        logger.info(f"Retrieved global model data with keys: {list(global_model_data.keys())}")
                        
                        # Extract ensemble weights from global model data with proper fallbacks
                        aggregated_weights = None
                        base_model_names = None
                        
                        # Try different possible locations for ensemble weights
                        if "ensemble_weights" in global_model_data:
                            aggregated_weights = global_model_data["ensemble_weights"]
                            logger.info(f"Found ensemble weights in global model data: {aggregated_weights}")
                        
                        # Try different possible locations for base model names
                        if "base_model_names" in global_model_data:
                            base_model_names = global_model_data["base_model_names"]
                            logger.info(f"Found base model names in global model data: {base_model_names}")
                        elif "info" in global_model_data and "base_model_names" in global_model_data["info"]:
                            base_model_names = global_model_data["info"]["base_model_names"]
                            logger.info(f"Found base model names in global model info: {base_model_names}")
                        
                        if aggregated_weights and base_model_names:
                            # Load saved base models if not already loaded
                            if not hasattr(self, 'base_models_loaded') or not self.base_models_loaded:
                                models_dir = os.path.join("models", f"client_{self.client_id}")
                                logger.info(f"Loading base models from {models_dir}")
                                
                                # Load base models using pipeline's method
                                self.ga_pipeline.load_base_models(models_dir)
                                self.base_models_loaded = True
                                
                                # Check if models were loaded successfully
                                if hasattr(self.ga_pipeline, 'trained_base_models'):
                                    logger.info(f"Loaded {len(self.ga_pipeline.trained_base_models)} trained base models")
                                else:
                                    logger.error("No trained_base_models found after loading")
                                    raise ValueError("Failed to load base models")
                            
                            # Check if we have trained base models before fine-tuning
                            if not hasattr(self.ga_pipeline, 'trained_base_models') or not self.ga_pipeline.trained_base_models:
                                logger.error("No trained_base_models available for fine-tuning")
                                raise ValueError("No trained base models available")
                            
                            # Log the models we have
                            logger.info(f"Fine-tuning with base models: {list(self.ga_pipeline.trained_base_models.keys())}")
                            
                            # Ensure weights are in the correct format
                            if isinstance(aggregated_weights, list):
                                # Convert to numpy array if needed
                                aggregated_weights = np.array(aggregated_weights)
                            
                            # Now perform the actual fine-tuning - explicitly passing the required parameters
                            logger.info(f"Fine-tuning with aggregated weights: {aggregated_weights}")
                            self.ga_results = self.ga_pipeline.fine_tune(
                                X_train=X_train, 
                                y_train=y_train, 
                                X_val=X_val, 
                                y_val=y_val,
                                base_models=self.ga_pipeline.trained_base_models,  # This is required
                                weights=aggregated_weights  # Pass the weights
                            )
                            
                            # Update ensemble state after fine-tuning
                            self.ensemble_state = self.ga_pipeline.get_ensemble_state()
                            
                            logger.info(f"Completed fine-tuning with {len(aggregated_weights)} base models")
                            logger.info(f"Updated ensemble weights: {self.ensemble_state.get('weights', [])}")
                            
                            # FINE-TUNING EVALUATION REPORTING
                            print(f"\n📈 ROUND {server_round} FINE-TUNING COMPLETED")
                            # Validation Set Evaluation after Fine-tuning
                            
                            print(f"\n📊 EVALUATING ON VALIDATION SET AFTER FINE-TUNING...")
                            y_pred_val = self.ga_pipeline.predict(X_val)
                            val_report = self._print_evaluation_report(
                                "Validation (After Fine-tuning)", self.y_val, y_pred_val, round_num=server_round
                            )
                            # Test Set Evaluation after Fine-tuning
                            print(f"\n🧪 EVALUATING ON TEST SET AFTER FINE-TUNING...")
                            y_pred_test = self.ga_pipeline.predict(self.X_test)
                            test_report = self._print_evaluation_report(
                                "Test (After Fine-tuning)", self.y_test, y_pred_test, round_num=server_round
                            )
                            
                        else:
                            logger.error(f"Missing required data in global model. Found keys: {list(global_model_data.keys())}")
                            if "info" in global_model_data:
                                logger.error(f"Info section contains keys: {list(global_model_data['info'].keys())}")
                    except Exception as e:
                        logger.error(f"Error retrieving global model from IPFS: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    logger.error("No global model IPFS hash provided for fine-tuning")
            except Exception as e:
                logger.error(f"Error in fine-tuning: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Evaluate on test data for metrics
        if hasattr(self, 'ga_pipeline') and self.ga_pipeline is not None and hasattr(self.ga_pipeline, 'predict'):
            y_pred = self.ga_pipeline.predict(self.X_test)
            test_metrics = evaluate_metrics(self.y_test, y_pred)
            
            accuracy = test_metrics['accuracy'] * 100.0
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = log_loss(self.y_test, y_pred_clipped)
            
            ga_stacking_metrics = {
                # Keep the data_points metric (used in our new reward system)
                "data_points": len(self.X_train),  # Use the raw number of training samples
                
                # Store recall value directly from test metrics
                "recall": float(test_metrics['recall']),  # Recall is critical for fraud detection
                
                # Keep other metrics for backward compatibility
                "data_size": len(self.X_train),
                "training_rounds": server_round if not hasattr(self, 'rounds_participated') else self.rounds_participated + 1,
                "evaluation_score": float(test_metrics['f1'])
            }
            
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(test_metrics['precision']),
                "recall": float(test_metrics['recall']),
                "f1_score": float(test_metrics['f1']),
            }
        else:
            # Fallback if pipeline isn't trained
            accuracy, loss = 0.0, float('inf')
            metrics = {
                "loss": float(loss),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
            ga_stacking_metrics = {
                "data_points": len(self.X_train),
                "recall": 0.0,
                "data_size": len(self.X_train),
                "training_rounds": server_round if not hasattr(self, 'rounds_participated') else self.rounds_participated + 1,
                "evaluation_score": 0.0
            }

        # Create and store client data in IPFS
        try:
            # Make sure ensemble weights and base model names are properly formatted
            ensemble_weights = self.ensemble_state.get("weights", [])
            if isinstance(ensemble_weights, np.ndarray):
                ensemble_weights = ensemble_weights.tolist()
            
            base_model_names = self.ensemble_state.get("model_names", [])
            
            # Create IPFS data structure that matches server expectations
            client_data = {
                "ensemble_weights": ensemble_weights,  # Store directly in root
                "base_model_names": base_model_names,  # Store directly in root
                "info": {
                    "round": server_round,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "version": "client-" + str(self.client_id),
                    "client_id": self.client_id,
                    "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                    "data_size": len(self.X_train),
                    "base_model_names": base_model_names  # Also store in info for server compatibility
                },
                "metrics": {
                    "loss": float(loss),
                    "recall": float(metrics.get('recall', 0.0)) if 'metrics' in locals() else 0.0,
                    "precision": float(metrics.get('precision', 0.0)) if 'metrics' in locals() else 0.0,
                    "f1_score": float(metrics.get('f1_score', 0.0)) if 'metrics' in locals() else 0.0,
                }
            }
            
            # Store in IPFS
            client_ipfs_hash = self.ipfs.add_json(client_data)
            logger.info(f"Stored client data in IPFS: {client_ipfs_hash}")
            
            # Verify data was stored correctly by retrieving it
            verification = self.ipfs.get_json(client_ipfs_hash)
            logger.info(f"Verification - stored data contains keys: {list(verification.keys())}")
            if "ensemble_weights" in verification:
                logger.info(f"Ensemble weights successfully stored, length: {len(verification['ensemble_weights'])}")
            else:
                logger.warning("Ensemble weights missing from stored data!")
        except Exception as e:
            logger.error(f"Failed to save client data to IPFS: {str(e)}")
            client_ipfs_hash = None

        # Update metrics history
        self.metrics_history.append({
            "round": server_round,
            "fit_loss": float(loss),
            "accuracy": float(accuracy),
            "ipfs_hash": client_ipfs_hash,
            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ga_stacking_metrics": ga_stacking_metrics
        })

        # Track participation for future rounds
        self.rounds_participated = server_round

        # Include necessary information in metrics
        metrics["client_ipfs_hash"] = client_ipfs_hash
        metrics["wallet_address"] = self.wallet_address if self.wallet_address else "unknown"
        metrics["ensemble_size"] = len(self.ensemble_state.get("model_names", [])) if (hasattr(self, 'ensemble_state') and self.ensemble_state is not None) else 0
        
        # Instead of directly sending lists in metrics, convert them to strings
        if hasattr(self, 'ensemble_state') and self.ensemble_state is not None:
            metrics["base_model_names_str"] = json.dumps(self.ensemble_state.get("model_names", []))
            metrics["ensemble_weights_str"] = json.dumps(self.ensemble_state.get("weights", []).tolist() 
                                                    if isinstance(self.ensemble_state.get("weights", []), np.ndarray) 
                                                    else self.ensemble_state.get("weights", []))
        else:
            metrics["base_model_names_str"] = json.dumps([])
            metrics["ensemble_weights_str"] = json.dumps([])
        
        # Add GA-Stacking specific metrics (NO AUC)
        for key, value in ga_stacking_metrics.items():
            metrics[key] = value

        # For IPFS-only approach, we only need to send metadata and hash
        # Instead of serializing the full ensemble state
        empty_parameters = [np.array([])]

        # Make sure metrics only contains serializable values
        for key in list(metrics.keys()):
            if not isinstance(metrics[key], (int, float, str, bool, list)):
                metrics[key] = str(metrics[key])

        # Return empty parameters with all information in metrics
        return empty_parameters, len(self.X_train), metrics

    def evaluate(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on the local test dataset using GA-Stacking.
        Uses global model from IPFS when available, with fallback to local model.
        """
        # Check if client is authorized (if blockchain is available)
        if self.blockchain and self.wallet_address:
            try:
                is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
                if not is_authorized:
                    logger.warning(f"Client {self.wallet_address} is not authorized to participate in evaluation")
                    return 0.0, 0, {"error": "client_not_authorized"}
            except Exception as e:
                logger.error(f"Failed to check client authorization: {e}")
        
        # Get configuration from server
        ipfs_hash = config.get("ipfs_hash", None)
        round_num = config.get("server_round", 0)
        is_evaluation_phase = config.get("evaluation_phase", False)
        
        logger.info(f"Starting evaluation for round {round_num}")
        if ipfs_hash:
            logger.info(f"Using model from IPFS: {ipfs_hash}")
        
        # TRY GLOBAL MODEL EVALUATION FIRST (if IPFS hash provided and it's evaluation phase)
        if ipfs_hash and is_evaluation_phase:
            try:
                logger.info("Attempting to evaluate using global ensemble model from IPFS")
                
                # Retrieve the global model from IPFS
                global_model_data = self.ipfs.get_json(ipfs_hash)
                logger.info(f"Retrieved global model data with keys: {list(global_model_data.keys())}")
                
                # Check if this is a proper ensemble model
                if "ensemble_weights" in global_model_data and "base_model_names" in global_model_data:
                    ensemble_weights = global_model_data["ensemble_weights"]
                    base_model_names = global_model_data["base_model_names"]
                    
                    logger.info(f"Evaluating global ensemble with {len(ensemble_weights)} models")
                    logger.info(f"Ensemble weights: {ensemble_weights}")
                    
                    # Ensure we have our trained base models loaded
                    if not hasattr(self, 'base_models_loaded') or not self.base_models_loaded:
                        models_dir = os.path.join("models", f"client_{self.client_id}")
                        if os.path.exists(models_dir):
                            logger.info(f"Loading base models from {models_dir}")
                            self.ga_pipeline.load_base_models(models_dir)
                            self.base_models_loaded = True
                            logger.info(f"Successfully loaded {len(self.ga_pipeline.trained_base_models)} base models")
                    
                    # Verify we have the required base models
                    if hasattr(self.ga_pipeline, 'trained_base_models') and self.ga_pipeline.trained_base_models:
                        # Generate meta-features using our trained base models
                        logger.info("Generating meta-features for global model evaluation")
                        meta_X_test = np.column_stack([
                            model.predict_proba(self.X_test)[:, 1]
                            for model in self.ga_pipeline.trained_base_models.values()
                        ])
                        
                        # Apply global ensemble weights
                        ensemble_predictions = np.dot(meta_X_test, np.array(ensemble_weights))
                        binary_predictions = (ensemble_predictions > 0.5).astype(int)
                        
                        # Print comprehensive evaluation report for global model
                        print(f"\n🌍 GLOBAL MODEL EVALUATION - ROUND {round_num}")
                        global_report = self._print_evaluation_report(
                            "Test (Global Ensemble)", self.y_test, ensemble_predictions, round_num=round_num
                        )
                        
                        # Calculate metrics using global ensemble
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
                        
                        accuracy = accuracy_score(self.y_test, binary_predictions)
                        precision = precision_score(self.y_test, binary_predictions, pos_label=1, zero_division=0)
                        recall = recall_score(self.y_test, binary_predictions, pos_label=1, zero_division=0)
                        f1 = f1_score(self.y_test, binary_predictions, pos_label=1, zero_division=0)
                        
                        # Calculate loss
                        clipped_predictions = np.clip(ensemble_predictions, 1e-7, 1 - 1e-7)
                        loss = log_loss(self.y_test, clipped_predictions)
                        
                        # Calculate confusion matrix
                        cm = confusion_matrix(self.y_test, binary_predictions)
                        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                        
                        # Create metrics dictionary (NO AUC)
                        metrics = {
                            "loss": float(loss),
                            "accuracy": float(accuracy * 100.0),  # Convert to percentage
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1_score": float(f1),
                            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                            
                            # Confusion matrix components
                            "true_positives": int(tp),
                            "false_positives": int(fp),
                            "true_negatives": int(tn),
                            "false_negatives": int(fn),
                            
                            # Global model information
                            "ensemble_size": len(ensemble_weights),
                            "evaluation_method": "global_ensemble",
                            "model_status": "global_model_evaluation",
                            "data_points": len(self.X_train),
                            "global_model_ipfs_hash": ipfs_hash
                        }
                        
                        # Add client contribution info from global model if available
                        if "client_contributions" in global_model_data:
                            contributions = global_model_data["client_contributions"]
                            metrics["global_model_contributors"] = len(contributions)
                            
                            # Find our contribution
                            for contrib in contributions:
                                if contrib.get("wallet_address") == self.wallet_address:
                                    metrics["our_contribution_score"] = float(contrib.get("score", 0.0))
                                    metrics["our_reward_amount"] = float(contrib.get("rewards", {}).get("eth_amount", 0.0))
                                    metrics["our_rank"] = int(contrib.get("rewards", {}).get("rank", 0))
                                    logger.info(f"Our contribution: Score={metrics['our_contribution_score']:.2f}, "
                                            f"Reward={metrics['our_reward_amount']:.6f} ETH, Rank={metrics['our_rank']}")
                                    break
                        
                        # Add reward summary if available
                        if "reward_summary" in global_model_data:
                            summary = global_model_data["reward_summary"]
                            metrics["global_rewards_distributed"] = float(summary.get("total_eth_distributed", 0.0))
                            metrics["global_average_score"] = float(summary.get("average_score", 0.0))
                        
                        # Add to metrics history
                        self.metrics_history.append({
                            "round": round_num,
                            "eval_loss": float(loss),
                            "accuracy": float(accuracy * 100.0),
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1_score": float(f1),
                            "eval_samples": len(self.X_test),
                            "ipfs_hash": ipfs_hash,
                            "evaluation_method": "global_ensemble",
                            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        
                        logger.info(f"Global model evaluation completed - Loss: {loss:.4f}, "
                                f"Precision: {precision:.4f}, "
                                f"Recall: {recall:.4f}, F1: {f1:.4f}")
                        
                        return float(loss), len(self.X_test), metrics
                    
            except Exception as e:
                logger.error(f"Failed to evaluate global model from IPFS: {e}")
                logger.info("Falling back to local model evaluation")
                # Continue to fallback evaluation below
        
        # FALLBACK: LOCAL MODEL EVALUATION (your existing logic)
        logger.info("Using local model evaluation")
        
        # Ensure parameters are in the correct format
        if isinstance(parameters, list):
            params = parameters
        else:
            params = parameters_to_ndarrays(parameters)
        
        # Set parameters if provided
        if len(params) > 0 and len(params[0]) > 0:
            self.set_parameters(params)
        
        # Check if the pipeline is trained
        pipeline_trained = (hasattr(self, 'ga_pipeline') and 
                        self.ga_pipeline is not None and 
                        hasattr(self.ga_pipeline, 'predict') and
                        hasattr(self, 'ensemble_state') and 
                        self.ensemble_state is not None)
        
        if not pipeline_trained:
            logger.warning("GA pipeline not trained yet, returning default evaluation metrics")
            return float('inf'), len(self.X_test), {
                "loss": float('inf'),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "model_status": "not_trained",
                "evaluation_method": "local_not_trained"
            }
        
        # Local model evaluation (your existing code)
        try:
            y_pred = self.ga_pipeline.predict(self.X_test)
            
            # Print comprehensive evaluation report for local model
            print(f"\n🏠 LOCAL MODEL EVALUATION - ROUND {round_num}")
            local_report = self._print_evaluation_report(
                "Test (Local Model)", self.y_test, y_pred, round_num=round_num
            )
            
            test_metrics = evaluate_metrics(self.y_test, y_pred)
            
            accuracy = test_metrics['accuracy'] * 100.0
            clipped_predictions = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = log_loss(self.y_test, clipped_predictions)
            
            # Calculate confusion matrix components
            threshold = 0.5
            y_pred_labels = (y_pred > threshold).astype(int)
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.y_test, y_pred_labels)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                fraud_idx = 1
                tp = cm[fraud_idx, fraud_idx]
                fp = cm[:, fraud_idx].sum() - tp
                fn = cm[fraud_idx, :].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
            
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(test_metrics['precision']),
                "recall": float(test_metrics['recall']),
                "f1_score": float(test_metrics['f1']),
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "ensemble_size": len(self.ensemble_state.get("model_names", [])) if hasattr(self, 'ensemble_state') else 0,
                
                # Confusion matrix components
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                
                # Additional info
                "data_points": len(self.X_train),
                "evaluation_method": "local_model",
                "model_status": "trained"
            }
            
            # Add GA-Stacking metrics if available (NO AUC)
            if hasattr(self, 'ga_results'):
                metrics.update({
                    "ensemble_accuracy": float(self.ga_results["val_metrics"]["accuracy"]),
                    "diversity_score": float(self.ga_results["diversity_score"]),
                    "generalization_score": float(self.ga_results["generalization_score"]),
                    "convergence_rate": float(self.ga_results["convergence_rate"])
                })
            
            # Add to metrics history
            self.metrics_history.append({
                "round": round_num,
                "eval_loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(test_metrics['precision']),
                "recall": float(test_metrics['recall']),
                "f1_score": float(test_metrics['f1']),
                "eval_samples": len(self.X_test),
                "ipfs_hash": ipfs_hash,
                "evaluation_method": "local_model",
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return float(loss), len(self.X_test), metrics
            
        except Exception as e:
            logger.error(f"Error during local evaluation: {e}")
            return float('inf'), len(self.X_test), {
                "loss": float('inf'),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "error": str(e),
                "model_status": "error",
                "evaluation_method": "local_error"
            }
        
    # def fit(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
    #     """
    #     Train the model using GA-Stacking in Round 1 and fine-tune in subsequent rounds.
    #     """
    #     # Check if client is authorized (if blockchain is available)
    #     if self.blockchain and self.wallet_address:
    #         try:
    #             is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
    #             if not is_authorized:
    #                 logger.warning(f"Client {self.wallet_address} is not authorized to participate")
    #                 # Return empty parameters with auth failure flag
    #                 return parameters_to_ndarrays(parameters), 0, {"error": "client_not_authorized"}
    #         except Exception as e:
    #             logger.error(f"Failed to check client authorization: {e}")
    #             # Continue anyway, server will check again

    #     # Get config parameters
    #     server_round = config.get("server_round", 0)
    #     is_first_round = server_round == 1
    #     global_model_ipfs_hash = config.get("global_model_ipfs_hash", None)
        
    #     # Prepare data
    #     X_train, y_train = self.X_train, self.y_train
    #     X_val, y_val = self.X_val, self.y_val
    #     initial_weights = None
    #     # ROUND 1: Train base models + GA-Stacking from scratch
    #     if is_first_round:
    #         logger.info("Round 1: Training base models with GA-Stacking")
    #         try:
    #             # Initialize the GA pipeline with config parameters
    #             self.ga_pipeline = GAStackingPipeline(
    #                 pop_size=10,
    #                 generations=5,
    #                 cv_folds=5,
    #                 verbose=True
    #             )
                
    #             # Generate random initial weights if none provided
    #             if initial_weights is None:
    #                 # Get the number of base models
    #                 num_models = len(self.ga_pipeline.base_models)
    #                 logger.info(f"Generating equal initial weights for {num_models} base models")
                    
    #                 # Set all weights to 1/num_models
    #                 initial_weights = np.full(num_models, 1 / num_models)
                    
    #                 logger.info(f"Generated equal initial weights: {initial_weights}")
                
    #             # Train the GA-Stacking pipeline with initial weights
    #             self.ga_results = self.ga_pipeline.train(
    #                 X_train, y_train, X_val, y_val, 
    #                 init_weights=initial_weights
    #             )
                
    #             # Save ensemble state
    #             self.ensemble_state = self.ga_pipeline.get_ensemble_state()
                
    #             model_dir = os.path.join("models", f"client_{self.client_id}")
    #             # Save base models for future rounds
    #             self.ga_pipeline.save_base_models(model_dir)
                
    #             logger.info(f"Completed GA-Stacking with {len(self.ensemble_state.get('model_names', []))} base models")
    #         except Exception as e:
    #             logger.error(f"Error in GA-Stacking: {str(e)}")
    #             import traceback
    #             traceback.print_exc()
        
    #     # ROUNDS 2+: Fine-tune with aggregated weights
    #     else:
    #         logger.info(f"Round {server_round}: Fine-tuning with aggregated weights")
    #         try:
    #             if global_model_ipfs_hash:
    #                 # Log the hash we're trying to retrieve
    #                 logger.info(f"Retrieving aggregated weights from IPFS hash: {global_model_ipfs_hash}")
                    
    #                 # Retrieve aggregated ensemble weights from IPFS
    #                 try:
    #                     global_model_data = self.ipfs.get_json(global_model_ipfs_hash)
    #                     logger.info(f"Retrieved global model data with keys: {list(global_model_data.keys())}")
                        
    #                     # Extract ensemble weights from global model data with proper fallbacks
    #                     aggregated_weights = None
    #                     base_model_names = None
                        
    #                     # Try different possible locations for ensemble weights
    #                     if "ensemble_weights" in global_model_data:
    #                         aggregated_weights = global_model_data["ensemble_weights"]
    #                         logger.info(f"Found ensemble weights in global model data: {aggregated_weights}")
                        
    #                     # Try different possible locations for base model names
    #                     if "base_model_names" in global_model_data:
    #                         base_model_names = global_model_data["base_model_names"]
    #                         logger.info(f"Found base model names in global model data: {base_model_names}")
    #                     elif "info" in global_model_data and "base_model_names" in global_model_data["info"]:
    #                         base_model_names = global_model_data["info"]["base_model_names"]
    #                         logger.info(f"Found base model names in global model info: {base_model_names}")
                        
    #                     if aggregated_weights and base_model_names:
    #                         # Load saved base models if not already loaded
    #                         if not hasattr(self, 'base_models_loaded') or not self.base_models_loaded:
    #                             models_dir = os.path.join("models", f"client_{self.client_id}")
    #                             logger.info(f"Loading base models from {models_dir}")
                                
    #                             # Load base models using pipeline's method
    #                             self.ga_pipeline.load_base_models(models_dir)
    #                             self.base_models_loaded = True
                                
    #                             # Check if models were loaded successfully
    #                             if hasattr(self.ga_pipeline, 'trained_base_models'):
    #                                 logger.info(f"Loaded {len(self.ga_pipeline.trained_base_models)} trained base models")
    #                             else:
    #                                 logger.error("No trained_base_models found after loading")
    #                                 raise ValueError("Failed to load base models")
                            
    #                         # Check if we have trained base models before fine-tuning
    #                         if not hasattr(self.ga_pipeline, 'trained_base_models') or not self.ga_pipeline.trained_base_models:
    #                             logger.error("No trained_base_models available for fine-tuning")
    #                             raise ValueError("No trained base models available")
                            
    #                         # Log the models we have
    #                         logger.info(f"Fine-tuning with base models: {list(self.ga_pipeline.trained_base_models.keys())}")
                            
    #                         # Ensure weights are in the correct format
    #                         if isinstance(aggregated_weights, list):
    #                             # Convert to numpy array if needed
    #                             aggregated_weights = np.array(aggregated_weights)
                            
    #                         # Now perform the actual fine-tuning - explicitly passing the required parameters
    #                         logger.info(f"Fine-tuning with aggregated weights: {aggregated_weights}")
    #                         self.ga_results = self.ga_pipeline.fine_tune(
    #                             X_train=X_train, 
    #                             y_train=y_train, 
    #                             X_val=X_val, 
    #                             y_val=y_val,
    #                             base_models=self.ga_pipeline.trained_base_models,  # This is required
    #                             weights=aggregated_weights  # Pass the weights
    #                         )
                            
    #                         # Update ensemble state after fine-tuning
    #                         self.ensemble_state = self.ga_pipeline.get_ensemble_state()
                            
    #                         logger.info(f"Completed fine-tuning with {len(aggregated_weights)} base models")
    #                         logger.info(f"Updated ensemble weights: {self.ensemble_state.get('weights', [])}")
    #                     else:
    #                         logger.error(f"Missing required data in global model. Found keys: {list(global_model_data.keys())}")
    #                         if "info" in global_model_data:
    #                             logger.error(f"Info section contains keys: {list(global_model_data['info'].keys())}")
    #                 except Exception as e:
    #                     logger.error(f"Error retrieving global model from IPFS: {str(e)}")
    #                     import traceback
    #                     logger.error(traceback.format_exc())
    #             else:
    #                 logger.error("No global model IPFS hash provided for fine-tuning")
    #         except Exception as e:
    #             logger.error(f"Error in fine-tuning: {str(e)}")
    #             import traceback
    #             logger.error(traceback.format_exc())
          
    #     # Evaluate on test data
    #     if hasattr(self, 'ga_pipeline') and self.ga_pipeline is not None and hasattr(self.ga_pipeline, 'predict'):
    #         y_pred = self.ga_pipeline.predict(self.X_test)
    #         test_metrics = evaluate_metrics(self.y_test, y_pred)
            
    #         accuracy = test_metrics['accuracy'] * 100.0
            
    #         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
                        
    #         loss = log_loss(self.y_test, y_pred_clipped)
            
    #         ga_stacking_metrics = {
    #             # Keep the data_points metric (used in our new reward system)
    #             "data_points": len(self.X_train),  # Use the raw number of training samples
                
    #             # Store recall value directly from test metrics
    #             "recall": float(test_metrics['recall']),  # Recall is critical for fraud detection
                
    #             # Keep other metrics for backward compatibility
    #             "data_size": len(self.X_train),
    #             "training_rounds": server_round if not hasattr(self, 'rounds_participated') else self.rounds_participated + 1,
    #             "evaluation_score": float(test_metrics['f1'])
    #         }
            
    #         metrics = {
    #             "loss": float(loss),
    #             "accuracy": float(accuracy),
    #             "precision": float(test_metrics['precision']),
    #             "recall": float(test_metrics['recall']),
    #             "f1_score": float(test_metrics['f1']),
    #         }
    #     else:
    #         # Fallback if pipeline isn't trained
    #         accuracy, loss = 0.0, float('inf')
    #         metrics = {
    #             "loss": float(loss),
    #             "accuracy": 0.0,
    #             "precision": 0.0,
    #             "recall": 0.0,
    #             "f1_score": 0.0
    #         }
    #         ga_stacking_metrics = {
    #             # Keep the data_points metric (used in our new reward system)
    #             "data_points": len(self.X_train),  # Use the raw number of training samples
                
    #             # Store recall value directly from test metrics
    #             "recall": float(test_metrics['recall']),  # Recall is critical for fraud detection
                
    #             # Keep other metrics for backward compatibility
    #             "data_size": len(self.X_train),
    #             "training_rounds": server_round if not hasattr(self, 'rounds_participated') else self.rounds_participated + 1,
    #             "evaluation_score": float(test_metrics['f1'])
    #         }

    #     # Create and store client data in IPFS
    #     try:
    #         # Make sure ensemble weights and base model names are properly formatted
    #         ensemble_weights = self.ensemble_state.get("weights", [])
    #         if isinstance(ensemble_weights, np.ndarray):
    #             ensemble_weights = ensemble_weights.tolist()
            
    #         base_model_names = self.ensemble_state.get("model_names", [])
            
    #         # Create IPFS data structure that matches server expectations
    #         client_data = {
    #             "ensemble_weights": ensemble_weights,  # Store directly in root
    #             "base_model_names": base_model_names,  # Store directly in root
    #             "info": {
    #                 "round": server_round,
    #                 "timestamp": datetime.now(timezone.utc).isoformat(),
    #                 "version": "client-" + str(self.client_id),
    #                 "client_id": self.client_id,
    #                 "wallet_address": self.wallet_address if self.wallet_address else "unknown",
    #                 "data_size": len(self.X_train),
    #                 "base_model_names": base_model_names  # Also store in info for server compatibility
    #             },
    #             "metrics": {
    #                 "loss": float(loss),
    #                 "recall": float(metrics.get('recall', 0.0)) if 'metrics' in locals() else 0.0,
    #                 "precision": float(metrics.get('precision', 0.0)) if 'metrics' in locals() else 0.0,
    #                 "f1_score": float(metrics.get('f1_score', 0.0)) if 'metrics' in locals() else 0.0,
    #             }
    #         }
            
    #         # Store in IPFS
    #         client_ipfs_hash = self.ipfs.add_json(client_data)
    #         logger.info(f"Stored client data in IPFS: {client_ipfs_hash}")
            
    #         # Verify data was stored correctly by retrieving it
    #         verification = self.ipfs.get_json(client_ipfs_hash)
    #         logger.info(f"Verification - stored data contains keys: {list(verification.keys())}")
    #         if "ensemble_weights" in verification:
    #             logger.info(f"Ensemble weights successfully stored, length: {len(verification['ensemble_weights'])}")
    #         else:
    #             logger.warning("Ensemble weights missing from stored data!")
    #     except Exception as e:
    #         logger.error(f"Failed to save client data to IPFS: {str(e)}")
    #         client_ipfs_hash = None

    #     # Update metrics history
    #     self.metrics_history.append({
    #         "round": server_round,
    #         "fit_loss": float(loss),
    #         "accuracy": float(accuracy),
    #         "ipfs_hash": client_ipfs_hash,
    #         "wallet_address": self.wallet_address if self.wallet_address else "unknown",
    #         "timestamp": datetime.now(timezone.utc).isoformat(),
    #         "ga_stacking_metrics": ga_stacking_metrics
    #     })

    #     # Track participation for future rounds
    #     self.rounds_participated = server_round

    #     # Include necessary information in metrics
    #     metrics["client_ipfs_hash"] = client_ipfs_hash
    #     metrics["wallet_address"] = self.wallet_address if self.wallet_address else "unknown"
    #     metrics["ensemble_size"] = len(self.ensemble_state.get("model_names", [])) if (hasattr(self, 'ensemble_state') and self.ensemble_state is not None) else 0
        
    #     # Instead of directly sending lists in metrics, convert them to strings
    #     if hasattr(self, 'ensemble_state') and self.ensemble_state is not None:
    #         metrics["base_model_names_str"] = json.dumps(self.ensemble_state.get("model_names", []))
    #         metrics["ensemble_weights_str"] = json.dumps(self.ensemble_state.get("weights", []).tolist() 
    #                                                 if isinstance(self.ensemble_state.get("weights", []), np.ndarray) 
    #                                                 else self.ensemble_state.get("weights", []))
    #     else:
    #         metrics["base_model_names_str"] = json.dumps([])
    #         metrics["ensemble_weights_str"] = json.dumps([])
        
    #     # Add GA-Stacking specific metrics
    #     for key, value in ga_stacking_metrics.items():
    #         metrics[key] = value

    #     # For IPFS-only approach, we only need to send metadata and hash
    #     # Instead of serializing the full ensemble state
    #     empty_parameters = [np.array([])]

    #     # Make sure metrics only contains serializable values
    #     for key in list(metrics.keys()):
    #         if not isinstance(metrics[key], (int, float, str, bool, list)):
    #             metrics[key] = str(metrics[key])

    #     # Return empty parameters with all information in metrics
    #     return empty_parameters, len(self.X_train), metrics
    
    # def evaluate(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
    #     """
    #     Evaluate the model on the local test dataset using GA-Stacking.
    #     Uses global model from IPFS when available, with fallback to local model.
    #     """
    #     # Check if client is authorized (if blockchain is available)
    #     if self.blockchain and self.wallet_address:
    #         try:
    #             is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
    #             if not is_authorized:
    #                 logger.warning(f"Client {self.wallet_address} is not authorized to participate in evaluation")
    #                 return 0.0, 0, {"error": "client_not_authorized"}
    #         except Exception as e:
    #             logger.error(f"Failed to check client authorization: {e}")
        
    #     # Get configuration from server
    #     ipfs_hash = config.get("ipfs_hash", None)
    #     round_num = config.get("server_round", 0)
    #     is_evaluation_phase = config.get("evaluation_phase", False)
        
    #     logger.info(f"Starting evaluation for round {round_num}")
    #     if ipfs_hash:
    #         logger.info(f"Using model from IPFS: {ipfs_hash}")
        
    #     # TRY GLOBAL MODEL EVALUATION FIRST (if IPFS hash provided and it's evaluation phase)
    #     if ipfs_hash and is_evaluation_phase:
    #         try:
    #             logger.info("Attempting to evaluate using global ensemble model from IPFS")
                
    #             # Retrieve the global model from IPFS
    #             global_model_data = self.ipfs.get_json(ipfs_hash)
    #             logger.info(f"Retrieved global model data with keys: {list(global_model_data.keys())}")
                
    #             # Check if this is a proper ensemble model
    #             if "ensemble_weights" in global_model_data and "base_model_names" in global_model_data:
    #                 ensemble_weights = global_model_data["ensemble_weights"]
    #                 base_model_names = global_model_data["base_model_names"]
                    
    #                 logger.info(f"Evaluating global ensemble with {len(ensemble_weights)} models")
    #                 logger.info(f"Ensemble weights: {ensemble_weights}")
                    
    #                 # Ensure we have our trained base models loaded
    #                 if not hasattr(self, 'base_models_loaded') or not self.base_models_loaded:
    #                     models_dir = os.path.join("models", f"client_{self.client_id}")
    #                     if os.path.exists(models_dir):
    #                         logger.info(f"Loading base models from {models_dir}")
    #                         self.ga_pipeline.load_base_models(models_dir)
    #                         self.base_models_loaded = True
    #                         logger.info(f"Successfully loaded {len(self.ga_pipeline.trained_base_models)} base models")
                    
    #                 # Verify we have the required base models
    #                 if hasattr(self.ga_pipeline, 'trained_base_models') and self.ga_pipeline.trained_base_models:
    #                     # Generate meta-features using our trained base models
    #                     logger.info("Generating meta-features for global model evaluation")
    #                     meta_X_test = np.column_stack([
    #                         model.predict_proba(self.X_test)[:, 1]
    #                         for model in self.ga_pipeline.trained_base_models.values()
    #                     ])
                        
    #                     # Apply global ensemble weights
    #                     ensemble_predictions = np.dot(meta_X_test, np.array(ensemble_weights))
    #                     binary_predictions = (ensemble_predictions > 0.5).astype(int)
                        
    #                     # Calculate metrics using global ensemble
    #                     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
                        
    #                     accuracy = accuracy_score(self.y_test, binary_predictions)
    #                     precision = precision_score(self.y_test, binary_predictions, pos_label=1, zero_division=0)
    #                     recall = recall_score(self.y_test, binary_predictions, pos_label=1, zero_division=0)
    #                     f1 = f1_score(self.y_test, binary_predictions, pos_label=1, zero_division=0)
                        
    #                     # Calculate loss
    #                     clipped_predictions = np.clip(ensemble_predictions, 1e-7, 1 - 1e-7)
    #                     loss = log_loss(self.y_test, clipped_predictions)
                        
    #                     # Calculate confusion matrix
    #                     cm = confusion_matrix(self.y_test, binary_predictions)
    #                     tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                        
    #                     # Create metrics dictionary
    #                     metrics = {
    #                         "loss": float(loss),
    #                         "accuracy": float(accuracy * 100.0),  # Convert to percentage
    #                         "precision": float(precision),
    #                         "recall": float(recall),
    #                         "f1_score": float(f1),
    #                         "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                            
    #                         # Confusion matrix components
    #                         "true_positives": int(tp),
    #                         "false_positives": int(fp),
    #                         "true_negatives": int(tn),
    #                         "false_negatives": int(fn),
                            
    #                         # Global model information
    #                         "ensemble_size": len(ensemble_weights),
    #                         "evaluation_method": "global_ensemble",
    #                         "model_status": "global_model_evaluation",
    #                         "data_points": len(self.X_train),
    #                         "global_model_ipfs_hash": ipfs_hash
    #                     }
                        
    #                     # Add client contribution info from global model if available
    #                     if "client_contributions" in global_model_data:
    #                         contributions = global_model_data["client_contributions"]
    #                         metrics["global_model_contributors"] = len(contributions)
                            
    #                         # Find our contribution
    #                         for contrib in contributions:
    #                             if contrib.get("wallet_address") == self.wallet_address:
    #                                 metrics["our_contribution_score"] = float(contrib.get("score", 0.0))
    #                                 metrics["our_reward_amount"] = float(contrib.get("rewards", {}).get("eth_amount", 0.0))
    #                                 metrics["our_rank"] = int(contrib.get("rewards", {}).get("rank", 0))
    #                                 logger.info(f"Our contribution: Score={metrics['our_contribution_score']:.2f}, "
    #                                         f"Reward={metrics['our_reward_amount']:.6f} ETH, Rank={metrics['our_rank']}")
    #                                 break
                        
    #                     # Add reward summary if available
    #                     if "reward_summary" in global_model_data:
    #                         summary = global_model_data["reward_summary"]
    #                         metrics["global_rewards_distributed"] = float(summary.get("total_eth_distributed", 0.0))
    #                         metrics["global_average_score"] = float(summary.get("average_score", 0.0))
                        
    #                     # Add to metrics history
    #                     self.metrics_history.append({
    #                         "round": round_num,
    #                         "eval_loss": float(loss),
    #                         "accuracy": float(accuracy * 100.0),
    #                         "precision": float(precision),
    #                         "recall": float(recall),
    #                         "f1_score": float(f1),
    #                         "eval_samples": len(self.X_test),
    #                         "ipfs_hash": ipfs_hash,
    #                         "evaluation_method": "global_ensemble",
    #                         "wallet_address": self.wallet_address if self.wallet_address else "unknown",
    #                         "timestamp": datetime.now(timezone.utc).isoformat()
    #                     })
                        
    #                     logger.info(f"Global model evaluation completed - Loss: {loss:.4f}, "
    #                             f"Precision: {precision:.4f}, "
    #                             f"Recall: {recall:.4f}, F1: {f1:.4f}")
                        
    #                     return float(loss), len(self.X_test), metrics
                    
    #         except Exception as e:
    #             logger.error(f"Failed to evaluate global model from IPFS: {e}")
    #             logger.info("Falling back to local model evaluation")
    #             # Continue to fallback evaluation below
        
    #     # FALLBACK: LOCAL MODEL EVALUATION (your existing logic)
    #     logger.info("Using local model evaluation")
        
    #     # Ensure parameters are in the correct format
    #     if isinstance(parameters, list):
    #         params = parameters
    #     else:
    #         params = parameters_to_ndarrays(parameters)
        
    #     # Set parameters if provided
    #     if len(params) > 0 and len(params[0]) > 0:
    #         self.set_parameters(params)
        
    #     # Check if the pipeline is trained
    #     pipeline_trained = (hasattr(self, 'ga_pipeline') and 
    #                     self.ga_pipeline is not None and 
    #                     hasattr(self.ga_pipeline, 'predict') and
    #                     hasattr(self, 'ensemble_state') and 
    #                     self.ensemble_state is not None)
        
    #     if not pipeline_trained:
    #         logger.warning("GA pipeline not trained yet, returning default evaluation metrics")
    #         return float('inf'), len(self.X_test), {
    #             "loss": float('inf'),
    #             "accuracy": 0.0,
    #             "precision": 0.0,
    #             "recall": 0.0,
    #             "f1_score": 0.0,
    #             "wallet_address": self.wallet_address if self.wallet_address else "unknown",
    #             "model_status": "not_trained",
    #             "evaluation_method": "local_not_trained"
    #         }
        
    #     # Local model evaluation (your existing code)
    #     try:
    #         y_pred = self.ga_pipeline.predict(self.X_test)
    #         test_metrics = evaluate_metrics(self.y_test, y_pred)
            
    #         accuracy = test_metrics['accuracy'] * 100.0
    #         clipped_predictions = np.clip(y_pred, 1e-7, 1 - 1e-7)
    #         loss = log_loss(self.y_test, clipped_predictions)
            
    #         # Calculate confusion matrix components
    #         threshold = 0.5
    #         y_pred_labels = (y_pred > threshold).astype(int)
            
    #         from sklearn.metrics import confusion_matrix
    #         cm = confusion_matrix(self.y_test, y_pred_labels)
            
    #         if cm.shape == (2, 2):
    #             tn, fp, fn, tp = cm.ravel()
    #         else:
    #             fraud_idx = 1
    #             tp = cm[fraud_idx, fraud_idx]
    #             fp = cm[:, fraud_idx].sum() - tp
    #             fn = cm[fraud_idx, :].sum() - tp
    #             tn = cm.sum() - (tp + fp + fn)
            
    #         metrics = {
    #             "loss": float(loss),
    #             "accuracy": float(accuracy),
    #             "precision": float(test_metrics['precision']),
    #             "recall": float(test_metrics['recall']),
    #             "f1_score": float(test_metrics['f1']),
    #             "wallet_address": self.wallet_address if self.wallet_address else "unknown",
    #             "ensemble_size": len(self.ensemble_state.get("model_names", [])) if hasattr(self, 'ensemble_state') else 0,
                
    #             # Confusion matrix components
    #             "true_positives": int(tp),
    #             "false_positives": int(fp),
    #             "true_negatives": int(tn),
    #             "false_negatives": int(fn),
                
    #             # Additional info
    #             "data_points": len(self.X_train),
    #             "evaluation_method": "local_model",
    #             "model_status": "trained"
    #         }
            
    #         # Add GA-Stacking metrics if available
    #         if hasattr(self, 'ga_results'):
    #             metrics.update({
    #                 "ensemble_accuracy": float(self.ga_results["val_metrics"]["accuracy"]),
    #                 "diversity_score": float(self.ga_results["diversity_score"]),
    #                 "generalization_score": float(self.ga_results["generalization_score"]),
    #                 "convergence_rate": float(self.ga_results["convergence_rate"])
    #             })
            
    #         # Add to metrics history
    #         self.metrics_history.append({
    #             "round": round_num,
    #             "eval_loss": float(loss),
    #             "accuracy": float(accuracy),
    #             "precision": float(test_metrics['precision']),
    #             "recall": float(test_metrics['recall']),
    #             "f1_score": float(test_metrics['f1']),
    #             "eval_samples": len(self.X_test),
    #             "ipfs_hash": ipfs_hash,
    #             "evaluation_method": "local_model",
    #             "wallet_address": self.wallet_address if self.wallet_address else "unknown",
    #             "timestamp": datetime.now(timezone.utc).isoformat()
    #         })
            
    #         return float(loss), len(self.X_test), metrics
            
    #     except Exception as e:
    #         logger.error(f"Error during local evaluation: {e}")
    #         return float('inf'), len(self.X_test), {
    #             "loss": float('inf'),
    #             "accuracy": 0.0,
    #             "precision": 0.0,
    #             "recall": 0.0,
    #             "f1_score": 0.0,
    #             "wallet_address": self.wallet_address if self.wallet_address else "unknown",
    #             "error": str(e),
    #             "model_status": "error",
    #             "evaluation_method": "local_error"
    #         }
    
    def save_metrics_history(self, filepath: str = "client_metrics.json"):
        """Save metrics history to a file."""
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Saved metrics history to {filepath}")


def load_csv_dataset(csv_file_path: str, target_col: str = "Class") -> pd.DataFrame:
    """
    Load a dataset from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
        target_col: Name of the target column
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_file_path}")
        
        # Load the CSV file
        data = pd.read_csv(csv_file_path)
        
        # Ensure the target column exists
        if target_col not in data.columns:
            if 'Class' in data.columns:
                target_col = 'Class'
                logger.warning(f"Target column '{target_col}' not found, using 'Class' instead")
            else:
                # Try to identify the target column based on common names
                potential_targets = ['target', 'label', 'y', 'class', 'fraud', 'is_fraud']
                for col in potential_targets:
                    if col in data.columns:
                        target_col = col
                        logger.warning(f"Target column not found, using '{target_col}' instead")
                        break
                else:
                    # If no known target column is found, use the last column
                    target_col = data.columns[-1]
                    logger.warning(f"Target column not found, using last column '{target_col}' as target")
        
        # Check if 'Time' column exists and add it if not
        if 'Time' not in data.columns:
            data['Time'] = range(len(data))
            logger.info("Added 'Time' column to dataset")
            
        logger.info(f"Successfully loaded dataset with {len(data)} samples and {len(data.columns)} features")
        logger.info(f"Target column: {target_col}, Class distribution: {data[target_col].value_counts().to_dict()}")
        
        return data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def start_client(
    server_address: str = "192.168.80.180:8088",
    ipfs_url: str = "http://127.0.0.1:5001",
    ganache_url: str = "http://127.0.0.1:7545",
    contract_address: Optional[str] = None,
    wallet_address: Optional[str] = None,
    private_key: Optional[str] = None,
    client_id: Optional[str] = None,
    ensemble_size: int = 8,
    ga_generations: int = 3,
    ga_population_size: int = 8,
    csv_file: Optional[str] = None,
    target_column: str = "Class",
    test_size: float = 0.2
) -> None:
    """
    Start a federated learning client with GA-Stacking ensemble optimization.
    
    Args:
        server_address: Server address (host:port)
        ipfs_url: IPFS API URL
        ganache_url: Ganache blockchain URL
        contract_address: Address of deployed EnhancedModelRegistry contract
        wallet_address: Client's Ethereum wallet address
        private_key: Client's private key (for signing transactions)
        client_id: Client identifier
        ensemble_size: Number of models in the ensemble
        ga_generations: Number of GA generations to run
        ga_population_size: Size of GA population
        csv_file: Path to the CSV data file
        target_column: Name of the target column in the CSV file
        test_size: Proportion of data to use for testing
    """
    # Extract client ID from CSV file path if not explicitly provided
    if client_id is None and csv_file is not None:
        # Try to extract client ID from the filename
        filename = os.path.basename(csv_file)
        if filename.startswith("client_"):
            # Extract client_X from filename like "client_1.csv"
            client_id = filename.split(".")[0]
            logger.info(f"Extracted client ID from CSV file: {client_id}")
        else:
            # Use a default client ID with timestamp to avoid collisions
            timestamp = int(time.time())
            client_id = f"client_{timestamp}"
            logger.info(f"Using generated client ID: {client_id}")
    elif client_id is None:
        # Default client ID if no CSV file or explicit ID provided
        client_id = f"client_{os.getpid()}"
        logger.info(f"Using process-based client ID: {client_id}")
    
    # Create metrics directory
    os.makedirs(f"metrics/{client_id}", exist_ok=True)
    
    # Initialize IPFS connector
    ipfs_connector = IPFSConnector(ipfs_api_url=ipfs_url)
    logger.info(f"Initialized IPFS connector: {ipfs_url}")
    
    # Initialize blockchain connector if contract address is provided
    blockchain_connector = None
    if contract_address:
        try:
            blockchain_connector = BlockchainConnector(
                ganache_url=ganache_url,
                contract_address=contract_address,
                private_key=private_key
            )
            logger.info(f"Initialized blockchain connector: {ganache_url}")
            logger.info(f"Using contract at: {contract_address}")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connector: {e}")
            logger.warning("Continuing without blockchain features")
    
    # Determine dataset file to load
    if csv_file is None:
        # If file isn't explicitly provided, derive from client ID
        derived_csv_file = f"{client_id}.csv"
        
        # Check if derived file exists
        if not os.path.exists(derived_csv_file):
            logger.warning(f"Derived dataset file not found: {derived_csv_file}")
            
            # Try to find in data directory
            data_dir_csv = f"data/{client_id}.csv"
            
            if os.path.exists(data_dir_csv):
                logger.info(f"Found dataset file in data directory")
                csv_file = data_dir_csv
            else:
                # Fall back to default client_0.csv file
                logger.warning(f"Using default dataset file")
                csv_file = "client_0.csv"
                
                # Check if default file exists in data directory
                if not os.path.exists(csv_file):
                    data_dir_default_csv = "data/client_0.csv"
                    
                    if os.path.exists(data_dir_default_csv):
                        csv_file = data_dir_default_csv
                    else:
                        raise FileNotFoundError(f"No dataset file found for {client_id}")
        else:
            csv_file = derived_csv_file
    
    logger.info(f"Using dataset file: {csv_file}")
    
    # Load and process dataset
    try:
        # Load CSV data
        data = load_csv_dataset(csv_file, target_column)
        
        # Use split_and_scale function to get train/val/test splits
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(
            data=data,
            target_col=target_column,
            test_size=test_size,
            random_state=42
        )
        
        logger.info(f"Data preprocessing complete:")
        logger.info(f"- Training set: {X_train.shape[0]} samples")
        logger.info(f"- Validation set: {X_val.shape[0]} samples")
        logger.info(f"- Test set: {X_test.shape[0]} samples")
        
        # For client interface, combine train and validation for X_train, y_train
        # The client's fit() method will split them again for GA-Stacking
        #X_train_combined = np.vstack([X_train, X_val])
        #y_train_combined = np.concatenate([y_train, y_val])
        
        #logger.info(f"Combined training set: {X_train_combined.shape[0]} samples")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise
    
    # Create client
    client = GAStackingClient(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        ensemble_size=ensemble_size,
        ipfs_connector=ipfs_connector,
        blockchain_connector=blockchain_connector,
        wallet_address=wallet_address,
        private_key=private_key,
        client_id=client_id,
        ga_generations=ga_generations,
        ga_population_size=ga_population_size
    )
    
    # Start client
    fl.client.start_client(server_address=server_address, client=client)
    
    # Save metrics after client finishes
    client.save_metrics_history(filepath=f"metrics/{client_id}/metrics_history.json")
    
    logger.info(f"Client {client_id} completed federated learning with GA-Stacking")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start FL client with GA-Stacking ensemble optimization")
    parser.add_argument("--server-address", type=str, default="192.168.80.180:8088", help="Server address (host:port)")
    parser.add_argument("--ipfs-url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--ganache-url", type=str, default="http://192.168.80.1:7545", help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, help="Address of deployed EnhancedModelRegistry contract")
    parser.add_argument("--wallet-address", type=str, help="Client's Ethereum wallet address")
    parser.add_argument("--private-key", type=str, help="Client's private key (for signing transactions)")
    parser.add_argument("--client-id", type=str, help="Client identifier")
    parser.add_argument("--ensemble-size", type=int, default=8, help="Number of models in the ensemble")
    parser.add_argument("--ga-generations", type=int, default=3, help="Number of GA generations to run")
    parser.add_argument("--ga-population-size", type=int, default=10, help="Size of GA population")
    parser.add_argument("--csv-file", type=str, help="Path to CSV data file")
    parser.add_argument("--target-column", type=str, default="Class", help="Name of target column in CSV file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for testing")
    
    # For backward compatibility
    parser.add_argument("--train-file", type=str, help="[DEPRECATED] Path to training data file")
    parser.add_argument("--test-file", type=str, help="[DEPRECATED] Path to test data file")
    args = parser.parse_args()
    
    # Print warning if using deprecated arguments
    if args.train_file or args.test_file:
        print("Warning: --train-file and --test-file are deprecated. Please use --csv-file instead.")
        print("Check our documentation for information on using CSV files with the updated client.")
    
    # Check if contract address is stored in file
    if args.contract_address is None:
        try:
            with open("contract_address.txt", "r") as f:
                args.contract_address = f.read().strip()
                print(f"Loaded contract address from file: {args.contract_address}")
        except FileNotFoundError:
            print("No contract address provided or found in file")
    
    # Handle backward compatibility for train/test files
    csv_file = args.csv_file
    if csv_file is None and args.train_file:
        print(f"Using train file in backward compatibility mode: {args.train_file}")
        # Here you would add logic to convert train/test files to a single CSV
        # For now, just print a warning
        print("Please convert your data to CSV format for best compatibility.")
    
    start_client(
        server_address=args.server_address,
        ipfs_url=args.ipfs_url,
        ganache_url=args.ganache_url,
        contract_address=args.contract_address,
        wallet_address=args.wallet_address,
        private_key=args.private_key,
        client_id=args.client_id,
        ensemble_size=args.ensemble_size,
        ga_generations=args.ga_generations,
        ga_population_size=args.ga_population_size,
        csv_file=args.csv_file,
        target_column=args.target_column,
        test_size=args.test_size
    )
