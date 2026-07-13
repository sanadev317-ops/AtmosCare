# ============================================================================
# GRU MODEL MODULE
# ============================================================================
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from .seq2seq_gru import (
    build_seq2seq_attention_gru,
    mc_dropout_predict,
    toggle_mc_dropout,
)

# Optional: Logger configuration (uncomment if using your custom logger)
# from ..utils import get_logger
# logger = get_logger(__name__)


class CompatBatchNormalization(tf.keras.layers.BatchNormalization):
    """Backward-compatible BatchNormalization that ignores legacy renorm args."""

    def __init__(self, *args, renorm=None, renorm_clipping=None, renorm_momentum=None, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(*args, **kwargs)


def _patch_batch_norm_init() -> None:
    """Patch the live Keras BatchNormalization class to ignore legacy kwargs."""
    try:
        import keras.src.layers.normalization.batch_normalization as bn_module

        original_init = bn_module.BatchNormalization.__init__

        def compat_init(self, *args, **kwargs):
            kwargs.pop("renorm", None)
            kwargs.pop("renorm_clipping", None)
            kwargs.pop("renorm_momentum", None)
            return original_init(self, *args, **kwargs)

        bn_module.BatchNormalization.__init__ = compat_init
        keras.layers.BatchNormalization = bn_module.BatchNormalization
        tf.keras.layers.BatchNormalization = bn_module.BatchNormalization
    except Exception:
        pass

class GRUModel:
    """
    Two-layer stacked GRU for PM2.5 forecasting.
    Supports single-step and 30-day multi-step forecasting.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.gru_config = config.get("gru_model", {})
        self.target = config.get("features", {}).get("target", "pm2_5")
        
        self.sequence_length = self.gru_config.get("sequence_length", 14)
        self.forecast_horizon = self.gru_config.get("forecast_horizon", 1)
        self.gru_units = self.gru_config.get("gru_units", [128, 64])
        self.dense_units = self.gru_config.get("dense_units", [64, 32])
        self.dropout_rate = self.gru_config.get("dropout_rate", 0.05)
        self.learning_rate = self.gru_config.get("learning_rate", 0.001)
        self.loss_fn = self.gru_config.get("loss", "huber")
        
        self.model = None
        self.history = None
        self.n_features = None
    
    def build(self, n_features: int) -> tf.keras.Model:
        """Build an encoder-decoder GRU with Bahdanau attention."""
        self.n_features = n_features
        self.model = build_seq2seq_attention_gru(
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            n_features=n_features,
            gru_units=self.gru_units,
            dense_units=self.dense_units,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            loss=self.loss_fn,
            name="Seq2Seq_GRU_Attention_SmogPredictor",
        )
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              patience: Optional[int] = None,
              checkpoint_dir: Optional[str] = None) -> tf.keras.callbacks.History:
        
        epochs = epochs or self.gru_config.get("epochs", 100)
        batch_size = batch_size or self.gru_config.get("batch_size", 16)
        patience = patience or self.gru_config.get("patience", 15)
        
        checkpoint_path = self._get_checkpoint_path(checkpoint_dir or "artifacts/models/gru")
        callbacks = self._build_callbacks(checkpoint_path, patience)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained GRU model."""
        if self.model is None:
            raise ValueError("Model has not been built or loaded.")
        preds = self.model.predict(X, verbose=0)
        return np.asarray(preds, dtype=float)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Return MC-dropout mean and 95% interval bounds."""
        if self.model is None:
            raise ValueError("Model has not been built or loaded.")

        mean, lower, upper, samples = mc_dropout_predict(self.model, X, n_samples=n_samples)
        return {
            "prediction": np.asarray(mean, dtype=float),
            "lower_bound": np.asarray(lower, dtype=float),
            "upper_bound": np.asarray(upper, dtype=float),
            "samples": np.asarray(samples, dtype=float),
        }

    def fine_tune(self, X_old, y_old, X_new, y_new, X_val, y_val, 
                  replay_ratio=0.2, freeze_base_layers=True, epochs=None):
        if freeze_base_layers:
            self._freeze_base_layers()
        self._reduce_learning_rate(factor=0.1)
        X_train, y_train = self._prepare_replay_data(X_old, y_old, X_new, y_new, replay_ratio)
        return self.train(X_train, y_train, X_val, y_val, epochs=epochs)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Comprehensive model evaluation with multiple metrics."""
        y_pred = self.predict(X_test)
        
        # Calculate metrics for each forecast horizon
        metrics = {}
        for horizon in range(self.forecast_horizon):
            y_true_h = y_test[:, horizon]
            y_pred_h = y_pred[:, horizon]
            
            mse = mean_squared_error(y_true_h, y_pred_h)
            mae = mean_absolute_error(y_true_h, y_pred_h)
            rmse = np.sqrt(mse)
            
            # Avoid division by zero in MAPE
            mask = y_true_h != 0
            mape = np.mean(np.abs((y_true_h[mask] - y_pred_h[mask]) / y_true_h[mask])) * 100
            
            # R² score
            ss_res = np.sum((y_true_h - y_pred_h) ** 2)
            ss_tot = np.sum((y_true_h - np.mean(y_true_h)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics[f'day_{horizon+1}'] = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2
            }
        
        # Overall metrics (averaged across horizons)
        overall_rmse = np.mean([m['rmse'] for m in metrics.values()])
        overall_mae = np.mean([m['mae'] for m in metrics.values()])
        overall_mape = np.mean([m['mape'] for m in metrics.values()])
        overall_r2 = np.mean([m['r2'] for m in metrics.values()])
        
        metrics['overall'] = {
            'rmse': overall_rmse,
            'mae': overall_mae,
            'mape': overall_mape,
            'r2': overall_r2
        }
        
        return metrics

    def save(self, model_dir: str) -> str:
        """Saves model in modern .keras format with timestamp."""
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(model_dir, f"gru_{timestamp}.keras")
        self.model.save(path)
        return path

    @classmethod
    def load(cls, path: str, config: Dict) -> "GRUModel":
        """Loads a .keras model and syncs properties."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        obj = cls(config)
        _patch_batch_norm_init()
        obj.model = tf.keras.models.load_model(
            path,
            compile=False,
        )
        obj.n_features = obj.model.input_shape[-1]
        obj.sequence_length = obj.model.input_shape[1]
        obj.forecast_horizon = obj.model.output_shape[-1]
        return obj

    def _freeze_base_layers(self) -> None:
        for layer in self.model.layers:
            if "encoder_gru_1" in layer.name:
                layer.trainable = False
        self.model.compile(optimizer=self.model.optimizer, loss=self.loss_fn, metrics=["mae"])

    def _reduce_learning_rate(self, factor: float = 0.1) -> None:
        old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        new_lr = max(old_lr * factor, 1e-6)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)

    def _get_checkpoint_path(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, f"gru_best_{datetime.now():%Y%m%d_%H%M%S}.keras")

    def _build_callbacks(self, checkpoint_path: Optional[str], patience: int) -> list:
        """Build comprehensive training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor="val_loss", 
                patience=patience, 
                restore_best_weights=True,
                min_delta=1e-4  # Minimum change to qualify as improvement
            ),
            ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5,  # More aggressive reduction
                patience=patience//3,  # Reduce LR more frequently
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_path, 
                    monitor="val_loss", 
                    save_best_only=True,
                    save_weights_only=False,
                    mode='min',
                    verbose=1
                )
            )
        
        return callbacks

    def _prepare_replay_data(self, X_old, y_old, X_new, y_new, replay_ratio):
        if replay_ratio <= 0 or len(X_old) == 0: return X_new, y_new
        old_size = min(int(len(X_new) * replay_ratio), len(X_old))
        idx = np.random.choice(len(X_old), old_size, replace=False)
        return self._shuffle(np.concatenate([X_old[idx], X_new]), np.concatenate([y_old[idx], y_new]))
