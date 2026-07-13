# Smog Prediction - Production ML Pipeline

**A comprehensive, modular, and reproducible machine learning system for PM2.5 forecasting in Pakistan.**

## 🏗️ Architecture

```
ml_pipeline/
├── data/                      # Data loading, preprocessing, feature engineering
│   ├── loader.py             # Data loading from CSV/API
│   ├── preprocessing.py      # Scaling, train/val/test split
│   └── feature_engineering.py # Lag features, rolling stats, interactions
├── models/                    # ML models (GRU, SARIMA, Hybrid)
│   ├── gru.py               # TensorFlow/Keras GRU model
│   ├── sarima.py            # SARIMA with online learning
│   └── hybrid.py            # Ensemble blending
├── training/                  # Training orchestration
│   ├── train.py             # Main training pipeline
│   ├── validation.py        # Walk-forward validation strategies
│   └── tuning.py            # Optuna hyperparameter tuning
├── evaluation/               # Model evaluation
│   ├── metrics.py           # RMSE, MAE, R², MAPE, directional accuracy
│   └── plots.py             # Publication-quality plots
├── inference/                # Prediction serving
│   └── predictor.py         # Production predictor class
├── api/                       # FastAPI REST service
│   └── api.py               # OpenAPI endpoints
├── config/                    # Configuration
│   └── config.yaml          # All hyperparameters and settings
├── utils/                     # Utilities
│   ├── logger.py            # Structured logging
│   ├── helpers.py           # Common utilities
│   └── __init__.py
├── train_pipeline.py         # Training entrypoint
└── run_api.py               # API server entrypoint
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and navigate
cd e:\Sana\smog

# Create virtual environment
python -m venv ml_env
ml_env\Scripts\activate

# Install dependencies
pip install -r requirements-production.txt
```

### 2. Configuration

Edit `ml_pipeline/config/config.yaml` to customize:
- Data paths and preprocessing
- Model architectures (GRU units, SARIMA order)
- Training hyperparameters
- Validation strategy (walk-forward, rolling window)
- Feature engineering settings
- Output directories

### 3. Train Models

```bash
# Run end-to-end training pipeline
python ml_pipeline/train_pipeline.py
```

This will:
- Load data from CSV
- Engineer features chronologically (no leakage)
- Split data (70% train, 15% val, 15% test)
- Train GRU model (with early stopping)
- Train SARIMA model
- Create hybrid ensemble
- Save models and results to `artifacts/results/run_YYYYMMDD_HHMMSS/`

### 4. Run Inference API

```bash
# Start FastAPI server
python ml_pipeline/run_api.py \
  --host 0.0.0.0 \
  --port 8000 \
  --models artifacts/models

# API documentation: http://localhost:8000/docs
```

### 5. Make Predictions

```python
from ml_pipeline.inference import Predictor
import pandas as pd

# Load trained models
predictor = Predictor(config, run_dir="artifacts/results/run_YYYYMMDD_HHMMSS")

# Prepare recent data (must be DataFrame with 48+ hours)
recent_data = pd.read_csv("recent_measurements.csv", parse_dates=["datetime"])
recent_data = recent_data.set_index("datetime")

# Predict next 24 hours
predictions = predictor.predict(recent_data, n_steps=24)
print(predictions)  # {"gru": [...], "sarima": [...], "hybrid": [...]}
```

---

## 🔑 Key Features

### 1. **Modular Codebase**
- Clean separation of concerns (data, models, training, evaluation)
- Reusable components for different experiments
- Easy to extend with new models or features

### 2. **Data Leakage Prevention**
- ✅ Chronological train/val/test split (no future info in train)
- ✅ Lag/rolling features computed per-city to prevent cross-contamination
- ✅ Scaler fitted on training data only
- ✅ Weather API enrichment applied before feature engineering

### 3. **Validation Strategies**
- **Chronological Split**: Single train/val/test split
- **Walk-Forward**: 5 folds with expanding window
- **Rolling Window**: Fixed window size with steps

```python
from ml_pipeline.training import get_validation_strategy

validator = get_validation_strategy(config)
for train_df, val_df, test_df in validator.split(df):
    # Train and evaluate per fold
    pass
```

### 4. **Hyperparameter Tuning (Optuna)**
```python
from ml_pipeline.training import HyperparameterTuner

tuner = HyperparameterTuner(config)
best_params = tuner.optimize(objective_fn, direction="minimize")
# Automatically tunes: GRU units, learning rate, dropout, batch size
```

### 5. **Experiment Tracking (MLflow ready)**
- Configuration versioning
- Metrics logging (RMSE, MAE, R², MAPE)
- Model artifacts persistence
- Run history tracking

### 6. **Feature Engineering (Chronologically Safe)**
- Cyclical encoding (hour, day-of-week, month)
- Autoregressive lags (1, 3, 6, 12, 24h)
- Rolling statistics (mean, std, min, max)
- Interaction terms (humidity×PM2.5, stagnation index, etc.)
- Domain flags (smog season, rush hours)

### 7. **Models**

#### **GRU (Gated Recurrent Unit)**
- 2-layer stacked GRU with batch normalization
- Huber loss (robust to outliers)
- Early stopping + learning rate scheduling
- Multi-city support via one-hot encoding

#### **SARIMA (Seasonal ARIMA)**
- Order: (2,1,2) × (1,1,1,24) [configurable]
- Exogenous variables: NH3, SO2, Temperature, Humidity, etc.
- Online learning via `extend()` for incremental updates
- Log transformation for stability

#### **Hybrid Ensemble**
- Weighted average of GRU (50%) + SARIMA (50%)
- Configurable weights and blending methods
- Provides conservative estimates

### 8. **Comprehensive Evaluation**
```python
from ml_pipeline.evaluation import MetricsEvaluator, PlotGenerator

metrics = MetricsEvaluator.compute_metrics(y_true, y_pred, label="Hybrid")
# RMSE, MAE, MAPE, R², directional accuracy, quantile losses

plotter = PlotGenerator(output_dir)
plotter.plot_predictions_vs_actual(y_true, y_pred, "Hybrid")
plotter.plot_error_distribution(y_true, y_pred)
plotter.plot_time_series(timestamps, y_true, y_pred)
plotter.plot_comparison_dashboard(y_true, {"GRU": pred_gru, "SARIMA": pred_sarima})
```

### 9. **Production REST API**

**Endpoints:**

- `GET /health` - Health check
- `GET /info` - API information
- `POST /predict` - Multi-step forecast from measurements
- `POST /predict_single` - Next-hour prediction
- `GET /docs` - Interactive API documentation (Swagger UI)

**Example:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "timestamp": "2024-01-15T10:00:00",
      "pm2_5": 85.5,
      "temperature": 15.2,
      "humidity": 65,
      "wind_speed": 3.5,
      "pressure": 1013.0
    }
  ]'
```

### 10. **Structured Logging**

All modules use standardized logging:

```python
from ml_pipeline.utils import get_logger

logger = get_logger(__name__, log_level="INFO", log_file="pipeline.log")
logger.info("Data loaded successfully")
logger.warning("Skipped missing values")
logger.error("Model training failed", exc_info=True)
```

---

## 📊 Configuration (config.yaml)

Key sections:

```yaml
data:
  raw_path: "datasets/processed/pakistan_air_quality_final_clean.csv"
  use_multi_city: true
  city_column: "City"

features:
  target: "PM2.5"
  forecast_horizon: 6
  lag_steps: [1, 3, 6, 12, 24]
  rolling_windows: [6, 12, 24]

gru_model:
  sequence_length: 48
  gru_units: [64, 32]
  dropout_rate: 0.15
  learning_rate: 0.005
  epochs: 100
  patience: 10

validation:
  strategy: "walk_forward"  # or "rolling_window", "chronological"
  walk_forward:
    n_folds: 5

tuning:
  use_optuna: true
  optuna:
    n_trials: 50

reproducibility:
  seed: 42
```

---

## 🔄 Workflow

### Training

```python
# 1. Load & preprocess
loader = DataLoader(config)
df = loader.load_data()

# 2. Feature engineering
engineer = FeatureEngineer(config)
df = engineer.engineer_features(df)

# 3. Scale & split
preprocessor = DataPreprocessor(config)
train, val, test = preprocessor.split_data(df)
train_s, val_s, test_s = preprocessor.scale_data(train, val, test)

# 4. Train models
gru = GRUModel(config)
gru.build(n_features=train_s.shape[1])
gru.train(X_train, y_train, X_val, y_val)

# 5. Evaluate
evaluator = MetricsEvaluator()
metrics = evaluator.compute_metrics(y_test, y_pred_gru)

# 6. Plot
plotter = PlotGenerator()
plotter.plot_predictions_vs_actual(y_test, y_pred_gru)
```

### Inference

```python
# Load predictor with trained models
predictor = Predictor(config, run_dir="path/to/run")

# Predict
recent_data = pd.read_csv("recent.csv", parse_dates=["datetime"]).set_index("datetime")
preds = predictor.predict(recent_data, n_steps=24)
# preds = {"gru": [...], "sarima": [...], "hybrid": [...]}
```

---

## 🎯 Best Practices Implemented

✅ **Reproducibility**
- Fixed random seeds
- Version-controlled configs
- Detailed run metadata logging

✅ **Data Quality**
- Chronological integrity (no leakage)
- Per-city independent preprocessing
- Sanity checks (PM2.5 ≥ 0, humidity 0-100, etc.)

✅ **Model Quality**
- Validation on unseen time periods
- Multiple evaluation metrics
- Early stopping to prevent overfitting
- Ensemble to reduce bias/variance

✅ **Code Quality**
- Type hints throughout
- Comprehensive docstrings
- Modular, DRY design
- Error handling and logging

✅ **Deployment Ready**
- Production-grade REST API
- Model persistence
- Scalable architecture
- Configuration management

---

## 📋 Requirements

See `requirements-production.txt`:

```
tensorflow>=2.12
pandas>=2.0
numpy>=1.24
scikit-learn>=1.2
statsmodels>=0.13
optuna>=3.0
fastapi>=0.100
uvicorn[standard]>=0.23
pydantic>=2.0
pyyaml>=6.0
python-dotenv>=1.0
joblib>=1.3
```

Install:

```bash
pip install -r requirements-production.txt
```

---

## 🔧 Troubleshooting

**Issue: "No dataset found"**
- Check `config.yaml` data paths
- Ensure CSV files exist in `datasets/processed/`

**Issue: "TensorFlow not available"**
- Install: `pip install tensorflow`

**Issue: "SARIMA fit failed"**
- Check for stationarity (ADF test output)
- Try different SARIMA order in config

**Issue: "API won't start"**
- Check port 8000 is not in use
- Run: `python ml_pipeline/run_api.py --port 8001`

---

## 📞 Support

For issues or questions, refer to:
- Code docstrings (comprehensive)
- Config file comments
- Log output (run_YYYYMMDD_HHMMSS/logs/)
- API Swagger docs: http://localhost:8000/docs

---

**Version:** 1.0.0  
**Updated:** 2024-01-15  
**Status:** Production-Ready ✅
