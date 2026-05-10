# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:48:55 2025

@author: pejma
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 13:41:44 2025

@author: pejma
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 11:27:19 2025

@author: pejma
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 20:27:45 2025

@author: pejma
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:40:58 2025

@author: pejma
"""

# -*- coding: utf-8 -*-
"""
Simplified ensemble methods with only Stacking Huber, Blending Huber, and base models.
Uses KFold cross-validation to avoid data leakage.
"""

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, make_scorer,median_absolute_error
)
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor


# class IQRWinsorizer(BaseEstimator, TransformerMixin):
#     """
#     Winsorize numeric columns by capping values at [Q1 - k*IQR, Q3 + k*IQR]
#     and normalize selected composition columns so their sum = 100.
#     """
#     def __init__(self, cols=None, whisker_coef=1.5, normalize_to_100=True):
#         self.cols = cols
#         self.whisker_coef = whisker_coef
#         self.normalize_to_100 = normalize_to_100
#         self.bounds_ = None
#         self.feature_names_in_ = None

#     def fit(self, X, y=None):
#         if isinstance(X, pd.DataFrame):
#             Xdf = X
#         else:
#             if self.cols is None:
#                 raise ValueError("When passing ndarray, 'cols' must be provided.")
#             Xdf = pd.DataFrame(X, columns=self.cols if isinstance(self.cols[0], str) 
#                                else [f"f{i}" for i in range(X.shape[1])])

#         self.feature_names_in_ = list(Xdf.columns)
#         num_cols = self.cols if self.cols is not None else Xdf.select_dtypes(include=[np.number]).columns.tolist()
#         self.bounds_ = {}

#         k = float(self.whisker_coef)
#         for c in num_cols:
#             if c not in Xdf.columns:
#                 continue
#             s = pd.to_numeric(Xdf[c], errors='coerce')
#             q1 = np.nanpercentile(s, 25)
#             q3 = np.nanpercentile(s, 75)
#             iqr = q3 - q1
#             lower = q1 - k * iqr
#             upper = q3 + k * iqr
#             self.bounds_[c] = (lower, upper)
#         return self

#     def transform(self, X):
#         if isinstance(X, pd.DataFrame):
#             Xdf = X.copy()
#         else:
#             if self.feature_names_in_ is None:
#                 raise RuntimeError("Transformer is not fitted yet.")
#             Xdf = pd.DataFrame(X, columns=self.feature_names_in_).copy()

#         if self.bounds_ is None:
#             raise RuntimeError("Transformer is not fitted yet.")

#         # Winsorize
#         for c, (lo, hi) in self.bounds_.items():
#             if c in Xdf.columns:
#                 Xdf[c] = Xdf[c].clip(lower=lo, upper=hi)

#         # Normalize compositions to sum = 100
#         if self.normalize_to_100 and self.cols is not None:
#             comp_sum = Xdf[self.cols].sum(axis=1).replace(0, np.nan)
#             Xdf[self.cols] = Xdf[self.cols].div(comp_sum, axis=0) * 100

#         return Xdf

# --------------------------
# Custom Blending Ensemble
# --------------------------
class BlendingRegressor(BaseEstimator, RegressorMixin):
    """
    Blending ensemble that uses a holdout validation set to train the meta-learner.
    """
    def __init__(self, estimators, meta_learner, test_size=0.2, random_state=42):
        self.estimators = estimators
        self.meta_learner = meta_learner
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        # Split training data into train and blend
        X_blend_train, X_blend_holdout, y_blend_train, y_blend_holdout = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Train base models on blend_train
        self.base_models_ = []
        blend_features = []
        
        for name, estimator in self.estimators:
            model = clone(estimator)
            model.fit(X_blend_train, y_blend_train)
            self.base_models_.append((name, model))
            
            # Predict on holdout set
            pred = model.predict(X_blend_holdout)
            blend_features.append(pred)
        
        # Create meta-features
        X_meta = np.column_stack(blend_features)
        
        # Train meta-learner
        self.meta_learner_ = clone(self.meta_learner)
        self.meta_learner_.fit(X_meta, y_blend_holdout)
        
        # Retrain base models on full training data
        self.final_models_ = []
        for name, estimator in self.estimators:
            model = clone(estimator)
            model.fit(X, y)
            self.final_models_.append((name, model))
        
        return self
    
    def predict(self, X):
        # Get predictions from all base models
        meta_features = []
        for name, model in self.final_models_:
            pred = model.predict(X)
            meta_features.append(pred)
        
        X_meta = np.column_stack(meta_features)
        return self.meta_learner_.predict(X_meta)

data = pd.read_excel(r"E:\AUT\msc project\Asphaltenes\complex data 2\excels\DATASETasph.xlsx")
data = data.dropna().drop_duplicates().drop(columns=['ref'])
Q1 = data['AOP'].quantile(0.25)
Q3 = data['AOP'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['AOP'] >= lower_bound) & (data['AOP'] <= upper_bound)]
data.info
X = data.drop(columns=['AOP'])
y = data['AOP']

selected_cols = ['N2 (mol%)', 'Co2', 'H2S', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7+']

# Train/Test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# --------------------------
# 1.5) Preprocess only training data with IQRWinsorizer
# --------------------------

# # Apply IQRWinsorizer only on training data
# winsorizer = IQRWinsorizer(cols=selected_cols, whisker_coef=1.5)
# X_train_winsorized = winsorizer.fit_transform(X_train_raw)  # Only apply on training data
# X_test_winsorized = X_test_raw.copy()  # No transformation on test data


# CV setup
cv = KFold(n_splits=5, shuffle=True, random_state=48)
r2_scorer = make_scorer(r2_score)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# --------------------------
# 2) Base Model Tuning
# --------------------------

# XGBoost tuning
def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("model", xgb.XGBRegressor(**params))
    ])
    scores = cross_val_score(pipe, X_train_raw, y_train, cv=cv, scoring=mae_scorer, n_jobs=-1)
    return scores.mean()

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=10)
best_xgb_params = {**study_xgb.best_params, 'objective': 'reg:squarederror', 
                   'booster': 'gbtree', 'random_state': 42, 'n_jobs': -1, 'verbosity': 0}

# RandomForest tuning
def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8]),
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    }
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("model", RandomForestRegressor(**params))
    ])
    scores = cross_val_score(pipe, X_train_raw, y_train, cv=cv, scoring=mae_scorer, n_jobs=-1)
    return scores.mean()

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=10)
best_rf_params = {**study_rf.best_params, "random_state": 42, "n_jobs": -1}

# LightGBM tuning
def objective_lgb(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
    }
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("model", lgb.LGBMRegressor(**params))
    ])
    scores = cross_val_score(pipe, X_train_raw, y_train, cv=cv, scoring=mae_scorer, n_jobs=-1)
    return scores.mean()

study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=10)
best_lgb_params = {**study_lgb.best_params, 'random_state': 42, 'n_jobs': -1, 'verbosity': -1}

# --------------------------
# 3) Create Base Model Pipelines
# --------------------------
xgb_pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("model", xgb.XGBRegressor(**best_xgb_params))
])

rf_pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("model", RandomForestRegressor(**best_rf_params))
])

lgb_pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("model", lgb.LGBMRegressor(**best_lgb_params))
])

et_pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("model", ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1))
])

base_estimators = [
    ('xgb', xgb_pipe),
    ('rf', rf_pipe),
    ('lgb', lgb_pipe),
    ('et', et_pipe)
]
# --------------------------
# 4) Train Base Models and Get Predictions
# --------------------------
print("=== Training Base Models ===")

base_train_preds = {}
base_test_preds = {}

for name, estimator in base_estimators:
    print(f"Training {name}...")
    est = clone(estimator)
    est.fit(X_train_raw, y_train)
    base_train_preds[name] = est.predict(X_train_raw)
    base_test_preds[name]  = est.predict(X_test_raw)

# --------------------------
# 5) Stacking with Huber Regressor (using sklearn's StackingRegressor)
# --------------------------
print("\n=== Training Stacking with Huber (sklearn) ===")

stacker = StackingRegressor(
    estimators=base_estimators,
    final_estimator=HuberRegressor(epsilon=1.35, max_iter=1000),
    cv=cv,              # از همان KFold که بالا ساختی استفاده می‌کنیم
    passthrough=True   # فقط پیش‌بینی‌های بیس‌مدل‌ها به متا-لِرنر می‌رسد
    # اگر با نسخهٔ اسکیکیت‌لرن سازگار است و خواستی: n_jobs=-1
)

stacker.fit(X_train_raw, y_train)
stacking_huber_train_pred = stacker.predict(X_train_raw)
stacking_huber_test_pred  = stacker.predict(X_test_raw)

# --------------------------
# 6) Blending with Huber Regressor
# --------------------------
print("\n=== Training Blending with Huber ===")

blender_huber = BlendingRegressor(
    estimators=base_estimators,
    meta_learner=HuberRegressor(epsilon=1.35, max_iter=1000),
    test_size=0.2,
    random_state=42
)
blender_huber.fit(X_train_raw, y_train)

blending_huber_train_pred = blender_huber.predict(X_train_raw)
blending_huber_test_pred = blender_huber.predict(X_test_raw)

# --------------------------
# 7) Comprehensive Results
# --------------------------
from sklearn.metrics import median_absolute_error

def enhanced_report_metrics(y_true, y_pred, label="Model"):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    
    print(f"{label:25} | R²: {r2:6.3f} | MAE: {mae:6.3f} | MEDAE: {medae:6.3f} | RMSE: {rmse:6.3f} | MAPE: {mape:6.2f}%")
    return {'R2': r2, 'MAE': mae, 'MEDAE': medae, 'RMSE': rmse, 'MAPE': mape}

# Example usage
print("\n" + "="*80)
print("RESULTS: BASE MODELS + STACKING HUBER + BLENDING HUBER")
print("="*80)

print(f"{'Method':<25} | {'R²':<6} | {'MAE':<6} | {'MEDAE':<6} | {'RMSE':<6} | {'MAPE':<6}")
print("-"*80)

# Base models results
print("\n--- BASE MODELS ---")
base_results = {}
for name in ['xgb', 'rf', 'lgb', 'et']:
    train_pred = base_train_preds[name]
    test_pred = base_test_preds[name]
    
    print(f"\n{name.upper()}:")
    train_metrics = enhanced_report_metrics(y_train, train_pred, f"{name} (Train)")
    test_metrics = enhanced_report_metrics(y_test, test_pred, f"{name} (Test)")
    base_results[name] = {'train': train_metrics, 'test': test_metrics}

# Ensemble results
print(f"\n--- ENSEMBLE METHODS ---")
print(f"\nSTACKING HUBER:")
stacking_train_metrics = enhanced_report_metrics(y_train, stacking_huber_train_pred, "Stacking (Train)")
stacking_test_metrics = enhanced_report_metrics(y_test, stacking_huber_test_pred, "Stacking (Test)")

print(f"\nBLENDING HUBER:")
blending_train_metrics = enhanced_report_metrics(y_train, blending_huber_train_pred, "Blending (Train)")
blending_test_metrics = enhanced_report_metrics(y_test, blending_huber_test_pred, "Blending (Test)")

# --------------------------
# 8) Best Model Selection and Visualization
# --------------------------
all_test_r2 = {
    'XGBoost': base_results['xgb']['test']['R2'],
    'RandomForest': base_results['rf']['test']['R2'],
    'LightGBM': base_results['lgb']['test']['R2'],
    'ExtraTrees': base_results['et']['test']['R2'],
    'Stacking_Huber': stacking_test_metrics['R2'],
    'Blending_Huber': blending_test_metrics['R2']
}

# Find best method
best_method = max(all_test_r2, key=all_test_r2.get)
best_score = all_test_r2[best_method]

print(f"\n" + "="*50)
print(f"BEST METHOD: {best_method}")
print(f"BEST TEST R² SCORE: {best_score:.4f}")
print(f"="*50)

# Visualization
plt.figure(figsize=(12, 8))
methods = list(all_test_r2.keys())
scores = list(all_test_r2.values())

# Sort by score
sorted_pairs = sorted(zip(methods, scores), key=lambda x: x[1], reverse=True)
methods, scores = zip(*sorted_pairs)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bars = plt.bar(range(len(methods)), scores, color=colors[:len(methods)])

plt.xlabel('Methods')
plt.ylabel('Test R² Score')
plt.title('Performance Comparison: Base Models + Ensembles')
plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (method, score) in enumerate(zip(methods, scores)):
    plt.text(i, score + 0.005, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nHyperparameter tuning completed!")
print(f"Best XGBoost params: {best_xgb_params}")
print(f"Best RandomForest params: {best_rf_params}")
print(f"Best LightGBM params: {best_lgb_params}")



# ========= 9) Error analysis by ranges (paste after predictions are ready) =========

# 9.1 جمع‌آوری پیش‌بینی‌های همه مدل‌ها در یک دیکشنری
preds_dict = {
    "XGBoost": base_test_preds['xgb'],
    "RandomForest": base_test_preds['rf'],
    "LightGBM": base_test_preds['lgb'],
    "ExtraTrees": base_test_preds['et'],
    "Stacking_Huber": stacking_huber_test_pred,
    "Blending_Huber": blending_huber_test_pred
}

# 9.2 تابع‌های کمکی برای محاسبه‌ی خطاها در بازه‌ها
def _metrics(y_true, y_pred):
    ae = np.abs(y_true - y_pred)
    mae = ae.mean()
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    # جلوگیری از تقسیم بر صفر برای MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.nanmean(np.where(y_true != 0, ae / np.abs(y_true), np.nan)) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def errors_by_bins(y_true, y_pred, bins, labels=None):
    """برگشت دیتافریم با MAE/RMSE/MAPE برای هر بازه."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["bin"] = pd.cut(df["y_true"], bins=bins, labels=labels, include_lowest=True)
    out = (
        df.groupby("bin")
          .apply(lambda g: pd.Series(_metrics(g["y_true"].values, g["y_pred"].values)))
          .reset_index()
          .rename(columns={"bin": "Range"})
    )
    out["Count"] = df.groupby("bin").size().values
    return out

def errors_by_feature_bins(X_test, y_true, y_pred, feature_name, use_quantile_bins=True, q=5, fixed_bins=None):
    """تحلیل خطا برحسب بازه‌های یک فیچر خاص."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, feature_name: X_test[feature_name].values})
    if use_quantile_bins:
        df["bin"] = pd.qcut(df[feature_name], q=q, duplicates="drop")
    else:
        if fixed_bins is None:
            raise ValueError("برای حالت fixed bins، آرایه‌ی fixed_bins را مشخص کن.")
        df["bin"] = pd.cut(df[feature_name], bins=fixed_bins, include_lowest=True)
    out = (
        df.groupby("bin")
          .apply(lambda g: pd.Series(_metrics(g["y_true"].values, g["y_pred"].values)))
          .reset_index()
          .rename(columns={"bin": f"{feature_name}_Range"})
    )
    out["Count"] = df.groupby("bin").size().values
    return out

# 9.3 تعریف بازه‌های y_true
use_quantile_bins = True   # اگر بازه‌های مساوی از نظر تعداد می‌خواهی
q = 5                      # پنج بازه (پنجکی)
if use_quantile_bins:
    y_bins = pd.qcut(y_test, q=q, duplicates="drop")
    # از qcut برای برچسب‌ها صرفاً حدود را برمی‌داریم:
    # اما برای groupby باید bin‌ها را دوباره بسازیم؛ بنابراین:
    y_bin_edges = sorted(set([y_test.min(), *[y_test.quantile(i/q) for i in range(1, q)], y_test.max()]))
    y_bins_edges = np.unique(y_bin_edges)
else:
    # نمونه: بازه‌های ثابت؛ این را بنا به مقیاس AOP تنظیم کن
    y_min, y_max = y_test.min(), y_test.max()
    y_bins_edges = np.linspace(y_min, y_max, num=6)  # 5 بازهٔ هم‌عرض
# DataFrame نمایش نتایج بازه‌ای (برحسب y_true) برای هر مدل
results_by_y = {}
for mname, y_pred in preds_dict.items():
    if use_quantile_bins:
        # برای qcut باید bin را با خود qcut بسازیم
        tmp = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        tmp["bin"] = pd.qcut(tmp["y_true"], q=q, duplicates="drop")
        out = (tmp.groupby("bin")
                    .apply(lambda g: pd.Series(_metrics(g["y_true"].values, g["y_pred"].values)))
                    .reset_index()
                    .rename(columns={"bin": "y_true_Range"}))
        out["Count"] = tmp.groupby("bin").size().values
    else:
        out = errors_by_bins(y_test.values, y_pred, bins=y_bins_edges)
        out = out.rename(columns={"Range": "y_true_Range"})
    results_by_y[mname] = out

# 9.4 نمونه: تحلیل خطا برحسب یک فیچر ورودی (مثلاً 'C1')
feature_name = 'C1'  # هر فیچر دلخواه از selected_cols
results_by_feat = {}
for mname, y_pred in preds_dict.items():
    out_feat = errors_by_feature_bins(
        X_test=X_test_raw.reset_index(drop=True),
        y_true=y_test.reset_index(drop=True).values,
        y_pred=y_pred,
        feature_name=feature_name,
        use_quantile_bins=True,  # یا False + fixed_bins=np.arange(...)
        q=5
    )
    results_by_feat[mname] = out_feat

# 9.5 چاپ نتایج جدولی خلاصه برای هر مدل (بازه‌های y_true)
print("\n" + "="*80)
print("ERROR DISTRIBUTION BY y_true RANGES (Quantile bins)")
print("="*80)
for mname, dfm in results_by_y.items():
    print(f"\n>>> {mname}")
    print(dfm.to_string(index=False))

# 9.6 رسم نمودار برای مقایسه‌ی MAE و MAPE در بازه‌های y_true (برای بهترین یا چند مدل)
def plot_bar_errors(df_errors, value_col, title):
    plt.figure(figsize=(10, 5))
    # تبدیل بازه‌ها به رشته برای برچسب محور x
    x_labels = df_errors.iloc[:, 0].astype(str).values
    plt.bar(x_labels, df_errors[value_col].values)
    plt.xlabel('y_true ranges')
    plt.ylabel(value_col)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# مثال: نمودار برای Stacking_Huber
if "Stacking_Huber" in results_by_y:
    plot_bar_errors(results_by_y["Stacking_Huber"], "MAE", "Stacking_Huber - MAE by y_true ranges")
    plot_bar_errors(results_by_y["Stacking_Huber"], "MAPE", "Stacking_Huber - MAPE by y_true ranges")

# مثال: نمودار برای LightGBM
if "LightGBM" in results_by_y:
    plot_bar_errors(results_by_y["LightGBM"], "MAE", "LightGBM - MAE by y_true ranges")
    plot_bar_errors(results_by_y["LightGBM"], "MAPE", "LightGBM - MAPE by y_true ranges")

# 9.7 همچنین می‌توانیم هیستوگرام باقی‌مانده‌ها را برای یک مدل ببینیم
def plot_residual_hist(y_true, y_pred, title):
    resid = y_true - y_pred
    plt.figure(figsize=(8,5))
    plt.hist(resid, bins=30)
    plt.xlabel('Residual (y_true - y_pred)')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_residual_hist(y_test.values, preds_dict["Stacking_Huber"], "Residuals Histogram - Stacking_Huber")


# Count per bin
bins = pd.qcut(y_test, q=5, duplicates='drop')
print(bins.value_counts().sort_index())

# Residual vs y_true
resid = y_test.values - stacking_huber_test_pred
plt.figure(figsize=(7,5)); plt.scatter(y_test.values, resid, s=20, alpha=0.6)
plt.axhline(0, ls='--'); plt.xlabel("y_true"); plt.ylabel("Residual"); plt.title("Residual vs y_true"); plt.show()



# XGBoost Feature Importance
xgb_model = xgb.XGBRegressor(**best_xgb_params)
xgb_model.fit(X_train_raw, y_train)
xgb_importance = xgb_model.feature_importances_

# RandomForest Feature Importance
rf_model = RandomForestRegressor(**best_rf_params)
rf_model.fit(X_train_raw, y_train)
rf_importance = rf_model.feature_importances_

# LightGBM Feature Importance
lgb_model = lgb.LGBMRegressor(**best_lgb_params)
lgb_model.fit(X_train_raw, y_train)
lgb_importance = lgb_model.feature_importances_

# ExtraTrees Feature Importance
et_model = ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
et_model.fit(X_train_raw, y_train)
et_importance = et_model.feature_importances_

# Feature names (assuming X_train_raw.columns exists and matches the feature importance length)
feature_names = X_train_raw.columns.tolist()

# Plotting the feature importance
plt.figure(figsize=(14, 8))

for i, (model_name, importance) in enumerate({
    'XGBoost': xgb_importance,
    'RandomForest': rf_importance,
    'LightGBM': lgb_importance,
    'ExtraTrees': et_importance
}.items()):
    plt.subplot(2, 2, i + 1)
    indices = np.argsort(importance)[::-1]
    sorted_importance = importance[indices]
    sorted_feature_names = np.array(feature_names)[indices]
    
    plt.barh(sorted_feature_names[:17], sorted_importance[:17], color='skyblue')
    plt.title(f'{model_name} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')

plt.tight_layout()
plt.show()


import shap
import numpy as np
import matplotlib.pyplot as plt

# ابتدا پیش‌بینی‌های مدل‌های پایه برای تولید ویژگی‌های مدل Blending
base_model_preds_train_blending = []
for name, estimator in base_estimators:
    # آموزش مدل پایه
    model = estimator.fit(X_train_raw, y_train)
    base_model_preds_train_blending.append(model.predict(X_train_raw))

# پیش‌بینی‌های مدل‌های پایه را برای ورودی‌های مدل Blending جمع می‌کنیم
X_train_blending = np.column_stack(base_model_preds_train_blending)

# SHAP explainer برای مدل Blending (استفاده از meta_learner که HuberRegressor است)
blending_shap = shap.Explainer(blender_huber.meta_learner_, X_train_blending)
blending_shap_values = blending_shap(X_train_blending)

# رسم نمودار SHAP برای مدل Blending
plt.figure(figsize=(10, 6))

# استخراج اهمیت ویژگی‌ها برای مدل Blending
importance = np.mean(np.abs(blending_shap_values.values), axis=0)
indices = np.argsort(importance)[::-1]
sorted_importance = importance[indices]
sorted_feature_names = np.array([f"Base Model {i+1}" for i in range(X_train_blending.shape[1])])[indices]

# رسم نمودار
plt.barh(sorted_feature_names[:17], sorted_importance[:17], color='skyblue')
plt.title('Blending Model Feature Importance (Huber Regressor)')
plt.xlabel('Importance')
plt.ylabel('Features')

plt.tight_layout()
plt.show()



# ===================== 10) Feature Importance for Stacking =====================
from sklearn.inspection import permutation_importance

# ---------- 10.1) نام‌گذاری متافچرها برای متالِرنر ----------
base_pred_names = [f"{name}_pred" for name, _ in base_estimators]

if getattr(stacker, "passthrough", False):
    # ترتیب رایجِ اسکیکیت‌لِرن: [ویژگی‌های خام X, سپس پیش‌بینی بیس‌مدل‌ها]
    meta_feature_names = list(X_train_raw.columns) + base_pred_names
else:
    meta_feature_names = base_pred_names

# ---------- 10.2) اهمیت متافچرها با ضرایب Huber (linear model) ----------
final_est = stacker.final_estimator_
if not hasattr(final_est, "coef_"):
    raise RuntimeError("final_estimator فاقد coef_ است. برای گرفتن اهمیت خطی، از مدلی مثل HuberRegressor/LinearRegression استفاده کنید.")

coefs = np.asarray(final_est.coef_).ravel()
if len(coefs) != len(meta_feature_names):
    # در صورت اختلاف، چک سازگاری
    raise RuntimeError(f"طول ضرایب ({len(coefs)}) با طول نام متافچرها ({len(meta_feature_names)}) برابر نیست.")

coef_df = (
    pd.DataFrame({"MetaFeature": meta_feature_names, "Coef": coefs})
      .assign(AbsCoef=lambda d: d["Coef"].abs())
      .sort_values("AbsCoef", ascending=False)
      .reset_index(drop=True)
)

print("\n=== Stacking meta-feature importances (by |coef| of Huber) ===")
print(coef_df.to_string(index=False))

# اگر passthrough=True است، یک خلاصه‌ی بلوکی هم چاپ کن:
if getattr(stacker, "passthrough", False):
    n_raw = X_train_raw.shape[1]
    block_summary = pd.DataFrame({
        "Block": ["Raw_X", "Base_Preds"],
        "L1_sum_abs_coef": [
            coef_df.loc[:n_raw-1, "AbsCoef"].sum(),
            coef_df.loc[n_raw:, "AbsCoef"].sum()
        ]
    })
    print("\n--- Block-level contribution (sum of |coef|) ---")
    print(block_summary.to_string(index=False))

# ---------- 10.3) Permutation Importance روی ورودی‌های خام نسبت به کل استکینگ ----------
# این روش، اهمیت «هر ویژگی ورودی اصلی» را با کاهش کارایی R^2 پس از درهم‌ریزی آن ویژگی، نسبت به کل استکینگ می‌سنجد.
perm = permutation_importance(
    estimator=stacker,
    X=X_test_raw,
    y=y_test,
    scoring="r2",
    n_repeats=25,
    random_state=42,
    n_jobs=-1
)

perm_df = (
    pd.DataFrame({
        "Feature": X_test_raw.columns,
        "Importance_mean": perm.importances_mean,
        "Importance_std": perm.importances_std
    })
    .sort_values("Importance_mean", ascending=False)
    .reset_index(drop=True)
)

print("\n=== Permutation importance of original input features (w.r.t Stacking) ===")
print(perm_df.to_string(index=False))

# ---------- 10.4) نمودارها ----------
plt.figure(figsize=(7.2, 5.5))
plt.barh(perm_df["Feature"][::-1], perm_df["Importance_mean"][::-1])
plt.xlabel("Permutation Importance (mean ΔR²)")
plt.title("Stacking — Permutation Importance of Original Features")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7.2, 6.2))
top_k = min(15, len(coef_df))
plt.barh(coef_df["MetaFeature"].head(top_k)[::-1], coef_df["AbsCoef"].head(top_k)[::-1])
plt.xlabel("|Coefficient| (Huber)")
plt.title("Stacking — Meta-feature Importance (Huber coefficients)")
plt.tight_layout()
plt.show()



# plots for thesis




# --- Best model preds ---
y_true = y_test.values
y_pred_best = preds_dict[best_method]  # از دیکشنریِ شما

# --- Parity plot ---
plt.figure(figsize=(6.5,6))
plt.scatter(y_true, y_pred_best, s=28, alpha=0.7)
minv, maxv = np.min([y_true, y_pred_best]), np.max([y_true, y_pred_best])
plt.plot([minv, maxv], [minv, maxv], 'k--', lw=1)
plt.xlabel('Observed AOP')
plt.ylabel('Predicted AOP')
plt.title(f'Parity Plot – {best_method}')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


resid_best = y_true - y_pred_best
plt.figure(figsize=(6.8,5))
plt.scatter(y_pred_best, resid_best, s=20, alpha=0.6)
plt.axhline(0, ls='--', c='k')
plt.xlabel('Predicted AOP')
plt.ylabel('Residual (y_true - y_pred)')
plt.title(f'Residuals vs Predicted – {best_method}')
plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()



df_cal = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred_best}).sort_values('y_pred')
df_cal['decile'] = pd.qcut(df_cal['y_pred'], 10, labels=False, duplicates='drop')
agg = df_cal.groupby('decile').agg(y_pred_mean=('y_pred','mean'),
                                   y_true_mean=('y_true','mean'),
                                   n=('y_true','size')).reset_index()
plt.figure(figsize=(7.2,5))
plt.plot(agg['y_pred_mean'], agg['y_true_mean'], marker='o')
plt.plot([agg['y_pred_mean'].min(), agg['y_pred_mean'].max()],
         [agg['y_pred_mean'].min(), agg['y_pred_mean'].max()], 'k--', lw=1)
plt.xlabel('Mean Predicted (per decile)')
plt.ylabel('Mean Observed (per decile)')
plt.title(f'Calibration by Predicted Deciles – {best_method}')
plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()


def plot_ecdf_abs_error(models_to_show=('LightGBM','Stacking_Huber','Blending_Huber')):
    plt.figure(figsize=(7,5))
    for m in models_to_show:
        ae = np.abs(y_true - preds_dict[m])
        x = np.sort(ae)
        y = np.arange(1, len(x)+1)/len(x)
        plt.step(x, y, where='post', label=m)
    plt.xlabel('Absolute Error')
    plt.ylabel('ECDF')
    plt.title('ECDF of Absolute Error')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout(); plt.show()

plot_ecdf_abs_error()



from optuna.importance import get_param_importances

def plot_optuna_simple(study, title):
    best_per_trial = [t.value for t in study.trials if t.value is not None]
    plt.figure(figsize=(6,4))
    plt.plot(best_per_trial, marker='o'); plt.title(f'Optimization History – {title}')
    plt.xlabel('Trial'); plt.ylabel('Objective (neg-MAE, higher is better)')
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    imp = get_param_importances(study)
    if len(imp):
        labels, vals = zip(*imp.items())
        plt.figure(figsize=(6,4))
        plt.barh(labels, vals)
        plt.title(f'Param Importance – {title}')
        plt.tight_layout(); plt.show()

plot_optuna_simple(study_lgb, 'LightGBM')
plot_optuna_simple(study_xgb, 'XGBoost')
plot_optuna_simple(study_rf,  'RandomForest')





# ===================== 10+) Permutation Importance for ALL models =====================
from sklearn.inspection import permutation_importance

def compute_perm_importance(estimator, X, y, feature_names, name, scoring="r2",
                            n_repeats=25, random_state=42, n_jobs=-1, top_k=20,
                            plot=True):
    """
    محاسبه و (اختیاری) رسم Permutation Importance برای یک estimator
    خروجی: DataFrame مرتب‌شده برحسب میانگین کاهش R^2
    """
    perm = permutation_importance(
        estimator=estimator,
        X=X, y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )
    df = (pd.DataFrame({
            "Feature": feature_names,
            "Importance_mean": perm.importances_mean,
            "Importance_std": perm.importances_std
         })
         .sort_values("Importance_mean", ascending=False)
         .reset_index(drop=True))

    print(f"\n=== Permutation Importance — {name} (metric: Δ{scoring.upper()}) ===")
    print(df.to_string(index=False))

    if plot:
        k = min(top_k, len(df))
        plt.figure(figsize=(7.8, 6))
        plt.barh(df["Feature"].head(k)[::-1], df["Importance_mean"].head(k)[::-1])
        plt.xlabel(f"Permutation Importance (mean Δ{scoring.upper()})")
        plt.title(f"{name} — Top-{k} features (Permutation Importance)")
        plt.tight_layout()
        plt.show()
    return df

# 10+.1) برای پایه‌ها: حتماً نسخهٔ «فیت‌شده» داخل پایپ‌لاین‌ها را داشته باشیم
# (اگر بالاتر نگه نداشتی، دوباره فیت می‌کنیم تا مطمئن باشیم روی X_train_raw آموزش دیده‌اند)
fitted_base = {}
for name, estimator in base_estimators:
    est = clone(estimator).fit(X_train_raw, y_train)
    fitted_base[name] = est

# 10+.2) محاسبه برای همهٔ مدل‌ها روی X_test_raw
perm_tables = {}  # برای تجمیع نتایج

# Base models (pipeline-based)
perm_tables["XGBoost"]     = compute_perm_importance(fitted_base["xgb"], X_test_raw, y_test, X_test_raw.columns, "XGBoost (Pipeline)")
perm_tables["RandomForest"]= compute_perm_importance(fitted_base["rf"],  X_test_raw, y_test, X_test_raw.columns, "RandomForest (Pipeline)")
perm_tables["LightGBM"]    = compute_perm_importance(fitted_base["lgb"], X_test_raw, y_test, X_test_raw.columns, "LightGBM (Pipeline)")
perm_tables["ExtraTrees"]  = compute_perm_importance(fitted_base["et"],  X_test_raw, y_test, X_test_raw.columns, "ExtraTrees (Pipeline)")

# Stacking & Blending نسبت به ورودی‌های خام
perm_tables["Stacking_Huber"]  = compute_perm_importance(stacker,        X_test_raw, y_test, X_test_raw.columns, "Stacking_Huber")
perm_tables["Blending_Huber"]  = compute_perm_importance(blender_huber,  X_test_raw, y_test, X_test_raw.columns, "Blending_Huber")

# 10+.3) جدول مقایسه‌ای (Feature × Model) — برای گزارش پایان‌نامه مفید است
# ستون‌ها: مدل‌ها، سطرها: فیچرها، مقدار: importance_mean
all_features = list(X_test_raw.columns)
models_in_order = ["XGBoost","RandomForest","LightGBM","ExtraTrees","Stacking_Huber","Blending_Huber"]

comp_df = pd.DataFrame(index=all_features, columns=models_in_order, dtype=float)
for m in models_in_order:
    tmp = perm_tables[m].set_index("Feature")["Importance_mean"]
    comp_df[m] = tmp.reindex(all_features)

# نمایش خلاصه‌ای از برترین فیچرها برحسب میانگین اهمیت بین مدل‌ها
comp_df["MeanAcrossModels"] = comp_df.mean(axis=1, skipna=True)
comp_top = comp_df.sort_values("MeanAcrossModels", ascending=False).head(20)

print("\n=== Aggregated view: top features by mean Permutation Importance across models ===")
print(comp_top.round(6).to_string())

# 10+.4) نمودار مقایسه‌ای ساده برای 10 فیچر برترِ میانگین‌گرفته
top10 = comp_top.head(10).index.tolist()
plt.figure(figsize=(10, 6))
x = np.arange(len(top10))
width = 0.13
for i, m in enumerate(models_in_order):
    plt.bar(x + i*width, comp_df.loc[top10, m].values, width=width, label=m)
plt.xticks(x + (len(models_in_order)-1)*width/2, top10, rotation=45, ha="right")
plt.ylabel("Mean ΔR² (Permutation Importance)")
plt.title("Top-10 features by mean importance — comparison across models")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout(); plt.show()


import matplotlib.pyplot as plt

plt.figure(figsize=(18, 12))  # تنظیم ابعاد برای ۶ نمودار کنار هم

# لیست مدل‌ها و رنگ‌ها
models = ['XGBoost', 'RandomForest', 'LightGBM', 'ExtraTrees', 'Stacking_Huber', 'Blending_Huber']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ایجاد ۶ زیرنمودار (subplots) در یک ردیف
for i, (model, color) in enumerate(zip(models, colors)):
    plt.subplot(2, 3, i + 1)  # ایجاد زیرنمودار در موقعیت مناسب
    y_pred = preds_dict[model]  # پیش‌بینی‌ها برای هر مدل
    plt.scatter(y_true, y_pred, color=color, alpha=0.6)  # رسم نمودار پراکندگی
    
    # خط یکتا برای مقایسه بهتر
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)  
    
    plt.xlabel('Observed Pressure (y_true)')
    plt.ylabel('Predicted Pressure (y_pred)')
    plt.title(f'{model} Parity Plot')  # عنوان نمودار برای هر مدل
    plt.grid(True)

plt.tight_layout()  # تنظیمات برای فاصله مناسب بین زیرنمودارها
plt.show()



plt.figure(figsize=(10, 8))

# لیست مدل‌ها و رنگ‌ها
models = ['XGBoost', 'RandomForest', 'LightGBM', 'ExtraTrees', 'Stacking_Huber', 'Blending_Huber']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# رسم نمودار پراکندگی برای هر مدل
for model, color in zip(models, colors):
    y_pred = preds_dict[model]  # پیش‌بینی‌ها برای هر مدل
    plt.scatter(y_true, y_pred, color=color, alpha=0.6, label=model)

# خط یکتا برای مقایسه بهتر
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)

plt.xlabel('Observed Pressure (y_true)')
plt.ylabel('Predicted Pressure (y_pred)')
plt.title('Comparison of Model Predictions vs Observed Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

for model, color in zip(models, colors):
    y_pred = preds_dict[model]
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, color=color, alpha=0.6, label=model)

plt.axhline(0, color='black', lw=2, linestyle='--')
plt.xlabel('Predicted Pressure')
plt.ylabel('Residuals (y_true - y_pred)')
plt.title('Residuals vs Predicted Pressure for Different Models')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))

for model in models:
    errors = np.abs(y_true - preds_dict[model])
    sorted_errors = np.sort(errors)
    y_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.step(sorted_errors, y_vals, where='post', label=model)

plt.xlabel('Absolute Error')
plt.ylabel('ECDF')
plt.title('ECDF of Absolute Errors for Different Models')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()








import joblib

# # # # # ذخیره‌سازی پایپ‌لاین‌های مدل‌ها (شامل مقیاس‌دهی و مدل)
# for name, estimator in base_estimators:
#     joblib.dump(estimator, f'{name}_article_aop.pkl')

# # # # # ذخیره‌سازی مدل‌های استکینگ و بلندینگ
# joblib.dump(stacker, 'stacking_huber_article_aop.pkl')
# joblib.dump(blender_huber, 'blending_huber_article_aop.pkl')



# # # # # بارگذاری مدل‌های استکینگ و بلندینگ (که شامل مدل‌های پایه و پیش‌پردازش‌ها هستند)import joblib

# # # # # بارگذاری مدل‌های استکینگ و بلندینگ (که شامل مدل‌های پایه و پیش‌پردازش‌ها هستند)# بارگذاری مدل‌ها از فایل
# stacker_model = joblib.load('stacking_huber_article_aop.pkl')
# blender_model = joblib.load('blending_huber_article_aop.pkl')

# # # # # انتخاب 5 داده اول از X_test_raw برای پیش‌بینی
# X_test_sample = X_test_raw.head(5)

# # # # # پیش‌بینی با استفاده از مدل‌های مختلف
# stacker_predictions = stacker_model.predict(X_test_sample)
# blender_predictions = blender_model.predict(X_test_sample)

# # # # # مقایسه پیش‌بینی‌ها با مقادیر واقعی
# print("=== Model Predictions vs Real Values ===")
# print(f"\n{'Sample':<10} | {'True Value':<12} | {'Stacking Prediction':<20} | {'Blending Prediction':<20}")
# print("="*80)

# for i, (true_val, stacker_pred, blender_pred) in enumerate(zip(y_test.head(5), stacker_predictions, blender_predictions)):
#     print(f"{i+1:<10} | {true_val:<12.3f} | {stacker_pred:<20.3f} | {blender_pred:<20.3f}")



import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error

# # # بارگذاری مدل استکینگ
stacker_model = joblib.load('stacking_huber_article_aop.pkl')
blendign_model = joblib.load('blending_huber_article_aop.pkl')

xgb_model = joblib.load('xgb_article_aop.pkl')
rf_model = joblib.load('rf_article_aop.pkl')
lgb_model = joblib.load('lgb_article_aop.pkl')
et_model = joblib.load('et_article_aop.pkl')

# # # پیش‌بینی با استفاده از مدل استکینگ
stacker_predictions = stacker_model.predict(X_test_raw)
blending_predictions = blendign_model.predict(X_test_raw)

xgb_model.fit(X_train_raw, y_train)
rf_model.fit(X_train_raw, y_train)
lgb_model.fit(X_train_raw, y_train)
et_model.fit(X_train_raw, y_train)

# حالا پیش‌بینی‌ها را انجام دهید
xgb_predictions = xgb_model.predict(X_test_raw)
rf_predictions = rf_model.predict(X_test_raw)
lgb_predictions = lgb_model.predict(X_test_raw)
et_predictions = et_model.predict(X_test_raw)

# # # محاسبه متریک‌ها برای مدل استکینگ
def calculate_metrics(y_true, y_pred, label="Model"):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    
    # Normalized MAE
    normalized_mae = mae / np.mean(y_true)
    
    # Normalized MSE
    normalized_mse = mse / np.mean(y_true)
    
    # Relative Error
    relative_error = np.mean(np.abs((y_pred - y_true) / y_true))
    
    # RMSLE (Root Mean Squared Logarithmic Error)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    
    print(f"{label:25} | R²: {r2:6.3f} | MAE: {mae:6.3f} | RMSE: {rmse:6.3f} | MAPE: {mape:6.2f}% | Normalized MAE: {normalized_mae:6.3f} | Normalized MSE: {normalized_mse:6.3f} | Relative Error: {relative_error:6.3f} | RMSLE: {rmsle:6.3f}")
    
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Normalized MAE': normalized_mae, 'Normalized MSE': normalized_mse, 'Relative Error': relative_error, 'RMSLE': rmsle}

# # # گزارش متریک‌ها برای مدل استکینگ
print(f"\n{'Method':<25} | {'R²':<6} | {'MAE':<6} | {'RMSE':<6} | {'MAPE':<6} | {'Normalized MAE':<15} | {'Normalized MSE':<15} | {'Relative Error':<15} | {'RMSLE':<6}")
print("-"*125)

calculate_metrics(y_test, stacker_predictions, "Stacking Huber")
calculate_metrics(y_test, blending_predictions, "blending Huber")
calculate_metrics(y_test, xgb_predictions, "xgb ")
calculate_metrics(y_test, rf_predictions, "rf ")
calculate_metrics(y_test, lgb_predictions, "lgb ")
calculate_metrics(y_test, et_predictions, "et ")


import matplotlib.pyplot as plt
import numpy as np

# رسم نمودار Parity Plot برای مدل استکینگ
def plot_parity(y_true, y_pred_best, best_method="Model"):
    plt.figure(figsize=(6.5, 6))  # تنظیم اندازه نمودار
    plt.scatter(y_true, y_pred_best, s=28, alpha=0.7)  # تنظیم اندازه و شفافیت نقاط
    minv, maxv = np.min([y_true, y_pred_best]), np.max([y_true, y_pred_best])  # پیدا کردن min و max مقادیر
    plt.plot([minv, maxv], [minv, maxv], 'k--', lw=1)  # رسم خط y=x به صورت نقطه‌چین مشکی
    
    plt.xlabel('Observed AOP (psia)')  # برچسب محور x
    plt.ylabel('Predicted AOP (psia')  # برچسب محور y
    plt.title(f'Parity Plot – {best_method}')  # عنوان نمودار
    plt.grid(alpha=0.3)  # تنظیم شفافیت شبکه
    plt.tight_layout()  # تنظیم فضای اطراف نمودار
    plt.show()

# استفاده از تابع برای رسم نمودار Parity Plot
plot_parity(y_test, stacker_predictions, "Stacking Huber")






