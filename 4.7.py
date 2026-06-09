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
study_xgb.optimize(objective_xgb, n_trials=100)
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
study_rf.optimize(objective_rf, n_trials=100)
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
study_lgb.optimize(objective_lgb, n_trials=50)
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
    ("model", ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
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


# =============================================================================
# JOURNAL-QUALITY PLOTTING SECTION
# Q1 Journal Standards: 300 dpi, clean axes, minimal gridlines, serif/sans fonts
# Replace all existing plotting blocks with this unified section
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# ---------------------------------------------------------------------------
# GLOBAL STYLE  (apply once, affects all subsequent figures)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "legend.frameon":     True,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.7",
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "figure.dpi":         150,          # screen preview
    "savefig.dpi":        300,          # saved files
    "savefig.bbox":       "tight",
    "axes.grid":          True,
    "grid.color":         "0.88",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
})

# ---------------------------------------------------------------------------
# PALETTE  — colorblind-safe, matches most Q1 journal style guides
# ---------------------------------------------------------------------------
MODEL_NAMES   = ['XGBoost', 'RandomForest', 'LightGBM', 'ExtraTrees',
                 'Stacking_Huber', 'Blending_Huber']
MODEL_LABELS  = ['XGBoost', 'Random Forest', 'LightGBM', 'Extra Trees',
                 'Stacking–Huber', 'Blending–Huber']
COLORS        = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
MARKERS       = ['o', 's', '^', 'D', 'v', 'P']

# Helper: build preds_dict from whatever names your code uses
# (adjust if your variable names differ)
preds_dict = {
    'XGBoost':        base_test_preds['xgb'],
    'RandomForest':   base_test_preds['rf'],
    'LightGBM':       base_test_preds['lgb'],
    'ExtraTrees':     base_test_preds['et'],
    'Stacking_Huber': stacking_huber_test_pred,
    'Blending_Huber': blending_huber_test_pred,
}
y_true = y_test.values


# ============================================================
#  FIGURE 1 — Parity Plots: individual (2×3) + combined (1)
# ============================================================

# ---- 1a) Individual parity panels in a 2×3 grid ---
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (mname, mlabel, col, mkr) in enumerate(
        zip(MODEL_NAMES, MODEL_LABELS, COLORS, MARKERS)):
    ax   = axes[idx]
    ypred = preds_dict[mname]
    r2    = r2_score(y_true, ypred)

    ax.scatter(y_true, ypred,
               color=col, marker=mkr, s=28, alpha=0.75,
               linewidths=0.3, edgecolors='white', label=mlabel)

    lims = [min(y_true.min(), ypred.min()) * 0.97,
            max(y_true.max(), ypred.max()) * 1.03]
    ax.plot(lims, lims, 'k--', lw=0.9, zorder=0)
    ax.set_xlim(lims); ax.set_ylim(lims)

    ax.set_xlabel('Observed AOP (psia)')
    ax.set_ylabel('Predicted AOP (psia)')
    ax.set_title(mlabel)
    ax.text(0.05, 0.92, f'$R^2$ = {r2:.3f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.85))
    ax.set_aspect('equal', adjustable='box')

fig.suptitle('Parity Plots — All Models', fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig1a_parity_individual.png')
plt.show()

# ---- 1b) All models overlaid on a single parity plot ---
fig, ax = plt.subplots(figsize=(6, 5.5))
all_vals = np.concatenate([y_true] + list(preds_dict.values()))
lims = [all_vals.min() * 0.97, all_vals.max() * 1.03]
ax.plot(lims, lims, 'k--', lw=0.9, zorder=0, label='Ideal')

for mname, mlabel, col, mkr in zip(MODEL_NAMES, MODEL_LABELS, COLORS, MARKERS):
    ypred = preds_dict[mname]
    r2    = r2_score(y_true, ypred)
    ax.scatter(y_true, ypred,
               color=col, marker=mkr, s=22, alpha=0.65,
               linewidths=0.2, edgecolors='white',
               label=f'{mlabel}  ($R^2$={r2:.3f})')

ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Observed AOP (psia)')
ax.set_ylabel('Predicted AOP (psia)')
ax.set_title('Parity Plot — All Models Combined')
ax.legend(loc='upper left', fontsize=7.5)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('fig1b_parity_combined.png')
plt.show()


# ============================================================
#  FIGURE 2 — Calibration Plots (predicted decile vs observed)
# ============================================================

def _calibration_data(y_true, y_pred, n_deciles=10):
    df = pd.DataFrame({'yt': y_true, 'yp': y_pred}).sort_values('yp')
    df['dec'] = pd.qcut(df['yp'], n_deciles, labels=False, duplicates='drop')
    agg = df.groupby('dec').agg(yp_mean=('yp','mean'), yt_mean=('yt','mean')).reset_index()
    return agg['yp_mean'].values, agg['yt_mean'].values

# ---- 2a) Individual calibration panels ---
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (mname, mlabel, col, mkr) in enumerate(
        zip(MODEL_NAMES, MODEL_LABELS, COLORS, MARKERS)):
    ax = axes[idx]
    xp, yo = _calibration_data(y_true, preds_dict[mname])
    lims = [min(xp.min(), yo.min()) * 0.97, max(xp.max(), yo.max()) * 1.03]

    ax.plot(lims, lims, 'k--', lw=0.9, zorder=0, label='Ideal')
    ax.plot(xp, yo, color=col, marker=mkr, ms=6, lw=1.2, label=mlabel)

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Mean Predicted AOP (psia)')
    ax.set_ylabel('Mean Observed AOP (psia)')
    ax.set_title(mlabel)
    ax.legend(fontsize=8)

fig.suptitle('Calibration Plots — All Models (Predicted Deciles)',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig2a_calibration_individual.png')
plt.show()

# ---- 2b) All models on one calibration plot ---
fig, ax = plt.subplots(figsize=(6, 5.5))
all_xp = np.concatenate([_calibration_data(y_true, preds_dict[m])[0] for m in MODEL_NAMES])
all_yo = np.concatenate([_calibration_data(y_true, preds_dict[m])[1] for m in MODEL_NAMES])
lims = [min(all_xp.min(), all_yo.min()) * 0.97,
        max(all_xp.max(), all_yo.max()) * 1.03]
ax.plot(lims, lims, 'k--', lw=0.9, zorder=0, label='Ideal')

for mname, mlabel, col, mkr in zip(MODEL_NAMES, MODEL_LABELS, COLORS, MARKERS):
    xp, yo = _calibration_data(y_true, preds_dict[mname])
    ax.plot(xp, yo, color=col, marker=mkr, ms=5, lw=1.1, label=mlabel)

ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Mean Predicted AOP (psia)')
ax.set_ylabel('Mean Observed AOP (psia)')
ax.set_title('Calibration Plot — All Models Combined')
ax.legend(fontsize=7.5)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('fig2b_calibration_combined.png')
plt.show()


# ============================================================
#  FIGURE 3 — Feature Importance (tree-based, gain)
#  Works for XGBoost, RF, LightGBM, ExtraTrees
# ============================================================

feat_importance = {
    'XGBoost':     xgb_pipe.fit(X_train_raw, y_train).named_steps['model'].feature_importances_,
    'RandomForest':rf_pipe.fit(X_train_raw, y_train).named_steps['model'].feature_importances_,
    'LightGBM':    lgb_pipe.fit(X_train_raw, y_train).named_steps['model'].feature_importances_,
    'ExtraTrees':  et_pipe.fit(X_train_raw, y_train).named_steps['model'].feature_importances_,
}
feature_names = list(X_train_raw.columns)

# ---- 3a) Individual feature importance panels ---
base_model_names  = ['XGBoost', 'RandomForest', 'LightGBM', 'ExtraTrees']
base_model_labels = ['XGBoost', 'Random Forest', 'LightGBM', 'Extra Trees']
base_colors       = COLORS[:4]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, (mname, mlabel, col) in enumerate(
        zip(base_model_names, base_model_labels, base_colors)):
    ax   = axes[idx]
    imp  = feat_importance[mname]
    order = np.argsort(imp)          # ascending → horizontal bar goes left→right
    top_n = min(15, len(order))
    order = order[-top_n:]

    bars = ax.barh(np.array(feature_names)[order],
                   imp[order],
                   color=col, alpha=0.85, edgecolor='white', linewidth=0.4)
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title(mlabel)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    # value labels
    for bar, v in zip(bars, imp[order]):
        ax.text(v + imp.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', ha='left', fontsize=7)

fig.suptitle('Feature Importance — Base Models', fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig3a_feat_importance_individual.png')
plt.show()

# ---- 3b) All base models side-by-side grouped bar ---
top_k = 12
# rank by mean importance across 4 base models
mean_imp = np.mean([feat_importance[m] for m in base_model_names], axis=0)
top_idx  = np.argsort(mean_imp)[-top_k:][::-1]
top_feats = np.array(feature_names)[top_idx]

x      = np.arange(top_k)
n_mod  = len(base_model_names)
width  = 0.18

fig, ax = plt.subplots(figsize=(12, 5))
for i, (mname, mlabel, col) in enumerate(
        zip(base_model_names, base_model_labels, base_colors)):
    offset = (i - n_mod / 2 + 0.5) * width
    ax.bar(x + offset, feat_importance[mname][top_idx],
           width=width, label=mlabel, color=col,
           alpha=0.85, edgecolor='white', linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(top_feats, rotation=35, ha='right')
ax.set_ylabel('Feature Importance (Gain)')
ax.set_title('Feature Importance — Base Models Comparison (Top 12 Features)')
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig('fig3b_feat_importance_combined.png')
plt.show()


# ============================================================
#  FIGURE 4 — Permutation Importance (all 6 models)
# ============================================================

# Re-fit base models if not already stored as fitted objects
fitted_models = {}
for mname, pipe in zip(['XGBoost','RandomForest','LightGBM','ExtraTrees'],
                       [xgb_pipe, rf_pipe, lgb_pipe, et_pipe]):
    fitted_models[mname] = clone(pipe).fit(X_train_raw, y_train)
fitted_models['Stacking_Huber']  = stacker
fitted_models['Blending_Huber']  = blender_huber

perm_results = {}
for mname in MODEL_NAMES:
    pi = permutation_importance(
        fitted_models[mname], X_test_raw, y_test,
        scoring='r2', n_repeats=20, random_state=42, n_jobs=-1)
    perm_results[mname] = pi.importances_mean

# ---- 4a) Individual permutation importance panels (2×3) ---
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for idx, (mname, mlabel, col) in enumerate(
        zip(MODEL_NAMES, MODEL_LABELS, COLORS)):
    ax   = axes[idx]
    imp  = perm_results[mname]
    order = np.argsort(imp)
    top_n = min(15, len(order))
    order = order[-top_n:]

    bars = ax.barh(np.array(feature_names)[order],
                   imp[order],
                   color=col, alpha=0.85, edgecolor='white', linewidth=0.4)
    ax.axvline(0, color='k', lw=0.6, ls='--')
    ax.set_xlabel('Permutation Importance (mean ΔR²)')
    ax.set_title(mlabel)
    ax.xaxis.set_major_locator(MaxNLocator(4))

fig.suptitle('Permutation Importance — All Models',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig4a_perm_importance_individual.png')
plt.show()

# ---- 4b) All models side-by-side grouped bar (top 10 by mean) ---
top_k  = 10
mean_pi = np.mean([perm_results[m] for m in MODEL_NAMES], axis=0)
top_idx = np.argsort(mean_pi)[-top_k:][::-1]
top_feats_pi = np.array(feature_names)[top_idx]

x     = np.arange(top_k)
n_mod = len(MODEL_NAMES)
width = 0.12

fig, ax = plt.subplots(figsize=(13, 5))
for i, (mname, mlabel, col) in enumerate(zip(MODEL_NAMES, MODEL_LABELS, COLORS)):
    offset = (i - n_mod / 2 + 0.5) * width
    ax.bar(x + offset, perm_results[mname][top_idx],
           width=width, label=mlabel, color=col,
           alpha=0.85, edgecolor='white', linewidth=0.3)

ax.axhline(0, color='k', lw=0.6, ls='--')
ax.set_xticks(x)
ax.set_xticklabels(top_feats_pi, rotation=35, ha='right')
ax.set_ylabel('Permutation Importance (mean ΔR²)')
ax.set_title('Permutation Importance — All Models (Top 10 Features)')
ax.legend(ncol=3, fontsize=7.5)
plt.tight_layout()
plt.savefig('fig4b_perm_importance_combined.png')
plt.show()


# ============================================================
#  FIGURE 5 — MAPE Comparison (individual + combined bar)
# ============================================================

mape_vals = {}
for mname in MODEL_NAMES:
    mape_vals[mname] = mean_absolute_percentage_error(y_true, preds_dict[mname]) * 100.0

# ---- 5a) Grouped bar: MAPE for all models ---
fig, ax = plt.subplots(figsize=(8, 4.5))
x_pos  = np.arange(len(MODEL_NAMES))
bars   = ax.bar(x_pos, [mape_vals[m] for m in MODEL_NAMES],
                color=COLORS, alpha=0.88,
                edgecolor='white', linewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(MODEL_LABELS, rotation=25, ha='right')
ax.set_ylabel('MAPE (%)')
ax.set_title('MAPE Comparison — All Models')

# value labels on bars
for bar, mname in zip(bars, MODEL_NAMES):
    v = mape_vals[mname]
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
            f'{v:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig('fig5a_mape_comparison.png')
plt.show()

# ---- 5b) Horizontal MAPE lollipop chart (alternative clean style) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
sorted_idx   = np.argsort([mape_vals[m] for m in MODEL_NAMES])
sorted_names = [MODEL_NAMES[i]  for i in sorted_idx]
sorted_labels= [MODEL_LABELS[i] for i in sorted_idx]
sorted_cols  = [COLORS[i]       for i in sorted_idx]
sorted_vals  = [mape_vals[m]    for m in sorted_names]

ax.hlines(range(len(MODEL_NAMES)), 0, sorted_vals,
          colors='0.80', linewidth=1.2)
ax.scatter(sorted_vals, range(len(MODEL_NAMES)),
           color=sorted_cols, s=80, zorder=5)

for y_pos, v in enumerate(sorted_vals):
    ax.text(v + 0.05, y_pos, f'{v:.2f}%', va='center', fontsize=8.5)

ax.set_yticks(range(len(MODEL_NAMES)))
ax.set_yticklabels(sorted_labels)
ax.set_xlabel('MAPE (%)')
ax.set_title('MAPE Comparison — All Models (Sorted)')
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout()
plt.savefig('fig5b_mape_lollipop.png')
plt.show()


# ============================================================
#  FIGURE 6 — MAPE per model as individual horizontal bar
#  (one subplot per model — for supplementary / appendix)
# ============================================================

# This one shows MAPE broken down by y_true quantile bins per model
def _mape_by_bins(y_true, y_pred, q=5):
    df = pd.DataFrame({'yt': y_true, 'yp': y_pred})
    df['bin'] = pd.qcut(df['yt'], q=q, duplicates='drop')
    out = df.groupby('bin', observed=True).apply(
        lambda g: mean_absolute_percentage_error(g['yt'], g['yp']) * 100.0
    ).reset_index(name='MAPE')
    out['label'] = out['bin'].astype(str)
    return out

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for idx, (mname, mlabel, col) in enumerate(
        zip(MODEL_NAMES, MODEL_LABELS, COLORS)):
    ax  = axes[idx]
    out = _mape_by_bins(y_true, preds_dict[mname], q=5)
    ax.barh(out['label'], out['MAPE'], color=col,
            alpha=0.85, edgecolor='white', linewidth=0.4)
    ax.set_xlabel('MAPE (%)')
    ax.set_title(mlabel)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    for i, v in enumerate(out['MAPE']):
        ax.text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=7.5)

fig.suptitle('MAPE by AOP Range (Quantile Bins) — All Models',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig6_mape_by_range_individual.png')
plt.show()

print("\n✓ All figures saved:")
print("  fig1a_parity_individual.png")
print("  fig1b_parity_combined.png")
print("  fig2a_calibration_individual.png")
print("  fig2b_calibration_combined.png")
print("  fig3a_feat_importance_individual.png")
print("  fig3b_feat_importance_combined.png")
print("  fig4a_perm_importance_individual.png")
print("  fig4b_perm_importance_combined.png")
print("  fig5a_mape_comparison.png")
print("  fig5b_mape_lollipop.png")
print("  fig6_mape_by_range_individual.png")



# =============================================================================
# ادامه کد: ذخیره‌سازی، بارگذاری مجدد، ارزیابی و رسم مجدد تمام نمودارها
# (نسخه اصلاح‌شده برای رفع خطای feature_names)
# =============================================================================

import joblib

# ---------------------------------------------------------------------------
# 1. ذخیره‌سازی تمام مدل‌ها
# ---------------------------------------------------------------------------
# print("\n" + "="*60)
# print("SAVING ALL MODELS TO DISK ...")
# print("="*60)

# # مدل‌های پایه را fit کن (اگر قبلاً fit نشده باشند)
# xgb_pipe_fitted = xgb_pipe.fit(X_train_raw, y_train)
# rf_pipe_fitted  = rf_pipe.fit(X_train_raw, y_train)
# lgb_pipe_fitted = lgb_pipe.fit(X_train_raw, y_train)
# et_pipe_fitted  = et_pipe.fit(X_train_raw, y_train)

# joblib.dump(xgb_pipe_fitted, 'xgb_pipe.pkl')
# joblib.dump(rf_pipe_fitted,  'rf_pipe.pkl')
# joblib.dump(lgb_pipe_fitted, 'lgb_pipe.pkl')
# joblib.dump(et_pipe_fitted,  'et_pipe.pkl')
# joblib.dump(stacker,         'stacking_huber.pkl')
# joblib.dump(blender_huber,   'blending_huber.pkl')

# print("All models saved as .pkl files.")

# # ---------------------------------------------------------------------------
# 2. بارگذاری مدل‌ها از دیسک
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("LOADING MODELS FROM DISK ...")
print("="*60)

loaded_xgb     = joblib.load('xgb_pipe.pkl')
loaded_rf      = joblib.load('rf_pipe.pkl')
loaded_lgb     = joblib.load('lgb_pipe.pkl')
loaded_et      = joblib.load('et_pipe.pkl')
loaded_stacker = joblib.load('stacking_huber.pkl')
loaded_blender = joblib.load('blending_huber.pkl')

print("All models loaded successfully.")

# ---------------------------------------------------------------------------
# 3. پیش‌بینی روی داده تست با مدل‌های بارگذاری‌شده
# ---------------------------------------------------------------------------
base_test_preds_loaded = {
    'xgb': loaded_xgb.predict(X_test_raw),
    'rf':  loaded_rf.predict(X_test_raw),
    'lgb': loaded_lgb.predict(X_test_raw),
    'et':  loaded_et.predict(X_test_raw),
}

stacking_huber_test_pred_loaded = loaded_stacker.predict(X_test_raw)
blending_huber_test_pred_loaded = loaded_blender.predict(X_test_raw)

# دیکشنری نهایی برای نمودارها (همان ساختار قبلی)
preds_dict_loaded = {
    'XGBoost':        base_test_preds_loaded['xgb'],
    'RandomForest':   base_test_preds_loaded['rf'],
    'LightGBM':       base_test_preds_loaded['lgb'],
    'ExtraTrees':     base_test_preds_loaded['et'],
    'Stacking_Huber': stacking_huber_test_pred_loaded,
    'Blending_Huber': blending_huber_test_pred_loaded,
}

# ---------------------------------------------------------------------------
# 4. چاپ معیارها روی داده تست (مدل‌های بارگذاری‌شده)
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("EVALUATION ON TEST DATA WITH LOADED MODELS")
print("="*80)
print(f"{'Method':<25} | {'R²':<6} | {'MAE':<6} | {'MEDAE':<6} | {'RMSE':<6} | {'MAPE':<6}")
print("-"*80)

for mname, mlabel in zip(MODEL_NAMES, MODEL_LABELS):
    ypred = preds_dict_loaded[mname]
    _ = enhanced_report_metrics(y_test, ypred, mlabel)

# ---------------------------------------------------------------------------
# 5. بازتعریف متغیرهای ضروری برای رسم نمودارها
# ---------------------------------------------------------------------------
preds_dict      = preds_dict_loaded
y_true          = y_test.values
feature_names   = list(X_train_raw.columns)   # <--- این خط اضافه شد تا خطا برطرف شود

# importance مبتنی بر gain از پایپلاین‌های بارگذاری‌شده
feat_importance = {
    'XGBoost':      loaded_xgb.named_steps['model'].feature_importances_,
    'RandomForest': loaded_rf.named_steps['model'].feature_importances_,
    'LightGBM':     loaded_lgb.named_steps['model'].feature_importances_,
    'ExtraTrees':   loaded_et.named_steps['model'].feature_importances_,
}

# مدل‌های fit شده برای permutation importance
fitted_models = {
    'XGBoost':        loaded_xgb,
    'RandomForest':   loaded_rf,
    'LightGBM':       loaded_lgb,
    'ExtraTrees':     loaded_et,
    'Stacking_Huber': loaded_stacker,
    'Blending_Huber': loaded_blender,
}

# ---------------------------------------------------------------------------
# 6. رسم مجدد تمام نمودارها با مدل‌های بارگذاری‌شده
#    (نام فایل‌ها با "_loaded" تغییر می‌کند)
# ---------------------------------------------------------------------------

# ============================================================
# FIGURE 1 — Parity Plots (بارگذاری‌شده)
# ============================================================

# ---- 1a) تکی در یک گرید 2×3 ----
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (mname, mlabel, col, mkr) in enumerate(
        zip(MODEL_NAMES, MODEL_LABELS, COLORS, MARKERS)):
    ax   = axes[idx]
    ypred = preds_dict[mname]
    r2    = r2_score(y_true, ypred)

    ax.scatter(y_true, ypred,
               color=col, marker=mkr, s=28, alpha=0.75,
               linewidths=0.3, edgecolors='white', label=mlabel)

    lims = [min(y_true.min(), ypred.min()) * 0.97,
            max(y_true.max(), ypred.max()) * 1.03]
    ax.plot(lims, lims, 'k--', lw=0.9, zorder=0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Observed AOP (psia)')
    ax.set_ylabel('Predicted AOP (psia)')
    ax.set_title(mlabel)
    ax.text(0.05, 0.92, f'$R^2$ = {r2:.3f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.85))
    ax.set_aspect('equal', adjustable='box')

fig.suptitle('Parity Plots — All Models (Loaded)', fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig1a_parity_individual_loaded.png')
plt.show()

# ---- 1b) همه در یک نمودار ----
fig, ax = plt.subplots(figsize=(6, 5.5))
all_vals = np.concatenate([y_true] + list(preds_dict.values()))
lims = [all_vals.min() * 0.97, all_vals.max() * 1.03]
ax.plot(lims, lims, 'k--', lw=0.9, zorder=0, label='Ideal')

for mname, mlabel, col, mkr in zip(MODEL_NAMES, MODEL_LABELS, COLORS, MARKERS):
    ypred = preds_dict[mname]
    r2    = r2_score(y_true, ypred)
    ax.scatter(y_true, ypred,
               color=col, marker=mkr, s=22, alpha=0.65,
               linewidths=0.2, edgecolors='white',
               label=f'{mlabel}  ($R^2$={r2:.3f})')

ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Observed AOP (psia)')
ax.set_ylabel('Predicted AOP (psia)')
ax.set_title('Parity Plot — All Models Combined (Loaded)')
ax.legend(loc='upper left', fontsize=7.5)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('fig1b_parity_combined_loaded.png')
plt.show()


# ============================================================
# FIGURE 2 — Calibration Plots (بارگذاری‌شده)
# ============================================================

def _calibration_data(y_true, y_pred, n_deciles=10):
    df = pd.DataFrame({'yt': y_true, 'yp': y_pred}).sort_values('yp')
    df['dec'] = pd.qcut(df['yp'], n_deciles, labels=False, duplicates='drop')
    agg = df.groupby('dec').agg(yp_mean=('yp','mean'), yt_mean=('yt','mean')).reset_index()
    return agg['yp_mean'].values, agg['yt_mean'].values

# ---- 2a) تکی ----
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (mname, mlabel, col, mkr) in enumerate(
        zip(MODEL_NAMES, MODEL_LABELS, COLORS, MARKERS)):
    ax = axes[idx]
    xp, yo = _calibration_data(y_true, preds_dict[mname])
    lims = [min(xp.min(), yo.min()) * 0.97, max(xp.max(), yo.max()) * 1.03]

    ax.plot(lims, lims, 'k--', lw=0.9, zorder=0, label='Ideal')
    ax.plot(xp, yo, color=col, marker=mkr, ms=6, lw=1.2, label=mlabel)

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Mean Predicted AOP (psia)')
    ax.set_ylabel('Mean Observed AOP (psia)')
    ax.set_title(mlabel)
    ax.legend(fontsize=8)

fig.suptitle('Calibration Plots — All Models (Loaded)',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig2a_calibration_individual_loaded.png')
plt.show()

# ---- 2b) همه در یک نمودار ----
fig, ax = plt.subplots(figsize=(6, 5.5))
all_xp = np.concatenate([_calibration_data(y_true, preds_dict[m])[0] for m in MODEL_NAMES])
all_yo = np.concatenate([_calibration_data(y_true, preds_dict[m])[1] for m in MODEL_NAMES])
lims = [min(all_xp.min(), all_yo.min()) * 0.97,
        max(all_xp.max(), all_yo.max()) * 1.03]
ax.plot(lims, lims, 'k--', lw=0.9, zorder=0, label='Ideal')

for mname, mlabel, col, mkr in zip(MODEL_NAMES, MODEL_LABELS, COLORS, MARKERS):
    xp, yo = _calibration_data(y_true, preds_dict[mname])
    ax.plot(xp, yo, color=col, marker=mkr, ms=5, lw=1.1, label=mlabel)

ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Mean Predicted AOP (psia)')
ax.set_ylabel('Mean Observed AOP (psia)')
ax.set_title('Calibration Plot — All Models Combined (Loaded)')
ax.legend(fontsize=7.5)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('fig2b_calibration_combined_loaded.png')
plt.show()


# ============================================================
# FIGURE 3 — Feature Importance (gain) (بارگذاری‌شده)
# ============================================================

base_model_names  = ['XGBoost', 'RandomForest', 'LightGBM', 'ExtraTrees']
base_model_labels = ['XGBoost', 'Random Forest', 'LightGBM', 'Extra Trees']
base_colors       = COLORS[:4]

# ---- 3a) تکی ----
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, (mname, mlabel, col) in enumerate(
        zip(base_model_names, base_model_labels, base_colors)):
    ax   = axes[idx]
    imp  = feat_importance[mname]
    order = np.argsort(imp)
    top_n = min(15, len(order))
    order = order[-top_n:]

    bars = ax.barh(np.array(feature_names)[order],
                   imp[order],
                   color=col, alpha=0.85, edgecolor='white', linewidth=0.4)
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title(mlabel)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    for bar, v in zip(bars, imp[order]):
        ax.text(v + imp.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', ha='left', fontsize=7)

fig.suptitle('Feature Importance — Base Models (Loaded)', fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig3a_feat_importance_individual_loaded.png')
plt.show()

# ---- 3b) مقایسه‌ای ----
top_k = 12
mean_imp = np.mean([feat_importance[m] for m in base_model_names], axis=0)
top_idx  = np.argsort(mean_imp)[-top_k:][::-1]
top_feats = np.array(feature_names)[top_idx]

x      = np.arange(top_k)
n_mod  = len(base_model_names)
width  = 0.18

fig, ax = plt.subplots(figsize=(12, 5))
for i, (mname, mlabel, col) in enumerate(
        zip(base_model_names, base_model_labels, base_colors)):
    offset = (i - n_mod / 2 + 0.5) * width
    ax.bar(x + offset, feat_importance[mname][top_idx],
           width=width, label=mlabel, color=col,
           alpha=0.85, edgecolor='white', linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(top_feats, rotation=35, ha='right')
ax.set_ylabel('Feature Importance (Gain)')
ax.set_title('Feature Importance — Base Models Comparison (Loaded)')
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig('fig3b_feat_importance_combined_loaded.png')
plt.show()


# ============================================================
# FIGURE 4 — Permutation Importance (بارگذاری‌شده)
# ============================================================

perm_results = {}
for mname in MODEL_NAMES:
    pi = permutation_importance(
        fitted_models[mname], X_test_raw, y_test,
        scoring='r2', n_repeats=20, random_state=42, n_jobs=-1)
    perm_results[mname] = pi.importances_mean

# ---- 4a) تکی ----
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for idx, (mname, mlabel, col) in enumerate(
        zip(MODEL_NAMES, MODEL_LABELS, COLORS)):
    ax   = axes[idx]
    imp  = perm_results[mname]
    order = np.argsort(imp)
    top_n = min(15, len(order))
    order = order[-top_n:]

    bars = ax.barh(np.array(feature_names)[order],
                   imp[order],
                   color=col, alpha=0.85, edgecolor='white', linewidth=0.4)
    ax.axvline(0, color='k', lw=0.6, ls='--')
    ax.set_xlabel('Permutation Importance (mean ΔR²)')
    ax.set_title(mlabel)
    ax.xaxis.set_major_locator(MaxNLocator(4))

fig.suptitle('Permutation Importance — All Models (Loaded)',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig4a_perm_importance_individual_loaded.png')
plt.show()

# ---- 4b) مقایسه‌ای ----
top_k  = 10
mean_pi = np.mean([perm_results[m] for m in MODEL_NAMES], axis=0)
top_idx = np.argsort(mean_pi)[-top_k:][::-1]
top_feats_pi = np.array(feature_names)[top_idx]

x     = np.arange(top_k)
n_mod = len(MODEL_NAMES)
width = 0.12

fig, ax = plt.subplots(figsize=(13, 5))
for i, (mname, mlabel, col) in enumerate(zip(MODEL_NAMES, MODEL_LABELS, COLORS)):
    offset = (i - n_mod / 2 + 0.5) * width
    ax.bar(x + offset, perm_results[mname][top_idx],
           width=width, label=mlabel, color=col,
           alpha=0.85, edgecolor='white', linewidth=0.3)

ax.axhline(0, color='k', lw=0.6, ls='--')
ax.set_xticks(x)
ax.set_xticklabels(top_feats_pi, rotation=35, ha='right')
ax.set_ylabel('Permutation Importance (mean ΔR²)')
ax.set_title('Permutation Importance — All Models (Loaded)')
ax.legend(ncol=3, fontsize=7.5)
plt.tight_layout()
plt.savefig('fig4b_perm_importance_combined_loaded.png')
plt.show()


# ============================================================
# FIGURE 5 — MAPE Comparison (بارگذاری‌شده)
# ============================================================

mape_vals = {}
for mname in MODEL_NAMES:
    mape_vals[mname] = mean_absolute_percentage_error(y_true, preds_dict[mname]) * 100.0

# ---- 5a) میله‌ای گروهی ----
fig, ax = plt.subplots(figsize=(8, 4.5))
x_pos  = np.arange(len(MODEL_NAMES))
bars   = ax.bar(x_pos, [mape_vals[m] for m in MODEL_NAMES],
                color=COLORS, alpha=0.88,
                edgecolor='white', linewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(MODEL_LABELS, rotation=25, ha='right')
ax.set_ylabel('MAPE (%)')
ax.set_title('MAPE Comparison — All Models (Loaded)')

for bar, mname in zip(bars, MODEL_NAMES):
    v = mape_vals[mname]
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
            f'{v:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig('fig5a_mape_comparison_loaded.png')
plt.show()

# ---- 5b) نمودار آبنباتی (lollipop) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
sorted_idx   = np.argsort([mape_vals[m] for m in MODEL_NAMES])
sorted_names = [MODEL_NAMES[i]  for i in sorted_idx]
sorted_labels= [MODEL_LABELS[i] for i in sorted_idx]
sorted_cols  = [COLORS[i]       for i in sorted_idx]
sorted_vals  = [mape_vals[m]    for m in sorted_names]

ax.hlines(range(len(MODEL_NAMES)), 0, sorted_vals,
          colors='0.80', linewidth=1.2)
ax.scatter(sorted_vals, range(len(MODEL_NAMES)),
           color=sorted_cols, s=80, zorder=5)

for y_pos, v in enumerate(sorted_vals):
    ax.text(v + 0.05, y_pos, f'{v:.2f}%', va='center', fontsize=8.5)

ax.set_yticks(range(len(MODEL_NAMES)))
ax.set_yticklabels(sorted_labels)
ax.set_xlabel('MAPE (%)')
ax.set_title('MAPE Comparison — All Models (Loaded, Sorted)')
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout()
plt.savefig('fig5b_mape_lollipop_loaded.png')
plt.show()


# ============================================================
# FIGURE 6 — MAPE by AOP Range (بارگذاری‌شده)
# ============================================================

def _mape_by_bins(y_true, y_pred, q=5):
    df = pd.DataFrame({'yt': y_true, 'yp': y_pred})
    df['bin'] = pd.qcut(df['yt'], q=q, duplicates='drop')
    out = df.groupby('bin', observed=True).apply(
        lambda g: mean_absolute_percentage_error(g['yt'], g['yp']) * 100.0
    ).reset_index(name='MAPE')
    out['label'] = out['bin'].astype(str)
    return out

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for idx, (mname, mlabel, col) in enumerate(
        zip(MODEL_NAMES, MODEL_LABELS, COLORS)):
    ax  = axes[idx]
    out = _mape_by_bins(y_true, preds_dict[mname], q=5)
    ax.barh(out['label'], out['MAPE'], color=col,
            alpha=0.85, edgecolor='white', linewidth=0.4)
    ax.set_xlabel('MAPE (%)')
    ax.set_title(mlabel)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    for i, v in enumerate(out['MAPE']):
        ax.text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=7.5)

fig.suptitle('MAPE by AOP Range (Quantile Bins) — All Models (Loaded)',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig6_mape_by_range_individual_loaded.png')
plt.show()

print("\n✓ All figures for loaded models saved:")
print("  fig1a_parity_individual_loaded.png")
print("  fig1b_parity_combined_loaded.png")
print("  fig2a_calibration_individual_loaded.png")
print("  fig2b_calibration_combined_loaded.png")
print("  fig3a_feat_importance_individual_loaded.png")
print("  fig3b_feat_importance_combined_loaded.png")
print("  fig4a_perm_importance_individual_loaded.png")
print("  fig4b_perm_importance_combined_loaded.png")
print("  fig5a_mape_comparison_loaded.png")
print("  fig5b_mape_lollipop_loaded.png")
print("  fig6_mape_by_range_individual_loaded.png")



