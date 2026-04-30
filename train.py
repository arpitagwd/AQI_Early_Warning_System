"""
train.py
Full ML training pipeline for AQI Early Warning System.

Steps:
  1. Load + clean Kaggle city_day.csv dataset
  2. Feature engineering (lag, rolling, seasonal, festival)
  3. SMOTE for class imbalance
  4. Train 3 models: Random Forest, XGBoost, LSTM
  5. Evaluate + compare all models
  6. SHAP feature importance
  7. Save best model + scaler + encoders

accuracy_score → correctness
f1_score → balance metric
confusion_matrix → errors
mean_squared_error → regression error
mean_absolute_error → average error



LSTM : Time-series model (VERY IMPORTANT for AQI trends)
Dense : Normal neural layer
Dropout : Prevents overfitting




🔹 PM2.5
Very tiny particles (dangerous )
Enter lungs & bloodstream
🔹 PM10
Larger dust particles
Cause breathing issues
Nitrogen Compounds
🔹 NO (Nitric Oxide)
🔹 NO2 (Nitrogen Dioxide)
🔹 NOx (Nitrogen Oxides)


Lung irritation
Smog formation
 Other Harmful Gases
🔹 NH3 (Ammonia)
From agriculture 
🔹 CO (Carbon Monoxide)
From incomplete burning (cars, fuel)
Very dangerous  (reduces oxygen in blood)
🔹 SO2 (Sulfur Dioxide)
From coal burning
Causes acid rain 
🔹 O3 (Ozone - ground level)
Not the good ozone layer 
Forms due to sunlight + pollution
Causes breathing problems
Toxic Chemicals (VOCs)
🔹 Benzene
From fuel, smoke
Cancer-causing 
🔹 Toluene
From paints, chemicals
Affects nervous system
"""

import os
import warnings
import numpy as np
import pandas as pd # Handles datasets (like Excel tables)
import matplotlib.pyplot as plt
import seaborn as sns
import shap #Explains ML predictions
import joblib #Saves trained models

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True) # Creates folder to save trained models
os.makedirs("plots",  exist_ok=True)

from sklearn.model_selection     import train_test_split
from sklearn.ensemble            import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors           import KNeighborsClassifier
from sklearn.preprocessing       import LabelEncoder, StandardScaler
from sklearn.metrics             import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error
)
from xgboost                     import XGBClassifier, XGBRegressor
from imblearn.over_sampling      import SMOTE #Fixes imbalanced data(balances the dataset)

import tensorflow as tf #deep learning framework
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import LSTM, Dense, Dropout
from tensorflow.keras.callbacks  import EarlyStopping

# CONSTANTS
DATA_PATH    = "data/city_day.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
LOOKBACK     = 7      # LSTM looks at the past 7 days to predict the next day.

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"
]

FESTIVAL_MONTHS = [10, 11]   # Oct–Nov spike season



# 1. LOAD + CLEAN

def load_and_clean(path: str) -> pd.DataFrame:
    print("\n[1] Loading data...")
    df = pd.read_csv(path, parse_dates=["Date"])

    print(f"    Raw shape: {df.shape}")
    print(f"    Cities   : {df['City'].nunique()}")
    print(f"    Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

    # Drop rows with no target
    df.dropna(subset=["AQI", "AQI_Bucket"], inplace=True)

    # City-wise median imputation for pollutants
    for col in POLLUTANTS:
        #for missing pollutant readings, it fills them with the city-wise median 
        df[col] = df.groupby("City")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Drop rows still missing after imputation
    df.dropna(subset=POLLUTANTS, inplace=True)
    df.sort_values(["City", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"    Clean shape: {df.shape}")
    return df


# 2. FEATURE ENGINEERING

# def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
#     print("\n[2] Engineering features...")

#     df["Month"]  = df["Date"].dt.month
#     df["DayOfWeek"] = df["Date"].dt.dayofweek
#     df["Season"] = df["Month"].map({
#         12: "Winter", 1: "Winter", 2: "Winter",
#         3:  "Spring", 4: "Spring", 5: "Spring",
#         6:  "Summer", 7: "Summer", 8: "Summer",
#         9:  "Autumn", 10:"Autumn", 11:"Autumn"
#     })

#     # Festival indicator (Oct–Nov Diwali / crop-burning)
#     df["IsFestivalSeason"] = df["Month"].isin(FESTIVAL_MONTHS).astype(int)

#     # Pollution ratios
#     df["PM_ratio"]  = df["PM2.5"] / (df["PM10"]  + 1e-5)
#     df["NOx_ratio"] = df["NO2"]   / (df["NOx"]   + 1e-5)

#     # Lag features (per city) — t-1, t-3, t-7
#     for lag in [1, 3, 7]:
#         df[f"AQI_lag{lag}"] = df.groupby("City")["AQI"].shift(lag)

#     # Rolling statistics (7-day window per city)
#     df["AQI_roll7_mean"] = (
#         df.groupby("City")["AQI"]
#         .transform(lambda x: x.rolling(7, min_periods=1).mean())
#     )
#     df["AQI_roll7_std"] = (
#         df.groupby("City")["AQI"]
#         .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
#     )

#     # AQI day-over-day delta
#     df["AQI_delta"] = df.groupby("City")["AQI"].diff().fillna(0)

#     df.dropna(subset=["AQI_lag1", "AQI_lag3", "AQI_lag7"], inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     print(f"    Features added. Shape: {df.shape}")
#     return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2] Engineering features...")

    df["Month"]     = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # India-specific season mapping 
    # Monsoon (Jul–Sep) is the most critical — rain cleans the air
    # Post-Monsoon (Oct–Nov) is worst — Diwali + crop burning spikes
    df["Season"] = df["Month"].map({
        12: "Winter",       # Cold inversion traps pollutants
        1:  "Winter",
        2:  "Winter",
        3:  "PreSummer",    # Dry, dusty, winds pick up
        4:  "PreSummer",
        5:  "Summer",       # Peak heat, high O3 formation
        6:  "Summer",       # Pre-monsoon heat
        7:  "Monsoon",      # Rain → AQI drops sharply
        8:  "Monsoon",
        9:  "Monsoon",
        10: "PostMonsoon",  # Diwali firecrackers + stubble burning
        11: "PostMonsoon",  # Worst AQI months of the year
    })

    # Binary flags for the two most polluted seasons
 
    df["IsMonsoon"]      = df["Month"].isin([7, 8, 9]).astype(int)
    df["IsPostMonsoon"]  = df["Month"].isin([10, 11]).astype(int)
    df["IsWinter"]       = df["Month"].isin([12, 1, 2]).astype(int)

   #Creates a new column IsFestivalSeason n Copies values from IsPostMonsoon
    df["IsFestivalSeason"] = df["IsPostMonsoon"]  # keep for backward compat

    # Pollution ratios
    #PM_ratio → fine particles vs coarse particles
    #NOx_ratio → NO2 vs total nitrogen oxides
#Instead of raw values: Gives relative pollution intensity
                        # Helps model understand composition

    df["PM_ratio"]  = df["PM2.5"] / (df["PM10"]  + 1e-5) # to avoid division by zero error
    df["NOx_ratio"] = df["NO2"]   / (df["NOx"]   + 1e-5)

    # Lag features (per city) — t-1(yesterdays), t-3( 3 day ago), t-7(1 week ago)
    for lag in [1, 3, 7]:
        df[f"AQI_lag{lag}"] = df.groupby("City")["AQI"].shift(lag) #shift(1) moves the entire column down by 1 row — so row i gets the value that was in row i-1.

    # Rolling statistics (7-day window per city) Computes 7-day moving average AQI
    df["AQI_roll7_mean"] = (
        df.groupby("City")["AQI"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["AQI_roll7_std"] = (
        df.groupby("City")["AQI"]
        .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
    )
# “How much did AQI change compared to the previous day?”
    df["AQI_delta"] = df.groupby("City")["AQI"].diff().fillna(0)

    df.dropna(subset=["AQI_lag1", "AQI_lag3", "AQI_lag7"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"    Features added. Shape: {df.shape}")
    return df

# 3. ENCODE + SPLIT

def encode_and_split(df: pd.DataFrame):
    print("\n[3] Encoding and splitting...")

    le_city   = LabelEncoder()
    le_season = LabelEncoder()
    le_target = LabelEncoder() # Encode AQI category

    df["City_enc"]   = le_city.fit_transform(df["City"]) # fittransform--Learn mapping + apply it
    df["Season_enc"] = le_season.fit_transform(df["Season"])
    df["Target"]     = le_target.fit_transform(df["AQI_Bucket"])

    FEATURES = (
        POLLUTANTS
        + ["Month", "DayOfWeek", "IsFestivalSeason",
           "PM_ratio", "NOx_ratio",
           "AQI_lag1", "AQI_lag3", "AQI_lag7",
           "AQI_roll7_mean", "AQI_roll7_std",
           "AQI_delta", "City_enc", "Season_enc"]
    )

    X = df[FEATURES]
    y_cls = df["Target"]       # classification target (AQI bucket)
    y_reg = df["AQI"]          # regression target (numeric AQI)

    print(f"    Feature count: {len(FEATURES)}")
    print(f"    Class distribution:\n{pd.Series(y_cls).value_counts()}")

    # Stratified split on classification target
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_cls, y_reg,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_cls # class distribution of AQI categories remains consistent across training and testing datasets
    )

    return (X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te,
            FEATURES, le_city, le_season, le_target)


# 4. SMOTE(Synthetic Minority Oversampling Technique)
# Creates fake (synthetic) samples for minority classes --Balances dataset
# Prevents bias--Model doesn’t ignore small classes

def apply_smote(X_train, y_train):
    print("\n[4] Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    print(f"    Before: {dict(pd.Series(y_train).value_counts())}")
    print(f"    After : {dict(pd.Series(y_sm).value_counts())}")
    return X_sm, y_sm


# 5. SCALE-- normalizes your data so that all features are on a similar scale.
# std scaler : z=x−μ/σ
#features have different ranges:so we get them in comparable range 

def scale(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


# 6. TRAIN RANDOM FOREST
"""
Precision	Correct positive predictions
Recall	How many actual positives found
F1-score	Balance of both
"""
def train_random_forest(X_tr_sc, yc_tr_sm, X_te_sc, yc_te, le_target): #Scaled training features,Training labels (after SMOTE),Scaled test features,Test labels,LabelEncoder (for class names).
    print("\n[5a] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, #Number of trees = 300
        max_depth=20,# Maximum depth of each tree
        min_samples_leaf=2,# Minimum samples in leaf node-- Improves generalization
        random_state=RANDOM_STATE,#Ensures same results every time
        n_jobs=-1 #Uses all CPU cores, to train trees in parallel.
    )
    rf.fit(X_tr_sc, yc_tr_sm)# model learns : Patterns between features → AQI categories
    preds = rf.predict(X_te_sc)#Predicts AQI category for test data
    acc = accuracy_score(yc_te, preds) #% of correct predictions
    f1  = f1_score(yc_te, preds, average="weighted") #Balance of precision + recall  & weighted: Gives importance based on class size
    print(f"    Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
    print(classification_report(yc_te, preds, target_names=le_target.classes_))
    return rf, acc, f1


# 7. TRAIN XGBOOST
"""
XGBoost builds an ensemble of weak decision trees sequentially. 
Each new tree attempts to correct the errors (residuals) of the preceding trees
 by minimizing a loss function.
"""
def train_xgboost(X_tr_sc, yc_tr_sm, X_te_sc, yc_te, le_target):
    print("\n[5b] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=7,# low than rf--prevents overfitting
        learning_rate=0.08,# each new tree corrects only 8% of the previous error, making learning slow but more accurate
        subsample=0.8,# each tree sees only 80% of the training rows (sampled randomly), adding variance and reducing overfitting
        colsample_bytree=0.8, #each tree uses only 80% of the features
        use_label_encoder=False, #Disables old encoding warning
        eval_metric="mlogloss", #Metric for multi-class classification  : Measures prediction probability quality
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb.fit(X_tr_sc, yc_tr_sm)
    preds = xgb.predict(X_te_sc) #Predict AQI categories
    acc = accuracy_score(yc_te, preds)
    f1  = f1_score(yc_te, preds, average="weighted")
    print(f"    Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
    print(classification_report(yc_te, preds, target_names=le_target.classes_))
    return xgb, acc, f1


# 8. TRAIN LSTM (time-series regression → numeric AQI)
"""
Long Short-Term Memory (LSTM) is a specialized type of Recurrent Neural Network (RNN) architecture 
designed to learn and retain long-term dependencies in sequential data, 
overcoming the "vanishing gradient" limitation of standard RNNs.
Using unique memory cells and three gates (input, forget, output), 
LSTMs selectively store and update information.

lstm : expects -- (samples, time_steps, features)
"""

# lookback : 7 -- “Because weekly patterns are common in pollution data, so 7 days capture short-term trends effectively.”
def build_lstm_sequences(df: pd.DataFrame, features: list, lookback: int = 7):
    """Build (X_seq-past 7 days data, y_seq--next day AQI) arrays for LSTM from per-city time-series."""
    all_X, all_y = [], [] # Will store sequences and targets

    for city, group in df.groupby("City"):
        group = group.sort_values("Date") # Ensures correct time order
        vals  = group[features + ["AQI"]].values #Converts dataframe → numpy array :  [features..., AQI]

        for i in range(lookback, len(vals)): #Starts from day 7
            all_X.append(vals[i-lookback:i, :-1])   # past 7 days of features ,-1 : Exclude last column (AQI)
            all_y.append(vals[i, -1])                # next-day AQI

    return np.array(all_X), np.array(all_y)


def train_lstm(df_full: pd.DataFrame, features: list, scaler: StandardScaler):
    print("\n[5c] Training LSTM (7-day forecast)...")

    # Re-scale the full dataset features for sequence building
    df_sc = df_full.copy()
    df_sc[features] = scaler.transform(df_full[features])

    X_seq, y_seq = build_lstm_sequences(df_sc, features, LOOKBACK)
    print(f"    Sequence shape: {X_seq.shape}  |  Target shape: {y_seq.shape}")

    split = int(len(X_seq) * (1 - TEST_SIZE))
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    # Scale target (AQI values) separately
    #AQI values vary widely ,Scaling improves training stability
    aqi_scaler = StandardScaler()
    #reshape()--Converts 1D → 2D
    #ravel()--converts bacck to 1d
    y_tr_sc = aqi_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel() 
    y_te_sc = aqi_scaler.transform(y_te.reshape(-1, 1)).ravel()

    """
Layer by layer: first LSTM has 128 memory units, return_sequences=True means it passes its
output at every timestep to the next layer. Dropout(0.2) randomly turns off 20% of neurons 
during each training step — prevents memorisation. Second LSTM has 64 units,
return_sequences=False means it only outputs the final timestep's value.
Two Dense layers reduce to a single number — the predicted AQI.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(LOOKBACK, X_seq.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"), # Dense Layers
        Dense(1)  
        #dense(32) processes learned patterns, and Dense(1) outputs the final AQI value
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])# optimizer: Efficient learning
    #Stops training if no improvement
    early_stop = EarlyStopping(
        monitor="val_loss", patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        X_tr, y_tr_sc,
        validation_split=0.1,#10% validation
        epochs=60, #Max training cycles
        batch_size=64, #Data per batch
        callbacks=[early_stop],
        verbose=1
    )

    preds_sc = model.predict(X_te)
    preds    = aqi_scaler.inverse_transform(preds_sc).ravel() #Convert back to real AQI

    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae  = mean_absolute_error(y_te, preds)
    print(f"    LSTM RMSE: {rmse:.2f}  |  MAE: {mae:.2f}")

    # Plot training loss
    plt.figure(figsize=(8, 3))
    plt.plot(history.history["loss"],     label="Train loss") # history.history → dictionary of metrics per epoch
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.title("LSTM training loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/lstm_loss.png", dpi=150)
    plt.close()

    return model, aqi_scaler, rmse, mae


# 9. CONFUSION MATRIX PLOT
def plot_confusion_matrix(model, X_te_sc, yc_te, le_target, name: str):
    preds = model.predict(X_te_sc)
    cm    = confusion_matrix(yc_te, preds) #cm = 2D array of counts
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le_target.classes_,
        yticklabels=le_target.classes_
    )
    plt.title(f"Confusion matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"plots/cm_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.close()
    print(f"    Saved: plots/cm_{name.replace(' ', '_').lower()}.png")


# 10. MODEL COMPARISON CHART
def plot_model_comparison(results: dict):
    res_df = pd.DataFrame(results).T[["Accuracy", "F1"]]
    ax = res_df.plot(
        kind="bar", figsize=(7, 4),
        color=["#5DCAA5", "#7F77DD"], edgecolor="none"
    )
    plt.title("Model comparison — Accuracy & F1 (weighted)")
    plt.ylabel("Score")
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=150)
    plt.close()
    print("    Saved: plots/model_comparison.png")


# 11. SHAP((SHapley Additive exPlanations), which tells us how each feature contributes to the prediction.) FEATURE IMPORTANCE
def plot_shap(model, X_te_sc, feature_names: list, model_name: str):
    print(f"\n[7] SHAP analysis on {model_name}...")
    sample_size = min(300, len(X_te_sc)) #SHAP can be slow on large datasets, so we sample up to 300 rows
    X_sample    = X_te_sc[:sample_size]

    explainer   = shap.TreeExplainer(model) # TreeExplainer → specialized for tree-based models (RF, XGBoost)
    shap_values = explainer.shap_values(X_sample) #shap_values → array showing impact of each feature on each prediction

    # For multiclass pick class 0 (Good) vs rest — shows most informative split
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(
        sv, X_sample,
        feature_names=feature_names,
        max_display=15,
        show=False
    )
    plt.title(f"SHAP feature impact — {model_name}")
    plt.tight_layout()
    plt.savefig("plots/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: plots/shap_summary.png")


# 12. ANOMALY SPIKE DETECTION (Z-SCORE)
def flag_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """Add a spike flag column using rolling z-score per city."""
    df = df.copy()
    df["AQI_zscore"] = (
        df.groupby("City")["AQI"].transform(
            lambda x: (x - x.rolling(30, min_periods=5).mean())
                      / (x.rolling(30, min_periods=5).std() + 1e-5)
        )
    )
    df["SpikeFlag"] = (df["AQI_zscore"].abs() > z_threshold).astype(int)
    n_spikes = df["SpikeFlag"].sum()
    print(f"\n[8] Anomaly detection: {n_spikes} spike events flagged "
          f"({100*n_spikes/len(df):.1f}% of days)")
    return df


# MAIN
def main():
    print("=" * 60)
    print("  AQI EARLY WARNING SYSTEM — TRAINING PIPELINE")
    print("=" * 60)

    # 1–2. Load and engineer
    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)
    df = flag_anomalies(df)

    # 3. Encode + split
    (X_tr, X_te,
     yc_tr, yc_te,
     yr_tr, yr_te,
     FEATURES,
     le_city, le_season, le_target) = encode_and_split(df)

    # 4. SMOTE on training set
    X_tr_sm, yc_tr_sm = apply_smote(X_tr, yc_tr)

    # 5. Scale
    X_tr_sc, X_te_sc, scaler = scale(X_tr_sm, X_te)

    # 6–7. Train classifiers
    rf,  rf_acc,  rf_f1  = train_random_forest(X_tr_sc, yc_tr_sm, X_te_sc, yc_te, le_target)
    xgb, xgb_acc, xgb_f1 = train_xgboost(X_tr_sc, yc_tr_sm, X_te_sc, yc_te, le_target)

    # 8. Train LSTM (uses full df with all features)
    # Filter df to same feature columns used in classifiers
    df_lstm = df[["City", "Date", "AQI"] + FEATURES].dropna()
    lstm_model, aqi_scaler, lstm_rmse, lstm_mae = train_lstm(df_lstm, FEATURES, scaler)

    # 9. Plots
    print("\n[6] Generating plots...")
    results = {
        "Random Forest": {"Accuracy": rf_acc,  "F1": rf_f1},
        "XGBoost":       {"Accuracy": xgb_acc, "F1": xgb_f1},
    }
    plot_model_comparison(results)
    plot_confusion_matrix(rf,  X_te_sc, yc_te, le_target, "Random Forest")
    plot_confusion_matrix(xgb, X_te_sc, yc_te, le_target, "XGBoost")

    # Pick best classifier for SHAP
    best_clf  = rf if rf_f1 >= xgb_f1 else xgb
    best_name = "Random Forest" if rf_f1 >= xgb_f1 else "XGBoost"
    plot_shap(best_clf, X_te_sc, FEATURES, best_name)

    # 10. Save everything
    print("\n[9] Saving models and encoders...")
    joblib.dump(best_clf,    "models/classifier.pkl")
    joblib.dump(scaler,      "models/scaler.pkl")
    joblib.dump(le_target,   "models/label_encoder.pkl")
    joblib.dump(le_city,     "models/city_encoder.pkl")
    joblib.dump(le_season,   "models/season_encoder.pkl")
    joblib.dump(FEATURES,    "models/feature_names.pkl")
    joblib.dump(aqi_scaler,  "models/aqi_scaler.pkl")

    lstm_model.save("models/lstm_model.keras")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best classifier : {best_name}")
    print(f"  Classifier F1   : {max(rf_f1, xgb_f1):.4f}")
    print(f"  LSTM RMSE       : {lstm_rmse:.2f} AQI points")
    print(f"  LSTM MAE        : {lstm_mae:.2f} AQI points")
    print("  All models saved to /models/")
    print("  All plots  saved to /plots/")
    print("=" * 60)
    print("\n  Run the dashboard:  streamlit run app.py\n")


if __name__ == "__main__":
    main()