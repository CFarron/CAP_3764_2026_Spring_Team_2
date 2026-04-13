import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay, f1_score, precision_score,
    recall_score, accuracy_score
)

# =====================================================
# 1. DATA LOADING
# =====================================================

def load_flight_data(path):
    df = pd.read_csv(path)
    return df


def load_clean_data(path="../data/processed/mia_flights_clean.csv"):
    df = pd.read_csv(path)
    df["fl_date"] = pd.to_datetime(df["fl_date"])
    return df


# =====================================================
# 2. PREPROCESSING
# =====================================================

def build_features(df):
    df = df.copy()
    df["dep_hour"] = (df["dep_time"] // 100).fillna(-1).astype(int)

    def get_time_of_day(hour):
        if hour < 6:    return 0
        elif hour < 12: return 1
        elif hour < 18: return 2
        else:           return 3

    df["time_of_day"]      = df["dep_hour"].apply(get_time_of_day)
    df["is_peak"]          = df["dep_hour"].isin([6, 7, 8, 16, 17, 18]).astype(int)
    df["is_weekend"]       = df["day_of_week"].isin([6, 7]).astype(int)
    df["is_early_morning"] = (df["dep_hour"] < 8).astype(int)

    def distance_bucket(d):
        if d < 500:    return 0
        elif d < 1500: return 1
        else:          return 2

    df["distance_bucket"] = df["distance"].apply(distance_bucket)
    return df


def build_target(df, threshold=15):
    df = df.copy()
    df["is_delayed"] = (df["late_aircraft_delay"] > threshold).astype(int)
    return df


def get_train_test_split(df, features, target="is_delayed",
                         test_size=0.20, random_state=42):
    model_df = df[features + [target]].dropna()
    X = model_df[features]
    y = model_df[target]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


# =====================================================
# 3. EVALUATION UTILITIES
# =====================================================

def evaluate_model(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"{model_name} — Classification Report")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["On Time", "Delayed"]))


def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }


def plot_confusion_matrix(model, X_test, y_test, title, cmap="Blues", save_path=None):
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["On Time", "Delayed"],
        cmap=cmap, ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def build_decile_summary(y_true, y_prob):
    df_eval = pd.DataFrame({"y": y_true.values, "proba": y_prob})
    df_eval = df_eval.sort_values("proba", ascending=False).reset_index(drop=True)
    df_eval["decile"] = pd.qcut(df_eval.index + 1, 10, labels=False) + 1
    overall_rate = df_eval["y"].mean()
    dec = df_eval.groupby("decile").agg(
        leads=("y", "size"),
        positives=("y", "sum")
    ).reset_index()
    dec["rate"]            = dec["positives"] / dec["leads"]
    dec["lift"]            = dec["rate"] / overall_rate
    dec["cum_pos"]         = dec["positives"].cumsum()
    dec["cum_pct_pos"]     = dec["cum_pos"] / dec["positives"].sum()
    dec["cum_pct_flights"] = dec["leads"].cumsum() / dec["leads"].sum()
    return dec


def overfit_report(model_name, model, X_train, X_test, y_train, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    gap = train_acc - test_acc
    print(f"\n{model_name}")
    print(f"  Train Accuracy : {train_acc:.3f}")
    print(f"  Test  Accuracy : {test_acc:.3f}")
    print(f"  Gap            : {gap:.3f}",
          "⚠️ Possible overfitting" if gap > 0.10 else "✅ Good generalization")


# =====================================================
# 4. SHAP HELPERS
# =====================================================

def compute_shap_values(model, X_test):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return explainer, shap_values


def plot_shap_summary(shap_values, X_test, title, plot_type="dot", save_path=None):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type=plot_type, show=False)
    plt.title(title, fontsize=12, pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
