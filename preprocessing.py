# ================================================================
# src/preprocessing.py
# Reusable preprocessing functions for the diabetes readmission
# prediction pipeline.
# ================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and replace '?' with NaN."""
    df = pd.read_csv(path, na_values="?")
    print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 3-class readmitted into binary target (1 = <30 days)."""
    df = df.copy()
    df["target"] = (df["readmitted"] == "<30").astype(int)
    df.drop(columns=["readmitted"], inplace=True)
    print(f"✓ Target: {df['target'].sum():,} positives ({df['target'].mean()*100:.1f}%)")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are either identifiers, have too many missing values,
    or are purely administrative with no clinical predictive value.
    """
    drop_cols = [
        "encounter_id",      # identifier — no predictive value
        "patient_nbr",       # identifier — no predictive value
        "weight",            # 97% missing — too sparse
        "payer_code",        # 52% missing + administrative
        "medical_specialty", # 53% missing + administrative
    ]
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)
    print(f"✓ Dropped {len(existing)} columns")
    return df


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove patients who died or went to hospice (they cannot be readmitted)
    and remove rows with invalid gender.
    """
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
    df = df[df["gender"].isin(["Male", "Female"])]
    print(f"✓ Removed {before - len(df):,} rows (deaths, hospice, invalid gender)")
    return df


def map_icd9(code: str) -> str:
    """Map ICD-9 diagnosis code to a clinical category (8 groups)."""
    try:
        c = str(code).strip().upper()
        if c.startswith("V") or c.startswith("E"):
            return "Other"
        n = float(c[:3])
        if   390 <= n <= 459: return "Circulatory"
        elif 460 <= n <= 519: return "Respiratory"
        elif 520 <= n <= 579: return "Digestive"
        elif 250 <= n <= 250.99: return "Diabetes"
        elif 800 <= n <= 999: return "Injury"
        elif 710 <= n <= 739: return "Musculoskeletal"
        elif 580 <= n <= 629: return "Genitourinary"
        elif 140 <= n <= 239: return "Neoplasms"
        else: return "Other"
    except:
        return "Other"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones:
    - ICD-9 diagnosis grouping (reduces 800+ codes to 8 categories)
    - age_mid: numeric midpoint of age bracket
    - n_active_meds: number of active medications
    - n_med_changes: number of medication dosage changes
    - prior_visits: total prior hospital visits
    """
    df = df.copy()

    # ICD-9 grouping
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("0").apply(map_icd9)

    # Age midpoint
    age_map = {
        "[0-10)": 5,  "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
        "[70-80)": 75, "[80-90)": 85, "[90-100)": 95
    }
    if "age" in df.columns:
        df["age_mid"] = df["age"].map(age_map).fillna(55)

    # Medication features
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide",
        "sitagliptin", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    med_cols = [c for c in med_cols if c in df.columns]
    df["n_active_meds"] = (df[med_cols] != "No").sum(axis=1)
    df["n_med_changes"]  = df[med_cols].isin(["Up", "Down"]).sum(axis=1)

    # Prior visits
    visit_cols = ["number_outpatient", "number_emergency", "number_inpatient"]
    if all(c in df.columns for c in visit_cols):
        df["prior_visits"] = df[visit_cols].sum(axis=1)

    print("✓ Feature engineering done (5 new features)")
    return df, med_cols


def encode_features(df: pd.DataFrame, med_cols: list) -> pd.DataFrame:
    """
    Encode all categorical features:
    - Medication columns → ordinal (No=0, Steady=1, Up=2, Down=-1)
    - Binary columns → label encoding
    - Other categoricals → one-hot encoding
    """
    df = df.copy()

    # Fill remaining missing values
    df["race"] = df["race"].fillna("Unknown")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != "target"]
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    cat_cols_imp = df.select_dtypes(include="object").columns.tolist()
    if cat_cols_imp:
        df[cat_cols_imp] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols_imp])

    # Ordinal encode medication columns
    med_order = {"No": 0, "Steady": 1, "Up": 2, "Down": -1}
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].map(med_order).fillna(0).astype(int)

    # Label encode binary columns
    for col in ["gender", "change", "diabetesMed"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # One-hot encode remaining categoricals
    ohe_cols = df.select_dtypes(include="object").columns.tolist()
    ohe_cols = [c for c in ohe_cols if c != "target"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)

    print(f"✓ Encoding done — {df.shape[1]} total features")
    return df


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to training set only."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"✓ SMOTE — Train size: {len(X_res):,} | "
          f"Class 0: {(y_res==0).sum():,} | Class 1: {(y_res==1).sum():,}")
    return X_res, y_res


def scale_features(X_train, X_val, X_test):
    """Fit StandardScaler on train, transform val and test."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    print("✓ StandardScaler applied (fitted on train only)")
    return X_train_sc, X_val_sc, X_test_sc, scaler
