import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score



def get_parent_column(encoded_feature_name):
    # remove transformer prefix
    if "__" in encoded_feature_name:
        transformer_prefix, rest = encoded_feature_name.split("__", 1)
        if transformer_prefix == "num":
            return rest  # numeric: column name is rest
        elif transformer_prefix == "cat":
            return rest.split("_")[0]  # categorical: column before first _
    return encoded_feature_name


def train_pipeline_top_n_original_columns(
    df,
    target_column,
    top_n=5,
    test_size=0.2,
    random_state=42,
    save_path="ML/churn_topn_original_pipeline.pkl"
):
    # -------------------------------
    # Separate target
    # -------------------------------

    y = df[target_column]
    X = df.drop(columns=[target_column, "customerID"])
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    X["TotalCharges"] = X["TotalCharges"].fillna(0)

    # -------------------------------
    # Identify numeric and categorical
    # -------------------------------
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    print(df.dtypes)
    print(numeric_features)
    # -------------------------------
    # Preprocessing
    # -------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # -------------------------------
    # Train full model
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    full_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", rf)
    ])
    full_pipeline.fit(X_train, y_train)
    preprocessor = full_pipeline.named_steps["preprocessing"]

    # Access the fitted OneHotEncoder inside ColumnTransformer
    encoder = preprocessor.named_transformers_["cat"]

    categories = encoder.categories_
    allowed_values = dict(zip(categorical_features, categories))
    print(allowed_values)
    # -------------------------------
    # Feature importances after encoding
    # -------------------------------
    model = full_pipeline.named_steps["model"]
    importances = model.feature_importances_
    feature_names = full_pipeline.named_steps["preprocessing"].get_feature_names_out()

    # Map back to original columns
    feature_df = pd.DataFrame({
        "encoded_feature": feature_names,
        "importance": importances
    })

    # Original column is everything before the first "__" in feature_names
    feature_df["original_column"] = feature_df["encoded_feature"].apply(get_parent_column)
    # Aggregate importance by original column
    original_importance = feature_df.groupby("original_column")["importance"].sum().sort_values(ascending=False)
    top_original_cols = original_importance.head(top_n).index.tolist()

    print(f"\nTop {top_n} original columns based on importance:")
    print(original_importance.head(top_n))

    # -------------------------------
    # Build new pipeline using only top N columns
    # -------------------------------
    top_numeric = [c for c in top_original_cols if c in numeric_features]
    top_categorical = [c for c in top_original_cols if c in categorical_features]

    preprocessor_top = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), top_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), top_categorical),
        ]
    )

    pipeline_top = Pipeline([
        ("preprocessing", preprocessor_top),
        ("model", RandomForestClassifier(random_state=random_state, class_weight="balanced"))
    ])

    pipeline_top.fit(X_train[top_original_cols], y_train)

    y_pred = pipeline_top.predict(X_test[top_original_cols])
    y_proba = pipeline_top.predict_proba(X_test[top_original_cols])[:, 1]
    print(y_pred)
    print(y_proba)
    print("\nClassification Report (Top-N original columns):")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_top, save_path)
    print(f"\nTop-N pipeline saved to: {save_path}")

    return pipeline_top, top_original_cols



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent # .resolve je potrebno jer ne znam absoluth path na tudjoj masini sa koje se ovo pokrece
    data_path = BASE_DIR / "data"/"WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)

    pipeline_top, top_original_cols = train_pipeline_top_n_original_columns(df, target_column="Churn",top_n=5, test_size=0.2, random_state=42, save_path= BASE_DIR / "ML" / "churn_topn_original_pipeline.pkl")