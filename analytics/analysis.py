from __future__ import annotations

import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
CSV_PATH = ROOT / "titanic.csv"
RANDOM_STATE = 42


def load_once() -> pd.DataFrame:
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    df = sns.load_dataset("titanic")
    df.to_csv(CSV_PATH, index=False)
    return df


def missing_strategy(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cleaned = df.copy()
    notes = []
    percentages = cleaned.isna().mean().mul(100)
    for column, percentage in percentages[percentages > 0].items():
        if percentage < 5:
            cleaned = cleaned.dropna(subset=[column])
            action = "dropped affected rows (<5%)"
        elif percentage <= 30:
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
                action = "median-imputed (5%–30%)"
            else:
                cleaned[column] = cleaned[column].fillna(cleaned[column].mode().iloc[0])
                action = "mode-imputed (5%–30%)"
        else:
            cleaned = cleaned.drop(columns=[column])
            action = "dropped column (>30%; imputation unreliable)"
        notes.append(f"- `{column}`: {percentage:.2f}% missing — {action}.")
    return cleaned, notes


def outlier_count(series: pd.Series) -> int:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


def save_eda(df: pd.DataFrame) -> list[str]:
    CHARTS.mkdir(exist_ok=True)
    notes = []
    for col in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(data=df, x=col, kde=True, ax=axes[0])
        sns.boxplot(data=df, x=col, ax=axes[1])
        fig.tight_layout(); fig.savefig(CHARTS / f"{col}_distribution.png"); plt.close(fig)
        notes.append(f"- `{col}` IQR outliers: **{outlier_count(df[col].dropna())}**.")

    numeric = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[numeric].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout(); fig.savefig(CHARTS / "correlation_heatmap.png"); plt.close(fig)
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    strongest = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(2)
    notes.extend([f"- Strong correlation: `{a}` and `{b}` = **{v:.3f}**." for (a, b), v in strongest.items()])

    charts = [
        ("survival_by_sex.png", lambda ax: sns.barplot(data=df, x="sex", y="survived", ax=ax),
         "Women show a higher average survival rate than men, indicating sex was strongly associated with survival."),
        ("survival_by_class.png", lambda ax: sns.barplot(data=df, x="pclass", y="survived", ax=ax),
         "First-class passengers have a higher survival rate than lower classes, suggesting access and location mattered."),
        ("fare_by_survival.png", lambda ax: sns.boxplot(data=df, x="survived", y="fare", ax=ax),
         "Survivors generally paid higher fares, which overlaps with the class effect."),
        ("age_class_survival.png", lambda ax: sns.scatterplot(data=df, x="age", y="fare", hue="survived", style="pclass", ax=ax),
         "Age, fare, class and survival overlap rather than forming a perfect boundary, supporting multivariate modeling."),
    ]
    for name, draw, interpretation in charts:
        fig, ax = plt.subplots(figsize=(7, 4)); draw(ax); fig.tight_layout(); fig.savefig(CHARTS / name); plt.close(fig)
        notes.append(f"- `{name}`: {interpretation}")
    return notes


def classifier_pipeline(model):
    numeric = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical = ["sex", "embarked"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def metrics_for(model, x_test, y_test) -> dict:
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, prob),
    }


def main() -> None:
    raw = load_once()
    raw.info()
    print(raw.describe(include="all"))
    print("Shape:", raw.shape)
    cleaned, missing_notes = missing_strategy(raw)
    report = ["# Analytics Report", "", f"Shape immediately after loading: `{raw.shape}`", "",
              "`df.info()`, `df.describe(include='all')`, and `df.shape` are printed during execution.",
              "", "## Missing values", *missing_notes]
    report += ["", "## EDA", *save_eda(cleaned)]
    fare = cleaned["fare"]
    report.append(f"- Fare mean={fare.mean():.3f}, median={fare.median():.3f}, mode={fare.mode().iloc[0]:.3f}. "
                  "Mean above median/mode indicates a right-skewed distribution.")
    sex_rates = {sex: cleaned.loc[cleaned["sex"] == sex, "survived"].mean() for sex in cleaned["sex"].dropna().unique()}
    class_rates = {int(cls): cleaned.loc[cleaned["pclass"] == cls, "survived"].mean() for cls in sorted(cleaned["pclass"].unique())}
    combined_rates = []
    for sex in cleaned["sex"].dropna().unique():
        for cls in sorted(cleaned["pclass"].unique()):
            mask = (cleaned["sex"] == sex) & (cleaned["pclass"] == cls)
            combined_rates.append({"sex": sex, "pclass": int(cls), "survival_rate": cleaned.loc[mask, "survived"].mean()})
    report += ["", "## Survival rates using Boolean masks", pd.Series(sex_rates, name="survival_rate").to_markdown(), "",
               pd.Series(class_rates, name="survival_rate").to_markdown(), "",
               pd.DataFrame(combined_rates).to_markdown(index=False)]

    standardized = StandardScaler().fit_transform(cleaned[["age", "fare"]])
    std_summary = pd.DataFrame(standardized, columns=["age_z", "fare_z"]).agg(["mean", "std"])
    report += ["", "## EDA standardization check", std_summary.to_markdown(),
               "Values are approximately mean 0 and standard deviation 1 (minor difference comes from sample-vs-population std conventions)."]

    features = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]
    x, y = raw[features], raw["survived"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    report += ["", "## Modeling", f"Class balance: `{y.value_counts(normalize=True).round(4).to_dict()}`.",
               "A stratified split preserves this class ratio in both train and test sets."]
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    fitted, scores = {}, {}
    for name, estimator in candidates.items():
        pipe = classifier_pipeline(estimator).fit(x_train, y_train)
        fitted[name], scores[name] = pipe, metrics_for(pipe, x_test, y_test)
        fig, ax = plt.subplots(figsize=(4, 4)); ConfusionMatrixDisplay.from_estimator(pipe, x_test, y_test, ax=ax)
        fig.tight_layout(); fig.savefig(CHARTS / f"{name.lower().replace(' ', '_')}_confusion.png"); plt.close(fig)
    comparison = pd.DataFrame(scores).T
    report += ["", "### Classifier comparison", comparison.to_markdown(floatfmt=".4f")]

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items(): RocCurveDisplay.from_estimator(pipe, x_test, y_test, name=name, ax=ax)
    fig.tight_layout(); fig.savefig(CHARTS / "roc_curves.png"); plt.close(fig)

    tree_pipe = fitted["Decision Tree"]
    names = tree_pipe.named_steps["preprocessor"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(24, 10)); plot_tree(tree_pipe.named_steps["model"], feature_names=names,
        class_names=["not survived", "survived"], filled=True, max_depth=3, ax=ax)
    fig.tight_layout(); fig.savefig(CHARTS / "decision_tree.png"); plt.close(fig)

    imbalance = {}
    for label, weight in [("baseline", None), ("class_weight_balanced", "balanced")]:
        pipe = classifier_pipeline(LogisticRegression(max_iter=1000, class_weight=weight, random_state=RANDOM_STATE)).fit(x_train, y_train)
        imbalance[label] = {k: v for k, v in metrics_for(pipe, x_test, y_test).items() if k in ["precision", "recall", "f1"]}
    prep = classifier_pipeline(LogisticRegression()).named_steps["preprocessor"]
    xt = prep.fit_transform(x_train); xv = prep.transform(x_test)
    xt_smote, yt_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(xt, y_train)
    smote_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(xt_smote, yt_smote)
    smote_pred = smote_model.predict(xv)
    imbalance["SMOTE_train_only"] = {"precision": precision_score(y_test, smote_pred),
        "recall": recall_score(y_test, smote_pred), "f1": f1_score(y_test, smote_pred)}
    imbalance_df = pd.DataFrame(imbalance).T
    best_imbalance = imbalance_df["f1"].idxmax()
    report += ["", "### Imbalance comparison", imbalance_df.to_markdown(floatfmt=".4f"),
               f"`{best_imbalance}` produced the best F1 balance on this split; SMOTE was applied only after training preprocessing."]

    rf = classifier_pipeline(RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE))
    grid = GridSearchCV(rf, {
        "model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10], "model__max_features": ["sqrt", "log2"]
    }, cv=5, scoring="f1", n_jobs=-1).fit(x_train, y_train)
    report += ["", "### Random Forest tuning", f"Best parameters: `{grid.best_params_}`",
               f"OOB score: **{grid.best_estimator_.named_steps['model'].oob_score_:.4f}**"]

    reg_features = ["pclass", "age", "sibsp", "parch", "survived", "sex", "embarked"]
    rx_train, rx_test, ry_train, ry_test = train_test_split(raw[reg_features], raw["fare"], test_size=0.2, random_state=RANDOM_STATE)
    reg_num, reg_cat = reg_features[:-2], ["sex", "embarked"]
    reg_prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), reg_num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), reg_cat),
    ])
    reg = Pipeline([("preprocessor", reg_prep), ("model", LinearRegression())]).fit(rx_train, ry_train)
    rpred = reg.predict(rx_test); mae = mean_absolute_error(ry_test, rpred); rmse = math.sqrt(mean_squared_error(ry_test, rpred)); r2 = r2_score(ry_test, rpred)
    n, p = len(ry_test), len(reg.named_steps["preprocessor"].get_feature_names_out()); adj = 1 - (1-r2)*(n-1)/(n-p-1)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(rpred, ry_test-rpred, alpha=.6); ax.axhline(0, color="red"); ax.set(xlabel="Predicted fare", ylabel="Residual")
    fig.tight_layout(); fig.savefig(CHARTS / "fare_residuals.png"); plt.close(fig)
    spread_corr = np.corrcoef(np.abs(ry_test-rpred), rpred)[0, 1]
    hetero = "suggests heteroscedasticity" if abs(spread_corr) >= 0.2 else "does not show strong evidence of heteroscedasticity"
    report += ["", "### Regression metrics", pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted R2": adj}]).to_markdown(index=False, floatfmt=".4f"),
               f"The residual plot {hetero}; correlation between absolute residual size and prediction is {spread_corr:.3f}."]

    best_name = comparison["f1"].idxmax(); best_pipe = fitted[best_name]
    joblib.dump(best_pipe, ROOT / "best_classifier_pipeline.joblib")
    reloaded = joblib.load(ROOT / "best_classifier_pipeline.joblib")
    check = int(reloaded.predict(x_test.iloc[[0]])[0])
    report += ["", "## Recommendation", f"Deploy **{best_name}** because it has the highest test F1 ({comparison.loc[best_name, 'f1']:.4f}) while achieving AUC {comparison.loc[best_name, 'auc']:.4f}. "
               "F1 balances precision and recall, making it more informative than accuracy alone for the minority survived class. "
               "The final choice should still be monitored for subgroup fairness and drift.",
               f"Reloaded raw-input prediction check succeeded: `{check}`."]
    (ROOT / "analysis_report.md").write_text("\n\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
