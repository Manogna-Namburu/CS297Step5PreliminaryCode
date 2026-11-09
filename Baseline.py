
import os
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from Preprocessing import prepare_data, SEED

K = 40

def create_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=1,
            random_state=SEED,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.2,
            max_depth=3,
            random_state=SEED
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=SEED
        )
    }


# Calculate evaluation metrics
def evaluate(name, y_true, predictions, train_time, predict_time):

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "FPR": fp / (fp + tn) if fp + tn > 0 else 0,
        "Train_Time": train_time,
        "Predict_Time": predict_time
    }


def run_baseline():

    os.makedirs("results", exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = (
        prepare_data()
    )
    start = time.perf_counter()

    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(
            X, y, random_state=SEED
        ),
        k=K
    )

    train = selector.fit_transform(X_train, y_train)
    val = selector.transform(X_val)
    test = selector.transform(X_test)

    selection_time = time.perf_counter() - start

    print("\n===== MUTUAL INFORMATION BASELINE =====")
    print("Selected features:", train.shape[1])
    print("Feature selection time:", round(selection_time, 4))

    models = create_models()
    training_time = selection_time
    validation_scores = []

    for name, model in models.items():

        start = time.perf_counter()
        model.fit(train, y_train)
        elapsed = time.perf_counter() - start

        training_time += elapsed

        val_predictions = model.predict(val)

        score = f1_score(
            y_val, val_predictions, zero_division=0
        )

        validation_scores.append(score)

        print(f"{name} trained in {elapsed:.4f} seconds")

    scores = np.array(validation_scores)

    if scores.sum() == 0:
        weights = np.ones(3) / 3
    else:
        weights = scores / scores.sum()

    print("Voting weights:", weights)

    start = time.perf_counter()

    predictions = np.array([
        model.predict(test)
        for model in models.values()
    ])

    hard = (predictions.sum(axis=0) >= 2).astype(int)

    hard_time = time.perf_counter() - start

    start = time.perf_counter()

    probabilities = np.array([
        model.predict_proba(test)[:, 1]
        for model in models.values()
    ])

    soft = (probabilities.mean(axis=0) >= 0.5).astype(int)

    soft_time = time.perf_counter() - start

    start = time.perf_counter()

    weighted_probabilities = np.array([
        model.predict_proba(test)[:, 1]
        for model in models.values()
    ])

    weighted = (
        np.average(
            weighted_probabilities,
            axis=0,
            weights=weights
        ) >= 0.5
    ).astype(int)

    weighted_time = time.perf_counter() - start

    # Evaluation
    voting_results = {
        "Hard Voting": (hard, hard_time),
        "Soft Voting": (soft, soft_time),
        "Weighted Voting": (weighted, weighted_time)
    }

    results = []

    for name, (pred, predict_time) in voting_results.items():

        metrics = evaluate(
            name,
            y_test,
            pred,
            training_time,
            predict_time
        )

        metrics["Feature_Selection"] = "Mutual Information"
        metrics["Features_Used"] = train.shape[1]

        results.append(metrics)

        print(
            f"{name}: "
            f"Accuracy={metrics['Accuracy']:.4f}, "
            f"Precision={metrics['Precision']:.4f}, "
            f"Recall={metrics['Recall']:.4f}, "
            f"F1={metrics['F1']:.4f}, "
            f"FPR={metrics['FPR']:.4f}, "
            f"Train={metrics['Train_Time']:.4f}s, "
            f"Predict={metrics['Predict_Time']:.4f}s"
        )

    df = pd.DataFrame(results)

    print("\n===== FINAL BASELINE RESULTS =====")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_baseline()
