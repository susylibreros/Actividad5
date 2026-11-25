# fuentes/train.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
)
from datos_prep import limpiar_y_preparar

def evaluar_y_registrar():
    # 1. Cargar datos limpios del Student Performance Dataset (UCI)
    df, mapa_clases = limpiar_y_preparar()

    # Seleccionar predictores estandarizados y binarios (excluyendo G3 para evitar data leakage)
    features = [c for c in df.columns if c.endswith("_std") or c.endswith("_bin")]
    X = df[features]
    y = df["intervencion_id"]
    nombres_clases = [k for k, v in sorted(mapa_clases.items(), key=lambda item: item[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. Configurar experimento en MLflow
    mlflow.set_experiment("Student_Performance_UCI_Clasificacion")

    modelos = {
        "Logistic_Regression_UCI": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.1, 1.0, 5.0, 10.0],
                "solver": ["lbfgs"]
            }
        },
        "Random_Forest_UCI": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 8]
            }
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados_cv = {}

    for nombre, config in modelos.items():
        with mlflow.start_run(run_name=nombre):
            print(f"\n--- Entrenando {nombre} con GridSearchCV (5-Fold) ---")
            grid = GridSearchCV(
                estimator=config["estimator"],
                param_grid=config["params"],
                cv=cv,
                scoring="f1_macro",
                n_jobs=1  # Ejecución estable en GitHub Codespaces
            )
            grid.fit(X_train, y_train)
            mejor_modelo = grid.best_estimator_

            best_idx = grid.best_index_
            fold_scores = [grid.cv_results_[f"split{i}_test_score"][best_idx] for i in range(5)]
            resultados_cv[nombre] = fold_scores

            y_pred = mejor_modelo.predict(X_test)
            y_prob = mejor_modelo.predict_proba(X_test)

            metricas = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "roc_auc_ovr": roc_auc_score(y_test, y_prob, multi_class="ovr"),
                "cv_f1_mean": np.mean(fold_scores),
                "cv_f1_std": np.std(fold_scores)
            }

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metricas)

            # Guardar diccionario de clases como artefacto JSON
            with open("clases_intervencion.json", "w", encoding="utf-8") as f:
                json.dump(mapa_clases, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact("clases_intervencion.json")
            os.remove("clases_intervencion.json")

            # Guardar Matriz de Confusión como artefacto PNG
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_estimator(
                mejor_modelo, X_test, y_test, display_labels=nombres_clases,
                ax=ax, cmap="Blues", xticks_rotation=20
            )
            plt.title(f"Matriz de Confusión - {nombre}\n(Student Performance UCI)")
            img_path = f"cm_{nombre}.png"
            plt.savefig(img_path, bbox_inches="tight")
            plt.close(fig)

            mlflow.log_artifact(img_path)
            mlflow.sklearn.log_model(
    mejor_modelo,
    name="modelo_uci",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
            os.remove(img_path)

            print(f"Mejores hiperparámetros: {grid.best_params_}")
            print(f"Métricas en Test: {metricas}")

    # 3. Prueba estadística t pareada sobre los 5 folds
    t_stat, p_val = stats.ttest_rel(
        resultados_cv["Logistic_Regression_UCI"],
        resultados_cv["Random_Forest_UCI"]
    )
    print("\n=== Comparación Estadística (CV F1-Macro - UCI Dataset) ===")
    print(f"Logistic Regression F1 (CV): {np.mean(resultados_cv['Logistic_Regression_UCI']):.4f}")
    print(f"Random Forest F1 (CV):       {np.mean(resultados_cv['Random_Forest_UCI']):.4f}")
    print(f"Estadístico t = {t_stat:.4f}, p-valor = {p_val:.4f}")

if __name__ == "__main__":
    evaluar_y_registrar()