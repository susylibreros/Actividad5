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
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
)
from datos_prep import limpiar_y_preparar

def evaluar_y_registrar():
    # 1. Cargar y preparar datos limpios
    df, mapa_categorias = limpiar_y_preparar()
    X = df["texto_limpio"]
    y = df["categoria_id"]
    nombres_clases = [k for k, v in sorted(mapa_categorias.items(), key=lambda item: item[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. Configurar experimento en MLflow
    mlflow.set_experiment("Tres_Lunas_Serenas_Clasificacion_Consultas")

    modelos = {
        "Logistic_Regression_TFIDF": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42))
            ]),
            "params": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.5, 1.0, 5.0]
            }
        },
        "Random_Forest_TFIDF": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", RandomForestClassifier(random_state=42))
            ]),
            "params": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [None, 10]
            }
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados_cv = {}

    for nombre, config in modelos.items():
        with mlflow.start_run(run_name=nombre):
            print(f"\n--- Entrenando {nombre} con GridSearchCV (5-Fold) ---")
            grid = GridSearchCV(
                estimator=config["pipeline"],
                param_grid=config["params"],
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            mejor_modelo = grid.best_estimator_

            # Guardar F1 por cada fold para comparación estadística
            best_idx = grid.best_index_
            fold_scores = [grid.cv_results_[f"split{i}_test_score"][best_idx] for i in range(5)]
            resultados_cv[nombre] = fold_scores

            # Predicciones en conjunto de prueba
            y_pred = mejor_modelo.predict(X_test)
            y_prob = mejor_modelo.predict_proba(X_test)

            # Métricas estandarizadas multiclase
            metricas = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "roc_auc_ovr": roc_auc_score(y_test, y_prob, multi_class="ovr"),
                "cv_f1_mean": np.mean(fold_scores),
                "cv_f1_std": np.std(fold_scores)
            }

            # Registrar parámetros y métricas en MLflow
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metricas)

            # Guardar diccionario de categorías como artefacto JSON
            with open("categorias.json", "w", encoding="utf-8") as f:
                json.dump(mapa_categorias, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact("categorias.json")
            os.remove("categorias.json")

            # Guardar Matriz de Confusión como artefacto visual
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_estimator(
                mejor_modelo, X_test, y_test, display_labels=nombres_clases,
                ax=ax, cmap="Purples", xticks_rotation=25
            )
            plt.title(f"Matriz de Confusión - {nombre}\n(Tres Lunas Serenas)")
            img_path = f"cm_{nombre}.png"
            plt.savefig(img_path, bbox_inches="tight")
            plt.close(fig)

            mlflow.log_artifact(img_path)
            mlflow.sklearn.log_model(mejor_modelo, artifact_path="modelo_nlp")
            os.remove(img_path)

            print(f"Mejores hiperparámetros: {grid.best_params_}")
            print(f"Métricas en Test: {metricas}")

    # 3. Comparación estadística entre modelos
    t_stat, p_val = stats.ttest_rel(
        resultados_cv["Logistic_Regression_TFIDF"],
        resultados_cv["Random_Forest_TFIDF"]
    )
    print("\n=== Comparación Estadística (CV F1-Macro) ===")
    print(f"Logistic Regression F1 (CV): {np.mean(resultados_cv['Logistic_Regression_TFIDF']):.4f}")
    print(f"Random Forest F1 (CV):       {np.mean(resultados_cv['Random_Forest_TFIDF']):.4f}")
    print(f"Estadístico t = {t_stat:.4f}, p-valor = {p_val:.4f}")

if __name__ == "__main__":
    evaluar_y_registrar()