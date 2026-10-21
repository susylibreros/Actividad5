import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# Configuración de MLflow
mlflow.set_experiment("Clasificacion_Consultas_TresLunasSerenas")

# Cargar datos limpios
df = pd.read_csv("datos/datos_limp/consultas_clientes_clean.csv")
X = df['text_clean']
y = df['label']

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)

# Validación Cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definición de Modelos e Hiperparámetros
modelos_grid = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    }
}

# Bucle de Entrenamiento y Registro en MLflow
for nombre_modelo, config in modelos_grid.items():
    with mlflow.start_run(run_name=nombre_modelo):
        print(f"🚀 Entrenando y ajustando: {nombre_modelo}")
        
        # Grid Search con CV
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1
        )
        grid.fit(X_vectorized, y)
        
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        
        # Predicciones y métricas sobre CV (Out-of-fold o promedio)
        y_pred = best_model.predict(X_vectorized)
        y_proba = best_model.predict_proba(X_vectorized)
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted')
        rec = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
        
        # Registrar Parámetros en MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("algoritmo", nombre_modelo)
        mlflow.log_param("vectorizador", "TF-IDF (max_features=500)")
        
        # Registrar Métricas en MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Registrar Modelo y Artefactos
        mlflow.sklearn.log_model(best_model, artifact_path="modelo")
        
        print(f"✅ {nombre_modelo} finalizado. F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

print("\n🎉 Todos los experimentos fueron registrados en MLflow.")
