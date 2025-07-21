# %%
import numpy as np
import pandas as pd

# %%

df = pd.read_csv(ruta_data_cliente, delimiter=";")

# Imputar nulos con mediana robusta
for col in ["limite_tc", "limite_acc", "deuda_tc"]:
    df[col] = df[col].fillna(df[col].median())

# Imputar 0 en importe_pf y importe_ca
for col in ["importe_pf", "importe_ca"]:
    df[col] = df[col].fillna(0)
# %%
train = df["2022-07":"2023-08"]
val = df["2023-09":"2024-03"]
test = df["2024-04":"2024-04"]  # Mes futuro real

# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Supongamos que df tiene una columna 'period' con formato AAAAMM (entero o string)
# Convertimos a fecha si es necesario para facilitar comparaciones
df["period_date"] = pd.to_datetime(df["periodo"].astype(str), format="%Y%m")

# Definir fecha de corte para entrenamiento (e.g., 2023-07-01)
cutoff_date = pd.to_datetime("2023-07-01")

# Conjunto de entrenamiento: periodos antes de julio 2023
train_df = df[df["period_date"] < cutoff_date]

# Conjunto de validación: periodos desde julio 2023 en adelante
val_df = df[df["period_date"] >= cutoff_date]

# Separar características (X) y etiqueta (y)
X_train = train_df.drop(columns=["bad", "periodo", "period_date"])
X_train = X_train.select_dtypes(
    include=[np.number]
)  # Seleccionar solo columnas numéricas
y_train = train_df["bad"]

X_val = val_df.drop(columns=["bad", "periodo", "period_date"])
X_val = X_val.select_dtypes(include=[np.number])  # Seleccionar solo columnas numéricas
y_val = val_df["bad"]

print("Observaciones entrenamiento:", len(X_train))
print("Observaciones validación:", len(X_val))

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Definir y entrenar el modelo de regresión logística
model_lr = LogisticRegression(solver="liblinear", random_state=0, penalty="l2", C=1.0)
model_lr.fit(X_train, y_train)

# Predecir probabilidades en el conjunto de validación
y_pred_proba_lr = model_lr.predict_proba(X_val)[:, 1]

# Evaluar rendimiento con AUC-ROC
auc_lr = roc_auc_score(y_val, y_pred_proba_lr)
print(f"AUC-ROC Regresi\u00f3n Log\u00edstica: {auc_lr:.3f}")

# %%

from sklearn.ensemble import RandomForestClassifier

# Definir y entrenar el modelo Random Forest
model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,  # Limita la profundidad máxima a 6 niveles
    min_samples_leaf=10,  # Mínimo de 10 muestras por hoja
    min_samples_split=20,  # Mínimo de 20 muestras para dividir un nodo
    max_features="sqrt",  # Usa la raíz cuadrada de las features en cada split
    random_state=42,
)
model_rf.fit(X_train, y_train)

# Predecir probabilidades en el conjunto de validaci\u00f3n


y_pred_proba_rf = model_rf.predict_proba(X_val)[:, 1]

# Calcular AUC en validaci\u00f3n
auc_rf = roc_auc_score(y_val, y_pred_proba_rf)
print(f"AUC-ROC Random Forest: {auc_rf:.3f}")

# %%

import lightgbm as lgb

model_lgb = lgb.LGBMClassifier(
    n_estimators=1000,  # Ponemos muchos árboles para usar early stopping
    learning_rate=0.05,  # Learning rate más bajo
    max_depth=6,  # Limitar profundidad
    num_leaves=31,  # Número de hojas máximo
    min_child_samples=20,  # Muestras mínimas por hoja
    feature_fraction=0.8,  # Usar solo el 80% de features por árbol
    bagging_fraction=0.8,  # Usar solo el 80% de datos por iteración
    bagging_freq=5,  # Aplicar bagging cada 5 iteraciones
    lambda_l1=0.1,  # Regularización L1
    lambda_l2=0.1,  # Regularización L2
    random_state=42,
)


model_lgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=10)],
)

y_pred_proba_lgb = model_lgb.predict_proba(X_val)[:, 1]


# %%
import pandas as pd
from sklearn.metrics import roc_auc_score

# Suponiendo que ya tenés definidos X_train, y_train, X_val, y_val
# y que los modelos están entrenados: model_lr, model_rf, model_lgb

# Calcular AUC en entrenamiento
auc_lr_train = roc_auc_score(y_train, model_lr.predict_proba(X_train)[:, 1])
auc_rf_train = roc_auc_score(y_train, model_rf.predict_proba(X_train)[:, 1])
auc_lgb_train = roc_auc_score(y_train, model_lgb.predict_proba(X_train)[:, 1])

# Calcular AUC en validación
auc_lr_val = roc_auc_score(y_val, model_lr.predict_proba(X_val)[:, 1])
auc_rf_val = roc_auc_score(y_val, model_rf.predict_proba(X_val)[:, 1])
auc_lgb_val = roc_auc_score(y_val, model_lgb.predict_proba(X_val)[:, 1])

# Crear cuadro comparativo de AUC
auc_comparison = pd.DataFrame(
    {
        "Modelo": ["Regresión Logística", "Random Forest", "LightGBM"],
        "AUC Entrenamiento": [auc_lr_train, auc_rf_train, auc_lgb_train],
        "AUC Validación": [auc_lr_val, auc_rf_val, auc_lgb_val],
    }
)

print("Cuadro comparativo de AUC:")
print(auc_comparison)

# Crear cuadro comparativo de hiperparámetros principales
params_comparison = pd.DataFrame(
    {
        "Modelo": ["Regresión Logística", "Random Forest", "LightGBM"],
        "Parámetros Principales": [
            "solver='liblinear', random_state=0",
            "n_estimators=100, max_depth=6, min_samples_leaf=10, min_samples_split=20, max_features='sqrt', random_state=42",
            "n_estimators=1000, learning_rate=0.05, max_depth=6, num_leaves=31, min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8, lambda_l1=0.1, lambda_l2=0.1, random_state=42",
        ],
    }
)

print("\nCuadro comparativo de hiperparámetros:")
print(params_comparison)
# %%
# Exportar cuadro comparativo de AUC a Excel
auc_comparison.to_excel("cuadro_auc.xlsx", index=False)

# Exportar cuadro comparativo de hiperparámetros a Excel
params_comparison.to_excel("cuadro_hiperparametros.xlsx", index=False)

print("Archivos Excel exportados exitosamente.")

# %%
# Interpretación de resultados
import pandas as pd

coef_df = pd.DataFrame({"Variable": X_train.columns, "Coeficiente": model_lr.coef_[0]})

coef_df["Importancia_absoluta"] = coef_df["Coeficiente"].abs()
coef_df = coef_df.sort_values(by="Importancia_absoluta", ascending=False)

for idx, row in coef_df.iterrows():
    print(f"{row['Variable']}: {row['Coeficiente']:.4f}")

# %%
# Obtenemos las importancias
importances_rf = model_rf.feature_importances_

# Creamos un DataFrame ordenado
importance_rf_df = pd.DataFrame(
    {"Variable": X_train.columns, "Importancia": importances_rf}
).sort_values(by="Importancia", ascending=False)

for idx, row in importance_rf_df.iterrows():
    print(f"{row['Variable']}: {row['Importancia']:.4f}")

# %%
importances_lgb = model_lgb.feature_importances_
importance_lgb_df = pd.DataFrame(
    {"Variable": X_train.columns, "Importancia": importances_lgb}
).sort_values(by="Importancia", ascending=False)

for idx, row in importance_lgb_df.iterrows():
    print(f"{row['Variable']}: {row['Importancia']:.0f}")
