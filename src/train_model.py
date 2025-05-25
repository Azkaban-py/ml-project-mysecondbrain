import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# ——— 1. Cargar tus datos ———
df = pd.read_csv('tus_datos.csv')  # Reemplazá con tu archivo

# ——— 2. Separar features y target ———
X = df.drop(columns='target')      # Reemplazá 'target' por el nombre real de tu variable objetivo
y = df['target']

# ——— 3. Detectar columnas categóricas ———
cat_cols = X.select_dtypes(include='object').columns.tolist()

# ——— 4. Train/Test split ———
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ——— 5. Crear el pipeline ———
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# ——— 6. Entrenar ———
pipeline.fit(X_train, y_train)

# ——— 7. Evaluar ———
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# ——— 8. Guardar el pipeline completo ———
joblib.dump(pipeline, 'modelo_xgb_pipeline.joblib')
print("Modelo guardado como 'modelo_xgb_pipeline.joblib'")