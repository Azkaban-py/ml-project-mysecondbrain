import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# ——— 1. Carga datos ———
df = pd.read_csv('datos.csv')

# ——— 2. Separar features y target ———
X = df.drop(columns='target')
y = df['target']

# ——— 3. Detectar columnas categóricas ———
cat_cols = X.select_dtypes(include='object').columns.tolist()

# ——— 4. Separar en train/test ———
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ——— 5. Pipeline con OneHotEncoder y XGBoost ———
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'  # deja pasar las columnas numéricas
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# ——— 6. Entrenamiento ———
pipeline.fit(X_train, y_train)

# ——— 7. Evaluación ———
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))