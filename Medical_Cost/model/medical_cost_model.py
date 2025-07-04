import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def check_data(df, target):
    np.set_printoptions(suppress=True)
    num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns]
    print("")
    print("🔍 THÔNG TIN CƠ BẢN:")
    print(df.info())
    print("-------------------------------\n")

    print("❓ GIÁ TRỊ THIẾU:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print("-------------------------------\n")

    print("📌 TRÙNG LẶP:", df.duplicated().sum(), "dòng")
    print("-------------------------------\n")

    print("⚠️ KIỂM TRA DỮ LIỆU:")
    for col in df.columns:
        print(f"- {col}:", df[col].unique())
    print("-------------------------------\n")

    print("⚠️ KIỂM TRA PHÂN PHỐI GIÁ TRỊ CÁC CỘT:")
    for col in df.columns:
        if col != target:
            print(f"🔹 {col} value_counts():")
            print(df[col].value_counts(dropna=False))
            print("")
    print("-------------------------------\n")

    print("📦 KIỂM TRA KIỂU DỮ LIỆU SAI:")
    for col in df.select_dtypes(include='object').columns:
        try:
            pd.to_numeric(df[col])
            print(f"- {col}: chứa số nhưng lưu dạng object")
        except:
            pass
    print("-------------------------------\n")

    print("📊 THỐNG KÊ MÔ TẢ CÁC CỘT DỮ LIỆU SỐ:")
    pd.set_option('display.max_columns', None)
    print(df[num_cols].describe())
    print("-------------------------------\n")

    print("🎯 KIỂM TRA ĐỘ MẤT CÂN BẰNG DỮ LIỆU:")
    class_counts = df[target].value_counts()
    print(class_counts)
    print("➡️ TỈ LỆ PHẦN TRĂM (%):")
    print(round(df[target].value_counts(normalize=True) * 100, 2))
    print("-------------------------------\n")

# --- Get the absolute path of the current directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Load & Explore Dataset ---
df = pd.read_csv(os.path.join(BASE_DIR, "..", "dataset", "medical_cost_dataset.csv"))
target = "charges"
# check_data(df,target)

# --- 2. Data Preprocessing ---
# 2_1. Handle missing values
df = df.drop_duplicates().reset_index(drop=True)

# 2_2. Data Partitioning
x = df.drop(target,axis=1)
y = np.log(df[target])  # log-transform target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# 2_3. Data Transformation
num_cols = ['age', 'bmi', 'children']
ord_cols = ['sex', 'smoker']
ord_categories = [['male', 'female'], ['no', 'yes']]
nom_cols = ['region']

num_scaler = StandardScaler()
ord_encoder = OrdinalEncoder(categories=ord_categories)
nom_encoder = OneHotEncoder(sparse_output=False)

preprocessor = ColumnTransformer(transformers=[
    ('num', num_scaler, num_cols),
    ('ord', ord_encoder, ord_cols),
    ('nom', nom_encoder, nom_cols)
])

# --- 3. Model Building ---
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

    # --- GridSearchCV ---
    # params = {
    #     'regressor__n_estimators': [100, 300],
    #     'regressor__max_depth': [None, 20, 40],
    #     'regressor__min_samples_split': [2, 5],
    #     'regressor__max_features': ['sqrt']
    # }
    # grid = GridSearchCV(model, params, scoring='r2', cv=3, n_jobs=-1, verbose=1)
    # grid.fit(x_train, y_train)
    # model = grid.best_estimator_

model.fit(x_train, y_train)
y_pred_log = model.predict(x_test)
y_pred = np.exp(y_pred_log)  # chuyển log back về giá trị gốc
y_test_true = np.exp(y_test)

# --- 4. Model Evaluation ---
# 4_1. Metrics
mae = mean_absolute_error(y_test_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
r2 = r2_score(y_test_true, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# 4_2. Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_true, y_pred, alpha=0.6, color="blue")
plt.plot([y_test_true.min(), y_test_true.max()],
         [y_test_true.min(), y_test_true.max()], 'r--')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "..", "report", "actual_vs_predicted.png"))
plt.close()

# 4_3. Distribution of Prediction Errors
errors = y_pred - y_test_true
plt.figure(figsize=(8, 6))
sns.histplot(errors, kde=True, color="orange", bins=30)
plt.xlabel("Error (Predicted - Actual)")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "..", "report", "error_distribution.png"))
plt.close()

# --- Save model ---
joblib.dump(model, "medical_cost_model.pkl")

# --- Save metrics ---
metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
with open(os.path.join(BASE_DIR, "..", "report", "medical_cost_metrics.json"), "w") as f:
    json.dump(metrics, f)
