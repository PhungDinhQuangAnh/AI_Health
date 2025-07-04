import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
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
def group_median_impute(df, target_col, group_cols, invalid_zero=False):
    # Gán các giá trị 0 thành NaN nếu có yêu cầu
    if invalid_zero:
        df[target_col] = df[target_col].replace(0, np.nan)

    # Áp dụng median theo từng nhóm
    df[target_col] = df.groupby(group_cols, observed=True)[target_col].transform(
        lambda x: x.fillna(x.median())
    )

    # Fallback: nếu vẫn còn NaN, dùng median toàn cục
    df[target_col] = df[target_col].fillna(df[target_col].median())
    return df
def remove_missing_invalid(df,col):
    return df[~((df[col].isnull()) | (df[col] == 0))]
def check_outliers(df):
    num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns]
    print("")
    print("📈 OUTLIERS (IQR method):")
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if not outliers.empty:
            print(f"- {col}: {len(outliers)} outliers")
    print("-------------------------------\n")

    print("🧾 BIỂU ĐỒ BOXPLOT (Phát Hiện Outliers Và Độ Lệch Dữ Liệu):")
    df[num_cols].plot(kind='box', subplots=True, layout=(3, 3), sharex=False)
    plt.show()
    print("-------------------------------\n")
def clip_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
    return df

# --- Get the absolute path of the current directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Load & Explore Dataset ---
df = pd.read_csv(os.path.join(BASE_DIR, "..", "dataset", "heart_dataset.csv"))
target = "HeartDisease"
check_data(df, target)

# --- 2. Data Preprocessing ---
# 2_1. Handle missing or invalid values
    # print("RestingBP: ", (df['RestingBP'] == 0).sum())
    # print("Cholesterol: ", (df['Cholesterol'] == 0).sum())

# RestingBP
df = remove_missing_invalid(df,"RestingBP")
# Cholesterol
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 35, 55, 120],
    labels=['Young', 'Middle', 'Old']
)
df = group_median_impute(df, target_col='Cholesterol', group_cols=['Sex','AgeGroup','HeartDisease'], invalid_zero=True)
df = df.drop(columns='AgeGroup')

# 2_2. Handle Outliers
for col in ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
    df = clip_outliers(df, col)

# 2_3. Data Partitioning
x = df.drop(target,axis=1)
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# 2_4. Data Transformation
num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns
            if df[col].nunique() > 2]

ord_cols = ["Sex", "ExerciseAngina", "RestingECG", "ST_Slope"]
ord_categories = [
    ["M", "F"],
    ["N", "Y"],
    ["Normal", "ST", "LVH"],
    ["Down", "Flat", "Up"]
]

nom_col = ["ChestPainType"]

num_scaler = StandardScaler()
ord_scaler = OrdinalEncoder(categories=ord_categories)
nom_scaler = OneHotEncoder(sparse_output=True)

preprocessor = ColumnTransformer(transformers=[
    ("num_features",num_scaler,num_cols),
    ("ord_features",ord_scaler,ord_cols),
    ("nom_features",nom_scaler,nom_col)
])

# --- 3. Model Building ---
model = Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier(max_depth=5, max_features = 'sqrt', min_samples_leaf = 4, min_samples_split = 5, n_estimators = 100))
])
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# --- 4. Model Evaluation ---
# 4.1. Evaluation Metrics
print("")
print(classification_report(y_test, y_pred))
    # Save Classification_Report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).T.round(2)
report_df.to_csv(os.path.join(BASE_DIR, "..", "report", "heart_classification_report.csv"))

# 4.2. Confusion Matrix Plot
cm = confusion_matrix(y_test,y_pred,labels=[0,1])
cm_to_df = pd.DataFrame(cm, index=["Not Heart", "Heart"], columns=["Not Heart", "Heart"])
sn.heatmap(cm_to_df, annot=True, fmt="g")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(BASE_DIR, "..", "report", "heart_confusion_matrix.jpg"))
plt.show()

# 4.3. Analyze Feature Importances
importances = model.named_steps["classifier"].feature_importances_
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feature_imp)
print("")

# 4.4. Assess Model Stability Using Cross-Validation
scores = cross_val_score(model, x, y, cv=5)
print("Cross-val accuracy each fold:", scores)
print("Mean:", scores.mean(), "Std:", scores.std())
print("")

# 4.5. Compare Accuracy Between Training and Test Sets --> Check Overfitting
print("Train Accuracy:", model.score(x_train, y_train))
print("Test Accuracy:", model.score(x_test, y_test))

# Load Model
joblib.dump(model, "heart_model.pkl")
