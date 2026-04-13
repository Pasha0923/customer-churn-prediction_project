import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model_LightGBM.joblib") 
FILE_PATH = os.path.join(BASE_DIR, "models", "models_summary.csv")  
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.joblib")

st.set_page_config(page_title="Churn Prediction", layout="wide")

# 🔹 Завантаження моделі та ознак
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

def manual_prediction(input_data: pd.DataFrame, model, feature_names):
    """
    Виконує прогноз відтоку клієнта
    Для моделі використовуються оброблені ознаки, але вихідні дані не змінюються.
    """
    data = input_data.copy()

    # 🔹 Логарифмування вихідних ознак для моделі
    for col in ["bill_avg", "upload_avg", "download_avg"]:
        if col in data.columns:
            data[f"{col}_log"] = np.log1p(data[col])

    # 🔹 Приведення типів для дискретних ознак
    int_cols = ["download_over_limit", "service_failure_count", "remaining_contract"]
    for col in int_cols:
        if col in data.columns:
            data[col] = data[col].astype(int)

    # 🔹 Приведення булевих ознак до int
    bool_cols = ["is_movie_package_subscriber", "is_tv_subscriber"]
    for col in bool_cols:
        if col in data.columns:
            data[col] = data[col].astype(int)

    # 🔹 Додавання відсутніх ознак
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0

    # 🔹 Упорядкування ознак
    data = data[feature_names]

    # 🔹 Передбачення
    prob = model.predict_proba(data)[:, 1][0]
    label = "🔴 Високий ризик" if prob > 0.5 else "🟢 Низький ризик"

    return prob, label

def personalized_recommendations(input_data: pd.DataFrame, probability: float):
    """
    Генерує персоналізовані рекомендації на основі вхідних даних клієнта.
    """
    recs = []

  
    bill_avg = input_data.get("bill_avg", [0])[0]
    subscription_age = input_data.get("subscription_age", [0])[0]
    remaining_contract = input_data.get("remaining_contract", [0])[0]
    download_over_limit = input_data.get("download_over_limit", [0])[0]
    service_failure_count = input_data.get("service_failure_count", [0])[0]
    is_movie = input_data.get("is_movie_package_subscriber", [0])[0]
    is_tv = input_data.get("is_tv_subscriber", [0])[0]

    # 🟢 Низький ризик
    if probability < 0.3:
        return ["✅ Профіль клієнта стабільний"]

    # 🟡 Середній ризик
    if 0.3 <= probability < 0.7:
        if remaining_contract < 3:
            recs.append("📄 Нагадати клієнту про можливість продовження контракту")

        if service_failure_count > 1:
            recs.append("🛠 Перевірити якість технічного обслуговування")

        if download_over_limit > 0:
            recs.append("📶 Запропонувати тариф з більшим обсягом інтернету")

        if not recs:
            recs.append("ℹ️ Рекомендується подальше спостереження за клієнтом")

        return recs

    # 🔴 Високий ризик
    if probability >= 0.7:
        recs.append("🚨 Негайно зв'язатися з клієнтом")

        if bill_avg > 50:  # Порог можна адаптувати під бізнес
            recs.append("💰 Запропонувати персональну знижку або вигідні умови")

        if subscription_age < 3:
            recs.append("🎁 Запропонувати бонус для нових клієнтів")

        if remaining_contract < 3:
            recs.append("📄 Терміново запропонувати продовження контракту")

        if download_over_limit > 0:
            recs.append("📶 Запропонувати тариф з більшим лімітом інтернету")

        if service_failure_count > 1:
            recs.append("🛠 Надати пріоритетну технічну підтримку")

        if not is_movie:
            recs.append("🎬 Запропонувати підписку на пакет фільмів")

        if not is_tv:
            recs.append("📺 Запропонувати підписку на TV пакет")

        return recs

# 🔹 Важливість ознак
def plot_feature_importance(model, feature_names):
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=fi.head(10), x="importance", y="feature")
    plt.title("Top-10 важливих ознак")
    st.pyplot(plt)

# 🔹 Аналітика
def data_analysis_section(df: pd.DataFrame):
    st.subheader("📊 Аналіз даних")

# 🔹 Округлення числових значень для зручного відображення
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_display = df.copy()
    df_display[numeric_cols] = df_display[numeric_cols].round(7)


    st.markdown("### 📋 Таблиця результатів метрик моделей")
    st.dataframe(df_display, use_container_width=True)

    if "target" in df.columns:
        plt.figure()
        sns.countplot(x="target", data=df)
        st.pyplot(plt)

    st.markdown("### 🔥 Кореляція між метриками")
    numeric_df = df.select_dtypes(include=[np.number])

    if not numeric_df.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,     
            fmt=".2f",       
            cmap="coolwarm",
            linewidths=0.5
        )
        plt.title("Кореляційна матриця")
        st.pyplot(plt)
    else:
        st.warning("В файлі відсутні числові ознаки для розрухунку кореляції.")

    st.markdown("### ⭐ Топ-10 найбільш важливих ознак")
    plot_feature_importance(model, feature_names)

# 🔹 UI

st.title("💡 Customer Churn Prediction")
menu = ["🔮 Прогноз", "📊 Аналітика"]
choice = st.sidebar.selectbox("Меню", menu)

# 🔹 Прогноз
if choice == "🔮 Прогноз":

    st.header("💻 Ввід даних клієнта")
    col1, col2 = st.columns(2)
    user_input = {}
    for feature in feature_names:

        if feature == "no_contract":
            continue
        # Логарифмовані ознаки 
        if feature in ["bill_avg_log", "upload_avg_log", "download_avg_log"]:
            base_feature = feature.replace("_log", "")
            user_input[base_feature] = st.number_input(base_feature, min_value=0, max_value=100000, value=0)
            # 🔹 Дискретні ознаки 
        elif feature in ["download_over_limit", "service_failure_count"]:
            user_input[feature] = st.number_input(feature, min_value=0, max_value=20, value=0, step=1)
             # 🔹 Бінарні ознаки
        elif feature in ["is_movie_package_subscriber", "is_tv_subscriber"]:
            user_input[feature] = st.checkbox(feature)
              # 🔹 Інші числові ознаки
        else:
            user_input[feature] = st.number_input(feature, min_value=0.0, max_value=100000.0, value=0.0)

    input_df = pd.DataFrame([user_input])

    if st.button("🚀 Зробити прогноз"):
        prob, label = manual_prediction(input_df, model, feature_names)

        st.success("Прогноз готовий")

        c1, c2 = st.columns(2)
        c1.metric("Вірогідність відтоку", f"{prob:.2%}")
        c2.metric("", label)

        st.subheader("📌 Рекомендації")
        recs = personalized_recommendations(input_df, prob)

        for r in recs:
            st.write(r)

elif choice == "📊 Аналітика":

    if os.path.exists(FILE_PATH):
        df = pd.read_csv(FILE_PATH)
        st.info(f"Файл моделей успішно завантажений ")
        data_analysis_section(df)
    else:
        st.warning(f"Файл {os.path.basename(FILE_PATH)} не знайден. Завантажте файл.")
        uploaded_file = st.file_uploader("Завантажте CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            data_analysis_section(df)