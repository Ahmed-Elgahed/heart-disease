from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# قراءة البيانات
data = pd.read_csv(r'C:\Users\ASUS\Downloads\heart.csv.xls')
# ملء القيم المفقودة
data['age'].fillna(data['age'].mean(), inplace=True)
data['sex'].fillna(data['sex'].mode()[0], inplace=True)
data['cp'].fillna(data['cp'].mode()[0], inplace=True)
data['trestbps'].fillna(data['trestbps'].mean(), inplace=True)
data['chol'].fillna(data['chol'].mean(), inplace=True)
data['fbs'].fillna(data['fbs'].mode()[0], inplace=True)
data['restecg'].fillna(data['restecg'].mode()[0], inplace=True)
data['thalach'].fillna(data['thalach'].mean(), inplace=True)
data['exang'].fillna(data['exang'].mode()[0], inplace=True)
data['oldpeak'].fillna(data['oldpeak'].mean(), inplace=True)
data['slope'].fillna(data['slope'].mode()[0], inplace=True)
data['ca'].fillna(data['ca'].mode()[0], inplace=True)
data['thal'].fillna(data['thal'].mode()[0], inplace=True)
# فصل الميزات والهدف
X = data.drop("target", axis=1)
y = np.array(data["target"])

# تحويل الأعمدة النصية إلى قيم رقمية باستخدام الـ dummy variables
X = pd.get_dummies(X, drop_first=True)

# تحجيم البيانات
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تدريب النموذج
clf = LogisticRegression(random_state=0, max_iter=1000, class_weight='balanced').fit(X_scaled, y)

# حساب الدقة
accuracy = clf.score(X_scaled, y)

# التنبؤات
y_pred = clf.predict(X_scaled)

# واجهة التطبيق
st.title("Heart Diseases Prediction")

# sidebar
nav = st.sidebar.radio('Options', ["Home", "Prediction", "About Us"])

# صفحة "Home"
if nav == "Home":
    st.image("C:/Users/ASUS/Downloads/camilo-jimenez-OermHGSUzhI-unsplash.jpg")
  # قم بتعديل مسار الصورة
    st.markdown("## Heart Disease Dataset")
    st.markdown("### About Dataset : ")
    st.text("""This dataset contains 76 attributes, including the predicted attribute.
              The target field indicates the presence of heart disease in the patient. 
              0 means no disease and 1 means disease.""")
    st.dataframe(data)

# صفحة "Prediction"
elif nav == "Prediction":
    st.header("Heart Disease Prediction")
    
    # المدخلات من المستخدم
    age = st.slider("Age", 18, 90, step=1)
    sex = st.slider("Sex: male = 0, female = 1", 0, 1, step=1)
    cp = st.slider("Chest pain type (0-3)", 0, 3, step=1)
    trestbps = st.slider("Resting blood pressure (mm Hg)", 90, 200, step=1)
    chol = st.slider("Serum cholesterol in mg/dl", 125, 565, step=1)
    fbs = st.slider("Fasting blood sugar > 120 mg/dl", 0, 1, step=1)
    restecg = st.slider("Resting electrocardiographic results (0-2)", 0, 2, step=1)
    thalach = st.slider("Maximum heart rate achieved", 70, 205, step=1)
    exang = st.slider("Exercise induced angina (0 = No, 1 = Yes)", 0, 1, step=1)
    oldpeak = st.slider("Oldpeak = ST depression induced by exercise", 0.0, 6.2, step=0.1)
    slope = st.slider("Slope of the peak exercise ST segment (0-2)", 0, 2, step=1)
    ca = st.slider("Number of major vessels (0-4) colored by fluoroscopy", 0, 4, step=1)
    thal = st.slider("Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect", 0, 3, step=1)
    
    # تحويل المدخلات إلى مصفوفة للتنبؤ
    val = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

    # التنبؤ
    # إضافة الأعمدة الناقصة في `val` وتعيين قيمها الافتراضية إلى 0
    missing_cols = set(X.columns) - set(pd.DataFrame(val, columns=X.columns[:len(val[0])]).columns)
    for col in missing_cols:
        val[0] = 0  # إضافة الأعمدة الناقصة كقيم صفرية

# تحويل البيانات إلى DataFrame وإعادة ترتيب الأعمدة لتطابق `X`
    val = pd.DataFrame(val, columns=X.columns)
    val = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

# تحويل val إلى DataFrame مع الأعمدة الناقصة وإضافة الأعمدة الناقصة كقيم صفرية
    val_df = pd.DataFrame(val, columns=X.columns[:13])  # 13 يمثل عدد الأعمدة الأصلية في val
    for col in X.columns[13:]:
        val_df[col] = 0  # إضافة الأعمدة الناقصة كقيم صفرية


# التنبؤ
   

    pred = clf.predict(val)[0]

    # عرض النتيجة
    if st.button("Predict"):
        st.markdown("### It is integer valued 0 = no disease and 1 = disease.")
        st.success(f"The prediction is: {pred}")
        if pred == 0:
            st.write("You are okay!")
        else:
            st.write("You may have heart disease.")

    # خيار لإضافة بيانات المستخدم
    st.title("Contribute to Improve Results")
    if st.checkbox("Do you want to add your data to our database?"):
        if st.button("Submit"):
            to_add = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
                      "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
                      "slope": slope, "ca": ca, "thal": thal, "target": pred}
            to_add = {k: [v] for k, v in to_add.items()}  # تحويل البيانات إلى DataFrame
            to_add = pd.DataFrame(to_add)
            to_add.to_csv("heart.csv", mode="a", index=False, header=False)
            st.success("Data Added Successfully")

# صفحة "About Us"
elif nav == "About Us":
    st.header("About Me")
    st.write("My name is Ahmed Elgahed")
    st.write("Contact me on WhatsApp: +201116115694")
    st.write("My Kaggle: https://www.kaggle.com/ahmedelgahed")
    st.write("My GitHub: https://github.com/Ahmed-Elgahed")
    st.write("My LinkedIn: www.linkedin.com/in/ahmed-mohamed-946890276")
    st.text_input("Your feedback")
    st.header("Thanks for your feedback!")



#streamlit run "C:\Users\ASUS\Downloads\from sklearn.py"
