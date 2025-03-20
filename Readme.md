# Diabetes Prediction using Support Vector Machine (SVM)

## 📌 Overview
This project implements a **Diabetes Prediction Model** using **Support Vector Machine (SVM)**. The model is trained on the **Pima Indians Diabetes Dataset**, which includes medical parameters such as glucose levels, BMI, insulin, and age to predict whether a person has diabetes.

## 📂 Dataset Information
The dataset contains the following features:
- **Pregnancies**: Number of pregnancies
- **Glucose**: Blood glucose level
- **BloodPressure**: Blood pressure measurement
- **SkinThickness**: Skin fold thickness
- **Insulin**: Insulin level in blood
- **BMI**: Body Mass Index
- **DiabetesPedigreeFunction**: Likelihood of diabetes based on family history
- **Age**: Patient's age
- **Outcome**: 1 (Diabetic), 0 (Non-Diabetic)

## 🛠 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/diabetes-svm.git
cd diabetes-svm
pip install -r requirements.txt
```

## 🚀 Running the Project
Run the Python script to train and test the SVM model:
```bash
python diabetes_svm.py
```

## 📊 Exploratory Data Analysis (EDA)
Before training, we perform data visualization using **Matplotlib**:
- **Histogram**: Distribution of glucose levels for diabetic vs. non-diabetic
- **Bar Chart**: Average glucose comparison

## 🏋️‍♂️ Model Training
We use **Support Vector Machine (SVM)** with a linear kernel:
```python
from sklearn.svm import SVC
model = SVC(kernel='linear', verbose=True)
model.fit(X_train, y_train)
```
### Model Evaluation:
- **Accuracy:** ~76%
- **Precision, Recall, F1-score:** Computed using classification report

#### **Latest Classification Report:**
```
               precision    recall  f1-score   support

           0       0.81      0.82      0.81        99
           1       0.67      0.65      0.66        55

    accuracy                           0.76       154
   macro avg       0.74      0.74      0.74       154
weighted avg       0.76      0.76      0.76       154
```

## 🔍 Making Predictions
You can **predict diabetes** for new data:
```python
new_data = [[5, 140, 80, 20, 85, 28.0, 0.45, 45]]  # Example input
prediction = model.predict(new_data)
print("Diabetes Prediction:", prediction)
```

## 📌 Future Improvements
- **Hyperparameter Tuning** to improve accuracy
- **Feature Engineering** for better insights
- **Deploying the Model** using Flask or FastAPI

## 🤝 Contributing
Feel free to fork the repo and submit pull requests.

## 📜 License
This project is open-source under the **MIT License**.

