# Diabetes Prediction ML Project 🩺

## 📌 Overview

This project predicts whether a patient is likely to have diabetes based on diagnostic measurements. Using machine learning algorithms, it analyzes patient data to provide accurate predictions. Users can also input their own data to check the likelihood of diabetes.

## 📊 Dataset

The project uses the Pima Indians Diabetes Dataset, which contains diagnostic measurements for female patients aged 21+ of Pima Indian heritage.

### Feature	Description
* Pregnancies	Number of times pregnant
* Glucose	Plasma glucose concentration (mg/dL)
* BloodPressure	Diastolic blood pressure (mm Hg)
* SkinThickness	Triceps skin fold thickness (mm)
* Insulin	2-Hour serum insulin (mu U/ml)
* BMI	Body Mass Index (weight in kg/(height in m)^2)
* DiabetesPedigreeFunction	Family history likelihood function
* Age	Age of the patient
* Outcome	1 = Diabetic, 0 = Non-Diabetic

Source: Kaggle Dataset

## 🛠 Features

* Data Preprocessing: Handle missing values, normalize and scale features.

* Exploratory Data Analysis (EDA): Visualizations of distributions, correlations, and diabetic patterns.

* Model Selection & Training: Logistic Regression, Decision Tree, Random Forest, SVM, KNN.

* Model Evaluation: Accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC.

* Prediction: Predict diabetes for new patient input data.

## 💻 Technologies Used

Python 3

Jupyter Notebook

Pandas, NumPy

## 🚀 Installation & Usage
### 1. Clone the repository:
git clone https://github.com/"your-username"/diabetes-prediction.git
cd diabetes-prediction

### 2. Install dependencies:
pip install -r requirements.txt

pip install pandas numpy scikit-learn matplotlib seaborn

### 3. Run the Jupyter Notebook:
jupyter notebook

Open Diabetes_Prediction.ipynb and execute all cells.

### 4. Predict for New Patients:

Enter feature values in the Prediction section

The model outputs:

Probability of having diabetes

* Classification: Diabetic or Non-Diabetic


## 📈 Sample Outputs

*Model Accuracy: 78% (Random Forest Classifier example)

*Confusion Matrix

*ROC Curve


## 📝 Future Enhancements

* Feature selection for improved model performance

* Advanced models like XGBoost or Neural Networks

* Web deployment for real-time predictions

* User input validation and error handling
