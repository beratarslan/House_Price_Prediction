# 🏡 House Price Prediction

## 📌 Project Overview
This project aims to predict house prices using various machine learning models. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and model evaluation.

---

## 📂 Dataset
The dataset contains various features related to house properties, including:
- **Size** (e.g., `GrLivArea`, `TotalBsmtSF`)
- **Quality & Condition** (e.g., `OverallQual`, `OverallCond`, `ExterCond`)
- **Location & Structure** (e.g., `Neighborhood`, `GarageType`)
- **Target Variable**: `SalePrice` (House price)

---

## 🛠 Project Workflow
### **1️⃣ Data Preprocessing**
- Load and inspect the dataset.
- Handle missing values through imputation.
- Detect and replace outliers using statistical thresholds.

### **2️⃣ Exploratory Data Analysis (EDA)**
- Identify numerical and categorical variables.
- Analyze distributions of categorical and numerical features.
- Compute and visualize feature correlations.
- Examine relationships between features and `SalePrice`.

### **3️⃣ Feature Engineering**
- Create new meaningful features to enhance predictive power.
- Apply **rare encoding** to handle infrequent categorical values.
- Perform **label encoding** and **one-hot encoding** for categorical variables.

### **4️⃣ Model Training & Evaluation**
- Split data into **training** and **test** sets.
- Train multiple regression models:
  - **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
  - **Tree-Based Models**: Decision Tree, Random Forest, Gradient Boosting
  - **Advanced Models**: XGBoost, LightGBM
  - **Other Models**: K-Nearest Neighbors (KNN)
- Evaluate models using **Root Mean Squared Error (RMSE)** via cross-validation.

### **5️⃣ Hyperparameter Optimization**
- Perform hyperparameter tuning using **GridSearchCV** for **LightGBM**.
- Train the final optimized model using the best parameters.

### **6️⃣ Model Testing**
- Make predictions on the test set.
- Reverse the **log transformation** applied to `SalePrice`.
- Compute the final RMSE to assess model performance.

