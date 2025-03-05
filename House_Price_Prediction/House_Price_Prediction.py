"""
#####################################################
# HOUSE PRICE PREDICTION
#####################################################

This script performs house price prediction using various machine learning models. 
It includes data preprocessing, exploratory data analysis (EDA), feature engineering, 
and model training with hyperparameter tuning.

### Steps:
1. Overview of the dataset
2. Analysis of Categorical Variables
3. Analysis of Numerical Variables
4. Analysis of Target Variable
5. Correlation Analysis
6. Model Training and Evaluation
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

# Pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Reading the dataset
df = pd.read_csv("datasets/house_price_prediction.csv")


######################################
# 1. Overview of the Dataset
######################################

def check_df(dataframe):
    """
    Prints general information about the dataset, including shape, data types, head/tail, 
    missing values, and quantile statistics.
    
    Parameters:
    dataframe (pd.DataFrame): The dataset to be analyzed.
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Removing outliers from the dataset
df = df.loc[df["SalePrice"] <= 400000]

check_df(df)


##################################
# IDENTIFYING NUMERICAL AND CATEGORICAL VARIABLES
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe: DataFrame to analyze
    :param cat_th: Threshold value for numerical but categorical variables
    :param car_th: Threshold value for categorical but cardinal variables
    :return: Lists of categorical, numerical, and categorical but cardinal variables
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'Categorical columns: {len(cat_cols)}')
    print(f'Numerical columns: {len(num_cols)}')
    print(f'Categorical but cardinal columns: {len(cat_but_car)}')
    print(f'Numerical but categorical columns: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = total number of variables.
    # num_but_cat is already included in cat_cols.
    # Thus, all variables are selected using these three lists: cat_cols + num_cols + cat_but_car.
    # num_but_cat is provided for reporting purposes only.

    return cat_cols, cat_but_car, num_cols, num_but_cat

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)



######################################
# 2. Analysis of Categorical Variables
######################################

def cat_summary(dataframe, col_name, plot=False):
    """
    Displays the frequency and ratio of unique values in a categorical variable.
    
    :param dataframe: The DataFrame containing the data.
    :param col_name: The name of the categorical column to be analyzed.
    :param plot: If True, displays a count plot of the variable.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)




######################################
# 3. Analysis of Numerical Variables
######################################

def num_summary(dataframe, numerical_col, plot=False):
    """
    Displays descriptive statistics and an optional histogram for a numerical variable.

    :param dataframe: The DataFrame containing the data.
    :param numerical_col: The name of the numerical column to be analyzed.
    :param plot: If True, displays a histogram of the variable.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col, True)



######################################
# 4. Analysis of Target Variable
######################################

def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Calculates and displays the mean of the target variable grouped by a categorical variable.

    :param dataframe: The DataFrame containing the data.
    :param target: The target variable to analyze.
    :param categorical_col: The categorical variable used for grouping.
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)

# Examining the distribution of the dependent (target) variable
df["SalePrice"].hist(bins=100)
plt.show()

# Examining the log transformation of the dependent variable
np.log1p(df['SalePrice']).hist(bins=50)
plt.show()


######################################
# 5. Analysis of Correlation
######################################

# Compute correlation matrix for numerical variables
corr = df[num_cols].corr()
corr

# Display correlation heatmap
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    """
    Identifies highly correlated columns based on a given threshold.

    :param dataframe: The DataFrame containing the data.
    :param plot: If True, displays a heatmap of the correlation matrix.
    :param corr_th: The correlation threshold above which variables are considered highly correlated.
    :return: A list of column names that are highly correlated.
    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    
    return drop_list

high_correlated_cols(df, plot=False)



######################################
# Outlier Analysis
######################################

def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    """
    Calculates the lower and upper limits for detecting outliers based on interquartile range.

    :param dataframe: The DataFrame containing the data.
    :param variable: The column for which outlier thresholds are calculated.
    :param low_quantile: Lower quantile value for threshold calculation (default: 0.10).
    :param up_quantile: Upper quantile value for threshold calculation (default: 0.90).
    :return: Lower and upper limit for detecting outliers.
    """
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Checks whether a given numerical column contains outliers.

    :param dataframe: The DataFrame containing the data.
    :param col_name: The numerical column to check for outliers.
    :return: True if outliers are found, False otherwise.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, variable):
    """
    Replaces outliers in a numerical column with threshold values.

    :param dataframe: The DataFrame containing the data.
    :param variable: The numerical column in which outliers will be replaced.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)


######################################
# Missing Value Analysis
######################################

def missing_values_table(dataframe, na_name=False):
    """
    Identifies missing values in the DataFrame and displays the count and percentage of missing values per column.

    :param dataframe: The DataFrame to analyze.
    :param na_name: If True, returns the list of columns with missing values.
    :return: List of columns with missing values (if na_name=True).
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


# Display missing values table
missing_values_table(df)

# Checking unique values in specific categorical columns
df["Alley"].value_counts()
df["BsmtQual"].value_counts()

# Some missing values indicate the absence of a particular feature in the house
no_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
           "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

# Filling missing values with "No" to indicate absence of the feature
for col in no_cols:
    df[col].fillna("No", inplace=True)

# Display missing values table after filling missing values
missing_values_table(df)


# This function fills missing values with median or mean for numerical variables and mode for categorical variables.

def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    """
    Imputes missing values in a DataFrame using different methods for categorical and numerical variables.

    :param data: The DataFrame containing missing values.
    :param num_method: The method to fill missing values for numerical variables ("mean" or "median").
    :param cat_length: The threshold for categorical variables; if the number of unique values is below this, mode is used.
    :param target: The target variable, which remains unchanged during imputation.
    :return: The DataFrame with missing values imputed.
    """
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # List of variables with missing values

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Number of missing values before imputation

    # Fill missing values with mode for categorical variables with unique values <= cat_length
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # Fill missing values with mean or median for numerical variables
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)


######################################
# RARE ENCODING
######################################

def rare_analyser(dataframe, target, cat_cols):
    """
    Analyzes the distribution of categorical variables and their relationship with the target variable.

    :param dataframe: The DataFrame containing the data.
    :param target: The target variable.
    :param cat_cols: List of categorical columns to analyze.
    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


# Analyze rare categorical variables
rare_analyser(df, "SalePrice", cat_cols)


# Encoding rare categories by grouping infrequent values

# ExterCond: Merging rare categories
df["ExterCond"] = np.where(df.ExterCond.isin(["Fa", "Po"]), "FaPo", df["ExterCond"])
df["ExterCond"] = np.where(df.ExterCond.isin(["Ex", "Gd"]), "Ex", df["ExterCond"])

# LotShape: Merging rare categories
df["LotShape"] = np.where(df.LotShape.isin(["IR1", "IR2", "IR3"]), "IR", df["LotShape"])

# GarageQual: Merging rare categories
df["GarageQual"] = np.where(df.GarageQual.isin(["Fa", "Po"]), "FaPo", df["GarageQual"])
df["GarageQual"] = np.where(df.GarageQual.isin(["Ex", "Gd", "TA"]), "ExGd", df["GarageQual"])

# BsmtFinType2: Merging rare categories
df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["GLQ", "ALQ"]), "RareExcellent", df["BsmtFinType2"])
df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["BLQ", "LwQ", "Rec"]), "RareGood", df["BsmtFinType2"])


# Nadir sınıfların tespit edilmesi
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


rare_encoder(df,0.01)


######################################
# Creating New Features
######################################

"""
This section creates new variables based on existing features to enhance data representation.
"""

# Interaction Features
df["NEW_1st*GrLiv"] = df["1stFlrSF"] * df["GrLivArea"]
df["NEW_Garage*GrLiv"] = df["GarageArea"] * df["GrLivArea"]

# Quality and Condition Aggregation
df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", 
                      "GarageQual", "GarageCond", "Fence"]].sum(axis=1)

df["TotalGarageQual"] = df[["GarageQual", "GarageCond"]].sum(axis=1)
df["Overall"] = df[["OverallQual", "OverallCond"]].sum(axis=1)
df["Exter"] = df[["ExterQual", "ExterCond"]].sum(axis=1)
df["Qual"] = df[["OverallQual", "ExterQual", "GarageQual", "Fence", "BsmtFinType1", 
                 "BsmtFinType2", "KitchenQual", "FireplaceQu"]].sum(axis=1)
df["Cond"] = df[["OverallCond", "ExterCond", "GarageCond", "BsmtCond", "HeatingQC", 
                 "Functional"]].sum(axis=1)

# Structural Area Features
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
df["NEW_TotalBsmtFin"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]
df["NEW_PorchArea"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["ScreenPorch"] + df["3SsnPorch"] + df["WoodDeckSF"]
df["NEW_TotalHouseArea"] = df["NEW_TotalFlrSF"] + df["TotalBsmtSF"]
df["NEW_TotalSqFeet"] = df["GrLivArea"] + df["TotalBsmtSF"]

# Bathroom Features
df["NEW_TotalFullBath"] = df["BsmtFullBath"] + df["FullBath"]
df["NEW_TotalHalfBath"] = df["BsmtHalfBath"] + df["HalfBath"]
df["NEW_TotalBath"] = df["NEW_TotalFullBath"] + (df["NEW_TotalHalfBath"] * 0.5)

# Area Ratios
df["NEW_LotRatio"] = df["GrLivArea"] / df["LotArea"]
df["NEW_RatioArea"] = df["NEW_TotalHouseArea"] / df["LotArea"]
df["NEW_GarageLotRatio"] = df["GarageArea"] / df["LotArea"]

# Masonry Veneer Ratio
df["NEW_MasVnrRatio"] = df["MasVnrArea"] / df["NEW_TotalHouseArea"]

# Unused Land Area
df["NEW_DifArea"] = df["LotArea"] - df["1stFlrSF"] - df["GarageArea"] - df["NEW_PorchArea"] - df["WoodDeckSF"]

# Low Quality Finished Square Footage Ratio
df["NEW_LowQualFinSFRatio"] = df["LowQualFinSF"] / df["NEW_TotalHouseArea"]

# Overall Scores
df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"]
df["NEW_KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]
df["NEW_FireplaceScore"] = df["Fireplaces"] * df["FireplaceQu"]

# Age & Renovation Features
df["NEW_Restoration"] = df["YearRemodAdd"] - df["YearBuilt"]
df["NEW_HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["NEW_RestorationAge"] = df["YrSold"] - df["YearRemodAdd"]
df["NEW_GarageAge"] = df["GarageYrBlt"] - df["YearBuilt"]
df["NEW_GarageRestorationAge"] = np.abs(df["GarageYrBlt"] - df["YearRemodAdd"])
df["NEW_GarageSold"] = df["YrSold"] - df["GarageYrBlt"]

# Dropping Unnecessary Features
drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope", "Heating", 
             "PoolQC", "MiscFeature", "Neighborhood"]

df.drop(drop_list, axis=1, inplace=True)




##################
# Applying Label Encoding & One-Hot Encoding
##################

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    """
    Applies label encoding to binary categorical variables.

    :param dataframe: The DataFrame containing the data.
    :param binary_col: The binary categorical column to be encoded.
    :return: DataFrame with the encoded column.
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# Identifying binary categorical variables
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

# Applying Label Encoding to binary categorical variables
for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Applies One-Hot Encoding to categorical variables.

    :param dataframe: The DataFrame containing the data.
    :param categorical_cols: List of categorical columns to be encoded.
    :param drop_first: If True, drops the first category to avoid multicollinearity.
    :return: DataFrame with One-Hot Encoded categorical columns.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# Applying One-Hot Encoding to categorical variables
df = one_hot_encoder(df, cat_cols, drop_first=True)




##################################
# MODELING
##################################

# Applying log transformation to the target variable
y = np.log1p(df['SalePrice'])
X = df.drop(["Id", "SalePrice"], axis=1)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# List of regression models
models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          # ('SVR', SVR()),  # Commented out
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))  # Commented out

def evaluate_models(models, X, y):
    """
    Evaluates multiple regression models using RMSE and cross-validation.

    :param models: A list of tuples containing model names and their corresponding regressor objects.
    :param X: Feature matrix.
    :param y: Target variable.
    """
    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")


# Evaluating models
evaluate_models(models, X, y)

# Displaying basic statistics of the target variable
df['SalePrice'].mean()
df['SalePrice'].std()


##################
# Hyperparameter Optimization
##################

# Initializing the LightGBM model
lgbm_model = LGBMRegressor(random_state=46)

# Evaluating baseline RMSE using cross-validation
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))

# Defining hyperparameter grid for optimization
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500]
               # "colsample_bytree": [0.5, 0.7, 1]  # Commented out for now
              }

def optimize_model(model, params, X_train, y_train, cv=3):
    """
    Performs hyperparameter optimization using GridSearchCV.

    :param model: The base model to be optimized.
    :param params: The hyperparameter grid for tuning.
    :param X_train: Training feature matrix.
    :param y_train: Training target variable.
    :param cv: Number of cross-validation folds (default: 3).
    :return: The best GridSearchCV model after optimization.
    """
    gs_best = GridSearchCV(model,
                           params,
                           cv=cv,
                           n_jobs=-1,
                           verbose=True).fit(X_train, y_train)
    return gs_best

# Optimizing LightGBM model
lgbm_gs_best = optimize_model(lgbm_model, lgbm_params, X_train, y_train)

# Training the final model with optimized hyperparameters
final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

# Evaluating the final model's RMSE
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))



##################
# Model Testing
##################

# Training the LightGBM model with the best hyperparameters
lgbm_tuned = LGBMRegressor(**lgbm_gs_best.best_params_).fit(X_train, y_train)

# Making predictions on the test set
y_pred = lgbm_tuned.predict(X_test)

# Applying inverse transformation of the log transformation
new_y = np.expm1(y_pred)
new_y_test = np.expm1(y_test)

def evaluate_rmse(y_true, y_pred):
    """
    Evaluates the model performance using Root Mean Squared Error (RMSE).

    :param y_true: The actual values.
    :param y_pred: The predicted values.
    :return: RMSE score.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Calculating RMSE
rmse = evaluate_rmse(new_y_test, new_y)
print(f"RMSE: {rmse}")

