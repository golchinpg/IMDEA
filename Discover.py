import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message=".*is_sparse is deprecated.*")


Dataset_path = "/home/pegah/Codes/imd/"
Dataset_names = ["Op1_CapDL_NoSteps.csv", "Op1_CapDL_NoSteps_wo_correlated_features.csv", "outliers_and_anomalies.csv"]
#Adding ground truth df
df2 = pd.read_csv(Dataset_path+Dataset_names[2], header=0, sep=',')
#print (df2.shape)
#print(df2.columns)
#print(df2.info)
#data with 135 columns
df = pd.read_csv(Dataset_path+Dataset_names[0], header=0, sep=',')
#print(df.shape)
#print(df.columns)
#add them to the target
df_new = pd.concat([df, df2.iloc[:, 1:]], axis=1)
#print(df_new.shape)
#print(df_new.columns)
print(f"Number of outliers:{df_new[df_new['outliers']==1.0].shape[0]}")
#print(df_new.info)
#cutting the df to bins
df_new['transfer.datarate.bins']= pd.cut(df_new['transfer.datarate'], bins=10, labels=False)
print(f"Columns:{df_new.columns}")

#find in which bins the outliers are 1
bin_values = df_new.loc[df_new['outliers']==1, 'transfer.datarate.bins']
print(f"In which bin, the outliers are 1:{bin_values}") #outliers are always in the bins 0, 1, 2 
# Use pd.cut with 50 bins and store the result
binned = pd.cut(df['transfer.datarate'], bins=10)

# Access bin ranges (categories)
bin_ranges = binned.cat.categories
print("Bin 0 range:", bin_ranges[0])
#print("Bin 1 range:", bin_ranges[1])
#print("Bin 2 range:", bin_ranges[2])
"""
#understanding data:
correlation_matrix = df_new.corr()
plt.figure(figsize=(15,10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.savefig(Dataset_path+'correlation.png')

#reduce the dimension to see the pattern
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_new)
pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_results, columns = ['PCA1', 'PCA2'])
plt.figure(figsize=(10,7))
sns.scatterplot(x='PCA1', y='PCA2', data=pca_df)
plt.title("PCA Scatter plot")
plt.savefig(Dataset_path+'PCA_plot.png')

#last 4 columns
last_4_cols = df_new.iloc[:, -4:]
sns.pairplot(last_4_cols)
plt.title("pair plot")
plt.savefig(Dataset_path+'last_4_cols.png')
"""
#"""
df = df.drop_duplicates() #there is no duplications
print(df.shape)
print (df["transfer.datarate"].max(), df["transfer.datarate"].min(), df["transfer.datarate"].mean())
#print(df.isnull().any())
scaler = MinMaxScaler()

"""
#write an xgboost for regression of the last column
X = df.iloc[:, :-1]
scaler.fit_transform(X)
y = df.iloc[:, -1]
plt.hist(y, bins=50, alpha= 0.7, color='blue')
plt.title("Target Distribution")
plt.savefig(Dataset_path+"target_distribution.png")
y = np.log(df.iloc[:, -1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.001,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
#1275028.049 2408.967 393434.7491348651 --> regressor results
"""
X_class = df_new.iloc[:, :-4]
y_class = df_new['transfer.datarate.bins']
print(y_class.value_counts())
"""
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2)
model = xgb.XGBClassifier(
    objective = 'multi:softprob', 
    num_classes = 10, 
    n_estimators=1000, 
    max_depth = 10,
    random_state = 42
)
model.fit(X_train_class, y_train_class)
y_pred_class = model.predict(X_test_class)
print(f"Classification Report: {classification_report(y_test_class, y_pred_class)}")
"""
"""
importance = model.feature_importances_
important_features = [X.columns[i] for i in np.argsort(importance)[::-1]]
Features_training = important_features[:20]
X_new_training = X_train[Features_training]
X_new_test = X_test[Features_training]
model.fit(X_new_training, y_train)

y_pred = model.predict(X_new_test)
y_pred_originalScale = np.exp(y_pred)
y_test_originalScale = np.exp(y_test)

mae= mean_absolute_error(y_test_originalScale, y_pred_originalScale)
mse = mean_squared_error(y_test_originalScale, y_pred_originalScale)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
"""
df_outlier = df_new[df_new['transfer.datarate.bins']==0]
print(f"Shape of data frame where the outliers can happer:{df_outlier.shape}")
df_outlier.drop(['transfer.datarate.bins', 'Unnamed: 0', 'anomalies'], axis=1, inplace=True)
print(df_outlier.columns)
#model.save_model('xgboostmodel_classification.json')
X_outlier = df_outlier.iloc[:, :-1]
y_outlier = df_outlier['outliers']
X_train_outlier, X_test_outlier, y_train_outlier, y_test_outlier = train_test_split(X_outlier, y_outlier, test_size=0.2)
model = xgb.XGBClassifier(
    objective = 'binary:logistic', 
    num_classes = 10, 
    n_estimators=1000, 
    max_depth = 10,
    random_state = 42
)
model.fit(X_train_outlier, y_train_outlier)
y_pred_class = model.predict(X_test_outlier)
print(f"Classification Report: {classification_report(y_test_outlier, y_pred_class)}")
print("\n AUROC Evaluation:")
print(roc_auc_score(y_test_outlier, y_pred_class))
