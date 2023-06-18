"""
Here is a list of all installed packages

pip install:

numpy
seaborn
scikit-learn
pandas


Versions:

contourpy           1.1.0
cycler              0.11.0
fonttools           4.40.0
importlib-resources 5.12.0
joblib              1.2.0
kiwisolver          1.4.4
matplotlib          3.7.1
numpy               1.24.3
packaging           23.1
pandas              2.0.2
Pillow              9.5.0
pip                 23.1.2
pyparsing           3.0.9
python-dateutil     2.8.2
pytz                2023.3
scikit-learn        1.2.2
scipy               1.10.1
seaborn             0.12.2
setuptools          67.8.0
six                 1.16.0
threadpoolctl       3.1.0
tzdata              2023.3
zipp                3.15.0
"""

# imports
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as sts
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
########################################################################################################################
'''''
OVERVIEW OF THE DATA

In this section the data set is initially inspected by looking at the shape, columns general statistics, etc.

'''''

data = pd.read_csv("../data/data_original_csv.csv", encoding='latin1')  # read data
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print('###################################################')
print('data description:')
print('data shape:', data.shape)  # The shape of the dataframe
print('data unique hospital types:', data['Typ'].nunique())  # unique hospital types
print('data unique hospitals (Inst)', data['Inst'].nunique())  # not realistic -> check out Adr
print('data unique hospitals (Adr)', data['Adr'].nunique(), 'according to BAG only 276')
print('Mean missing values per column:', data.isna().sum().mean())  # mean missing values per column

# print(data.describe())                                                             # General statistics of the data
# print('data types:', data.dtypes)                                                  # Data types
# print('data missing values:', data.isna().sum())                                   # missing values
# print('data columns:', data.columns)                                               # The columns of the dataframe
# print('data head:', data.head())                                                   # The head of the dataframe
# print('data hospital types:', data['Typ'].value_counts())                          # Verschiedene Spitaltypen (Anzahl)
print('###################################################')
########################################################################################################################
'''''
MERGING COLUMS

Since the equation to calculate some features changed over the year the data is in different columns. In this step 
the columns are merged back together.

This is applied for:
Ptage, Aust, Neug, PtageStatA, AustStatA
'''''

tags = [['Ptage', 'PtageStatT', 'PtageStatMST'],  # Name of columns to combine
        ['Aust', 'AustStatT', 'AustStatMST'],
        ['Neug', 'NeugStatT', 'NeugStatMST'],
        ['PtageStatA', 'PtageStatMSA'],
        ['AustStatA', 'AustStatMSA']
        ]

jahre = [['Ptage_comb', 2010, 2014],  # New column name &
         ['Aust_comb', 2010, 2014],  # years at which the columns are combined
         ['Neug_comb', 2010, 2014],
         ['PtageStatA_comb', 2014],
         ['AustStatA_comb', 2014]
         ]

for j in range(len(jahre)):
    indexes = []
    for i in range(len(jahre[j]) - 1):
        index = min(data.index[data['JAHR'] == jahre[j][i + 1]])
        indexes.append(index)

        if data.loc[index - 1, 'JAHR'] != jahre[j][i + 1] - 1:
            print('ERROR in combining columns')
            break

    comb = []
    for k in range(len(indexes) + 1):
        if k == 0:
            col = (data[tags[j][k]][0:indexes[0]])
        elif k != 0 and k != len(indexes):
            col = (data[tags[j][k]][indexes[k - 1]:indexes[k]])
        elif k == len(indexes):
            col = (data[tags[j][k]][indexes[k - 1]:len(data)])

        comb.append(col)

    data[jahre[j][0]] = pd.concat(comb)
    data = data.drop(columns=tags[j], axis=1)

########################################################################################################################
'''
FUNCTIONS

Functions are defined for later use.

'''


def mean_missing_values(column):
    global data
    """
    compute the mean missing values of the data set orderd by the attributes of a column

    :param      column: column to compute the missing values
    :type       column: pandas column containing objects

    :return:    Returns a summary of the average missing values orderd by the column attributes

                Example: Mean missing values orderd by hospital type
                K211    17.508850
                ...        ...
                K112     1.077748
                Name: missing_values, dtype: float64

    :rtype:     Pandas Series
    """
    data['missing_values'] = data.isna().sum(axis=1)
    return (data.groupby(column)['missing_values'].mean()).sort_values(ascending=False)


def missing_values_percentage():
    global data
    """
    Calculates the percentage of missing values in each column

    :return:    Returns a dataframe containing columns (% missing values & Index [feature names])

                Example: % of missing values per column
                       Values                     Index
                0    0.000000                Unnamed: 0
                1    0.000000                      JAHR
                        ...                     ...
                201  0.333544           PtageStatA_comb
                202  0.333544            AustStatA_comb


    :rtype:     Pandas Dataframe
    """
    missing_values_percentages = []
    for p in data.columns:
        missing_values_percentages.append((data[p].isna().sum(axis=0)) / len(data[p]))
    return pd.DataFrame(data={'Values': missing_values_percentages, 'Index': data.columns})


# drop all columns which have more than Threshold% missing values
def screening(column, threshold):
    global data
    """
    Drops the column if the %missing values is higher than the threshold.

    :param      column: column to check
    :type       column: pandas column containing float64

    :param      data: data set
    :type       data: pandas dataframe
    
    :param      threshold: percentage missing values at which to drop column
    :type       thrshold: float

    :return:    Returns a dataframe containing columns (% missing values and a the feature names)

                Example: Mean missing values orderd by hospital type
                       Values                     Index
                0    0.000000                Unnamed: 0
                1    0.000000                      JAHR
                        ...                     ...
                201  0.333544           PtageStatA_comb
                202  0.333544            AustStatA_comb


    :rtype:     Pandas Dataframe
    """

    if (data[column].isna().sum(axis=0) / len(data[column])) > threshold:
        data = data.drop(column, axis=1)
    return data


drop_log_df = pd.DataFrame({}, index=['rows', 'columns'])


def drop_log(method):
    global drop_log_df
    global data
    """
    Keeps a log of all the adaptions made to the data

    :param      method: What was changed in the data
    :type       method: Str
    
    :return:    Returns the updated drop_log_df dataframe

    :rtype:     Pandas Dataframe
    """
    drop_log_df[method] = data.shape
    return drop_log_df


def add_identity(axes, *line_args, **line_kwargs):
    """
    Function taken from notebook
    """

    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def get_score(model, x_train, y_train, x_test, y_test, name):
    """
    :param      model: choose necessary model
    :type       model:

    :param      x_train: features of the training dataset
    :type       x_train: pandas dataframe

    :param      y_train: label of the training dataset
    :type       y_train: pandas dataframe

    :param      x_test: features of the test dataset
    :type       x_test: pandas dataframe

    :param      y_test: label of the test dataset
    :type       y_test: pandas dataframe

    :param      name: Plot name
    :type       name: str

    :return     returns the R2-score and RMSE for the Training and Test dataset
    """
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    plt.clf()
    fige = sns.scatterplot(x=y_pred_test, y=y_test)
    add_identity(fige, linestyle='--', color='black')
    fige.set_xlabel('Ground Truth CHF')
    fige.set_ylabel('Predicted CHF')
    fige.set_title(name + ' - Residual Analysis')
    plt.savefig('../Output/' + name + '.png')
    plt.close()

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=True)

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=True)

    print('Training set score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_train, rmse_train))
    print('Test set score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_test, rmse_test))


def plt_roc_curve(clf, name, x_test_roc, y_test_roc):
    """
    Plots the ROC curve for the classification models

    :param      clf: choose necessary model
    :type       clf:

    :param      x_test_roc: x_test
    :type       x_test_roc:

    :param      y_test_roc: y_test
    :type       y_test_roc:

    :param      name: Name of model used
    :type       name: str
    """

    y_pred_proba = clf.predict_proba(x_test_roc)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_roc, y_pred_proba)
    auc = roc_auc_score(y_test_roc, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve: ' + name)
    plt.legend(loc='lower right')
    plt.savefig('../Output/ROC_Curve_' + name)
    plt.close()
    return


def eval_metircs_classification(true_values, predicted_values, model):
    """
    prints all the evaluation metrics for the classification models

    :param      true_values: values for the label in the split data set
    :type       true_values: pandas column

    :param      predicted_values: predicted values for the label
    :type       predicted_values: pandas column

    :param      model: Model name
    :type       model: str
    """
    accuracy = accuracy_score(true_values, predicted_values)
    precision = precision_score(true_values, predicted_values)
    recall = recall_score(true_values, predicted_values)
    f1 = f1_score(true_values, predicted_values)
    cm = confusion_matrix(true_values, predicted_values)

    print('###################################################')
    print(model)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:", cm)
    print('\n###################################################')
    return


########################################################################################################################
'''''
CLEANING 

The preprocessing was conduceted in the following manner:

                             rows  columns
Data Shape                  3798      203       Orignial Dataframe         
Drop Years 2008/2009        3166      203       The first two year where removed due to lacking data       
Threshold Columns 50%       3166       53       A threshold of 50% of missing values was set for the columns. 
Drop Specialized Hospitals  1199       54       All rows containing specialized hospitals where removed
Drop additional columns     1199       44       Additional columns where identified and removed
Drop specific rows          1185       44       Additional rows where identified and removed  
Imputation of 'KostLangA'   1185       44       The missing values in 'KostLangA' where imputated using KNN

'''''

drop_log('Data Shape')  # initiate drop log

plt.figure(figsize=(15, 10))  # Check Heatmap for missing values
sns.heatmap(data.isna(), cmap='viridis', cbar=False)
plt.xlabel("Features", fontsize=16)
plt.ylabel("Row Index", fontsize=16)
plt.title("Missing values before Preprocessing", fontsize=22)
plt.tight_layout()
plt.savefig('../Output/Heatmap_beginning.png')
plt.close()

data = data[~data.isin([2008, 2009]).any(axis=1)]  # Remove years 2008 and 2009

drop_log('Drop Years 2008/2009')  # update drop log

missing_df = missing_values_percentage()  # calculate & plot
plt.figure(figsize=(15, 10))  # percentage of missing values
fig = sns.barplot(y='Values',
                  x='Index',
                  data=missing_df,
                  order=missing_df.sort_values('Values', ascending=False)['Index'])
label_axes = fig.set(xlabel='Features',
                     ylabel='% Missing values',
                     title='% of missing values per feature orderd by size')
plt.xticks(fontsize=4, rotation=90)
plt.axhline(y=0.50, color='blue')
plt.xlabel("Features", fontsize=16)
plt.ylabel("Percentage of missing values in the columns", fontsize=16)
plt.title("Ordered features by percentage of missing values in the columns", fontsize=22)
plt.tight_layout()
plt.savefig('../Output/Missing_Values.png')
plt.close()

col = data.columns  # Screening columns
for i in range(len(col)):  # with Threshold 50%
    data = screening(col[i], 0.5)

drop_log('Threshold Columns 50%')  # update drop log

print('Mean missing values sorted by hospital type', mean_missing_values('Typ'))    # Mean missing values per row
print('###################################################')                        # ordered by hospital type

plt.figure(figsize=(15, 10))  # Plot the mean missing values per
missing_mean_typ = (mean_missing_values('Typ')).reset_index()  # row ordered by hospital type
missing_mean_typ['Color'] = ['Specialized Hospitals',
                             'Specialized Hospitals',
                             'Specialized Hospitals',
                             'Specialized Hospitals',
                             'Specialized Hospitals',
                             'Specialized Hospitals',
                             'Specialized Hospitals',
                             'General Hospitals',
                             'General Hospitals',
                             'General Hospitals',
                             'General Hospitals',
                             'General Hospitals',
                             'General Hospitals']
fig = sns.barplot(y='missing_values', x='Typ', hue='Color', data=missing_mean_typ)
plt.xlabel('Hospital Type', fontsize=16)
plt.ylabel('Mean missing values per row', fontsize=16)
plt.title('Mean missing values per row ordered by hospital type', fontsize=20)
plt.xticks(fontsize=12, rotation=45)
plt.savefig('../Output/Missing_Values_Typ.png')
plt.close()

data = data[~data['Typ'].isin(['K211', 'K212', 'K221', 'K231',  # Drop Specialized Hospitals rows
                               'K232', 'K233', 'K234', 'K235'])]
# print('Sum of missing values',data.isna().sum())

drop_log('Drop Specialized Hospitals')  # update drop log

plt.figure(figsize=(15, 10))  # plot remaining missing values
sns.heatmap(data.isna(), cmap='viridis', cbar=False)  # in a heatmap
plt.xlabel("Features", fontsize=16)
plt.ylabel("Row Index", fontsize=16)
plt.title("Remaining missing values after preprocessing Steps 1-3", fontsize=22)
plt.tight_layout()
plt.savefig('../Output/Heatmap_missing_values.png')
plt.close()

data = data.drop(['Adr', 'RForm', 'StdBelA', 'StdBelP', 'ErlZvOKPStatVA',  # Drop additonal Columns & rows
                  'ErlKVGStatVA', 'WB', 'Ort', 'Inst', 'Unnamed: 0'], axis=1)  # Drop specific Columns

drop_log('Drop additional columns')  # update drop log

# print(data['missing_values'].sort_values(ascending=False).head(6))
specific_rows = data['missing_values'].sort_values(ascending=False).head(5).index  # Drop specific rows
data = data.drop(specific_rows, axis=0)

data = data.dropna(subset=data.columns.difference(['KostLangA']))  # Drop missing values from all
# columns except 'KostLangA'
drop_log('Drop specific rows')  # update drop log

plt.figure(figsize=(15, 10))  # Check Heatmap again for
sns.heatmap(data.isna(), cmap='viridis', cbar=False)  # remaining missing values
plt.xlabel("Features", fontsize=16)
plt.ylabel("Row Index", fontsize=16)
plt.title("Remaining missing values after preprocessing Step 4", fontsize=22)
plt.tight_layout()
plt.savefig('../Output/Heatmap_Remaining_missing_values.png')
plt.close()

########################################################################################################################
''''' 
IMPUATIONS 

Imputaing missing values in 'KostLangA'

Imputate the missing values of the column 'KostLangA' using k nearest neighbours (KNN). To find the optimal number of
neighbours (n_neighbours) the distributions of the column 'KostLangA' without missing values was compared to the 
distribution of column after imputating the missing values. The k (= 4) with least deviation in distribution was chosen to 
imputate the missing values.
'''''

column_data = data['KostLangA'].dropna()  # column to imputate data

column_mean = column_data.mean()  # distibution characteristics
column_median = column_data.median()  # before imputations
column_std = column_data.std()

metrics_df = pd.DataFrame(columns=['k', 'Metrics Difference', 'Mean Difference',
                                   'Median Difference', 'Standard Deviation Difference'])

for k in range(1, 20):  # hyperparameter tuning
    num_col = data.select_dtypes(include=[np.float64]).columns  # find optial number of
    num_data = data[num_col].copy()  # neighbours (k)
    imp = KNNImputer(n_neighbors=k)
    imp.fit(num_data)
    imputed_data = imp.transform(num_data)

    imputed_column_data = pd.Series(imputed_data[:, num_col.get_loc('KostLangA')])

    imputed_mean = imputed_column_data.mean()  # calculate distribution
    imputed_median = imputed_column_data.median()  # characteristics
    imputed_std = imputed_column_data.std()

    df = pd.DataFrame({
        'k': [k],
        'Metrics Difference': [
            abs(column_mean - imputed_mean) + abs(column_median - imputed_median) + abs(column_std - imputed_std)],
        'Mean Difference': [abs(column_mean - imputed_mean)],
        'Median Difference': [abs(column_median - imputed_median)],
        'Standard Deviation Difference': [abs(column_std - imputed_std)]
    })

    metrics_df = pd.concat([metrics_df, df], ignore_index=True)

plt.figure(figsize=(10, 6))  # plot k and calculated metrics
plt.plot(metrics_df['k'], metrics_df['Metrics Difference'], marker='o', label='Sum of all metrics')
plt.plot(metrics_df['k'], metrics_df['Mean Difference'], marker='o', label='Mean Difference')
plt.plot(metrics_df['k'], metrics_df['Median Difference'], marker='o', label='Median Difference')
plt.plot(metrics_df['k'], metrics_df['Standard Deviation Difference'], marker='o',
         label='Standard Deviation Difference')
plt.xticks(range(2, 17, 2))
plt.xlabel('Number of neighbours [k]')
plt.ylabel('Difference in metrics [CHF]')
plt.title('Difference in distribution metrics of the imputated data compared to original data')
plt.legend()
plt.savefig('../Output/Imputations_metrics.png')
plt.close()

num_col = data.select_dtypes(include=[np.float64]).columns  # imputate missing data in
num_data = data[num_col]  # 'KostLangA' with k = 4
imp = KNNImputer(n_neighbors=4)
imp.fit(num_data)
data[num_col] = imp.transform(num_data)
########################################################################################################################
plt.figure(figsize=(15, 10))  # Check Heatmap for zeros
sns.heatmap(data == 0, cmap='viridis', cbar=False)
plt.xlabel("Columns: Features")
plt.ylabel("Rows: Hospital Index")
plt.title("Zeros in the data set")
plt.tight_layout()
plt.savefig('../Output/Heatmap_Zeros.png')
plt.close()


########################################################################################################################
''''' 
LABEL 

Before checking the label for its distribution outliers where removed by excluding the vales smaller than the 2.5 
and larger than 97.5 percentile. A Shapiro test was conducted to test normality
'''''

# Plot the distribution of the
fig = sns.histplot(data, x='FiErg')  # Revenue(Ertrag) with
fig.set(xlabel='Revenue [CHF]',  # outliers
        ylabel='Number of Hospitals',
        title='Distribution of Revenue in Swiss Hospitals with outliers after processing')
plt.tight_layout()
plt.savefig('../Output/Distribution.png')
plt.close()

IQ01 = data[data['FiErg'] < data['FiErg'].quantile(q=0.025)].index  # Remove potential outliers
IQ99 = data[data['FiErg'] > data['FiErg'].quantile(q=0.975)].index  # Keep all rows in Quaniles
data = data.drop(IQ01)  # 1%-99%
data = data.drop(IQ99)

drop_log('Drop Outliers')  # update drop log

# Plot the distribution of the
fig = sns.histplot(data, x='FiErg')  # Revenue (Ertrag) without
fig.set(xlabel='Revenue [CHF]',  # potential outliers
        ylabel='Number of Hospitals',
        title='Distribution of Revenue in Swiss Hospitals without potential outliers')
plt.savefig('../Output/Distribution_no_outliers.png')
plt.close()

statistic, p_value = sts.shapiro(data['FiErg'])  # Shapiro Wilk Test to test
print('Shapiro Test after cleaning: p-value =', p_value)  # for normality

print('###################################################')
print('This is a log of the cleaning procedure:\n\n', drop_log_df.transpose())  # print drop log
print('\n###################################################')
########################################################################################################################
'''''
LABEL ENCODING

For the classification models the label (continous) was converted to categorical. The encoding is 0/Loss, 1/break-even 
and 2/Gain.
'''''
FiErg_encoded = []
for i in range(len(data['FiErg'])):
    if data.iloc[i]['FiErg'] >= 0:
        FiErg_encoded.append(1)
    else:
        FiErg_encoded.append(0)
data['FiErg_encoded'] = FiErg_encoded

print('Label after transformed into categoricall data type (balancing)')
print(data['FiErg_encoded'].value_counts())
########################################################################################################################
''''' 
FEATURE SELECTION 

For the classification models a univariate feature selection was conducted. The top seven features where selected.
'''''
########################################################################################################################
'''''
~ Classificaton: 
        
    Label:                      y_class, 
    Split features:             X_train_class, X_test_class, y_train_class, y_test_class
    
Use as reference for Classification models:
    
    post-feature-selection
    
    New Data Set:               data_class
    Label:                      y_class
    Features                    X_class
    Split features:             X_train_class_uvfs, X_test_class_uvfs, y_train_class, y_test_class
'''''
########################################################################################################################

''''' UNIVARIATE FEATURE SELECTION '''''
# Categorical and Numerical values together
y_class = data['FiErg_encoded']
X_class = (pd.get_dummies(data, columns=['KT', 'Typ', 'Akt'], drop_first=True)).drop(
    ['missing_values', 'FiErg_encoded', 'FiErg'], axis=1)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2,
                                                                            random_state=420)

col_class = X_class.columns  # Standardize the model for the
sc = StandardScaler()  # numerical data

X_train_class[col_class] = sc.fit_transform(X_train_class[col_class])
X_test_class[col_class] = sc.transform(X_test_class[col_class])

import random
random.seed(420)  # Set the random seed for Python's random module
np.random.seed(420)
select_uvfs = SelectKBest(mutual_info_classif, k=7)

X_uvfs = select_uvfs.fit_transform(X_train_class, y_train_class)  # Fit the Classification Model
X_uvfs_test = select_uvfs.transform(X_test_class)

score = select_uvfs.scores_  # evaluate the performance
score /= score.max()

sorted_indices = np.argsort(score)[::-1]  # Sort the features in descending
# order based on scores
sorted_scores = score[sorted_indices]  # Get the sorted scores and
sorted_features = X_class.columns[sorted_indices]  # feature names

X_index = np.arange(X_class.shape[-1])  # Visualize the results in a plot
plt.figure(figsize=(12, 6))
plt.clf()
plt.bar(X_index - 0.05, sorted_scores, width=0.2)
plt.title("Univariate feature selection")
plt.xlabel("Features")
plt.ylabel("Mutial Information Score")
plt.xticks(X_index[:7], sorted_features[:7], rotation=90, fontsize=10)
plt.axhline(y=0.585, color='blue')
plt.tight_layout()
plt.savefig("../Output/uvfs.png")
plt.close()

selected_features = X_class.columns[select_uvfs.get_support(indices=True)]
data_class = X_class[selected_features]

X_train_class_uvfs = X_train_class[selected_features]  # filter data for selected features
X_test_class_uvfs = X_test_class[selected_features]  # for use in models
########################################################################################################################
''''' LOGISTIC REGRESSION'''''
param_grid = {  # Grid search for C
    'C': [0.8, 0.9, 1, 1.035, 1.045, 1.05, 1.055, 1.1, 1.15, 1.2, 1.5, 2, 4, 6],
}
import random
random.seed(420)
np.random.seed(420)
logistic_regression = LogisticRegression(random_state=420)
grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, cv=5)
grid_search.fit(X_train_class_uvfs, y_train_class)

best_model_log = grid_search.best_estimator_
best_params_log = grid_search.best_params_

print('###################################################')
print("Best Model:", best_model_log)
print("Best Hyperparameters:", best_params_log)

log_reg = LogisticRegression(C=2, random_state=420, class_weight='balanced')
log_reg.fit(X_train_class_uvfs, y_train_class)
y_pred_log_reg = log_reg.predict(X_test_class_uvfs)
eval_metircs_classification(y_test_class, y_pred_log_reg, 'Logistic Regression')
plt_roc_curve(log_reg, 'Logistic_Regression', X_test_class_uvfs, y_test_class)
########################################################################################################################
''''' RANDOM FOREST '''''
param_grid = {  # Grid search
    'n_estimators': list(range(10, 121, 5)),
    'max_features': ['sqrt', 'log2', None]

}
'''''                                                                               # hyperparameters not implemented
    'bootstrap':[False,True],
    'criterion': ['gini','entropy','log_loss'],
    'min_samples_leaf':[1,2,3],
    'min_weight_fraction_leaf':[0.0,0.05,0.1],
    'max_depth': [None,5,10,20,30],
    'min_samples_split': [1,2,4,5,6,10],
    'max_leaf_nodes':[None,1,5,10],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'oob_score': [True, False],
'''''
Random_Forest = RandomForestClassifier(random_state=420)
grid_search = GridSearchCV(estimator=Random_Forest, param_grid=param_grid, cv=5)
grid_search.fit(X_train_class_uvfs, y_train_class)

best_model_log = grid_search.best_estimator_
best_params_log = grid_search.best_params_


print("Best Model:", best_model_log)
print("Best Hyperparameters:", best_params_log)
RF = RandomForestClassifier(bootstrap=True, max_depth=None, min_samples_split=5, n_estimators=80, random_state=420,
                            class_weight='balanced')
RF.fit(X_train_class_uvfs, y_train_class)
y_pred_RF = RF.predict(X_test_class_uvfs)
eval_metircs_classification(y_test_class, y_pred_RF, 'Random Forest')
plt_roc_curve(RF, 'Random_forest', X_test_class_uvfs, y_test_class)

########################################################################################################################
''''' CLUSTERING '''''

data_cluster = data  # drop non numeric values in data
data_cluster = data_cluster.drop(['JAHR', 'KT', 'Typ', 'Akt', 'missing_values', 'FiErg_encoded'], axis=1)
# print(data_cluster.dtypes)

feature_names = data_cluster.columns

train_data_cluster, test_data_cluster = train_test_split(data_cluster, test_size=0.2, random_state=420)

X_train = train_data_cluster.values  # exclude target variable if present                # select the columns for clustering
X_test = test_data_cluster.values  # exclude target variable if present

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)  # pca
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

pc1_name = feature_names[pca.components_[0].argmax()]  # get names of PC1 and PC2
pc2_name = feature_names[pca.components_[1].argmax()]
print('PCA: 2 Components that contain the most variability')
print(f"Name of PC1: {pc1_name}")
print(f"Name of PC2: {pc2_name}")
# fit and train K-means clustering model
# on PCA-transformed train data
max_clusters = 10  # max number of clusters to try                               # find optimal K-means
wcss_values = []  # list to store the WCSS values

for n_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=420, n_init='auto')
    kmeans.fit(X_train_pca)
    wcss = kmeans.inertia_
    wcss_values.append(wcss)

plt.figure(figsize=(8, 6))  # Plot the WCSS values
plt.plot(range(1, max_clusters + 1), wcss_values, marker='o')
plt.xlabel('Number of Clusters')
plt.xticks(range(1, max_clusters + 1))
plt.ylabel('WCSS')
plt.title('WCSS vs. Number of Clusters')
plt.grid(True)
plt.tight_layout()
plt.savefig('../output/WCSS.png')
plt.close()
# fit and train K-means clustering model
# on PCA-transformed train data
n_clusters = 3  # --> 3 looking at the elbow plot wcss
kmeans = KMeans(n_clusters=n_clusters, random_state=420, n_init='auto')
kmeans.fit(X_train_pca)

test_clusters = kmeans.predict(X_test_pca)  # apply K-means clustering model
test_data_cluster['Cluster'] = test_clusters  # to PCA-transformed test data
# print(test_clusters)

unique_clusters = np.unique(test_clusters)  # get unique cluster labels

plt.figure(figsize=(8, 6))  # plot results
for cluster_label in unique_clusters:
    cluster_indices = np.where(test_clusters == cluster_label)
    plt.scatter(X_test_pca[cluster_indices, 0], X_test_pca[cluster_indices, 1], label=f'Cluster {cluster_label}')

plt.xlabel('Principal Component 1 (KostStatA)')  # plot results
plt.ylabel('Principal Component 2 (pPatLKP)')
plt.title('K-means Clustering Results (after PCA)')
plt.legend()
plt.tight_layout()
plt.savefig('../Output/Clustering.png')
plt.close()

fi_erg_index = np.where(feature_names == 'FiErg')[0][0]  # get index of 'FiErg' feature

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # plot results

for cluster_label in unique_clusters:  # clustering plot
    cluster_indices = np.where(test_clusters == cluster_label)
    ax1.scatter(X_test_pca[cluster_indices, 0], X_test_pca[cluster_indices, 1], label=f'Cluster {cluster_label}')

ax1.set_xlabel('KostStatA')
ax1.set_ylabel('pPatLKP')
ax1.set_title('K-means Clustering Results (after PCA)')
ax1.legend()

# FiErg plot
ax2.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=X_test[:, fi_erg_index], cmap='viridis')
ax2.set_xlabel('KostStatA')
ax2.set_ylabel('pPatLKP')
ax2.set_title('FiErg Distribution')
cbar = plt.colorbar(ax2.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=X_test[:, fi_erg_index], cmap='viridis'))
cbar.set_label('FiErg')
plt.tight_layout()
plt.savefig('../Output/Clustering_with_FiErg.png')
########################################################################################################################
