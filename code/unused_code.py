########################################################################################################################
# Univariate als funktion implementieren (noch unvollständig, mit Categorischen Daten funktionierts nicht!)
def univariate_feature_selection(X, X_train, X_test, y_train, y_test,model, plot_name):

    select = SelectKBest(model, k = 10)
    #fit the Classification Model
    X_UVFS = select.fit_transform(X_train,y_train)
    X_UVFS_test = select.transform(X_test)
    score = -np.log10(select.pvalues_)
    score /= score.max()

    #Visualize the results in a plot
    X_index = np.arange(X.shape[-1])
    plt.figure()
    plt.clf()
    plt.bar(X_index - 0.05, score, width = 0.2)
    plt.title("Feature univariante score")
    plt.xlabel("Feature")
    plt.ylabel("-Log(Univariante score)")
    plt.xticks(X_index, X.columns, rotation = 90)
    plt.tight_layout()
    plt.savefig("../Output/uvfs_"+plot_name+".png")

    return print(select.get_feature_names_out())


# Classification
y_class = data['FiErg_encoded']
X_class_n = data.drop(['KT', 'Typ', 'Akt', 'missing_values', 'FiErg_encoded','FiErg'], axis = 1)
X_class_c = data[['KT', 'Typ', 'Akt']]

#Hot encode categorical data
X_class_c = pd.get_dummies(X_class_c)

#Split data into Training and Test Data
X_train_class_c, X_test_class_c, y_train_class_c, y_test_class_c= train_test_split(X_class_c, y_class, test_size = 0.2, random_state=420)
X_train_class_n, X_test_class_n, y_train_class_n, y_test_class_n= train_test_split(X_class_n, y_class, test_size = 0.2, random_state=420)

columns_class_n = X_class_n.columns

#Standardize the model for the numerical data
sc = StandardScaler()
X_train_class_n[columns_class_n] = sc.fit_transform(X_train_class_n[columns_class_n])
X_test_class_n[columns_class_n] = sc.transform(X_test_class_n[columns_class_n])

univariate_feature_selection(X_class_c,X_train_class_c, X_test_class_c, y_train_class_c, y_test_class_c,chi2,'Categorical')
univariate_feature_selection(X_class_n,X_train_class_n, X_test_class_n, y_train_class_n, y_test_class_n,f_classif, 'Numerical')


########################################################################################################################
'''''encoding for mulitclass & Splitting data'''''
FiErg_encoded_multiclass = []
for i in range(len(data['FiErg'])):
    if data.iloc[i]['FiErg'] > 0:
        FiErg_encoded_multiclass.append(2)
    elif data.iloc[i]['FiErg'] == 0:
        FiErg_encoded_multiclass.append(1)
    else:
        FiErg_encoded_multiclass.append(0)
data_multiclass = pd.DataFrame({'FiErg_encoded_multiclass':FiErg_encoded_multiclass})
value_counts_multiclass = data_multiclass.value_counts()

y_multiclass = data_multiclass['FiErg_encoded_multiclass']
X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = train_test_split(X_class, y_multiclass, test_size=0.2, random_state=420)
X_train_multiclass[col_class] = sc.fit_transform(X_train_multiclass[col_class])
X_test_multiclass[col_class] = sc.transform(X_test_multiclass[col_class])

#after feature selection
X_train_multiclass_uvfs = X_train_multiclass.iloc[:,select_uvfs.get_support(indices=True)]
X_test_multiclass_uvfs = X_test_multiclass.iloc[:,select_uvfs.get_support(indices=True)]
########################################################################################################################
'''''ROC Curve und evaluationmetrics for multiclass'''''
from scipy.special import softmax
def ROC_Curve(clf,Name,X_test_roc,y_test_roc,multiclass,OvO):
    """
    Plots the ROC curve for the classification models

    :param      clf: choose necessary model
    :type       clf:

    :param      Name: Name of model used
    :type       Name: str
    """
    if OvO:
        y_pred_proba = softmax(clf.decision_function(X_test_roc), axis=1)
    else:
        y_pred_proba = clf.predict_proba(X_test_roc)

    if multiclass:
        y_test_binarized = label_binarize(y_test_roc, classes=np.unique(y_test_roc))        # Binarize the true labels
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        colors = ['blue', 'orange', 'green']  # Adjust the colors as needed
        for i in range(3):
            plt.plot(fpr[i], tpr[i], color=colors[i], label='Class %d (AUC = %0.2f)' % (i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

    else:
        y_pred_proba = y_pred_proba[:,1]
        fpr, tpr, thresholds = roc_curve(y_test_roc, y_pred_proba)

        AUC = roc_auc_score(y_test_roc, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {AUC:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve: '+Name)
    plt.legend(loc='lower right')
    plt.savefig('../Output/ROC_Curve_'+Name)
    plt.close()
    return

def eval_metircs_classification(true_values, predicted_values, Model):
    """
    prints all the evaluation metrics for the classification models

    :param      true_values: values for the label in the split data set
    :type       true_values: pandas column

    :param      predicted_values: predicted values for the label
    :type       predicted_values: pandas column

    :param      Model: Model name
    :type       Model: str
    """
    accuracy = accuracy_score(true_values, predicted_values)
    precision = precision_score(true_values, predicted_values,average = 'weighted')
    recall = recall_score(true_values, predicted_values,average = 'weighted')
    f1 = f1_score(true_values, predicted_values,average = 'weighted')
    cm = confusion_matrix(true_values, predicted_values)

    #average = None für multiclass verwenden
    print('############################################')
    print(Model)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:",cm)
    print('\n############################################')
    return

best_estimators = 110
RF_multi = RandomForestClassifier(n_estimators=best_estimators, random_state=420,class_weight='balanced')
RF_multi.fit(X_train_multiclass_uvfs, y_train_multiclass)
y_pred_RF_multi = RF_multi.predict(X_test_multiclass_uvfs)
eval_metircs_classification(y_test_multiclass,y_pred_RF_multi, 'Random Forest')
ROC_Curve(RF_multi,'Random_forest Multiclass',X_test_multiclass_uvfs,y_test_multiclass,True,False)

RF_ovo_multi = RandomForestClassifier(n_estimators=best_estimators,random_state=420,class_weight='balanced')
RF_ovo = OneVsOneClassifier(RF_ovo_multi)
RF_ovo.fit(X_train_multiclass_uvfs, y_train_multiclass)
ROC_Curve(RF_ovo,'Random_forest_Multiclass_OvO',X_test_multiclass_uvfs,y_test_multiclass,True,True)

log_reg_multi = LogisticRegression(random_state=420, max_iter=1000,class_weight='balanced')
log_reg_multi.fit(X_train_multiclass_uvfs, y_train_multiclass)
y_pred_log_reg_multi = log_reg_multi.predict(X_test_multiclass_uvfs)
eval_metircs_classification(y_test_multiclass,y_pred_log_reg_multi,'Logistic Regression Multiclass')
ROC_Curve(log_reg_multi, 'Logistic_Regression_Multiclass',X_test_multiclass_uvfs,y_test_multiclass,True,False)

log_reg_ovo_multi = LogisticRegression(random_state=420, max_iter=1000,class_weight='balanced')
log_reg_ovo = OneVsOneClassifier(log_reg_ovo_multi)
log_reg_ovo.fit(X_train_multiclass_uvfs, y_train_multiclass)
ROC_Curve(log_reg_ovo,'Logistic_Regression_Multiclass_OvO',X_test_multiclass_uvfs,y_test_multiclass,True,True)

'''''
~ Regression:     
    Label:                      y_reg, 
    Split features:             X_train_reg, X_test_reg, y_train_reg, y_test_reg
    
Use as reference for Regression models:
    post-feature-selection
    
    New Data Set:               data_reg
    Label:                      y_reg
    Split features              X_train_L1, X_test_L1, y_train_reg, y_test_reg   
'''''
'''''
Use as reference for Classification models:
    
    post-feature-selection 
    
    New Data Set:               data_class
    Label:                      y_class
    Split features:gi           X_train_class_uvfs, X_test_class_uvfs, y_train_class, y_test_class
    
'''''
########################################################################################################################
''''' LASSO / L1 regularization '''''
# Choosing data and splitting into Training and Test Set
y_reg = data['FiErg']
X_reg = (pd.get_dummies(data, columns=['KT', 'Typ', 'Akt'], drop_first=True)).drop(['missing_values', 'FiErg_encoded','FiErg'], axis = 1)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=420)

# Perform feature scaling using StandardScaler
col_reg = X_reg.columns
sc = StandardScaler()
X_train_reg[col_reg] = sc.fit_transform(X_train_reg[col_reg])
X_test_reg[col_reg] = sc.transform(X_test_reg[col_reg])

# Create and fit the Lasso regression model
lasso = linear_model.Lasso(alpha=250000, max_iter= 5000)  # Set the regularization strength (alpha) and number of Iteration
lasso.fit(X_train_reg, y_train_reg)

# Retrieve the learned coefficients (feature importance)
coefficients = lasso.coef_

# Identify and print all selected features
selected_features = X_reg.columns[coefficients != 0]
print("Selected Features after L1:", selected_features)

# Evaluate the performance
get_score(lasso, X_train_reg, y_train_reg, X_test_reg, y_test_reg, 'Lasso_Evaluation')

# Get the absolute values of the coefficients and their indices
coef_abs = np.abs(lasso.coef_)
indices = np.argsort(coef_abs)[::-1]  # Sort indices in descending order

# Select top N features and their corresponding coefficients
N = min(10, len(indices))
top_features = [selected_features[i] for i,b in enumerate(indices[:N])]
top_coef_abs = coef_abs[indices][:N]

# Create a bar plot to visualize the top N features and coefficients
plt.figure(figsize=(12, 6))
plt.bar(top_features, top_coef_abs)
plt.xlabel("Features")
plt.ylabel("Absolute Coefficient")
plt.title(f"Top {N} Selected Features with Absolute Coefficients")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../Output/Lasso.png')
plt.close()


#saving selected features and adjusting the data for linear Regression
data_reg = X_reg[top_features]
X_train_L1 = X_train_reg[top_features]
X_test_L1 = X_test_reg[top_features]

########################################################################################################################
'''''LINEAR REGRESSION'''''
X_train_linear = X_train_reg[selected_features]
X_test_linear = X_test_reg[selected_features]

# fit Linear Regression model
LR = LinearRegression()
LR.fit(X_train_linear, y_train_reg)

# Evaluate Performance with RMSE and MSE
get_score(LR, X_train_linear, y_train_reg, X_test_linear, y_test_reg,'Linear_Reg_UVFS')

########################################################################################################################
'''''
n_estimators manual search

best_accuracy = 0
best_estimators = 0

for n_estimators in range(10, 121, 10):
    RF = RandomForestClassifier(n_estimators=n_estimators, random_state=420)
    #base_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=420)
    #ovo_classifier = OneVsOneClassifier(base_classifier)

    cv_scores = cross_val_score(RF, X_train_class_uvfs, y_train_class, cv=5)
    mean_accuracy = cv_scores.mean()

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_estimators = n_estimators
'''''
########################################################################################################################
#loop cluster over every label
for label in data_cluster.columns:
    #print(f'Label: {label}')

    #get index of the label in the feature_names array
    label_index = np.where(feature_names == label)[0][0]

    #plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    #clustering plot
    for cluster_label in unique_clusters:
        cluster_indices = np.where(test_clusters == cluster_label)
        ax1.scatter(X_test_pca[cluster_indices, 0], X_test_pca[cluster_indices, 1], label=f'Cluster {cluster_label}')

    ax1.set_xlabel('Principal Component 1 (KostStatA)')
    ax1.set_ylabel('Principal Component 2 (pPatLKP)')
    ax1.set_title('K-means Clustering Results (after PCA)')
    ax1.legend()



    #plot with label
    ax2.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=X_test[:, label_index], cmap='viridis')
    ax2.set_xlabel('Principal Component 1 (KostStatA)')
    ax2.set_ylabel('Principal Component 2 (pPatLKP)')
    ax2.set_title(f'{label} Distribution')
    cbar = plt.colorbar(ax2.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=X_test[:, label_index], cmap='viridis'))
    cbar.set_label(label)

    plt.tight_layout()
    plt.savefig(f'../Output/Clustering_with_{label}.png')
    plt.close()
########################################################################################################################
'''''
#Standardize the model for the numerical data
col_reg = X_reg.columns
sc = StandardScaler()
X_train_reg[col_reg] = sc.fit_transform(X_train_reg[col_reg])
X_test_reg[col_reg] = sc.transform(X_test_reg[col_reg])

# Use L1 regualarization in a logistic regression classifier to select features.
sel_ = SelectFromModel(
    LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=10)
)
sel_.fit(X_train_reg, y_train_reg)

# features with non-zero coefficients are "support":
selected_features = X_reg.columns[sel_.get_support()]
X_train_L1 = X_train_reg[selected_features]
X_test_L1 = X_test_reg[selected_features]
'''''

def ROC_Curve_One_vs_One(clf,Name,X_test,y_test):
    """
    Plots the ROC curve for the classification models

    :param      clf: choose necessary model
    :type       clf:

    :param      Name: Name of model used
    :type       Name: str
    """

    y_pred_prob = softmax(clf.decision_function(X_test), axis=1)
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))        # Binarize the true labels

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['blue', 'orange', 'green']  # Adjust the colors as needed
    for i in range(3):
        plt.plot(fpr[i], tpr[i], color=colors[i], label='Class %d (AUC = %0.2f)' % (i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve: '+Name)
    plt.legend(loc='lower right')
    plt.savefig('../Output/ROC_Curve_'+Name)
    plt.close()
    return
