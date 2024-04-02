#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install imbalanced-learn==0.6.0')
get_ipython().system('pip install scikit-learn==0.22.1')
get_ipython().system('pip install imblearn')
get_ipython().system('pip install pandas')
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# In[2]:


df = pd.read_csv('C:/Users/lyasm/OneDrive/Desktop/mimic.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


#Dropping group and ID columns as these do not provide any useful info and will no be used for analysis
df_clean = df.drop(['group','ID'], axis=1)
df_clean


# In[7]:


#Converting gender entry from as 1 and 2 to 0 and 1 for consisitency with other binary columns
df_clean['gendera'] = df_clean['gendera'] - 1
df_clean.head()


# In[8]:


#Determining the number of na values for each column
df_clean.isnull().sum()


# In[9]:


#Since the outcome column is the target variable, the single missing value will not be replaced/calculated.
#It will be dropped to preserve the authencity of the outcome column.
df_clean = df_clean.dropna(subset=['outcome'])
df_clean.isnull().sum()


# In[10]:


#Visualizing the distribution of all columns to determine which columns are normally distributed and which columns are categorical
df_clean.hist(figsize=(15,15))


# In[11]:


#Correlation Plot to show features correlation with outcome
fig = plt.figure(figsize = (10, 10))
target_corr = pd.DataFrame(df.corr()['outcome'].sort_values(ascending = True))
plt.barh(target_corr.index, target_corr['outcome'],color="#FF9912")
plt.title('Correlation with outcome')
plt.show()


# ## Data processing

# In[12]:


#Creating demographic , vital signs and comorbidity data subsets for seperate analysis to determine if there is a significant effect on ICU patient mortality.

df_demo = df_clean[['outcome', 'age', 'gendera','BMI']]
df_vital = df_clean[['outcome','heart rate', 'Systolic blood pressure','Diastolic blood pressure', 'Respiratory rate','temperature','SP O2','Urine output']]
df_comorbidities = df_clean[['outcome','hypertensive','atrialfibrillation','CHD with no MI','diabetes','deficiencyanemias','depression','Hyperlipemia','Renal failure','COPD']]


# In[13]:


#Imputing the missing values for using k nearest neighbours algorithm for demographic , vital and comorbidity data subsets 

df_demo_imputer = KNNImputer(missing_values=np.nan, n_neighbors=2)
df_demo_imputer.fit(df_demo)
df_demo = df_demo_imputer.transform(df_demo)

df_vital_imputer = KNNImputer(missing_values=np.nan, n_neighbors=2)
df_vital_imputer.fit(df_vital)
df_vital = df_vital_imputer.transform(df_vital)

df_comorbidities_imputer = KNNImputer(missing_values=np.nan, n_neighbors=2)
df_comorbidities_imputer.fit(df_comorbidities)
df_comorbidities = df_comorbidities_imputer.transform(df_comorbidities)


# In[14]:


#Determining if demographic features have a significant effect on ICU patient mortality
#Chi-Square test on Demographic details - age

contingency_table_age = pd.crosstab(df_demo[1], df_demo[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table_age)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[15]:


#Determining if demographic features have a significant effect on ICU patient mortality
#Chi-Square test on Demographic details - gendera

contingency_table_age = pd.crosstab(df_demo[2], df_demo[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table_age)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[16]:


#Determining if demographic features have a significant effect on ICU patient mortality
#Chi-Square test on Demographic details - BMI

contingency_table_age = pd.crosstab(df_demo[3], df_demo[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table_age)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[17]:


#Determining if and which vital signs have significance when predicting ICU patient mortality
# Correlation Analysis on Vital Signs - heart rate
vital_signs_hr = df_vital[1]
mortality = df_vital[0]

# Perform correlation analysis (Pearson)
correlation_coefficient, p_value = stats.pearsonr(vital_signs_hr, mortality)

# Display results
print("Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)


# In[18]:


# Correlation Analysis on Vital Signs - Systolic blood pressure
vital_signs_hr = df_vital[2]
mortality = df_vital[0]

# Perform correlation analysis (Pearson)
correlation_coefficient, p_value = stats.pearsonr(vital_signs_hr, mortality)

# Display results
print("Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)


# In[19]:


# Correlation Analysis on Vital Signs - Diastolic blood pressure
vital_signs_hr = df_vital[3]
mortality = df_vital[0]

# Perform correlation analysis (Pearson)
correlation_coefficient, p_value = stats.pearsonr(vital_signs_hr, mortality)

# Display results
print("Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)


# In[20]:


# Correlation Analysis on Vital Signs - Respiratory rate 
vital_signs_hr = df_vital[4]
mortality = df_vital[0]

# Perform correlation analysis (Pearson)
correlation_coefficient, p_value = stats.pearsonr(vital_signs_hr, mortality)

# Display results
print("Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)


# In[21]:


# Correlation Analysis on Vital Signs - temperature 
vital_signs_hr = df_vital[5]
mortality = df_vital[0]

# Perform correlation analysis (Pearson)
correlation_coefficient, p_value = stats.pearsonr(vital_signs_hr, mortality)

# Display results
print("Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)


# In[22]:


# Correlation Analysis on Vital Signs - SP O2 
vital_signs_hr = df_vital[6]
mortality = df_vital[0]

# Perform correlation analysis (Pearson)
correlation_coefficient, p_value = stats.pearsonr(vital_signs_hr, mortality)

# Display results
print("Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)


# In[23]:


# Correlation Analysis on Vital Signs - Urine output 
vital_signs_hr = df_vital[7]
mortality = df_vital[0]

# Perform correlation analysis (Pearson)
correlation_coefficient, p_value = stats.pearsonr(vital_signs_hr, mortality)

# Display results
print("Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)


# In[24]:


#Determining if pre-existing condition/Comorbidities have a signicant effect on ICU patient mortality
#Chi-Square test on Comorbidities - hypertensive

contingency_table = pd.crosstab(df_comorbidities[1], df_comorbidities[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[25]:


#Chi-Square test on Comorbidities - atrialfibrillation

contingency_table = pd.crosstab(df_comorbidities[2], df_comorbidities[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[26]:


#Chi-Square test on Comorbidities - CHD with no MI

contingency_table = pd.crosstab(df_comorbidities[3], df_comorbidities[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[27]:


#Chi-Square test on Comorbidities - diabetes 

contingency_table = pd.crosstab(df_comorbidities[4], df_comorbidities[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[28]:


#Chi-Square test on Comorbidities - deficiencyanemias

contingency_table = pd.crosstab(df_comorbidities[5], df_comorbidities[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[29]:


#Chi-Square test on Comorbidities - depression

contingency_table = pd.crosstab(df_comorbidities[6], df_comorbidities[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[30]:


#Chi-Square test on Comorbidities - Hyperlipemia

contingency_table = pd.crosstab(df_comorbidities[7], df_comorbidities[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[31]:


#Chi-Square test on Comorbidities - Renal failure

contingency_table = pd.crosstab(df_comorbidities[8], df_comorbidities[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# In[32]:


#Chi-Square test on Comorbidities - COPD 

contingency_table = pd.crosstab(df_comorbidities[9], df_comorbidities[0])

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Display results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)


# ## Building the Model

# In[33]:


get_ipython().run_cell_magic('time', '', '# Splitting the dataset into 80% train set and 20% test set before imputing missing enteries to avoid data leakage\n\nX = df_clean.iloc[:, 1:]\ny = df_clean.iloc[:, 0]\n\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)\n')


# In[34]:


get_ipython().run_cell_magic('time', '', '#Imputing the missing values using k nearest neighbours algorithm.\n#This will take into account the relationship and pattern of the data, to resemble more real world measurements.\n\ntrain_imputer_features = KNNImputer(missing_values=np.nan, n_neighbors=2)\ntrain_imputer_features.fit(X_train)\nX_train = train_imputer_features.transform(X_train)\n\n#Important: test and train are imputed seperately to avoid overfitting\ntest_imputer_features = KNNImputer(missing_values=np.nan, n_neighbors=2)\ntest_imputer_features.fit(X_test)\nX_test = test_imputer_features.transform(X_test)\n')


# In[35]:


get_ipython().run_cell_magic('time', '', '#Standardize the data to make the mean = 0 and the standard deviation = 1\n\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.fit_transform(X_test)\n')


# In[36]:


get_ipython().run_cell_magic('time', '', "#Setting the evaluation parameters that will be used on all of the models to compare performance\n\ndef evaluate(gt, pred):\n    print('Precision: ', precision_score(gt, pred, average = 'micro'))\n    print('Recall: ', recall_score(gt, pred, average = 'micro'))\n    print('Accuracy: ', accuracy_score(gt, pred))\n    print('F1 Score: ', f1_score(gt, pred, average = 'micro'))\n")


# In[37]:


#Setting the roc curve to be used on all of the models to compare performance

def roc_auc(model, real_X, real_y):
    y_scores = model.predict_proba(real_X)[:, 1]
    fpr, tpr, thresholds = roc_curve(real_y, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=fpr, y=tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--', color='navy')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


# In[37]:


get_ipython().run_cell_magic('time', '', '#Logistic Regression Model\n\nlr_model = LogisticRegression()\nlr_model.fit(X_train, y_train.ravel())\n\ny_pred = lr_model.predict(X_test)\n')


# In[38]:


#Evaluating the Logistic Regression Model performance

evaluate(y_test, y_pred)


# In[39]:


get_ipython().run_cell_magic('time', '', '#Conducting feature selection to determine the top 10 most influential features for predicting ICU patient mortality.\n#Logistic Regression Model - Recursive Feature Elimination (RFE)\n\ncols = list(X.columns)\n\nlr_model_rfe = LogisticRegression()\nlr_model_rfe.fit(X_train, y_train.ravel())\n\n\n# Use RFE to select the top 10 features\nrfe = RFE(lr_model_rfe, n_features_to_select=10)\nrfe.fit(X_train, y_train.ravel())\n\ntemp = pd.Series(rfe.support_,index = cols)\nselected_features_rfe = temp[temp==True].index\n\n# Print the selected features\nprint(rfe.support_)\nprint(rfe.ranking_)\nprint(selected_features_rfe)\n')


# In[40]:


get_ipython().run_cell_magic('time', '', '#Gradient Boosting Classifier\n\ngb_model = GradientBoostingClassifier()\ngb_model.fit(X_train, y_train)\n\ny_pred_gb = gb_model.predict(X_test)\n')


# In[41]:


#Evaluating the Gradient Boosting Classifier performance

evaluate(y_test, y_pred_gb)


# In[42]:


get_ipython().run_cell_magic('time', '', '#Random Forest Classifier\n\nrf_model = RandomForestClassifier()\nrf_model.fit(X_train, y_train)\n\ny_pred_rf = rf_model.predict(X_test)\n')


# In[43]:


#Evaluating the Random Forest Classifier performance

evaluate(y_test, y_pred_rf)


# In[44]:


from sklearn import metrics

#set up plotting area
plt.figure(0).clf()

#fit logistic regression model and plot ROC curve
model = LogisticRegression(random_state=123)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))

#fit gradient boosted model and plot ROC curve
model = GradientBoostingClassifier(random_state=123)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Gradient Boosting Classifier, AUC="+str(auc))

#fit random forest classifier and plot ROC curve
model = RandomForestClassifier(random_state=123)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Random Forest Classifier, AUC="+str(auc))

plt.plot([0,1], [0,1], color='black', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

#add legend
plt.legend()


# ## Smote application 

# In[45]:


get_ipython().run_cell_magic('time', '', "df_clean['outcome'].value_counts()\n")


# In[46]:


print("Number enteries in X_train dataset: ", X_train.shape) 
print("Number enteries in y_train dataset: ", y_train.shape) 
print("Number enteries in X_test dataset: ", X_test.shape) 
print("Number enteries in y_test dataset: ", y_test.shape) 


# In[47]:


get_ipython().run_cell_magic('time', '', 'from imblearn.over_sampling import SMOTE\nsmote = SMOTE(random_state= 123)\n\n#Applying smote to the training set\n\nX_train_res,y_train_res = smote.fit_sample(X_train, y_train.ravel())\n')


# In[48]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 


# In[49]:


get_ipython().run_cell_magic('time', '', 'lr_model_res = LogisticRegression()\nlr_model_res.fit(X_train_res, y_train_res.ravel())\n\ny_pred_res = lr_model_res.predict(X_test)\n')


# In[50]:


evaluate(y_test, y_pred_res)


# In[51]:


get_ipython().run_cell_magic('time', '', '#Gradient Boosting Classifier\n\ngb_model_res = GradientBoostingClassifier()\ngb_model_res.fit(X_train_res, y_train_res)\n\ny_pred_gb_res = gb_model_res.predict(X_test)\n')


# In[52]:


#Evaluating the Gradient Boosting Classifier performance

evaluate(y_test, y_pred_gb_res)


# In[53]:


get_ipython().run_cell_magic('time', '', '#Random Forest Classifier\n\nrf_model_res = RandomForestClassifier()\nrf_model_res.fit(X_train_res, y_train_res)\n\ny_pred_rf_res = rf_model_res.predict(X_test)\n')


# In[54]:


#Evaluating the Random Forest Classifier performance

evaluate(y_test, y_pred_rf_res)


# In[55]:


#set up plotting area
plt.figure(0).clf()

#fit logistic regression model and plot ROC curve
model = LogisticRegression(random_state=123)
model.fit(X_train_res, y_train_res)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))

#fit gradient boosted model and plot ROC curve
model = GradientBoostingClassifier(random_state=123)
model.fit(X_train_res, y_train_res)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Gradient Boosting Classifier, AUC="+str(auc))

#fit random forest classifier and plot ROC curve
model = RandomForestClassifier(random_state=123)
model.fit(X_train_res, y_train_res)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Random Forest Classifier, AUC="+str(auc))

plt.plot([0,1], [0,1], color='black', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis (SMOTE)', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

#add legend
plt.legend()


# ## Hyperparameter Tuning - Logistic Regression Model

# In[56]:


get_ipython().run_cell_magic('time', '', '# Hyperparameter tuning for logistic regression model\nparam_grid = {"C": np.logspace(-4, 4, 20)}\n\n\n# Perform grid search with 5-fold cross-validation\nlogistic_model = LogisticRegression(solver="liblinear")\ngrid_search = GridSearchCV(logistic_model, param_grid, cv=5)\ngrid_search.fit(X_train_res, y_train_res)\n\n\n# Get the best hyperparameters\nbest_params = grid_search.best_params_\n\n\n# Train the model with the best hyperparameters\nbest_logistic_model = LogisticRegression(C=best_params["C"])\nbest_logistic_model.fit(X_train_res, y_train_res)\n\n\n# Make predictions on the test set\ny_pred = best_logistic_model.predict(X_test)\n\n\n# Calculate accuracy on the test set\naccuracy = accuracy_score(y_test, y_pred)\nprint("Best Hyperparameters:", best_params)\nprint("Test Accuracy:", accuracy)\n')


# ## Hyperparameter Tuning - Random Forest Classifier Model

# In[57]:


get_ipython().run_cell_magic('time', '', "#Define the hyperparameter grid\nparam_grid = { \n    'n_estimators': [25, 50, 100, 150],\n    'max_features': ['sqrt', 'log2', None],\n    'max_depth': [None, 10, 20],\n    'max_leaf_nodes': [2, 5, 10], \n} \n\n#Initalize GridSearchCV\ngrid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)\n\n#fit the model to the data\ngrid_search.fit(X_train_res, y_train_res)\n\n#Get the best parameters and the best estimator\nbest_params = grid_search.best_params_\nbest_rf = grid_search.best_estimator_\n\n#Make predictions\ny_pred = best_rf.predict(X_test)\n\n#Evaluate the model\naccuracy = accuracy_score(y_test, y_pred)\nprint(f'Accuracy: {accuracy:.2f}')\n")


# ## Hyperparameter Tuning - Gradient Boosted Classifier

# In[58]:


get_ipython().run_cell_magic('time', '', '#import xgboost as xgb\nfrom sklearn.model_selection import GridSearchCV\n\n# Define the hyperparameter grid\nparam_grid = {\n    \'max_depth\': [3, 5, 7],\n    \'learning_rate\': [0.1, 0.01, 0.001],\n    \'subsample\': [0.5, 0.7, 1]\n}\n\n# Create the XGBoost model object\ngb_model = GradientBoostingClassifier()\n\n# Create the GridSearchCV object\ngrid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring=\'accuracy\')\n\n# Fit the GridSearchCV object to the training data\ngrid_search.fit(X_train, y_train)\n\n# Print the best set of hyperparameters and the corresponding score\nprint("Best set of hyperparameters: ", grid_search.best_params_)\nprint("Best score: ", grid_search.best_score_)\n')


# ## Logistic Regression Model  - selected features

# In[59]:


get_ipython().run_cell_magic('time', '', '#Building model using selected features\nX= df_clean[[\'Renal failure\', \'COPD\', \'heart rate\', \'Urine output\', \'Creatinine\',\n       \'Urea nitrogen\', \'Blood calcium\', \'Anion gap\', \'Bicarbonate\', \'PCO2\']]\ny= df_clean[[\'outcome\']]\n\n#split the dataset into train and test\nX_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)\n\n\n#Imputing the missing values using k nearest neighbours algorithm.\n#This will take into account the relationship and pattern of the data, to resemble more real world measurements.\n\ntrain_imputer_features = KNNImputer(missing_values=np.nan, n_neighbors=2)\ntrain_imputer_features.fit(X_train)\nX_train = train_imputer_features.transform(X_train)\n\n#Important: test and train are imputed seperately to avoid overfitting\ntest_imputer_features = KNNImputer(missing_values=np.nan, n_neighbors=2)\ntest_imputer_features.fit(X_test)\nX_test = test_imputer_features.transform(X_test)\n\n#Standardize the data to make the mean = 0 and the standard deviation = 1\n\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.fit_transform(X_test)\n\n#Logistic Regression Model\n\nlr_model = LogisticRegression()\nlr_model.fit(X_train, y_train)\n\ny_pred = lr_model.predict(X_test)\n\n\n#Evaluating the Logistic Regression Model performance\n\nevaluate(y_test, y_pred)\n\n\n#Displaying the ROC for the Logistic Regression Model\n\n#set up plotting area\nplt.figure(0).clf()\n\n#fit logistic regression model and plot ROC curve\nmodel = LogisticRegression(random_state = 123)\nmodel.fit(X_train, y_train)\ny_pred = model.predict_proba(X_test)[:, 1]\nfpr, tpr, _ = metrics.roc_curve(y_test, y_pred)\nauc = round(metrics.roc_auc_score(y_test, y_pred), 4)\nplt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))\n\nplt.plot([0,1], [0,1], color=\'black\', linestyle=\'--\')\n\nplt.xticks(np.arange(0.0, 1.1, step=0.1))\nplt.xlabel("False Positive Rate", fontsize=15)\n\nplt.yticks(np.arange(0.0, 1.1, step=0.1))\nplt.ylabel("True Positive Rate", fontsize=15)\n\nplt.title(\'ROC Curve Analysis - feature selection\', fontweight=\'bold\', fontsize=15)\nplt.legend(prop={\'size\':13}, loc=\'lower right\')\n\n#add legend\nplt.legend()\n')

