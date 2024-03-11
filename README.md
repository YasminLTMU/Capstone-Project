# CIND820 Capstone Project Repository

This repository contains the CIND820 Capstone Project files which consists of the MIMIC III dataset, EDA html file, Python/Jupyter notebooks and an outline of the tenative stages of this project.

## Research Questions

This aim of this project is ICU patient mortality prediction. There are several research questions that are to be answered during this project. These are: 

-Is it possible to create a highly accurate ICU admitted patient survival model?

-What are the most influential predictors of ICU patient survival?

-How do demographic factors, such as age and gender, impact ICU patient survival?

-Does the presence of chronic illnesses and comorbidities, impact ICU patient resilience and response to treatment?


## Outline of tenative steps for this project: 

-Create demographic, vital signs and comorbidity data subsets for seperate analysis to determine if there is a significant effect on ICU patient mortality.

-Impute missing values for demographic , vital and comorbidity data subsets 

-Conduct Chi-Square test to determine if demographic features have a significant effect on ICU patient mortality

-Conduct Correlation Analysis to determine if and what vital signs have signicance when predicting ICU patient mortality

-Conduct Chi-Square test on Comorbidities to determine if pre-existing conditions/Comorbidities have a signicant effect on ICU patient mortality

-Split MIMIC III dataset into 80% train set and 20% test set, then impute missing enteries to avoid data leakage

-Standardize data

-Run Logistic Regression, Gradient Boosted Classifier and Random Forest Classifier models

-Evaluate and compare model performance

-Conduct oversampling and re-evalute model performance

-Conduct feature selection to determine the most influential features for predicting ICU patient mortality

- Conclusion and recommendations
- 
