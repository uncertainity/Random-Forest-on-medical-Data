import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix



df = pd.read_csv("EyeData_1.csv")
print(df.head())
print(df.dtypes)
df = df.dropna()
is_nan = df.isnull()
print(is_nan.sum())
y = np.array(df["MorphologicalEdited"])
df = df.drop("MorphologicalEdited",axis = 1)
print(df.columns)
X = np.array(df)
print("Shape of the data:",X.shape)
print("Shape of the labels:",y.shape)

## Random Oversmapling ##
strategy = {1:60,2:110,3:65,4:60,5:100}
oversample = RandomOverSampler(sampling_strategy = strategy)
X_over,y_over = oversample.fit_resample(X,y)
print("Shape of the data:",X_over.shape)
print("Shape of the labels:",y_over.shape)
df_1 = pd.DataFrame(X_over)
df_1["Y"] = y_over
df_1.to_csv("SampledEye.csv")


##Using CV to see whether decision tree/random forest is a good model type
train_features,test_features,train_labels,test_labels = train_test_split(X_over,y_over,test_size = 0.1,random_state = 42)
rf = RandomForestClassifier(n_estimators = 100,random_state = 100)
cv_score = cross_val_score(rf,train_features,train_labels,cv = 5)
print("Cv score for a basic random forest for 10 estimators:",cv_score)
print("Mean CV:",np.mean(cv_score))
rf.fit(train_features,train_labels)
predicted_labels = rf.predict(test_features)
print("Error:",np.sum(test_labels != predicted_labels)/test_labels.shape[0])
print(confusion_matrix(test_labels,predicted_labels))

rf.fit(X_over,y_over)
shap_explainer = shap.TreeExplainer(rf)
shap_values = shap_explainer.shap_values(X_over)
print(len(shap_values))
shap.summary_plot(shap_values,X_over)
shap.summary_plot(shap_values[0],X_over)
shap.dependence_plot("Feature 14",shap_values[0],X_over,interaction_index=None, color="k")






#rf_random = RandomForestClassifier(random_state = 100)
#n_estimators = [int(x) for x in np.linspace(start = 10,stop = 30, num = 5)]
#max_depth = [int(x) for x in np.linspace(start = 3,stop = 20,num = 1)]
#min_samples_split = [2,3,5]
#min_samples_leaf = [1,2,3]
#bootstrap = [True,False]
#randomGrid = {"n_estimators":n_estimators,"max_depth":max_depth,"min_samples_split":min_samples_split,"min_samples_leaf":
#             min_samples_leaf,"bootstrap":bootstrap}
#rf_random = RandomizedSearchCV(estimator = rf_random,param_distributions = randomGrid,n_iter = 100,cv = 5,verbose = 2,random_state = 100)
#rf_random.fit(train_features,train_labels)
#print(rf_random.best_params_)

#rf_1 = RandomForestClassifier(n_estimators = 30,min_samples_split = 2, min_samples_leaf = 3,max_depth = 3, bootstrap = "True",random_state = 100)
#cv_score = cross_val_score(rf_1,train_features,train_labels,cv = 5)
#print("Cv score for a basic random forest for 10 estimators:",cv_score)

#dt = DecisionTreeClassifier(max_depth = 10)
#cv_score = cross_val_score(dt,train_features,train_labels,cv = 5)
#print("Cv score for a basic random forest for 10 estimators:",cv_score)
#dt.fit(train_features,train_labels)
#predicted_labels = dt.predict(test_features)
#print("Error:",np.sum(test_labels != predicted_labels)/test_labels.shape[0])

#fig = plt.figure(figsize=(15,10))
#_ = tree.plot_tree(dt,filled=True)
#text_representation = tree.export_text(dt)
#print(text_representation)
#print(dt.feature_importances_)

#X_imp1 = X[:,X.shape[1]-1]
#X_imp2 = X[:,4]
#X_imp2
#plt.scatter(X_imp2,y)





