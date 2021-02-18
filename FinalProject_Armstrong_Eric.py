"""
@author: Eric Armstrong
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

#Import dataset
col_names = ['fixedAcidity', 'volatileAcidity', 'citricAcid', 'residualSugar', 'chlorides', 'freeSulfurDioxide', 'totalSulfurDioxide', 'density','pH','sulphates','alcohol','quality','label']
# load dataset
wine = pd.read_csv("winequality.csv", header=None, names=col_names)

#Pick features to use (with help from Zach Combs)
feature_cols = ['volatileAcidity', 'citricAcid', 'residualSugar', 'density','pH','sulphates']
X = wine[feature_cols]
Y = wine.label

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

LogRegression = LogisticRegression()
LogRegression.fit(x_train, y_train)
y_pred = LogRegression.predict(x_test)
print("Accuracy of logistic regection classifier on test set: {:.2f}\n".format(LogRegression.score(x_test, y_test)))

#ALL following code from 2_Logostic_ExSKLearn_Demo.py
y_pred = LogRegression.predict(x_test)
print(y_pred)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
#sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu" ,fmt='g')
#ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print(metrics.classification_report(y_test,y_pred))
plt.figure()
y_pred_proba = LogRegression.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()