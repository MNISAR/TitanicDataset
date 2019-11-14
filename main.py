import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

titanic = pd.read_csv('titanic_train.csv')
	
titanic.set_index('PassengerId', drop = True, inplace = True)

del titanic['Name']
del titanic['Ticket']
titanic['Embarked'].dropna(inplace = True)
del titanic['Cabin']

titanic['gender'] = titanic.Sex.apply(lambda x: 1 if x=='male' else 2)
del titanic['Sex']
class_1_surv = titanic[(titanic['Pclass'] == 1)].mean()['Survived']
class_2_surv = titanic[(titanic['Pclass'] == 2)].mean()['Survived']
class_3_surv = titanic[(titanic['Pclass'] == 3)].mean()['Survived']

my_xticks = ['Class 1','Class 2','Class 3']
x = [1,2,3]
plt.xticks(x, my_xticks)
plt.plot(x, [class_1_surv, class_2_surv, class_3_surv])
plt.show()

males = len(titanic[titanic['gender'] == 1])
females = len(titanic[titanic['gender'] == 2])

plt.pie([males,females],
       labels = ['Male', 'Female'],
       explode = [0.10, 0],
       startangle = 0)
plt.show()

C_surv = titanic[titanic['Embarked'] == 'C'].mean()['Survived']
Q_surv = titanic[titanic['Embarked'] == 'Q'].mean()['Survived']
S_surv = titanic[titanic['Embarked'] == 'Q'].mean()['Survived']
my_xticks = ['C','Q','S']
x = [1,2,3]
plt.xticks(x, my_xticks)
plt.scatter(x, [C_surv, Q_surv, S_surv])
plt.show()


plt.scatter(titanic['Age'], titanic['gender'])
plt.show()


survived = titanic[titanic['Survived'] == 1]
surv_avg = survived.mean()['Age']
not_survived = titanic[titanic['Survived'] == 0]
nsurv_avg = not_survived.mean()['Age']


titanic['avg'] = titanic.Survived.apply(lambda x: surv_avg if x==1 else nsurv_avg)

titanic.Age.fillna(titanic['avg'], inplace = True)
del titanic['avg']

print(titanic.head())
############################################################
from sklearn import datasets
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection 
from sklearn import tree
import pydotplus
from sklearn import tree

clf = RandomForestClassifier()
survived = titanic['Survived']
del titanic['Survived']
del titanic['Embarked']

X_Train, X_Test, Y_Train, Y_Test = model_selection.train_test_split(titanic, survived, test_size = 0.2)
clf.fit(X_Train, Y_Train)
print("For Random Forest Classifier: ",clf.score(X_Test, Y_Test))

#######################################################
clf1 = tree.DecisionTreeClassifier()
clf1.fit(X_Train, Y_Train)
print("For Decision Tree Classifier: ",clf1.score(X_Test, Y_Test))
dot_data = tree.export_graphviz(clf1, out_file=None,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('titanic - Decision Forest.pdf')

##########################################################
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_Train, Y_Train)
print("For Logistic Regression: ",logreg.score(X_Test, Y_Test))


#############################################################
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier

X_Train, X_Test, Y_Train, Y_Test = model_selection.train_test_split(titanic, survived, test_size = 0.2)
clf = MLPClassifier(hidden_layer_sizes = (100,100,100,), solver = 'lbfgs', activation = 'logistic')
clf.fit(X_Train, Y_Train)
print("for Neural Network",clf.score(X_Test, Y_Test))


