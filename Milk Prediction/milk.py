import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt

# load dataset
milk = pd.read_csv('milknew.csv')

# data cleaning
milk.rename(columns={'Temprature': 'Temperature', 'Fat ': 'Fat'}, inplace=True)

# feature selection
features = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
X = milk[features]
Y = milk.Grade

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# visualizing the tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   feature_names=features,
                   class_names=milk.Grade,
                   filled=True)
fig.savefig("milk_tree.png")