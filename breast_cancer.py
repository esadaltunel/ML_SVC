from sklearn.svm import LinearSVC # to create support vector classification model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd


dataset = pd.read_csv("breast_cancer.csv")
label_encoder = LabelEncoder()
dataset["diagnosis"] = label_encoder.fit_transform(dataset.diagnosis.values)

train, test = train_test_split(dataset, test_size = 0.3)

x_train = train.drop("diagnosis", axis = 1)
y_train = train.loc[:, "diagnosis"]


x_test = test.drop("diagnosis", axis = 1)
y_test = test.loc[:, "diagnosis"]

model = LinearSVC()

model.fit(x_train, y_train)

predictions = model.predict(x_test) 

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))
