from classes.tree import TreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("Dataset\iris.csv", skiprows = 1, header = None, names = col_names)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
training_X, test_X, training_Y, test_Y = train_test_split(X, Y, test_size = .25, random_state = 30)

classifier = TreeClassifier(minSamplesSplit = 1, maxDepth = 4)
classifier.Fit(training_X, training_Y)
classifier.PrintTree()

pred_Y = classifier.Predict(test_X) 
print("Accuracy score: " + str(accuracy_score(test_Y, pred_Y)))