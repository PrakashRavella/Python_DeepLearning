from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris_datasets = datasets.load_iris()
x = iris_datasets.data
y = iris_datasets.target


train_x, test_x, train_y, test_y=train_test_split(x, y, test_size=0.2, random_state=21)

model=SVC(kernel='linear')
model.fit(train_x, train_y)
prediction=model.predict(test_x)

print("linear kernel Accuracy score is", accuracy_score(prediction, test_y))