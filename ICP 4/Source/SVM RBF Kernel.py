from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris_datasets = datasets.load_iris()

x = iris_datasets.data
y = iris_datasets.target

train_x, test_x, train_y, test_y=train_test_split(x, y, test_size=0.2, random_state=2)

rmodel=SVC(kernel='rbf')
rmodel.fit(train_x, train_y)
pred=rmodel.predict(test_x)

print("RBF kernel accuracy score is", accuracy_score(pred, test_y))