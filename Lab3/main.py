import scipy.io
from sklearn import svm
from collections import OrderedDict

from process_email import get_dictionary
from process_email import email_features
from process_email import process_email



def clf_(P, y):
    a = sum(P == y);
    return a / len(y) * 100;

with open('email.txt', 'r') as file:
    email = file.read().replace('\n', '');

features = process_email(email);
print(features)
features = email_features(features);
sum_ = sum(features > 0);
print('Длинна вектора признаков: 1899.');
print('Количество не нулевых  элементов:', sum_);

data = scipy.io.loadmat('train.mat');
X = data['X'];
y = data['y'].flatten();

print('Тренировка SVM-классификатора с линейным ядром...');
clf = svm.SVC(C=0.1, kernel='linear', tol=1e-3);
model = clf.fit(X, y);
p = model.predict(X);

CLF = clf_(p, y);
print('точность на обучающей выборке: ', CLF);

data = scipy.io.loadmat('test.mat');
X = data['Xtest'];
y = data['ytest'].flatten();

print('Тренировка SVM-классификатора с линейным ядром...');
P = model.predict(X);

a = sum(P == y);
c = a / len(y) * 100;
print('точность на тестовой выборке: ', c);

t = sorted(list(enumerate(model.coef_[0])), key=lambda e: e[1], reverse=True);
d = OrderedDict(t);
idx = list(d.keys())
weight = list(d.values())
dictionary = get_dictionary()

print('Топ-15 слов в письмах со спамом')
[print(' %-15s (%f)' %(dictionary[idx[i]], weight[i])) for i in range(15)]


