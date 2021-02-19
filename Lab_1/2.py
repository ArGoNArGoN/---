import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    x1 = np.zeros(x.shape[0]);
    min_x = min(x); 
    max_x = max(x);
    for i in range(0, x.shape[0]):
        x1[i] = (x[i] - min_x) / (max_x - min_x)
    return x1

fName = "ex1data2.txt"
data = np.matrix(np.loadtxt(fName, delimiter=','))

#берем столбцы из матрицы
X = data[:, 0]
y = data[:, 1]

X_norm = normalize(X);
y_norm = normalize(X);

X_std = np.std(X)
y_std = np.std(y)

Std = np.std(data)

print('X: ', X_std)
print('y: ', y_std)

X_1 = X_norm / X_std
y_1 = y_norm / y_std
print(X_1)
print(y_1)


