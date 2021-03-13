import numpy as np
import matplotlib.pyplot as plt

# TODO 1

# C:\\ИТБ-230N\\Максим\\4 Сем\\Основы коллективной ИТ-деятельности\\Задания\\Лаб_1\\
fName = "ex1data1.txt"

# считываем данные из файла
data = np.matrix(np.loadtxt(fName, delimiter=','))
# print(data);

# TODO 2

from matplotlib import rc

font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)

# берем столбцы из матрицы
x = data[:, 0]
y = data[:, 1]

# показываем данные пользователю
plt.plot(x, y, 'b.')
plt.xlabel('Численность в тыс.')
plt.ylabel('Прибыльность')
plt.title('Зависимость прибыльности от численности')
plt.grid()
plt.show()

# TODO 3
print('\n')


# что десь происходит?
# нам нужно найти значение гипотезы для каждой строки по x!
# как это сделать? математика Карл!
# перемножаем матрицы!

# находит cost (хех)
def compute_cost(x, y, theta):
    m = x.shape[0]
    h_x = x * theta
    cost = np.nansum(np.power(h_x - y, 2)) / (2 * m)
    return cost


# количество элементов в x
m = x.shape[0]

# добавляем единичный столбец к x
x_ones = np.c_[np.ones((m, 1)), x]

# 'k' theta представляют собой вектор таблиц из 2 элементов
theta = np.matrix([[1], [2]])

cost = compute_cost(x_ones, y, theta)
print(cost)

# TODO 4
print('\n')


def H0_X(x, theta):
    '''
     Расчет: h_x
     theta - Коэффициент
     x - некий вектор 
    '''
    h_x = theta * x.T
    return h_x


def J_theta(h_x, y):
    '''
    Расчет: j_theta
    
    h_x - матрица m на 2
    y - матрица m на 1

    '''
    m = y.shape[0]
    j_theta = np.sum(np.power(h_x - y, 2)) / (2 * m)

    return j_theta


def Theta_j(theta, alpha, hTheta_x, y, X):
    '''
     рассчитывает по формуле 4 новые
     коэффициенты для каждого столбца
    '''

    # иниц. новые значения в tmp_theta
    tmp_theta = theta
    n = X.shape[1]
    # используем старые значения для кажд. столбца j для новых значений
    for j in range(n):
        sum_ = float((hTheta_x - y).T * X[:, j])
        # формула 4
        tmp_theta[j] = tmp_theta[j] - (alpha / X.shape[0]) * sum_

    return tmp_theta


def gradient_descent2(X, y, alpha, it):
    n = X.shape[1]

    # иниц. theta (0, 1,...1)
    theta = np.ones(n)
    theta[0] = 0

    # иниц. j_theta (0,...0)
    j_theta = np.zeros(it)  # return

    for i in range(it):
        # формула 3
        h_x = H0_X(X, theta).T

        # формула 1
        j_theta[i] = J_theta(h_x, y)
        # рассчитываем новую theta
        theta = Theta_j(theta, alpha, h_x, y, X)
    return j_theta, theta


X = np.c_[np.ones((x.shape[0], 1)), x]
res = gradient_descent2(X, y, 0.001, 500)
J_theta = res[0]
THETA = res[1]

plt.plot(J_theta)
plt.title('Снижение ошибки при градиентном спуске')
plt.xlabel('Итерации')
plt.ylabel('Ошибки')
plt.grid()
plt.show()

# TODO 5
print('\n')


# x0_1 = float(input('Введите x1[0]: '))

# x1_1 = float(input('Введите x2[0]: '))
# data1 = np.matrix([[x0_1, 0],[x1_1, 0]])

# эксперемент! найдем y по x

# Перемножим матрицы и найдем y
def h_theta__X(theta, x):
    m = x.shape[0]
    x_ones = np.c_[np.ones((m, 1)), x]
    h = theta * x_ones.T
    return h


D = np.matrix([5])  # Возьмем рандомича с x = 5
D1 = np.matrix([8])  # Возьмем рандомича с x = 8
print(D)
print(h_theta__X(THETA, D))
print(D1)
print(h_theta__X(THETA, D1))

# TODO 6

# берем столбцы из матрицы
a = data[:, 0]
b = data[:, 1]

x1 = np.arange(min(a), max(a))
plt.plot(x1, THETA[1] * x1 + THETA[0], 'g--')
plt.plot(a, b, 'b.')
plt.grid()
plt.show()


# TODO 7

def gradient_descent3(x, y, alpha, it):
    # Добавим единичный вектор и найдем m and n
    x_ones = np.c_[np.ones((x.shape[0], 1)), x]
    m = x_ones.shape[0]
    n = x_ones.shape[1]

    # иниц. theta [[0, 1,...1][1,...1]]
    theta = np.ones((n, 1))
    theta[0][0] = 0

    # иниц. j_theta [[0,...0][1,..1]]
    J_theta = np.zeros((it, 1))  # return

    for i in range(it):
        # Воспользуемся старой формулой
        J_theta[i] = compute_cost(x_ones, y, theta);
        temp = theta
        for j in range(n):
            # Разделим все вычисление на маленькие, ибо так легче манипулировать данными
            b = x_ones * theta - y
            a = b.T * x_ones[:, j]
            temp[j][0] = theta[j][0] - alpha * np.nansum(a) / m

            # не забываем переприсвоить theta
            theta = temp
    return J_theta, theta


fName = "ex1data2.txt"

data = np.matrix(np.loadtxt(fName, delimiter=','))
x = data[:, :2]
y = data[:, 2]

# x содержит 2 столбца лощадь в футах
# y содержит 1 столбец (последний) цена дома
# 3 стобец зависит от первых 2, то есть x - данное, а y зависимое 


# найдем среднее значение среднеквадр. отклонение 
# (если я правильно понял - квадратное)

xmean = np.nanmean(x)  # среднее арифм.
xstd = np.nanstd(x)  # среднеквадр. отклонение

ymean = np.nanmean(y)  # среднее арифм.
ystd = np.nanstd(y)  # среднеквадр. отклонение

# по формуле найдем x нормальное
xnorm = (x - xmean) / xstd
ynorm = (y - ymean) / ystd

# TODO 8

# TODO 8.1

# Запускаем шайтан машину
Chtoto = gradient_descent3(xnorm, ynorm, 0.01, 500)
J_theta = Chtoto[0]
theta = Chtoto[1]

# чертим график
plt.plot(J_theta)
plt.title('Снижение ошибки при градиентном спуске')
plt.xlabel('Итерации')
plt.ylabel('Ошибки')
plt.grid()
plt.show()

print('theta:', theta)

# TODO 8.2

# найдем y для 2 известных через theta
X1 = np.array([[5000, 5], [2000, 2]])

# проведем нормализацию по x (будем использовать xmean and xstd (полученные в TODO 7))
X1norm = (X1 - xmean) / xstd

X1_1 = (theta[0] * 1 + theta[1] * X1norm[:, 0] + theta[2] * X1norm[:, 1]) * ystd + ymean  # Градиентное опускание
print('x1: ', X1[0])
print('x2: ', X1[1])
print('y1, y2: ', X1_1)

# TODO 9
x = data[:, :2]
y = data[:, 2]

x_ones = np.c_[np.ones((len(x), 1)), x]  # Добавим 1 столбец
newTheta = np.linalg.pinv(x_ones.T * x_ones) * x_ones.T * y  # Расчитаем newTheta по формуле

X1_2 = (newTheta[0] * 1 + newTheta[1] * X1[:, 0] + newTheta[2] * X1[:, 1])  # МНК

print('x1: ', X1[0])
print('x2: ', X1[1])
print('y1, y2: ', X1_2[0])

# Task 10
print('x1: ', X1[0])
print('x2: ', X1[1])
print('Градиентное опускание: ', X1_1)
print('МНК: ', X1_2)
