import numpy as np
import matplotlib.pyplot as plt

#TODO 1

#C:\\ИТБ-230N\\Максим\\4 Сем\\Основы коллективной ИТ-деятельности\\Задания\\Лаб_1\\
fName = "ex1data1.txt"

#считываем данные из файла
data = np.matrix(np.loadtxt(fName, delimiter=','))
#print(data);

#TODO 2

from matplotlib import rc
font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)

#берем столбцы из матрицы
x = data[:, 0]
y = data[:, 1]

#показываем данные пользователю
plt.plot(x, y, 'b.')
plt.plot('Зависимость прибыльности от численности')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')
plt.grid()
plt.show()

#TODO 3
print('\n')

#что здесь происходит?
#нам нужно найти значение гипотезы для каждой строки по x!
#как это сделать? математика Карл!
#перемножаем матрицы!


#находит cost (хех)
def compute_cost(x_ones, y, theta):
    
    #так можно вычислить значение гипотезы для всех городов сразу
    h_x = x_ones * theta
    print(h_x)
    
    cost = np.sum(np.power(h_x-y,2)) / (2 * (np.power(h_x-y,2)).shape[0])
    return cost

#количество элементов в x
m = x.shape[0]

#добавляем единичный столбец к x
x_ones = np.c_[np.ones((m,1)), x]

#'k' theta представляют собой вектор таблиц из 2 элементов
theta = np.matrix([[1],[2]])

cost = compute_cost(x_ones,y,theta)
print(cost)

#TODO 4
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
    j_theta = np.sum(np.power(h_x-y,2))/ (2 * m) 
    
    return j_theta

def Theta_j(theta, alpha, hTheta_x, y, X):
    '''
     рассчитывает по формуле 4 новые
     коэффициенты для каждого столбца
    '''
    
    # иниц. новые значения в tmp_theta
    tmp_theta = theta
    
    # используем старые значения для кажд. столбца j для новых значений
    for j in range(2):
        sum_ = float((hTheta_x - y).T * X[:, j])
        # формула 4
        tmp_theta[j] = tmp_theta[j] - (alpha / X.shape[0]) * sum_
        
    return tmp_theta

def gradient_descent(X, y, alpha, it):
    n = X.shape[1]
    
    # иниц. theta (0, 1,...1)
    theta = np.ones(n)
    theta[0] = 0
    
    # иниц. j_theta (0,...0)
    j_theta = np.zeros(it) # return
    
    for i in range(it):
        
        # формула 3
        h_x = H0_X(X, theta).T
        
        # формула 1
        j_theta[i] = J_theta(h_x, y)
        if(i < 100):
            print(theta)
        # рассчитываем новую theta
        theta = Theta_j(theta, alpha, h_x, y, X)
    return j_theta, theta

X = np.c_[np.ones((x.shape[0],1)), x]
res = gradient_descent(X, y, 0.02, 500)
J_theta = res[0]
THETA = res[1]

plt.plot(J_theta)
plt.title('Снижение ошибки при градиентном спуске')
plt.xlabel('Итерации')
plt.ylabel('Ошибки')
plt.grid()
plt.show()

#TODO 5
print('\n')

#x0_1 = float(input('Введите x1[0]: '))

#x1_1 = float(input('Введите x2[0]: '))

#data1 = np.matrix([[x0_1, 0],[x1_1, 0]])

# эксперемент!

def h_theta__X(theta, x):
    m = x.shape[0]
    x_ones = np.c_[np.ones((m, 1)), x]
    h = theta * x_ones.T
    return h

D = np.matrix([5]) #17.929
print(D)
print(h_theta__X(THETA, D))

#TODO 6

#берем столбцы из матрицы
a = data[:, 0]
b = data[:, 1]

x1 = np.arange(min(a), max(a))
print(THETA[1]*x1 + THETA[0])
plt.plot(x1, THETA[1]*x1 + THETA[0], 'g--')
plt.plot(a, b,'b.')

#TODO7




