import numpy as np
import scipy.io as sio
import svm

fName1 = "dataset1.mat";
fName2 = "dataset2.mat";
fName3 = "dataset3.mat";

fPatch = "C:\\ИТБ-230N\\Максим\\4 Сем\\reposGit\\Python---Collective-IT-Dev\\Lab2\\"
C = 1;

# TODO 1


# считываем данные из файла
data = sio.loadmat(fName1);
X = data['X']; # 2 столбика 54 * 2
y = data['y']; # 2 столбика 54 * 1

y = y.astype(np.float64);

print(X)
print(y)

# покажем на грфике через файл "svm"
svm.visualize_boundary_linear(X, y, None, title='Исходные данные');

# TODO 2

# отобразить границу *\+
# найдем model из svm.svm_train
model = svm.svm_train(X, y, C, svm.linear_kernel, 0.001, 20) # C = 1
# отобразим границу *\+
print(model)
svm.visualize_boundary_linear(X, y, model, title='Разделяющая граница');

# TODO 3
C = 100; # теперь C = 100

# делаем все тож самое что и в TODO 2
model = svm.svm_train(X, y, C, svm.linear_kernel, 0.001, 20);
svm.visualize_boundary_linear(X, y, model, title='Разделяющая граница');

# TODO 4

# используем готовую функцию contour (она уже реализована)
svm.contour(1);
svm.contour(3);

# TODO 5

# считываем 2 данные
data = sio.loadmat(fName2);
X = data["X"];
y = data["y"];

y = y.astype(np.float64) # приводим к типу float

svm.visualize_boundary_linear(X, y, None, title='Исходные данные'); # покажем данные на графике

# TODO 6
C = 1.0
sigma = 0.1

# покажем границу по уже известным данным
# код взят из методички
gaussian = svm.partial(svm.gaussian_kernel, sigma=sigma)
gaussian.__name__ = svm.gaussian_kernel.__name__
model = svm.svm_train(X, y, C, gaussian)
svm.visualize_boundary(X, y, model)

# TODO 7
# считаем данные из 3 файла
data = sio.loadmat(fName3);

# обучаемая выборка
X = data['X']
y = data['y']
y = y.astype(np.float64)

# тестовая выбока
Xval = data['Xval']
yval = data['yval']
yval = yval.astype(np.float64)

# покажем на графике
svm.visualize_boundary_linear(X, y, None, title='Исходные данные');
svm.visualize_boundary_linear(Xval, yval, None, title='Исходные данные');

# TODO 8
C = 1.0
sigma = 0.5

# выполним метод Гаусса для обучаемой
gaussian = svm.partial(svm.gaussian_kernel, sigma=sigma)
gaussian.__name__ = svm.gaussian_kernel.__name__
model = svm.svm_train(X, y, C, gaussian)
svm.visualize_boundary(X, y, model)

# TODO 9

# найдем оптимальные C и sigma для отображения линии
errorMin = 100000000000.0; # кол. ошибок по умолч. 10 ^ дофига
Cmin = 0.0; # запоминаем значения, найденное в цикле
sigmaMin = 0.0; # запоминаем sigma
modelMin = 0.0; # запоминаем model, которую уже посчитали

for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
    for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        # найдем по методу Гаусса model
        gaussian = svm.partial(svm.gaussian_kernel, sigma=sigma)
        gaussian.__name__ = svm.gaussian_kernel.__name__
        model = svm.svm_train(X, y, C, gaussian)


        ypred = svm.svm_predict(model, Xval);
        error = np.mean(ypred != yval.ravel()); # найдем кол. ошибок

        if(errorMin > error): # запомним мин. кол. ошибок и его значения
            errorMin = error;
            Cmin = C;
            sigmaMin = sigma;
            modelMin = model;

# нарисуем граффик тестовых и обуч. данных по найденной матрице
svm.visualize_boundary(X, y, modelMin);
svm.visualize_boundary(Xval, yval, modelMin);