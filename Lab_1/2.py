import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

data = np.matrix(np.loadtxt('ex1data2.txt', delimiter = ','))
x = data [:, :2]
y = data[:, 2]

# x содержит 2 столбца лощадь в футах
# y содержит 1 столбец (последний) цена дома
# 3 стобец зависит от первых 2, то есть x - данное, а y зависимое 


# найдем среднее значение среднеквадр. отклонение 
# (если я правильно понял - квадратное)

xmean = np.nanmean(x)
xstd = np.nanstd(x)
ystd = np.nanstd(y) #среднеквадр. отклонение
ymean = np.nanmean(y) # среднее арифм.

# по формуле найдем x нормальное
xnorm = (x - xmean)/xstd

print(xnorm)
## 9.64150080e-01