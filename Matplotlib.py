# matplotlib
# 파이썬에서 데이터과학 관련 시각화 패키지

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %matplotlib inline # 주피터노트북에서 show() 호출없이도
                     # 그래프를 그릴 수 있게 해줌
data = np.arange(10)
plt.plot(data)
plt.show()


# 산점도 - 100의 표준정규분포 난수 생성
list = []
for i in range(100):           # 0 ~ 99
    x = np.random.normal(0, 1) # 표준정규분포 난수
    y = x + 0.1 + 0.2 + np.random.normal(0, 1)
    list.append([x, y])

x_data = [ v[0] for v in list ]
y_data = [ v[1] for v in list ]

plt.plot(x_data, y_data, 'ro')
plt.show()