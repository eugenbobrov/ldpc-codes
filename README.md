# ldpc_codes
loopy belief propagation in error-correction coding

Вероятностная модель информационного канала с шумом:

$$p(e, s) \propto p(e)p(s|e) \propto \prod\limits_n p(e_n)\prod\limits_m [h_m^Te = s_m]$$

$e \in \R^N $ -- вектор ошибок, $s \in \R^M$ -- вектор синдромов

$h_m \in \R^N$ -- вектор, $m$-я строка проверочной матрицы $H \in \R^{M, N}$


$p(e_n) = q^{e_n}(1-q)^{1-e_n} $ -- априорное распределение шума в канале связи


![](data/spoon.png)
![](data/0.png)
![](data/2.png)
![](data/3.png)
![](data/4.png)
![](data/5.png)
