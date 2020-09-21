---
layout:     post
title:      "高斯回归(Gaussian Processes)"
subtitle:   "chaitong"
date:       2020-04-09
author:     "CT"
header-img: "img/blog-bg.jpg"
tags:
    - 机器学习
---

- 本文参考《PRML》，并加入了一些自己的理解，如有错误恳请指出。
# 1 前言
### 1.1 怎么理解高斯过程？高斯过程和随机过程的关系是什么？
- **高斯分布**：随机变量服从高斯分布，意味着在**当前时刻**，该随机变量**可能取的值**服从高斯分布，但它只能从分布中**取一个**！
- **随机过程**：在一条时间线上，或一个数据集中，随机变量在每个位置上服从一定的分布，但在每一个位置只能取一个值，假设共有m个时间点（或数据集容量为m），则一共会产生m个取值结果，这m个取值结果便称为一个**过程**，因为在每一个点的取值是随机的，因此称为**随机过程**。我们用联合概率分布来描述随机过程：
$$p(x^1=t_1,x^2=t_2,...,x^m=t_m)=p(x^1=t_1)p(x^2=t_2)...p(x^m=t_m)$$
- **高斯过程**：对于一个随机过程，如果随机变量在每个位置服从的分布是高斯分布，那该随机过程就称为**高斯过程**；但在高斯过程中，对于m个随机变量产生的结果$\mathbf{x}=\{x^1,x^2,...,x^m\}$，我们不再使用联合概率分布来描述，而是使用**多维高斯分布**的二阶统计量来描述： $\mathbb{E}[\mathbf{x}]\in\mathbb{R}^m$ 和 $cov[\mathbf{x}]\in \mathbb{R}^{m\times m}$；

### 1.2 贝叶斯线性回归与高斯过程
- 在[贝叶斯线性回归](https://blog.csdn.net/qq_36013249/article/details/105932395)问题中，我们建立了如下模型：
$$y(x)=w^T\Phi(x),\quad\Phi(x)\in R^n,w\in R^n\tag{1.2.1}$$
参数$w$服从一个高斯先验分布：
$$p(w)=\mathcal{N}(w|0,\alpha^{-1}I^{n\times n})\tag{1.2.2}$$

- 因此对于一个数据集 $\mathbf{x}=\{x^1,x^2,...,x^m\}$，模型的输出  $\mathbf{y}=[y^1,y^2,...,y^m]^T$ （注意这里的模型的输出不是数据集中的标签或目标值，这里只描述了一个线性高斯模型的输出，并没有描述噪声）服从$m$维高斯分布：
$$\mathbf{y}=\Phi w=\left(
\begin{matrix}
\Phi(x^1)^T\\
...\\
\Phi(x^m)^T
\end{matrix}
\right)_{m\times n}\cdot\quad\left(
\begin{matrix}
w_1\\
...\\
w_n
\end{matrix}
\right)_{n\times 1}=\left(
\begin{matrix}
y^1\\
...\\
y^m
\end{matrix}
\right)_{m\times 1}\sim\mathcal{N}(\mathbb{E}[\mathbf{y}],cov[\mathbf{y}])\tag{1.2.3}$$
其中：
$$\mathbb{E}[\mathbf{y}]=\Phi\mathbb{E}[\mathbf{w}]=\mathbf{0}\in \mathbb{R}^m$$
$$cov[\mathbf{y}]=\frac{1}{\alpha}\Phi\Phi^T=K\in \mathbb{R}^{m\times m}$$
其中$K$是Gram矩阵：
$$K_{ij}=k(x^i,x^j)=\frac{1}{\alpha}\Phi(x^i)^T\Phi(x^j),\quad i,j\in[1,m]$$
$k(x^i,x^j)$称为核函数；
- 因此，模型的输出 $\mathbf{y}=[y^1,y^2,...,y^m]^T$ 可以看作一个高斯过程;
### 1.3 贝叶斯线性回归与高斯过程：二维场景举例 (看懂了可以跳过这一节)

- 考虑简单的二维场景： $\mathbf{x}=\{x^1,x^2,x^3\}=\{1,2,3\},\quad\Phi=\left[
\begin{matrix}
\Phi(1)^T\\
\Phi(2)^T\\
\Phi(3)^T
\end{matrix}
\right]_{3\times 2}=\left[
\begin{matrix}
1 & 1\\
1 & 2\\
1 & 3
\end{matrix}
\right]_{3\times 2}$
- 假设参数$w$服从高斯先验分布 $\mathcal{N}(\mathbb{E}[\mathbf{w}],cov[\mathbf{w}])$：
$$\mathbb{E}[\mathbf{w}]=\left[
\begin{matrix}
0\\
1
\end{matrix}
\right]$$
$$cov[\mathbf{w}]=\alpha^{-1}I=\left[
\begin{matrix}
0.5 & 0\\
0& 0.5
\end{matrix}
\right]$$

- 根据公式$(1.2.3)$，可以直接计算得到模型输出$\mathbf{y}=[y^1,y^2,y^3]^T$ 的多元高斯分布的均值和协方差矩阵：
$$\mathbb{E}[\mathbf{y}]=\Phi\mathbb{E}[\mathbf{w}]=\left[
\begin{matrix}
1\\
2\\
3
\end{matrix}
\right]$$
$$cov[\mathbf{y}]=\frac{1}{2}\Phi\Phi^T=\frac{1}{2}\left[
\begin{matrix}
1+x^1x^1 & 1+x^1x^2 & 1+x^1x^3\\
1+x^1x^2 & 1+x^2x^2 & 1+x^2x^3\\
1+x^1x^3 & 1+x^2x^3 & 1+x^3x^3
\end{matrix}
\right]=\left[
\begin{matrix}
1& 1.5 & 2\\
1.5 & 2.5 & 3.5\\
2 & 3.5 & 5
\end{matrix}
\right]$$
- 根据多元高斯分布的均值和协方差矩阵，可以得出高斯过程在三个位置上的概率分布如图所示（三条红色曲线）,可以看出，$y^3$的方差大于$y^2$的方差大于$y^1$的方差；且由于三个变量的协方差都为正，因此三个变量之间成正相关，即当$y^1$小于均值的时候，$y^2，y^3$都会大概率小于各自的均值（当然这取决于相关性的强弱），如图中$y^1,y^2,y^3$所示为一组可能的采样点。
![img](/img/gaussian_1.png)
### 1.4 高斯过程：直接定义核函数
- 回顾一下前面对模型输出的分布建模的思路：选择basis function $\Phi(x)$，确定参数$w$的先验分布，然后使用线性回归模型$(1.2.1)$定义一个高斯过程；然后得到模型输出 $\mathbf{y}=[y^1,y^2,...,y^m]^T$ 的多元高斯分布。
- 我们也可以不定义basis function，而是直接定义核函数，得到输出的多元高斯分布；
- 常用核函数：
	- 高斯核：
	$$k(x,x')=exp(\frac{-||x-x'||^2}{2\sigma^2})$$
	- 指数核：
	$$k(x,x')=exp(-\theta|x-x'|)$$
- 高斯核描述了数据点之间的相关性强弱，
- 使用高斯核和指数核的模型输出的采样结果如下
![img](/img/gaussian_2.png)
# 2 高斯过程采样的定性分析
- 下面来说说上图的曲线是怎么得到的。仍然考虑简单的场景，假设数据是一维的，仍然假设参数$w$服从$\mathcal{N}(\mathbf{0},\alpha^{-1}I^{n\times n})$的先验分布，且不考虑噪声（noise-free）；
1. 当数据集只有一个数据 $\mathbf{x}=\{x^1\}$时，根据高斯核函数可以直接得到输出的分布:
$$\mathbb{E}[y(\mathbf{x})]=0\quad\quad cov[y(\mathbf{x})]=k(x^1,x^1)$$
然后对输出 $y^1$ 进行采样，如图所示：
![img](/img/gaussian_3.png)
2. 然后数据集中加入第二个点$\mathbf{x}=\{x^1,x^2\}$，根据高斯核函数可以直接得到输出的联合分布：
$$\mathbb{E}[y(\mathbf{x})]=\mathbf{0}\quad\quad cov[y(\mathbf{x})]= \left[
\begin{matrix}
k(x^1,x^1) & k(x^1,x^2)\\
k(x^2,x^1)& k(x^2,x^2)
\end{matrix}
\right]$$

我们得到了输出$y(x^1)$和$y(x^2)$的联合概率分布，该分布取决于高斯核函数的计算结果，即如果$x^1，x^2$相距较近，则$y(x^1)$和$y(x^2)$的相关性更强，则可能得到如下左图的联合分布，如果$x^1，x^2$相距较远，则$y(x^1)$和$y(x^2)$的相关性较弱，则可能得到如下右图的联合分布（值得注意的是，由于高斯核函数计算的对角线元素均为1，因此$y(x^1)$和$y(x^2)$各自的方差是相同的，对应下图的阴影部分在横纵坐标上跨度相同）：
![img](/img/gaussian_4.png)
由于目前已经有了 $y^1$ 的采样结果，再对 $y^2$ 进行采样的时候，就要按照**条件高斯分布**进行采样，如下图所示：
![img](/img/gaussian_5.png)
如果$x^2$和$x^1$距离很近，则会出现上图左图中的情况，相关性很强，即此时的$y(x^2)$极其依赖于$y(x^1)$，进行采样得到 $y^2$ 几乎和 $y^1$大小相同；如果$x^2$和$x^1$距离较远，则会出现上图右图中的情况，相关性较弱，即此时的$y(x^2)$不太依赖于$y(x^1)$，进行采样得到 $y^2$ 和 $y^1$也会有一定差别。反映到输入输出坐标轴上，如下图所示：（注意！！下图中看起来采样点$y^1$似乎等于$y^2$的采样均值，但实际上两者并不相等！具体计算见3.2节，这里是为了展示方便）
![img](/img/gaussian_6.png)

3. 假设我们已经采样完成了如上图左图中的两个点$y^1$ 和 $y^2$，现在我们加入第三个数据，$\mathbf{x}=\{x^1,x^2,x^3\}$，根据高斯核函数可以直接得到输出的联合分布：
$$\mathbb{E}[y(\mathbf{x})]=\mathbf{0}\quad\quad cov[y(\mathbf{x})]= \left[
\begin{matrix}
k(x^1,x^1) & k(x^1,x^2) & k(x^1,x^3) \\
k(x^2,x^1)& k(x^2,x^2) & k(x^2,x^3) \\
k(x^3,x^1)& k(x^3,x^2) & k(x^3,x^3) 
\end{matrix}
\right]$$
如下图所示，黄色椭球代表三个变量的联合分布，根据已有的采样点$y^1$ 和 $y^2$，可以求出 $y^3$的采样范围（图中绿色范围所示）：条件高斯分布$p(y(x^3)|y^1,y^2)$
![img](/img/gaussian_7.png)
根据协方差矩阵，$x^3$ 和 $x^1,x^2$的距离决定了$y(x^3)$和$y(x^1),y(x^2)$的相关性，在输出输出坐标轴上表示（定性表示）如下图所示，左图中，$x^3$与$x^1$，$x^2$的距离都较远，因此相关性较弱，采样范围较大；右图中，$x^3$与$x^1$的距离较远，与$x^2$的距离较近，因此$y(x^3)$和$y(x^1)$相关性弱，与$y(x^2)$相关性强，因此采样范围主要取决于$y^2$的位置；（这里同样的，样本点$y^2$并不等于$y^3$的采样均值，只是为了展示方便）
![img](/img/gaussian_8.png)
4. 目前我们已经得到了三个采样点$\{y^1,y^2,y^3\}$，如果数据集中继续加入元素，将无法用图像直观描述，但是我们可以以此类推，来分析下图（左边的高斯核采样曲线）：（1）当新加入的点 $x^{new}$ 和已有的点 $x^{old}$ 无限接近的时候，则$y(x^{new})$和$y(x^{old})$也将无限接近，这就是为什么我们在采样图（如下图）中看到的采样曲线是连续的平滑的；（2）下图的曲线实际上是在x轴上无限采样的结果，实际上是离散的，而不是连续的；（3）新加入的点与离其最近的点集相关性最强，例如如果最近的点集的输出是单调递增，则新加入的点的输出也大概率满足单调递增；

![img](/img/gaussian_9.png)
# 3 高斯回归
### 3.1 对模型输出添加噪声
- 在前面的定性分析中，我们描述了模型输出 $\mathbf{y}=[y^1,y^2,...,y^m]^T$的联合概率分布，没有考虑噪声，而在回归问题中，训练集中的标签值 $\mathbf{t}$ 都是带有固定噪声的，因此我们需要将噪声 $\epsilon$ 加入到模型的输出 $\mathbf{y}$ 中：
$$t^i=y(x^i)+\epsilon^i \quad for \quad i\in\{1,2,...,m\}$$
其中：$\epsilon^i\sim\mathcal{N}(0,\beta^{-1})$
因此，可以得到：
$$p(\mathbf{t}|\mathbf{y})=\mathcal{N}(\mathbf{t}|\mathbf{y},\beta^{-1}I_{m})$$
又根据模型高斯过程的定义：
$$p(\mathbf{y})=\mathcal{N}(\mathbf{y}|\mathbf{0},\mathbf{K}_m)$$
 因此，标签值$\mathbf{t}$的联合概率分布为：
 $$p(\mathbf{t})=\int p(\mathbf{t}|\mathbf{y})p(\mathbf{y})d\mathbf{y}=\mathcal{N}(\mathbf{t}|\mathbf{0},\mathbf{C}_m)$$
 其中：$C(x^i,x^j)=k(x^i,x^j)+\beta^{-1}\delta_{ij}$，即：
 $$\mathbf{C}_m=\left[
\begin{matrix}
k(x^1,x^1) & ... & k(x^1,x^m) \\
...& ... & ... \\
k(x^m,x^1)& ... & k(x^m,x^m) 
\end{matrix}
\right]+\left[
\begin{matrix}
\beta^{-1} & ... & 0 \\
...& ... & ... \\
0 & ... & \beta^{-1} 
\end{matrix}
\right]$$
这样我们就得到了数据集中的==标签值$\mathbf{t}$的联合概率分布==；
- 下图蓝色的曲线为高斯过程采样的结果，红色的点为数据集中的点 $\mathbf{x}$ 对应的模型输出 $\mathbf{y}$，绿色的点为加入独立噪声$\epsilon$的结果 $\mathbf{t}$：
![img](/img/gaussian_10.png)
### 3.2 高斯回归
- 问题定义：给定数据集$\mathbf{x}=\{x^1,x^2,...,x^m\}$，$\mathbf{t}=\{t^1,t^2,...,t^m\}$和新的测试数据 $x^{m+1}$，预测 $t^{m+1}$的分布，即计算$p(t^{m+1}|\mathbf{x},\mathbf{t},x^{m+1})$；
- 高斯回归过程：
1. 计算目标值$\{t^1,t^2,...,t^m,t^{m+1}\}$的联合概率分布：
$$\left[
\begin{matrix}
t^1 \\
...\\
t^m\\
t^{m+1}
\end{matrix}
\right]\sim\mathcal{N}(\mathbf{0},\mathbf{C}_{m+1})$$
协方差矩阵$\mathbf{C}_{m+1}$可分解为如下部分：
$$\mathbf{C}_{m+1}=\left[
\begin{matrix}
\mathbf{C}_{m}&\mathbf{k} \\
\mathbf{k}^T& c
\end{matrix}
\right]$$
2. 条件高斯分布$p(t^{m+1}|\mathbf{x},\mathbf{t},x^{m+1})$便可以由下式计算：
$$p(t^{m+1}|\mathbf{x},\mathbf{t},x^{m+1})=\mathcal{N}(t^{m+1}|\mu_{m+1},\sigma_{m+1})$$
其中输出的均值和方差可由下式给出：
$$\mu_{m+1}=\mathbf{k}^T\mathbf{C}_{m}^{-1}\mathbf{t}$$
$$\sigma_{m+1}=c-\mathbf{k}^T\mathbf{C}_{m}^{-1}\mathbf{k}$$
- 上式放在二维场景下，就是计算下图中的绿色曲线的过程。（可以看出$t^2$的采样均值并不等于$t^1$）
![img](/img/gaussian_11.png)
- 在[贝叶斯线性回归](https://blog.csdn.net/qq_36013249/article/details/105932395)中，我们曾经提到“==预测分布的均值是训练集中每个标签值的线性组合==”，高斯过程预测的均值 $\mu_{m+1}$，也是标签值$\{t^1,t^2,...,t^m\}$的线性组合：
$$\mu_{m+1}=\mathbf{k}^T\mathbf{C}_{m}^{-1}\mathbf{t}=\left[
\begin{matrix}
k(x^1,x^{m+1}) & ... &k(x^m,x^{m+1})
\end{matrix}
\right]
\left[
\begin{matrix}
k(x^1,x^1) & ... & k(x^1,x^m) \\
...& ... & ... \\
k(x^m,x^1)& ... & k(x^m,x^m) 
\end{matrix}
\right]^{-1}\cdot\left[
\begin{matrix}
t^1 \\
...\\
t^m
\end{matrix}
\right]$$
其中各项组合的权重是由核函数计算的，且该权重是新输入测试数据$x^{m+1}$的函数，即$\mu_{m+1}$还可以写成如下形式：
$$\mu_{m+1}=f(x^{m+1})=\sum^m_{i=1}a_i k(x_i,x_{m+1})$$
其中 $a_i$是 $\mathbf{C}_{m}^{-1}\mathbf{t}$ 中的第 $i$ 个元素；
### 3.3 超参数的选择
- 高斯过程的超参数包括噪声误差精度$\beta$，还有核函数中的参数，这些参数的选择方法是通过**极大似然估计**，具体过程省略。
# 4 极大似然线性回归、贝叶斯线性回归、高斯回归的区别和联系

|  | 极大似然线性回归 |贝叶斯线性回归| 高斯回归|
|--|--|--|--|
|研究空间  | 参数空间 |参数空间|函数空间|
|使用方法|极大似然估计|极大后验估计|高斯过程|


### 4.1 极大似然线性回归
- 我们之前最熟悉的是[极大似然线性回归](https://blog.csdn.net/qq_36013249/article/details/105903452)，代表频率学派的观点，其目的是在参数空间中寻找一组能拟合当前训练集的最优参数，学习到参数之后，使用判别式模型，预测新数据；但是当拿来一组新的训练集时，就要重新学习参数；
### 4.2 贝叶斯线性回归
- 贝叶斯线性回归代表贝叶斯学派的观点，其目的并不是寻找能拟合训练集的最优参数，而是认为，参数没有“最优”，只有“更优”；贝叶斯线性回归认为参数存在先验，学习的过程就是不断计算参数的后验分布，然后把计算得到的后验分布作为下一次学习的先验分布的过程，每次拿到一组新的训练集，都能在原来的基础上继续学习，有一种“学习永不停止”的意思；与极大似然回归相比，其学习的结果不是一组确定的参数，而是参数的（后验）分布，用这组不确定的参数对新数据进行预测时，得到的预测结果也是不确定的，即得到的预测结果也是一组分布，而不是某个确定的预测值。
### 4.3 高斯回归
- 前两种回归方法都是在参数空间中对参数进行估计，这样的弊端在于必须自己选择模型（basis-function），有时模型的选择会对预测结果产生较大的影响，而如何选择模型则完全靠经验。高斯回归直接放弃对模型的选择，而是相信一个“真理”：我只要用和新数据相近的训练集数据预测输出，结果肯定大差不差。即使用核方法对训练集中的目标值的联合概率分布进行建模，当输入新的数据时，使用贝叶斯公式得出预测值的条件高斯分布。预测结果中除了核函数的超参数外不包含任何需要学习的参数。