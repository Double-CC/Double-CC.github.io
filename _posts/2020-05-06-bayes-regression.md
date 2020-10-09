---
layout:     post
title:      "贝叶斯线性回归（贝叶斯学派）"
subtitle:   "lets begin"
date:       2020-05-06
author:     "CT"
header-img: "img/blog-bg.jpg"
tags:
    - 机器学习
---

# 1 前言
- 极大似然线性回归中，我们使用线性高斯判别模型，似然函数为（注意大写的$X,Y$表示训练集）：
$$p(Y|X,\theta,\beta)=\prod_{i=1}^n\mathcal{N}(y_i|\theta^T\Phi (x_i),\beta^{-1})$$
其中$\Phi (x_i)$表示**basis-function**，$\Phi(x_i)\in R^{n},\theta\in R^{n},\beta\in R^{n\times n}$；
![img](/img/bayes_1.png)
- 频率学派认为参数$\theta$是固定不变的，一个数据集可以训练出一套参数，如果换一个数据集，就会训练出另一套参数，因此，参数是随着数据集的变化而变化的。因此**模型的不确定性**就用**不同数据集之间的差别**来表示，我们可以基于不同数据集进行偏差-方差分解（Bias-Variance Decomposition），来研究模型的性能，如图所示（图片取自PRML），左列 代表在不同正则化权重下，从20个不同数据集学习到的20条曲线，右列 红色代表这20条曲线的均值，绿色代表目标曲线。可以看出基于不同数据集的方差越大，偏差就越小；反之方差越小，偏差就越大，因此需要权衡两者之间的关系，找到最优的$\lambda$值。
![img](/img/bayes_2.png)

# 2 贝叶斯线性回归
### 2.1 参数$\theta$的最大后验估计MAP
- 贝叶斯学派认为参数 $\theta$ 是个变量，仍然使用线性高斯作为似然函数：
$$p(Y|X,\theta,\beta)=\prod_{i=1}^n\mathcal{N}(y_i|\theta^T\Phi (x_i),\beta^{-1})$$
其中，$\Phi(x_i)\in R^{n},\theta\in R^{n},\beta\in R^{n\times n}$；
现在，我们的目标不再是最大化似然，而是最大化后验概率：
$$p(\theta|Y)=\frac{p(Y|\theta)p(\theta)}{p(Y)}$$
由于似然函数为高斯分布，其共轭先验为高斯分布，因此假设先验分布服从高斯分布：
$$p(\theta)=\mathcal{N}(\theta|\mu_0,S_0)$$
均值为$\mu_0\in R^n$，协方差为$S_0=\alpha^{-1}I\in R^{n\times n}$;
- 则后验概率仍服从高斯分布（根据共轭分布定理）：
$$p(\theta|Y)\approx p(Y|\theta)p(\theta)=\mathcal{N}(\theta|\mu_m,S_m)$$
其中$\mu_m$代表经过容量为m的数据集训练得到的参数后验分布均值，$S_m$代表经过容量为m的数据集训练得到的参数后验分布的协方差矩阵：
$$\mu_m=S_m(S_0^{-1}\mu_0+\beta\Phi^TY)$$
$$S_m^{-1}=S_0^{-1}+\beta\Phi^T\Phi$$
其中，$\Phi=\left(
\begin{matrix}
\Phi(x_1)^T\\
...\\
\Phi(x_m)^T
\end{matrix}
\right)_{m\times n}$，m为数据集容量，n为basis-function得到的特征数；
- 因此==参数 $\theta$ 最大后验估计$\theta_{MAP}$就是 $\mu_m$==
### 分析
- 当先验概率的$S_0=\alpha^{-1}I$的 $\alpha\rightarrow0$时，最大后验估计退化为极大似然估计:
$$\theta_{MAP}=\theta_{ML}=(\Phi^T\Phi)^{-1}\Phi^TY$$
- 当$m=0$时，即不使用训练集训练，后验概率退化为先验概率；
- 如果有多个数据集，则上一个数据集学习到的$\theta$的极大后验估计可以作为下一个数据集的先验估计；
### 初始状态：均值为0的先验估计
- 如果没有任何数据集，那我们假设参数 $\theta$ 服从均值为0，协方差矩阵为$\alpha^{-1}I$的高斯分布：
$$p(\theta|\alpha)=\mathcal{N}(\theta|0,\alpha^{-1}I)$$
- 此时后验概率的均值$\mu_m$和协方差$S_m$为：
$$p(\theta|Y)\approx p(Y|\theta)p(\theta)=\mathcal{N}(\theta|\mu_m,S_m)$$
$$\mu_m=S_m\beta\Phi^TY$$
$$S_m^{-1}=\alpha I+\beta\Phi^T\Phi$$
- 对数后验：
$$lnp(\theta|Y)=-\frac{\beta}{2}\sum_{i=1}^m[y_i-\theta^T\Phi(x_i)]^2-\frac{\alpha}{2}\theta^T\theta+const$$
- 可以看出，==最大后验就是最小二乘加平方正则化项==，其中正则化项权重 $\lambda=\alpha/\beta$
### 贝叶斯线性回归的学习过程
- 增量式的学习：每行都引入新的数据，每行得到的后验估计作为下一行的先验，随着数据集的增大，参数的后验概率估计趋于收缩，如果数据集无限大，最终会收敛到准确的参数（图中的十字点）
![img](/img/bayes_3.png)
### 2.2 预测分布 (Predictive Distribution)
- 使用贝叶斯线性回归得到的参数$\theta$是不确定的，服从一个后验概率分布，因此对新的数据 $x$ 进行预测得到的预测结果 $y$ 也是不确定的，也服从一个分布(注意这里的$x$表示新数据，$X,Y$表示训练集中的数据)：
$$p(y|x,Y,\alpha,\beta)=\int p(y|x,\theta,\beta)p(\theta|Y,\alpha,\beta)d\theta$$
其中$p(\theta|Y,\alpha,\beta)$就是后验概率分布，$p(y|x,\theta,\beta)=\mathcal{N}(y|\theta^T\Phi(x),\beta^{-1})$为判别模型；
- 因此，预测分布的计算结果：
$$p(y|x,Y,\alpha,\beta)=\mathcal{N}(y|\mu_m^T\Phi(x),\sigma_m^2(x))$$
其中：
$$\sigma_m^2(x)=\frac{1}{\beta}+\Phi(x)^TS_m\Phi(x)$$
第一项表示数据的噪声，这里 $1/\beta\in R^{1}$ (单输出)；第二项表示参数$\theta$的不确定性；
- 随着数据集的增加，预测分布的均值（红线）和方差（红色阴影）变化：
![img](/img/bayes_4.png)
### 2.3 Equivalent kernel 核方法
- 上节得出了对于新数据 $x$ 的预测结果 $y$ 的概率分布：$p(y|x,Y,\alpha,\beta)$，其均值（即上图中的红线）可以写为：
$$y(x,\mu_m)=\mu_m^T\Phi(x)=\beta\Phi(x)^TS_m\Phi^TY=\sum_{i=1}^m\beta\Phi(x)^TS_m\Phi(x_i)y_i$$
其中，$\Phi=\left(
\begin{matrix}
\Phi(x_1)^T\\
...\\
\Phi(x_m)^T
\end{matrix}
\right)_{m\times n}$,$\mu_m$是参数$\theta$的后验估计均值；
- 可以发现==预测分布 $y$ 的均值是训练集中每个标签值的线性组合==：
$$y(x,\mu_m)=\sum_{i=1}^mk(x,x_i)y_i$$
其中$x_i,y_i$是训练集中的数据，$x$是新输入的数据，$k(x,x_i)$称为==核==：
$$k(x,x_i)=\beta\Phi(x)^TS_m\Phi(x_i)$$
- 因此，给定一个数据集，我们可以通过定义核函数 $k(x,x')$，来直接对新输入的数据$x$进行预测，得到预测分布的均值，而不需要选择**basis-function**。
![img](/img/bayes_5.png)
- 由上图可以看出，三个输入数据所对应的核函数，都是在该输入数据的位置附近权重最高，在远离输入数据的位置权重很低，可以这样理解：==使用贝叶斯线性回归对新的输入数据进行预测时，预测结果的均值就是训练集中所有的标签值的带权线性组合，而核函数则描述了该线性组合的权重分布情况，对于一个输入数据$x$，核函数希望他更多的使用（更信任）数据集中和$x$相近的数据来得出它的预测结果==；这也符合我们的预测逻辑：例如你要对自己的人生进行一个预测，你肯定会去关注跟你特质比较像的人的人生是什么样子，而不是去参考那些跟你差别很大的人，核函数就是起到了这样一个作用，它会告诉你关注谁更多一点，谁更少一点。