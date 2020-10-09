---
layout:     post
title:      "极大似然估计解决线性回归问题（频率派）"
subtitle:   "lets begin"
date:       2020-05-03
author:     "CT"
header-img: "img/blog-bg.jpg"
tags:
    - 机器学习
---
# 1 前言
- 之前一篇文章[LR推导](https://blog.csdn.net/qq_36013249/article/details/105903536)讲了如何使用极大似然估计推导逻辑回归的损失函数：交叉熵 ；
- 本文讲解如何使用极大似然估计推导线性回归的损失函数：最小二乘。
- 本文参考自PRML
# 2 问题定义
- 给定数据集$\{x_1,x_2, ...,x_m\}$和标签$\{y_1,y_2, ...,y_m\}$，训练一个模型$y(x,\theta)$，使得输入新的 $x$，输出对应的预测值。其中 $x_i\in R^n$，标签 $y_i\in R$；
![img](/img/regression_1.png)

# 3 建立判别模型
- 建立一个判别模型，使得输入一个数据 $x$，输出所有可能的 $y$ 值对应的概率，如果 $y$ 值连续，则输出 $y$ 值的概率密度函数。
- 我们使用线性高斯分布（Linear Gaussian）来建立该判别模型：
$$p(y|x,\theta,\beta)=\mathcal{N}(y|\theta^Tx,\beta^{-1})$$
其中，$\theta^Tx$表示线性模型，也可以是其他形式的线性组合（注意，这里的线性指的是对参数$\theta$线性），$\beta$表示精度（方差的倒数）；
# 4 建立似然函数
$$L(\theta,\beta)=\prod_{i=1}^m\mathcal{N}(y_i|\theta^Tx_i,\beta^{-1})$$
- 对数似然：
$$lnL(\theta,\beta) = \frac{N}{2}ln\beta-\frac{N}{2}ln2\pi-\beta E_D(\theta)$$
其中：$E_D(\theta)=\frac{1}{2}\sum_{i=1}^m[y_i-\theta^Tx_i]^2$
- 可以看出：==极大似然就是最小二乘==，同样的在逻辑回归问题中：==极大似然就是最小化交叉熵==
- 求导：
$$\bigtriangledown lnL(\theta,\beta)=\sum_{i=1}^m[y_i-\theta^Tx_i]x_i$$
- 令导数等于0，求得：
$$\theta_{ML}=(X^TX)^{-1}X^TY$$
其中，
$$X=
\left(
\begin{matrix}
x^1_1  & ... & x^1_n\\
... & &...\\
x^m_1 & ... & x^m_n
\end{matrix}
\right)_{m\times n}$$
$$Y=
\left(
\begin{matrix}
y_1\\
...\\
y_m
\end{matrix}
\right)_{m\times 1}$$
$m$为样本个数，$n$为样本特征数；
# 5 极大似然和最小二乘的关系
- 在线性回归问题中： 极大似然就是最小二乘；
- 极大似然估计的计算结果$\theta_{ML}=(X^TX)^{-1}X^TY$，就是最小二乘法求解线性回归问题的解，投影矩阵$P = (X^TX)^{-1}X^T$，$\theta_{ML}$就是向量$Y$向$X$张成的子空间的投影，但实际计算的时候一般使用**梯度下降法**；
- 此外，极大似然估计还能估计参数 $\beta$ 的结果。通过最小二乘损失函数学习到的模型，使用时输入一个 $x$，只能输出一个$y(x,\theta)$，但是通过线性高斯建立的判别模型，使用极大似然估计可以学习到两个参数$\theta,\beta$，使用时输入一个$x$，会输出一个条件高斯分布 $p(y|x,\theta,\beta)$，可以得到所有可能估计值y的概率分布；
### 普通预测模型：
![img](/img/regression_2.png)
### 线性高斯预测模型：
![img](/img/regression_3.png)