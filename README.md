基于梯度搜索的自适应算法原理及实现

序、论文写作背景

&emsp;    作为电子信息专业的学生，笔者对信号处理相关理论十分感兴趣。试想，含有各种噪声的信号，在经过数学家们的理论分析和工程师们的软硬件实现，噪声能被几乎完全消除，同时解析出完美的有效信号，高效率地完成去噪过程。在*Machine Learning*这一课程的学习过程中，笔者接触到了LMS*(Least mean squares)*算法，在使用LMS算法进行过一些简单的信号去噪应用后，笔者为这一性能优异的自适应算法吸引，本着爱屋及乌的学习态度及生活精神，又对自适应系统及自适应信号处理进行了较深的了解。本文还将介绍其他基本自适应算法，如与LMS同样使用梯度搜索方法的RLS*(Recursive least squares)*算法，和基于随机搜索的遗传优化算法*(Genetic optimizer algorithm)* 。本文对于LMS和RLS算法进行了较为详细的介绍，并使用MATLAB进行了性能测试，对于遗传优化算法仅给出了理论介绍和推导。在第三部分LMS算法原理阐述及代码实现部分，为了让公式推导及过程起来不那么枯燥，在每一个算法数学公式下，都附上了用代码实现的过程。在最后，通过实现一个

一、自适应系统及滤波器简介

​  &emsp;  自适应系统*(Adaptive System)*是一类结构可变或可以调整的系统，通过自身与外界环境的接触，改善自身对信号处理的性能。这类系统通常为随时间变化的非线性系统，可以自动适应信号传送变化的环境和要求，无需详细知道信号结构和信号的实际知识，无需精确设计信号处理系统本身。

​  &emsp;  对于一个自适应滤波器来说，如果输入仅包含有效信号，没有噪声存在，那么这个滤波器仅仅是一个全通滤波器，对信号没有任何影响；若该输入中含有噪声，这个滤波器可以变成一个带通或带阻滤波器。自适应系统的时变特性往往由其‘’自适应学习‘’过程确定，当学习过程结束，系统不再调整，系统参数不再变化后，有一类自适应系统可成为线性系统。

​ &emsp;   在滤波方面，经典的滤波算法包括维纳滤波，卡尔曼滤波，这些滤波算法都需要对输入信号的相关系数，噪声功率等参数进行估计，而实际中很难实现这些参数的准确估计，而这些参数的准确估计直接影响到滤波器的滤波效果。另一方面，这两类滤波器一般设计完成，参数便不可改变，实际应用中，希望滤波器的参数能够随着输入信号的变化而改变，以取得较好的实时性处理效果。此时，自适应滤波的出现可弥补上述传统算法的不足，同时较好地满足信号处理的要求。

​   &emsp;对于一个闭环自适应滤波器来说，它不仅可以通过测得的信息来形成特有的传递函数系数，还可以通过反馈通道，根据某一时刻滤波器的输出与期望信号的误差调整滤波器的系数，从而实现滤波器系数的动态调整。

> Generally speaking, the closed loop adaptive process involves the use of a cost function which is a criterion for optimum performance of the filter, to feed an algorithm, which determines how to modify filter transfer function to minimize the cost on the next iteration. The most common cost function is the mean square of the error signal.
>
> 引用自https://en.wikipedia.org/wiki/Adaptive_filter

  

二、均方误差性能测度(MSE)——LMS算法测度基础

​      &emsp; 首先介绍MSE的大致概念，可见如下维基百科定义:

> In statistics, the Mean Squared Error (MSE) or Mean Squared Deviation (MSD) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors—that is, the average squared difference between the estimated values and what is estimated. MSE is a risk function, corresponding to the expected value of the squared error loss. The fact that MSE is almost always strictly positive (and not zero) is because of randomness or because the estimator does not account for information that could produce a more accurate estimate.
>
> The MSE is a measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.
>
> The MSE is the second moment (about the origin) of the error, and thus incorporates both the variance of the estimator (how widely spread the estimates are from one data sample to another) and its bias (how far off the average estimated value is from the truth). For an unbiased estimator, the MSE is the variance of the estimator. Like the variance, MSE has the same units of measurement as the square of the quantity being estimated. In an analogy to standard deviation, taking the square root of MSE yields the root-mean-square error or root-mean-square deviation (RMSE or RMSD), which has the same units as the quantity being estimated; for an unbiased estimator, the RMSE is the square root of the variance, known as the standard error.
>
> 引用自https://en.wikipedia.org/wiki/Mean_squared_error

&emsp;    均方误差性能测度适用于一个总的系统输出为期待响应和自适应系统的输出之差，对于第三部分中将要研究的系统，可绘制简单示意图如下:![1](图片/1.png)

定义期待响应为$d_k$,希望系统输出$y_k$与之接近，并且在k时刻，系统输出误差为:

$$\varepsilon =d_k - y_k\tag{1}​$$

输入经过加权后，系统输出为:

$$y_k = W_kX^T_k\tag{2}$$

其中

$$X_k = [x_{0k}  x_{1k}  ……  x_{Lk}]\tag{3}$$

$$W_k = [w_{0k} w_{1k} …… w_{Lk}]\tag{4}$$

故名思意，”最小均方误差测度“即，输出误差模值平方达到最小时，系统性能达到最佳，故有下式:

$$E(|\varepsilon_k|^2) = E[(d_k - y_k)^*(d_k - y_k)] \\= E(|d_k|^2)+W_k^HE[X^*_kX^T_k]W_k-2Re(W^T_kE[d^*_kX_k])\tag{5}$$

观察到上式中仍含有两个期望值，现定义输入信号自相关矩阵$R$及输入信号与期待响应互相关矩阵$P$如下:

$$R = E[X^*_kX^T_k]\tag{6}$$

$$P = E[d^*_kX_k]\tag{7}$$

由式(5)(6)(7)可得系统性能函数$$\xi(W) =E[(d_k - y_k)^*(d_k - y_k)] \\ = E(|d_k|^2)+W_k^HRW_k-2Re(W^T_kR)   \tag{8}$$

由上式可知系统性能函数为系数矩阵(权值向量)的二次函数。根据采用的MSE测量方法，系统自适应调整权值向量，以使性能函数$\xi(W)$达到最小值，此时得到最佳权向量$W_{opt}$，性能函数输出$\xi(W) = \xi_{min}$ 。可用梯度下降的方式求得这个性能函数最小值，梯度向量位于权向量$W$构成的空间之内。在性能表面上任取一点，若其梯度不为零，其负梯度方向为性能表面在该处最陡下降时权向量的变化方向；当某一点梯度为零时，权向量$W = W_{opt}$，得到最佳系数矩阵。下面使用数学关系式进行推导,

性能函数梯度为，

$\nabla =  \dfrac{\partial}{\partial W}[\xi(W)] = 2RW - 2P^*\tag{9}$

在最佳权向量处梯度值$\nabla$为零，

$$\nabla = 2RW_{opt} - 2P^* = 0\tag{10}$$

$$W_{opt} =R^{-1} P^* \tag{11}$$

将式(11)带入式(8)，

$$\xi_{min} = E(|d_k|^2) + P^TR^{-1}RW_{opt} -2W^T_{opt}P \\ E(|d_k|^2) -P^TW_{opt} \tag{12}$$

输入有噪的情况下，$\xi_{min} \neq 0 $，输出与期待响应总有微小的不同，这一点也将在第三部分算法实现中以矩阵和图像的形式显示出来。

三、LMS算法原理及MATLAB实现

1.LMS算法原理

&emsp;&emsp;为了让公式推导及过程

&emsp;&emsp;与一般梯度估值的自适应算法不同，LMS简单直接地采用单次采样数据$|\varepsilon_k|^2$来代替均方误差$\xi_k$，故有下式

$$\nabla_k = \dfrac{\partial}{\partial W_k}|\varepsilon_k|^2\\=\dfrac{\partial}{\partial W_k}[|d_k|^2 + W^H_kX^*_kX^T_kW_k - 2Re(d_kX^*_k)]\\2X^*_kX^T_kW_k - 2d_kX^*_k = -2\varepsilon_kX^*_k\tag{13}$$

LMS算法采用这个简单地梯度估值，可以得到其数学表示式，也即其权值迭代公式，

$$W_{k+1} = W_k = \mu\nabla_k\\=W_k + 2\mu\varepsilon_kX^*_k\tag{14}$$

以下为其代码实现:

```matlab
W(:,k) = W(:,k-1) + 2*mu*en(k)*x;
```

其中$\mu$为用于控制自适应速度和稳定性的增益常数，经过推导和分析可知其取值范围，

$$0 < \mu <\dfrac{1}{\lambda_{max}}\tag{15}$$

其中$\lambda_{max}$为信号自相关相关矩阵$R$的最大特征值。

```
R = y*y.';                   %输入相关矩阵
lambda_max = max(eig(R));    %输入信号相关矩阵的最大特征值
```

注意到$\lambda_{max}$一般不会大于输入自相关矩阵的迹$tr(R)$，故如下取$\mu$的值可更好地保证权向量的收敛性,

$$0 < \mu <\dfrac{1}{tr(R)}\tag{16}$$

```
lambda_max2 = tr(R)
```

可由式(14)画出简单的LMS算法实现图，![2](图片/2.png)

由图可知，该算法计算简便，效率较高。梯度向量的每个分量由单个数据样本得到，由此可知式(13)中梯度的估值包含了输入的噪声成分，但是在自适应过程中，随着权值的不断优化，该噪声成分会逐渐衰减。

LMS算法具体实现代码如下,其余解释已在注释中标明，

```matlab
function [yn,W,en]=LMS(xn,dn,N,mu,itr)
% LMS(Least Mean Squre)算法
% 输入参数:
%     xn   输入信号序列      
%     dn   期望响应序列       
%     N    滤波器的阶数        
%     mu   收敛因子/步长           
%     itr  迭代次数            
% 输出参数:
%     W    滤波器的权值矩阵;大小为N x itr,
%     en   误差序列  
%     yn   实际输出序列
% 初始化参数
itr = length(xn);              %迭代次数默认为xn的长度，尽量取大一些，保证收敛
en = zeros(itr,1);             % 误差序列,en(k)表示第k次迭代时预期输出与实际输入的误差
W  = zeros(M,itr);             % 每一行代表一个加权参量,每一列存储一次迭代结果,初始为0
% 迭代计算
for k = N:itr                  % 第k次迭代
    x = xn(k:-1:k-N+1);        % 滤波器M个抽头的输入
    y = W(:,k-1).' * x;        % 滤波器的输出
    en(k) = dn(k) - y ;        % 第k次迭代的误差
    % 滤波器权值计算的迭代式
    W(:,k) = W(:,k-1) + 2*mu*en(k)*x;
end
% 求最优时滤波器的输出序列
yn = inf * ones(size(xn));
for k = N:length(xn)
    x = xn(k:-1:k-N+1);
    yn(k) = W(:,end).'* x;
end
```



2.算法实现及效果检测

1.处理被高斯白噪声污染的正弦信号(注释丰富)

```matlab
%LMS main()
close all;clc;clear;
%洁净信号产生
t=0:399;
S=10*sin(0.3*t);
%加噪生成输入信号X 
X = awgn(S,1)%加入噪声，SNR = 1
% 自适应滤波
X = X.' ;   % 输入信号序列
S = S.' ;   % 预期结果序列
N  = 50 ; % 滤波器的阶数
R = X*X.'; %输入信号自相关矩阵
lambda_max = max(eig(R));   % 收敛因子的上界：输入信号相关矩阵的最大特征值
lambda_max2 = trace(R);       % 进一步保证收敛
mu1 = 0.1*(1/lambda_max)   ; % 收敛因子 0 < mu < 1/lambda_max
%mu2= 0.1*(1/lambda_max2) ; 比mu1更加保守的收敛因子取值
%调用函数计算最佳权向量及滤波结果
for i = 1:100% 循环原因:每一次加噪效果不同,循环一百次以取统计平均
    X = awgn(S,1)%加入噪声，SNR = 1
    [ynb,Wb,enb] = LMS(X,S,N,mu1);%输出 yn,W,en
    A(:,i) = ynb% A维度100*100存储矩阵 ； ynb为100*1行向量 ，
end
Y = mean(A,2);%矩阵按行求均值
%%
%期望信号与含噪信号
figure(1);
subplot(2,1,1);plot(t,S);grid;axis([0 length(t) -20 20]);
title('洁净信号S/期望信号d');%Desired signal
subplot(2,1,2);plot(t,X);grid;axis([0 length(t) -20 20])
title('滤波器输入信号(含噪声)');
% 绘制滤波器输入、输出信号
figure(2);
subplot(2,1,1);plot(t,X);grid;axis([0 length(t) -20 20])
title('滤波器输入信号');
subplot(2,1,2);plot(t,Y);grid;axis([0 length(t) -20 20])
title('自适应滤波器输出信号');
% 绘制自适应滤波器输出信号+预期输出信号+两者的误差
figure(3); 
plot(t,Y,'b',t,S,'g',t,S-Y,'r');grid;axis([0 length(t) -20 20])
legend('自适应滤波器输出','预期输出','误差');
ylabel('幅值');
xlabel('时间');
title('自适应滤波器效果');
```

效果如下图:


2.处理被两个高频余弦信号污染的低频正弦信号

绘图部分与第一部分代码相同，不再给出

```
%洁净信号产生
t=0:399;
S=10*sin(0.3*t);
%加噪生成输入信号X 
X = S + 8*cos(600*t)+7*sin(215*t);%加入噪声，SNR = 1
%%
% 自适应滤波
X = X.' ;   % 输入信号序列
S = S.' ;   % 预期结果序列
N  = 50 ; % 滤波器的阶数
R = S*S.'; %输入信号自相关矩阵
lambda_max = max(eig(R));   % 收敛因子的上界：输入信号相关矩阵的最大特征值
mu = 0.1*(1/lambda_max)   ; 
%调用函数计算最佳权向量及滤波结果
[Y,W,en] = LMS(X,S,N,mu);
```

效果如下图:
