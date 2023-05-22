# Table of Contents <!-- omit in toc -->
- [1. Delta Method](#1-delta-method)
  - [1.1. Univariate](#11-univariate)
  - [1.2. Multivariate](#12-multivariate)
    - [1.2.1. Variance property](#121-variance-property)
- [2. Ratio metrics](#2-ratio-metrics)
- [3. CUPED](#3-cuped)
  - [3.1. Count metrics](#31-count-metrics)
  - [3.2. Points to note](#32-points-to-note)
  - [3.3. CUPED on Ratio metrics](#33-cuped-on-ratio-metrics)
- [4. CUPED vs Other Methods](#4-cuped-vs-other-methods)
- [Misc. notes](#misc-notes)
  - [Ratio-of-averages v. Average-of-ratios](#ratio-of-averages-v-average-of-ratios)

# 1. Delta Method

## 1.1. Univariate
From the central limit theorem, the random variables such as the sample average are asymptotically normal. Thus if $X_{n}$ is a sequence of averages where $n\to\infty$:

```math
\begin{align*}
X_{n}   &\sim {\sf N}(\mu, \frac{\sigma^{2}}{n})\\
X_{n}-\mu   &\sim {\sf N}(0, \frac{\sigma^{2}}{n})\\
X_{n}-\mu   &\sim \frac{\sigma}{\sqrt{n}}{\sf N}(0, 1)\\
\frac{\sqrt{n}(X_{n}-\mu)}{\sigma}   &\sim {\sf N}(0, 1)\\

\end{align*}
```


The Delta method then gives that for any continuous transformation $g(x)$:

```math
\begin{align*}
\frac{\sqrt{n}(g(X_{n})-g(\mu))}{|g'(\mu)|\sigma}   &\sim {\sf N}(0, 1)\\
g(X_{n})  &\sim {\sf N} \left( g(\mu), \frac{g'(\mu)^{2}\sigma^{2}}{n} \right) \\

\end{align*}
```


The above applies because from the Taylor series:

```math
\begin{align*}
g(X_{n}) &= g(\mu) + \frac{g'(\mu)}{1!}(X_{n}-\mu) + \frac{g''(\mu)}{2!}(X_{n}-\mu)^{2} + \dots\\
\frac{g(X_{n})-g(\mu)}{g'(\mu)} &\approx (X_{n}-\mu)\\
\frac{\sqrt{n}(g(X_{n})-g(\mu))}{g'(\mu)\sigma} &\approx \frac{\sqrt{n}}{\sigma}(X_{n}-\mu) \sim {\sf N}(0, 1)\\
\end{align*}
```

---



## 1.2. Multivariate
Similarly, for the multivariate case:

```math
\frac{\sqrt{n}(B-\beta)}{\Sigma} \sim {\sf N}(0,1)
```
where:
* $B$: vector of $k$ random variables
* $\beta$: vector of $k$ constants
* $\Sigma$: covariance matrix of $B$

For any continuous function, $h(B)$, first order Taylor series implies:

```math
h(B) \approx h(\beta) + \nabla{h(\beta)}^{T}(B-\beta)
```

where:
* $\nabla h(B)$: gradient; i.e., vector of partial differentiation of $h(B)$ by each element of $B$

The variance of both sides should also be equal:

```math
var(h(B)) = var(\nabla{h(\beta)}^{T}B)\\
```

$\because$ all other terms on the _rhs_ are constants. Then by variance property:

```math
\begin{align*}
var(h(B)) &= \nabla{h(\beta)}^{T}var(B)\nabla{h(\beta)}\\
&= \nabla{h(\beta)}^{T} \Sigma \nabla{h(\beta)}\\
\end{align*}
```

---


### 1.2.1. Variance property
Variance when multiplying scalar with random variable.

```math
\begin{align*}
var(\sum_{i}{a_{i}X_{i}}) &\\
&= \sum_{i,j}{a_{i}a_{j}cov(x_{i},x_{j})}\\
&= \sum_{i}{a_{i}^{2}var(x_{i})} + \sum_{i,j, i \neq j}{a_{i}a_{j}cov(x_{i},x_{j})}\\
&= \sum_{i}{a_{i}^{2}var(x_{i})} + 2\sum_{i<j<N}{a_{i}a_{j}cov(x_{i},x_{j})}\\
\end{align*}
```

Equivalently, in matrix notation:

```math
var(a^{T}X) = a^{T} \Sigma a
```

___
___




# 2. Ratio metrics
Calculating the treatment effect with the Delta method

* Random variables: $B = [\bar{Y}, \bar{N}]$ (thus, $k=2$)
* Let mean vector: $\beta = [\mu_{Y}, \mu_{N}]$
* Covariance: $\Sigma$

Averages of randomization unit level quantities (which are $\therefore$ i.i.d):
* $\bar{Y} = \frac{\sum_{i}{Y_{i+}}}{n}$
* $\bar{N} = \frac{\sum_{i}{N_{i+}}}{n}$

Let

```math
\begin{align*}
h(B)=h(\bar{Y}, \bar{N}) &= \frac{\bar{Y}}{\bar{N}} \\
\therefore var(h(B)) &= \nabla{h(\beta)}^{T} \Sigma \nabla{h(\beta)}\\

\end{align*}
```

where

```math
\begin{align*}
\nabla{h(\beta)} &= \left[ 
    \frac{\partial{\frac{\bar{Y}}{\bar{N}}}}{\partial{\bar{Y}}},
    \frac{\partial{\frac{\bar{Y}}{\bar{N}}}}{\partial{\bar{N}}} 
    \right]\\
    &= \left[ 
    \frac{1}{\mu_{N}},
    \frac{-\mu_{Y}}{\mu_{N}^{2}} 
    \right]\\

\end{align*}
```

This way calculate $h(B)$ and $var(h(B))$ for the treatment and control sets. The difference of the former gives the mean effect (_delta_) and the latter can be combined to get the pooled variance. The delta and pooled variance can be combined to calculate  the z-statistic.
___
---




# 3. CUPED
## 3.1. Count metrics
Control variates is a variance reduction method commonly applied in Monte Carlo procedures to reduce the error of estimates. Let's say we are interested in estimating $\mu_{Y}=\mathbb{E}(\bar{Y})$. A typical estimator would be the average of sample observations, $\bar{Y} = \frac{\sum_{i}{Y_{i+}}}{n}$.

Let $\bar{X}$ be some other metric where $\mathbb{E}(\bar{X})$ is known. Then $Y_{cv}=\bar{Y} + \theta(\bar{X} - \mathbb{E}(\bar{X}))$, where $\theta$ is a fixed value, is also an unbiased estimator of $\mu_{Y}$. To prove this, take the expectation on both sides:

```math
\begin{align*}
\mathbb{E}(Y_{cv}) &= \mathbb{E}(\bar{Y} + \theta(\bar{X} - \mathbb{E}(\bar{X})))\\
&= \mathbb{E}(\bar{Y}) + \theta\mathbb{E}(\bar{X}) - \theta\mathbb{E}(\bar{X})\\
&= \mathbb{E}(\bar{Y})

\end{align*}
```

The variance of this estimator is:

```math
\begin{align*}
var(Y_{cv}) &= var(\bar{Y} + \theta(\bar{X} - \mathbb{E}(X)))\\
&= var(\bar{Y} + \theta\bar{X})\\
&= var(\bar{Y}) + var(\theta\bar{X}) + cov(\bar{Y},\theta\bar{X})\\
&= var(\bar{Y}) + \theta^{2} var(\bar{X}) + 2\theta cov(\bar{Y},\bar{X})

\end{align*}
```

We would like to minimize the variance of any estimator. As $\theta$ is a free parameter we can optimize to minimize the variance:

```math
\begin{align*}
\frac{\mathrm{d}[var(Y_{cv})]}{\mathrm{d}\theta} &= 0\\
\frac{\mathrm{d}\left[var(\bar{Y}) + \theta^{2} var(\bar{X}) + 2\theta cov(\bar{Y},\bar{X})\right]}{\mathrm{d}\theta} &= 0\\
2\theta var(\bar{X}) + 2cov(\bar{Y},\bar{X}) &= 0\\
\theta &= - \frac{cov(\bar{Y},\bar{X})}{var(\bar{X})}\\

\end{align*}
```

Replacing theta in the variance formula with this optimal value:

```math
\begin{align*}
var(Y_{cv}) &= var(\bar{Y}) + 
\frac{cov(\bar{Y},\bar{X})^{2}}{var(\bar{X})} - 
2\frac{cov(\bar{Y},\bar{X})^{2}}{var(\bar{X})}\\

&= var(\bar{Y}) - 
\frac{cov(\bar{Y},\bar{X})^{2}}{var(\bar{X})}\\

&= var(\bar{Y}) \left[1-\frac{cov(\bar{Y},\bar{X})^{2}}{var(\bar{X})var(\bar{Y})} \right]\\

&= var(\bar{Y}) (1-\rho_{\bar{Y},\bar{X}}^{2})

\end{align*}
```
---


## 3.2. Points to note
* We don't know the terms required to calculate $\theta$ exactly.
    * Typically the sample estimates for the terms can be used.
* We don't know $\mathbb{E}(X)$.
    * If we just use the sample mean to estimate it then we just get the old metric back (i.e., sample mean of $\bar{Y}$).
    * We could use some other data than $X_1, X_2, \dots$ to estimate $\mathbb{E}(\bar{X})$ with some variance.
    * However, for A/B tests this is not a problem if $\bar{X}$ is some pre-experimental data. This is because, we are typically interested in estimating $\mathbb{E}(\bar{Y_t}) - \mathbb{E}(\bar{Y_c})$ (referring to the metric in treatment and control).
    * With the new estimator that becomes as shown below; and $\mathbb{E}(\bar{X_c}) - \mathbb{E}(\bar{X_t}) = 0$ because using pre-experimental data, there should be no treatment effect and random assignment would assure that there are no differences.
    
```math
\begin{align*}
\mathbb{E}(\bar{Y}_{cv,t}) - \mathbb{E}(\bar{Y}_{cv,c}) &= 
(\mathbb{E}(\bar{Y}_{t}) + \theta(\bar{X}_{t} - \mathbb{E}(\bar{X}_{t}))) - 
(\mathbb{E}(\bar{Y}_{c}) + \theta(\bar{X}_{c} - \mathbb{E}(\bar{X}_{c})))\\

&= (\mathbb{E}(\bar{Y}_{t}) + \theta\bar{X}_{t}) - 
(\mathbb{E}(\bar{Y}_{c}) + \theta\bar{X}_{c}) +
(\mathbb{E}(\bar{X}_{c}) - \mathbb{E}(\bar{X}_{t}))

\end{align*}
```

* Which X to use?
    * Typically the same metric but pre-experiment will give the most variance reduction.

___




## 3.3. CUPED on Ratio metrics
We combine CUPED and the Delta method. Two additional random variables are going to be included from the pre-experimental data.

* Random variables: $B = [\bar{Y}, \bar{N}, \bar{X}, \bar{M}]$ (thus, $k=4$)
    * Here $\bar{Y}$ and $\bar{X}$ refer to the numerator of the ratio metric in the experimental and pre-experimental data, respectively. Correspondingly, $\bar{N}$ and $\bar{M}$ refer to the denominator.
* Let mean vector: $\beta = [\mu_{Y}, \mu_{N}, \mu_{X}, \mu_{M}]$
* Covariance: $\Sigma$

To apply CUPED:
* Let $s(B)=\frac{\bar{Y}}{\bar{N}}$ and $r(B)=\frac{\bar{X}}{\bar{M}}$ then using the Delta method:
    * $var(s(B)) = \nabla{s(\beta)}^{T} \Sigma \nabla{s(\beta)}$
    * $var(r(B)) = \nabla{r(\beta)}^{T} \Sigma \nabla{r(\beta)}$
    * $covar(s(B), r(B)) = \nabla{s(\beta)}^{T} \Sigma \nabla{r(\beta)}$

* The gradients calculations are thus:

```math
\begin{align*}
\nabla{s(\beta)} &= \left[ 
    \frac{\partial{\frac{\bar{Y}}{\bar{N}}}}{\partial{\bar{Y}}},
    \frac{\partial{\frac{\bar{Y}}{\bar{N}}}}{\partial{\bar{N}}},
    \frac{\partial{\frac{\bar{Y}}{\bar{N}}}}{\partial{\bar{X}}},
    \frac{\partial{\frac{\bar{Y}}{\bar{N}}}}{\partial{\bar{M}}}
    \right]\\
    &= \left[ 
    \frac{1}{\mu_{N}},
    \frac{-\mu_{Y}}{\mu_{N}^{2}},
    0,
    0
    \right]\\

\nabla{r(\beta)} &= \left[ 
    \frac{\partial{\frac{\bar{X}}{\bar{M}}}}{\partial{\bar{Y}}},
    \frac{\partial{\frac{\bar{X}}{\bar{M}}}}{\partial{\bar{N}}},
    \frac{\partial{\frac{\bar{X}}{\bar{M}}}}{\partial{\bar{X}}},
    \frac{\partial{\frac{\bar{X}}{\bar{M}}}}{\partial{\bar{M}}}
    \right]\\
    &= \left[
    0,
    0,
    \frac{1}{\mu_{M}},
    \frac{-\mu_{X}}{\mu_{M}^{2}},
    \right]\\

\end{align*}
```

* $\theta = -\frac{covar(s(B), r(B))}{var(r(B))}$
* Replacing theta in the variance formula:

```math
\begin{align*}
var \left( \left[\frac{\bar{Y}}{\bar{N}}\right]_{cv} \right) &= var \left( \frac{\bar{Y}}{\bar{N}} \right) + 
\frac{cov\left( \frac{\bar{Y}}{\bar{N}}, \frac{\bar{X}}{\bar{M}} \right)^{2}}{var \left( \frac{\bar{X}}{\bar{M}} \right)} - 
2\frac{cov\left( \frac{\bar{Y}}{\bar{N}}, \frac{\bar{X}}{\bar{M}} \right)^{2}}{var \left( \frac{\bar{X}}{\bar{M}} \right)}\\

&= var \left( \frac{\bar{Y}}{\bar{N}} \right) \left(1-\rho_{\frac{\bar{Y}}{\bar{N}},\frac{\bar{X}}{\bar{M}}}^{2} \right)

\end{align*}
```


___
___
# 4. CUPED vs Other Methods
ANCOVA
Difference-in-differences
Autoregression


# Misc. notes
## Ratio-of-averages v. Average-of-ratios
To calculate the variance of ratio-of-averages we have to use the Delta method while the variance of average-of-ratios can be estimated using the sample variance.

Let $i$ be the independent randomization unit. Let there be $K$ clusters.

Then in each unit, $Y_{j}$ are independent such that, $\mathbb{E}(Y_{j})=\mu_i$ and $var(Y_{j})=\sigma_{i}^2$. Thus, the between-cluster variance, $\tau^2=var(\mu_i)$. Further assume that $N_i=m, \sigma_{i}^2=\sigma^2; \forall{i}$.

Based on the [Variance property](#121-variance-property) noted above, 

```math
\begin{align*}
var\left(\sum_{i}{a_{i}X_{i}}\right) &= \sum_{i}{a_{i}^{2}var(x_{i})} + 2\sum_{i<j<N}{a_{i}a_{j}cov(x_{i},x_{j})}\\

var\left(\frac{\sum_{i=1}^{K}{\mu_i}}{K}\right) &=\\

&= \frac{1}{K^2}    var\left(\sum_{i=1}^{K}{\mu_i}\right) \\
&= \frac{1}{K^2}   
\left[ 
    \sum_{i=1}^{K}{var({\mu_i})} + 
    2\sum_{i<j<\frac{K}{2}}{cov(\mu_{i},\mu_{j})}
\right]\\
&= \frac{1}{K^2}   
\left[ 
    \sum_{i=1}^{K}{var({\mu_i})} + 
    \sum_{p=1}^{K}\sum_{q=1,q\neq p}^{K}{cov(\mu_{p},\mu_{q})}
\right]\\
&= \frac{1}{K^2}   
\left[ 
    K \tau^2 + 
    K(K-1)  corr(\mu_{p},\mu_{q})   \sqrt{var(\mu_p) var(\mu_q)}
\right]\\
&= \frac{\tau^2}{K}   
\left[ 
    1 + (K-1)   corr(\mu_{p},\mu_{q})
\right]\\
&= \frac{\tau^2}{K}   
\left[ 
    1 + (K-1)\rho
\right]\\
\end{align*}
```



```math
\begin{align*}
var\left(\frac{\sum_{i=1}^{K}\sum_{j=1}^{m}{Y_{ij}}}{Km}\right) &=\\

&= \frac{1}{(Km)^2}    var\left(\sum_{i=1}^{K}\sum_{j=1}^{m}{Y_{ij}}\right) \\
&= \frac{1}{(Km)^2} 
\left[ 
    \sum_{i=1}^{K} var\left(\sum_{j=1}^{m}{Y_{ij}}\right) + 
    2 \sum_{p<q<\frac{K}{2}}^{K} cov\left(\sum_{p=1}^{m}{Y_{pj}}, \sum_{q=1}^{m}{Y_{qj}} \right)
\right]\\
&= \frac{1}{(Km)^2} 
\left[ 
    (K m \sigma^2 + 0) +
    2 \sum_{p<q<\frac{K}{2}}^{K} cov\left(\sum_{p=1}^{m}{Y_{pj}}, \sum_{q=1}^{m}{Y_{qj}} \right)
\right]\\
&= \frac{1}{(Km)^2}   
\left[ 
    K m \sigma^2 + 
    \sum_{p=1}^{K}\sum_{q=1,q\neq p}^{K} cov\left(\sum_{p=1}^{m}{Y_{pj}}, \sum_{q=1}^{m}{Y_{qj}} \right)
\right]\\
&= \frac{1}{K^2}   
\left[ 
    K \tau^2 + 
    K(K-1)  corr(\mu_{p},\mu_{q})   \sqrt{var(\mu_p) var(\mu_q)}
\right]\\
&= \frac{\tau^2}{K}   
\left[ 
    1 + (K-1)   corr(\mu_{p},\mu_{q})
\right]\\
&= \frac{\tau^2}{K}   
\left[ 
    1 + (K-1)\rho
\right]\\
\end{align*}
```



the coefficient of intra-cluster correlation, $\rho=\frac{\tau^2}{\tau^2 + \sigma^2}$, where $\tau^2=var(\mu_i)$ (i.e., the between ) and $\sigma^2=var(Y_{ij})$. 

```math
var_c = (Y_sigma2/(N_bar**2) - 2*YN_cov*Y_bar/(N_bar**3) + (Y_bar**2)*N_sigma2/(N_bar**4))
```

