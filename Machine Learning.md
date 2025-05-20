"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, improves with experience E".

ML is a sub-field of AI where the knowledge comes from experience and induction. It is a bottom-up approach. It can extract information from data, not create information: if something is not observed, it cannot be generated.
It is used when:
- there is NO human expert (e.g. DNA analysis).
- humans can perform the task but cannot explain how to the machine (e.g. character recognition).
- desired function changes frequently (e.g. predicting stock prices based on recent training data).
- each user needs a customized function (e.g. email filtering).

Every ML algorithm has the following components:
- Representation (e.g. linear models, instance-based, Gaussian processes, SVM, model ensembles, ..).
- Evaluation (e.g. accuracy, precision and recall, SSE, likelihood, posterior probability, ..).
- Optimization (e.g. combinatorial as greedy search, convex as gradient descent or constrained as linear/quadratic programming).
# ML Models
## Supervised Learning
The goal is to estimate the unknown model that maps known inputs to known outputs. 
It is the largest, most mature, most widely used sub-field of ML. 
Input variables $x$ are also called features, predictors or attributes. Output variables $t$ are also called target, responses or labels. The nature of the output $t$ determine the type of problem:
- Classification: if $t$ is discrete.
- Regression: if $t$ is continuous.
- Probability estimation: if $t$ is the probability of $x$.

We want to approximate $f$ given the dataset $\mathcal{D}$. The steps are:
1. Define a loss function $L$, which is the objective function we want to minimize.
2. Choose some hypothesis space $\mathcal{H}$, a subspace of all possible functions.
3. Optimize to find an approximate model $h$.
A bigger hypothesis space will yield better performance on seen data, but it might perform poorly with unseen data. In general, having a smaller hypothesis space is better, as with a bigger one the optimal $f$ moves with the data, thus we are learning noise. 
## Unsupervised Learning
The goal is to learn a better (more efficient) representation of a set of unknown input data. 
## Reinforcement Learning
The goal is to learn the optimal policy, in order to automatize the decision making process. 
## Dichotomies in ML
- **Parametric** vs. **Non parametric**
	- Parametric: fixed and finite number of parameters. They require training to estimate those parameters. The complexity of the model is fixed regardless of the dataset size. 
	- Non parametric: the number of parameters depends on the training set (it requires storing the entire dataset and perform queries on it to performs prediction so it might be memory demanding, but it is faster since it does not require training). The complexity of the model grows with the number of training samples.
- Frequentist vs. Bayesian
	- Frequentist: use probabilities to model the sampling process.
	- Bayesian: use provability to model uncertainty about estimate.
- Generative vs. Discriminative
	- Generative: learns the joint probability distribution $p(x, t)$.
	- Discriminative: learns the conditional probability distribution $p(t|x)$.
- Empirical Risk Minimization vs. Structural Risk Minimization
	- Empirical risk: error over the training set.
	- Structural risk: error over the training error with model complexity.

# Linear Models for Regression
## Linear Regression
The goal of regression is to learn a mapping from input $x$ to a continuous output $t$.

Many real processes can be approximated with linear models. A model is said to be linear if it is linear in the parameters (coefficients that multiplies $x$ or functions of $x$). Linear problems can be solved analytically. Augmented with kernels, it can model non-linear relationships.

A linear function in the parameters $w$ can be written as:
$$
y(x, w) = w_0 + \sum_{j=1}^{D-1} w_jx_j = w^T x
$$
where $w_0$ is the offset and $x = (1, x_1, .., x_{D-1})$.

To quantify how well or poorly we are doing on a task we define a loss function $L(t, y(x))$. The average expected loss is given by:
$$
\mathbb{E}[L] = \int\int L(t, y(x))p(x, t)dxdt
$$
where $p(x, t)$ is the joint probability of observing both $x$ and $t$.

A common choice for regression is the square loss function $L = (t - y(x))^2$. 
The optimal solution (if we assume a completely flexible function) is the conditional average:
$$
y(x) = \int t p(t|x)dt = \mathbb{E}[t|x]
$$
![400](./assets/optimal_solution_LS.png)

A simple generalization of the squared loss is the **Minkowski loss**:
$$
\mathbb{E}[L] = \int \int |t-y(x)|^q p(x, t)dxdt
$$
where the minimum of $\mathbb{E}[L]$ is given by:
- the conditional mean for $q=2$.
- the conditional median for $q=1$.
- the conditional mode for $q \rightarrow 0$.
### Basis Functions
To consider non-linear functions, we can use non-linear basis function:
$$
y(x, w) = w_0 + \sum_{j=1}^{M-1} w_j \phi_j (x) = w^T\phi(x)
$$
where $\phi(x) = (1, \phi_1(x), .., \phi_{M-1}(x))^T$ are called features (e.g. polynomial, Gaussian, sigmoidal, ..). 
In this way, we extend the class of models by considering linear combinations of fixed non-linear functions (basis functions) of the input variable.
### Approaches
A **direct** approach (which is not a statistical method) consists in finding a regression function $y(x)$ directly from the training data. 
It is computationally efficient, but it has no probabilistic interpretation and lacks flexibility.

The **generative** approach consists in:
1. Model the joint probability distribution: $p(x, t) = p(x|t)p(t)$.
2. Infer the conditional density (using Bayes theorem): $p(t|x) = {p(x, t) \over p(x)}$.
3. Marginalize to find the conditional mean: $\mathbb{E}[t|x] = \int t p(t|x)dt$.
It is useful for augmenting data since it can generate new samples, so it is robust to missing data. However, the assumptions about data distribution (point 1.) might be unrealistic. 

In a **discriminative** approach we try to predict the target given the input. It consists in:
1. Model the conditional probability: $p(t|x)$.
2. Marginalize to find the conditional mean: $\mathbb{E}[t|x] = \int tp(t|x)dt$.
It finds decision boundaries that best classify $y$ given $X$, without modeling the distribution of $X$.
Typically achieves better classification accuracy than generative models.

| Approach       | What it models                       |
| -------------- | ------------------------------------ |
| Direct         | direct parameter estimation          |
| Generative     | joint probability distribution       |
| Discriminative | conditional probability distribution |
## Minimizing Least Squares
Given a dataset with $N$ samples, we consider the following error (loss) function
$$
L(w) = {1 \over 2} \sum_{n=1}^N (y(x_n, w))^2
$$
which is (half) the **residual sum of squared errors** (**RSE**) a.k.a. **sum of squared errors** (**SSE**). It can also be written as the sum of the $l_2$-norm of the vector of residual errors:
$$
RSS(w) = ||\epsilon||^2_2 = \sum_{i=1}^N \epsilon^2_i
$$
### Ordinary Least Squares
Let's write RSS in matrix form, to do so we need:
- $\Phi = (\phi(x_1), ..., \phi(x_N))^T$, as the $N \times d$ design matrix (each row is $x_n^T$).
- $t=(t_1, .., t_N)^T$ as the $N \times 1$ vector of target values.
- $w=(w_1, .., w_N)^T$ as the $d \times 1$ vector of parameters (one for each feature).
Then:
$$
L(w) = {1\over2} RSS(w) = {1\over2} (t - \Phi w)^T (t - \Phi w)
$$
Note that $||t- \Phi w||^2 = (t - \Phi w)^T (t - \Phi w)$, one term is transposed otherwise we cannot make the matrix multiplication. 
Moreover, the factor $1 \over 2$ is used to remove the factor $2$ that will come out the derivative.
The goal is to find $w$ in order to minimize $L(w)$. To do so, we compute first and second derivative:
$$
L(w) = {1 \over 2}[t^Tt -2t^T\Phi w + w^T\Phi^T \Phi w]
$$
$$
{\partial L(w) \over \partial w} = {1 \over 2}[\overbrace{{\partial \over \partial w}t^Tt}^{A} - \overbrace{{\partial \over \partial w} 2t^T \Phi w}^{\text{B}} + \overbrace{{\partial \over \partial w} w^T \Phi^T \Phi w}^{\text{C}}]
$$
where:
- A does not depend on $w$, therefore its contribute is 0.
- B can be solved taking into account that $-2t^T \Phi = a^T$ is a constant vector, therefore ${\partial \over \partial w} a^T w = a$, so $\text{B} = -2\Phi^T t$.
- C can be solved taking into account that $w^t \Phi^t \Phi w = a$ is a symmetric matrix by construction, therefore ${\partial \over \partial w} w^t a w = 2aw$ if $a$ is symmetric, so $\text{C} = 2 \Phi^T \Phi w$.
Putting all together:
$$
{\partial L(w) \over \partial w} = -\Phi^T t + \Phi^T \Phi w
$$
Assuming **$\Phi^T\Phi$ is non singular**, then the minimum is reached for:  $$
\hat{w}_{OLS} = (\Phi^T\Phi)^{-1} \Phi^Tt
$$The assumption is important, otherwise we cannot invert the matrix.
To clarify, $\Phi^T\Phi$ is singular if:
- we have more features ($d$) than samples ($N$): then $\Phi^T \Phi$ is singular because the design matrix $\Phi$ has more columns than rows, making it rank deficient.
- if features are linearly dependent (redundant features): then $\Phi^T \Phi$ is singular because some columns of $\Phi$ are linear combination of others. 

OLS is a closed-form solution (direct approach). It is efficient only when $d$ is small since the matrix inversion is $O(d^3)$. For big data (large $d$), iterative approaches like gradient descent or stochastic gradient descent are preferred. 

> [!TIP] A bad-condition (or ill-condition) matrix
> A matrix is said to be bad-conditioned when its condition number is large:
> $$
> cond(A) = {\sigma_{max} \over \sigma_{min}}
> $$
> where $\sigma_{max}$ and $\sigma_{min}$ are the largest and smallest singular values of matrix A. Moreover, if $\sigma_{min}$ is close to 0, then $cond(A)$ becomes large.
> 
> A large condition number means that small changes in input can lead to large changes in output. 
> In this case, the matrix $\Phi^T \Phi$ is nearly singular (not invertible or almost not), so computing the inverse if numerically unstable. This happens especially when:
> - some features are highly correlated (multicollinearity).
> - there are redundant or nearly redundant features.
> - there are more features than datapoints (so $\Phi^T \Phi$ is rank deficient).
### Gradient Optimization
It is an algorithm with sequential (online) updates.
If the loss function can be expressed as a sum over samples $L(x) = \sum_n L(x_n)$, then we can write the following rules:
$$
w^{(k+1)} = w^k - \alpha^{(k)} \nabla L(x_n)  
$$
$$
w^{(k+1)} = w^{(k)} -\alpha^{(k)}({w^{(k)}}^T \phi(x_n) - t_n) \phi(x_n)
$$
where $k$ is the iteration and $\alpha$ is the learning rate. 
For convergence, the learning rate has to satisfy two boundaries:
$$
\sum_{k=0}^\infty a^{(k)} = + \infty
$$
$$
\sum_{k=0}^\infty a^{(k)^2} < + \infty
$$
Advantages: it is cheaper and since the problem is convex it will find the optimal solution. 
#### Geometric interpretation
Let's assume that $t$ is an N-dimensional vector. 
Let's denote:
- $\varphi_j$ as the $j_{th}$ column of $\Phi$.
- $\hat{t}$ as the N-dimensional vector whose $n_{th}$ element is $y(x_n, w)$.
So, we can say that:
- $\hat{t}$ is a linear combination of $\varphi_1, .., \varphi_M$, which are the columns of $\Phi$.
- $\hat{t}$ lives in a M-subspace $\mathcal{S}$.
- since $\hat{t}$ minimizes SSE w.r.t. $t$, it represents the orthogonal projection of $t$ onto the subspace $\mathcal{S}$ $$
\hat{t} = \Phi \hat{w} = H t
$$where $H = \Phi (\Phi ^T \Phi)^{-1} \Phi ^T$ is called the hat matrix.
![600](./assets/hat_matrix.png)
### Maximum Likelihood Estimation (MLE)
It is a discriminative approach. 
The output variable $t$ can be modeled as a deterministic function $y$ of the input $x$ and random noise $\epsilon$: $t = f(x) + \epsilon$.
We want to approximate $f(x)$ with $y(x, w)$ assuming that $\epsilon \sim \mathcal{n}(0, \sigma^2)$ (white, Gaussian noise).

Given N samples, with inputs $X = \{x_1, ..., x_N\}$ and outputs $t = (t_1, ..., t_N)^T$, the likelihood function is:
$$
p(t|X, w, \sigma^2) = \prod_{n=1}^N \mathcal{N}(t_n|w^T \phi(x_n), \sigma^2)
$$
Assuming the samples to be independent and identically distributed (i.i.d.), we can consider the log-likelihood (by applying the log we do not change the position of the max):
$$
l(w) = \ln p(t|X, w, \sigma^2) = \sum_{n=1}^N \ln p(t_n|x_n, w, \sigma^2) = -{N \over 2} \ln (2\pi\sigma^2) - {1 \over 2\sigma^2} RSS(w) 
$$
To find the maximum likelihood, we compute the gradient and put it to zero:
$$
\nabla l(w) = \sum_{n=1}^N t_n \phi(x_n)^T - w^T (\sum_{n=1}^N \phi(x_n)\phi(x_n)^T) = 0 
$$
$$
w_{ML} = (\phi^T \phi)^{-1} \phi^T t
$$
The results is the same [[#Ordinary Least Squares|OLS]] we have computed before.
### Variance of parameters: statistical tests on coefficients
Given a limited number of parameters, uncertainty arises. 
In general, we assume that:
- the observation $t_i$ are uncorrelated and have constant variance $\sigma^2$.
- the $x_i$ are fixed (non random).
The variance-covariance matrix of the least-squares estimates is:
$$
Var(\hat{w}_{OLS}) = (\Phi^T\Phi)^{-1} \sigma^2
$$
In fact, more samples lower the variance of the parameters. 
Usually, the variance $\sigma^2$ is estimated by $\hat{\sigma^2} = {1 \over N-M} \sum_{n=1}^N (t_n - \hat{w}^T \phi(x_n))^2$.

Assuming that the model is linear in the features $\phi_1(), ..., \phi_M()$ and that the noise is additive and Gaussian we can say that:
$$
\hat{w} \sim \mathcal{N}(w, (\phi^T\phi)^{-1}\sigma^2)
$$
$$
(N-M)\hat{\sigma^2} \sim \sigma^2 \chi_{N-M}^2
$$
This allow us to formulate some statistical tests.
#### Single coefficients
Given the following hypothesis test: 
$$ H_0: w_j = 0 \qquad \text{ vs. } \qquad H_1: w_j \neq 0 $$
it determines if the single coefficient is relevant or not: $$ t_{stat} = \frac{\hat{w}_j - w_j}{\hat{\sigma} \sqrt{v_j}} \sim t_{N - M - 1} $$where $t_{N - M - 1}$ is the T-Student distribution with $N-M-1$ degrees of freedom.
We do not reject the null hypothesis (reject the coefficient) if we have:
$$
|t_{stat}| \le z_{1-\alpha / 2}
$$
where $z_{\sigma}$ is the quantile of order $\sigma$ of the Normal distribution. 
#### Overall significance of the model (F-statistic):
It considers the following hypothesis test:
$$
\begin{align}
& H_0: w_1 = \dots = w_M = 0 \ \ \ \ \ \ \text{none of the predictors are useful}\\
& H_1: \exists w_j \neq 0 \ \ \ \ \ \text{at least one predictor is useful}
\end{align}
$$
In other words, under $H_0$​ the model does not explain more variability than would be explained by the mean alone.
The F-statistic can be computed (is distributed) as follows:
$$

F = \frac{N-M-1}{M }\frac{TSS - RSS}{RSS} \sim F_{M, N-M-1}

$$
where $N$ is the total number of observations, $M$ is the total number of predictor, and $F_{M, N-M-1}$ is the Fisher-Snedecor distribution.

Both these statistics are interpreted via the p-value, that is the probability of obtaining a statistic as extreme as (or more extreme than) the observed value, under the null hypothesis ($H_0$).
- If the p-value is small (e.g., < 0.05), reject $H_0$ → the predictor/model is significant.
- If the p-value is large, the predictor/model is likely not significant.
#### Gauss-Markov theorem

> [!THEOREM] 
> The least square estimate (LSE) of $w$ has the smallest variance among all linear unbiased estimates. 

It follows that LSE has the lower MSE of all linear estimator with NO bias. However, there may exist a biased estimator with smaller MSE. So, introducing bias can be beneficial since it reduce the variance (bias-variance trade-off).
### Multiple outputs
In case of multiple outputs, we could use a different set of basis functions for each output, thus having independent regression problems.
Usually, a single set of basis functions is considered $\hat{W}_{ML} = (\Phi^T\Phi)^{-1} \Phi^T T$.
For each output $t_k$, which is a N-dimensional column vector, we have $\hat{w}_k = (\Phi^t\Phi)^{-1} \Phi^T t_k$.

The solution decouples between different outputs. The pseudo inverse $(\Phi^t\Phi)^{-1} \Phi^T$ needs to be computed only once. 
### Overfitting vs. Underfitting
We want to have a good generalization: this is the problem of model selection, which consist in identifying the proper hypothesis space. 
## Regularization
It is used to reduce the MSE by adding a penalty term to the loss function as follow:
$$
L(w) = L_D(w) + \lambda L_W(w)
$$
In this way, we prevent coefficient to reach large values. 
### Ridge
By taking $L_W(w) = {1 \over 2}w^Tw = {1 \over 2}||w||_2^2$ we get:
$$
L(w) = {1 \over 2} \sum_{i=1}^N (t_i - w^T \phi(x_i))^2 + {\lambda \over 2} ||w||_2^2
$$
It is called ridge regression or [[Artificial Neural Networks & Deep Learning#Weight decay limiting overfitting by weights regularization|weight decay]].
It helps by distributing weights among correlated features, reducing variance but keeping all features.
The loss function is still quadratic in $w$:
$$
\hat{w}_{ridge} = (\lambda I + \Phi^t\Phi)^{-1} \Phi^T t
$$
Note that the matrix $\Phi^T \Phi + \lambda I$ is definite positive, so its eigenvalues must be greater than $\lambda$. Moreover, $\lambda I$ adds a small positive value to the diagonal improving the matrix conditioning, as well as numerical stability.
### Lasso
By taking $L_W(w) = {1 \over 2}\sum_{j=1}^M w_j = {1 \over 2}||w||_1$ we get:
$$
L(w) = {1 \over 2} \sum_{i=1}^N (t_i - w^T \phi(x_i))^2 + {\lambda \over 2} ||w||_1
$$
Differently from Ridge, Lasso is non-linear in $t_i$ and no closed form solution exists (it is a quadratic programming problem).
However, it has the advantage of making some weights equal to zero for values of $\lambda$ sufficiently large. In fact, it can be used for feature selection, by excluding the features which have coefficient equal to zero.
Lasso yields sparse models. 

|           | What it does in practice                                                                                                                                            | When to use and why                                                                                          | Penalty term                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| **Ridge** | It **helps by distributing weights among correlated features**, reducing variance but keeping both features.                                                        | When we have **highly correlated features** and want to **keep all of them** while controlling their impact. | Adds a penalty term proportional to the **square of the coefficients**.          |
| **Lasso** | It **tends to arbitrarily drop one of the correlated features** because of its sparsity effect. This can be problematic if both variables carry useful information. | When we want to **perform automatic feature selection** and eliminate unnecessary variables.                 | Adds a penalty term proportional to the **absolute values of the coefficients**. |

## Bayesian Linear Regression
### Bayesian approach
- Formulate the knowledge about the world in a probabilistic way.
	- Define the model that expresses the knowledge quantitatively.
	- The model will have some unknown parameters.
	- Capture the assumptions about unknown parameters by specifying the prior distribution over those parameters before seeing the data.
- Observe the data.
- Compute posterior probability distribution for the parameters, given observed data.
- Use the posterior distribution to:
	- Make prediction by averaging over the posterior distribution.
	- Examine/Account for uncertainty in the parameter values.
	- Make decisions by minimizing expected posterior loss.

The posterior distribution can be obtained by combining the prior with the likelihood for the parameters, given the data (Bayes' rule):
$$
p(parameters|data) = {p(data|parameters)p(parameters) \over p(data)}
$$
$$
p(w|\mathcal{D}) = {p(\mathcal{D}|w)p(w) \over p(\mathcal{D})}
$$
where:
- $p(w|\mathcal{D})$ is the posterior probability of parameters given training data.
- $p(\mathcal{D}|w)$ is the probability (likelihood) of observing the training data given the parameters.
- $p(w)$ is the prior probability over the parameters.
- $p(\mathcal{D})$ is the marginal likelihood (normalizing constant): $p(\mathcal{D}) = \int p(\mathcal{D}|w)p(w)dw$.

In words: $posterior \propto likelihood \times prior$.
We are searching for the most probable value of $w$ given the data: maximum a posteriori (MAP) which is the mode of the posterior.
### Bayesian Linear Regression (BLR)
Another approach to avoid overfitting is to use a Bayesian Linear Regression (BLR).
In the Bayesian approach the parameters of the model are considered as drawn from some distribution. 

Assuming Gaussian likelihood model, the conjugate prior is Gaussian too $p(w) = \mathcal{N}(w|w_0, S_0)$.
Given the data $\mathcal{D}$, the posterior is still Gaussian:
$$
p(w|t, \Phi, \sigma^2) \propto \mathcal{N}(w|w_0, S_0)\mathcal{N}(t|\Phi_w, \sigma^2 I_N) = \mathcal{N}(w|w_N, S_N)
$$
$$
w_N = S_n(S_0^{-1}w_0 + {\Phi^Tt \over \sigma^2})
$$
$$
S_N^{-1} = S_0^{-1} + {\Phi^T\Phi \over \sigma^2}
$$
For sequential data, the posterior acts as prior for the next iteration. 

In Gaussian distributions the mode coincides with the mean. It follows that $w_N$ is the MAP estimator. Moreover:
- If the prior has infinite variance, $w_N$ reduces to the ML estimator. 
- If $w_0 = 0$ and $S_0 = \tau^2I$, then $w_N$ reduces to the [[Machine Learning#Ridge|ridge estimate]], where $\lambda = \sigma^2 / \tau^2$.

We are interested in the posterior predictive distribution:
$$
	p(t|x, \mathcal{D}, \sigma^2) = \int \mathcal{N}(t|w^T \phi(x), \sigma^2)\mathcal{N}(w | w_N, S_N) dw = \mathcal{N}(t|w_N^T\phi(x), \sigma^2_N(x))
$$
where:
$$
\sigma^2 = \underbrace{\sigma^2}_{\text{noise in the} \atop \text{ target values}} + \underbrace{\phi(x)^T S_N\phi(x)}_{\text{uncertainty associated} \atop \text{with parameter values}}
$$
- In the limit, as $N \rightarrow \infty$, the second term goes to zero.
- The variance of the predictive distribution arises only from the additive noise governed by parameter $\sigma$.
#### Modeling challenges
The first challenge is in specifying:
- a suitable **model**, which should admit all possibilities that thought to be all likely.
- a suitable **prior distribution**, which should avoid giving zero or very small probabilities to possible events, but should also avoid spreading out the probability over all possibilities.
To avoid uninformative priors, we may need to model dependencies between parameters. One strategy is to introduce latent variables into the model and hyperparameters into the prior. 
Both of these represents the ways of modeling dependencies in a tractable way. 
#### Computational challenges
The other big challenge is computing the posterior distribution. There are several approaches:
- **Analytical integration**: if we use conjugate priors, the posterior distribution can be computed analytically. It only works for simple models. 
- **Gaussian (Laplace) approximation**: approximate the posterior distribution with a Gaussian. It works well when there are lot of data compared to the model complexity.
- **Monte Carlo integration**: once we have a sample from the posterior distribution, we can simulate a Markov chain that converges to the posterior distribution (Markov Chain Monte Carlo, MCMC).
- **Variational approximation**: usually faster than MCMC, but it is less general.

In summary:
- Advantages:
	- Closed-form solution.
	- Tractable Bayesian treatment.
	- Arbitrary non-linearity with the proper basis functions.
- Disadvantages:
	- Basis functions are chosen independently from the training set. 
	- Curse of dimensionality. 

| Method                              | Category            |
| ----------------------------------- | ------------------- |
| Ordinary Least Squares (OLS)        | Direct approach     |
| Gradient-based optimization         | Direct approach     |
| Maximum Likelihood Estimation (MLE) | Generative approach |
| Bayesian Linear Regression          | Generative approach |
# Linear Classification
## Classification Problem
The goal of classification is to assign an input $x$ into of of $K$ discrete classes $C_k$, where $k = 1, .., K$. 

In linear classification, the input space is divided into decision regions whose boundaries are called **decision boundaries** or decision surfaces. 
In classification, we need to predict discrete class labels, or posterior probabilities that lie in the range of $(0, 1)$, so we use a nonlinear function. 
$$
y(x, w) = g(x^Tw + w_0)
$$
It is a **generalized linear model**, which is **not linear in $x$**:
- it will give us **decision boundaries linear in $x$** which corresponds to $y(x, w) = \text{constant}$.
- but the function $y(x, w)$ is NOT linear in $x$ because of the non-linear function $g$.
They have more complex analytical and computational properties than regression. As in regression, we can consider fixed nonlinear basis functions to transform the input space while maintaining a linear relationship in the new space. 

- In two class problems, we have a binary target value $t \in \{0, 1\}$, such that $t=1$ is the positive class and $t=0$ is the negative class. 
  We can interpret the value of t as the probability of the positive class. 
  The output of the model can be represented as the probability that the model assigns to the positive class. 
- If there are $K$ classes, we can use a 1-of-$K$ encoding scheme.
  $t$ is a vector of length $K$ and contains a single 1 for the correct class and 0 elsewhere. $t$ is the vector of class probabilities. 

We can use three approaches to classification:
- **Discriminant functions approach**: build a function that directly maps each input to a specific class (e.g. [[Machine Learning#Artificial Neural Networks & Deep Learning Perceptron Learning Algorithm The Perceptron Algorithm|perceptron algorithm]], [[Machine Learning#K-nearest neighbors|k-nearest neighbors]]). 
- **Probabilistic approach**: model the conditional probability distribution $p(C_k|x)$ and use it to make optimal decisions. There are two alternatives:
	- **Probabilistic discriminative approach**: model $p(C_k|x)$ directly using parametric models (e.g. [[Machine Learning#Logistic Regression|logistic regression]]).
	- **Probabilistic generative approach**: model class conditional densities $p(x|C_k)$ together with prior probabilities $p(C_k)$ for the classes, then infer the posterior $p(x|C_k)$ using Bayes' rule (e.g. [[Machine Learning#|naive Bayes]]). 
## Discriminant Functions approach
### Two classes
Given $y(x) = x^Tw + w_0$, the decision boundaries can be found with $y(x) = 0$. We assign $x$ to $C_1$ if $y(x) \ge 0$, to $C_2$ otherwise. 
Moreover, given two points on the decision surface $x_A, x_B$ we can say that $y(x_A) = y(x_B) = 0$ and $w^T(x_A - x_B) = 0$: since the scalar product is null (perpendicular vectors) it means that $w$ is orthogonal to the decision surface. 
If $x$ is on the decision surface, then $w^Tx / ||w||_2 = - w_0 / ||w||_2$ which means that $w_0$ determines the location of the decision surface (translation w.r.t. the origin).  
![500](./assets/2class_classification.png)
### Multiple classes
Consider the extension to $K>2$ classes. We can adapt the previous solution in two ways:
- One-versus-the-rest: $K-1$ classifiers, each of which solves a two class problem. The idea is to separate points in class $C_k$ from points not in that class. However, there are regions in input space that are ambiguously classified. ![300](./assets/one-versus-the-rest.png)
- One-versus-one: $K(K-1)/2$ binary discriminant functions. However, ambiguity arises. ![300](./assets/one-versus-one.png)
A simple solution is to use $K$ linear discriminant functions of the form: 
$$
y_k(x) = x^Tw_k + w_{k0}
$$
where $k = 1, ..., K$.
The idea is to assign $x$ to class $C_k$, if $y_k(x)>y_j(x)$ $\forall j \neq i$. 
The result decision boundaries are **singly connected** and **convex** (any straight line between two points inside the region lie inside that region). In fact, for any two points $x_A, x_B$ that lie inside the region $\mathcal{R}_k$, so $y_k(x_A) > y_j(x_B)$ and $y_k(x_B) > y_j(x_B)$, taken any positive $\alpha$ we can say
$$
y_k(\alpha x_A + (1 - \alpha)x_B) > y_j(\alpha x_A + (1 - \alpha)x_B)
$$
due to linearity of the discriminant functions. 
![300](./assets/multiclass_classification.png)
### Least Squares for Classification
Least squares approximates the conditional expectation $\mathbb{E}[t|x]$.
Consider a general classification problem with $K$ classes using 1-of-$K$ encoding scheme for the target vector $t$. Each class is described by its own linear model:
$$
y_k(x) = w_k^Tx + w_{k0}
$$
where $k = 1, .., K$. 
Using vector notation $y(x) = \tilde{W}^T\tilde{x}$ where:
- $\tilde{W}$ is a $D \times K$ matrix whose $k_{th}$ column is $\tilde{w}_k = (w_{k0}, w_{k}^T)^T$.
- $\tilde{x} = (1, x^T)^T$.

Given a dataset $\mathcal{D} = \{x_i, t_i\}$, $i = 1, .., N$. 
We have already seen how to minimize [[Machine Learning#Minimizing Least Squares|least squares]] $\tilde{W} = (\tilde{X}^T\tilde{X})^{-1}\tilde{X}^T T$ where:
- $\tilde{X}$ is an $N \times D$ matrix whose $i_{th}$ row is $\tilde{x}_i^T$.
- T is an $N \times D$ matrix whose $i_{th}$ row is $t_i^T$.
A new input is assigned to a class for which $t_k = \tilde{x}^T\tilde{w}_k$ is the largest. 

However, some problems arises in using least squares:
- it is highly sensitive to outliers, unlike logistic regression. 
- it does not works well with non-Gaussian distribution since LS is derived as the [[#Maximum Likelihood Estimation (MLE)|MLE]] under the assumption of Gaussian noise. LS regression assumes that the errors are normally (Gaussian) distributed around the predicted values. This assumption is valid for continuous targets but not for binary targets. This is the reason why LS fails in classification. 

So far, we have considered classification models that work directly in the input space. 
All considered algorithms are equally applicable if we first make a fixed non-linear transformation of the input space using vector of basis functions $\phi(x)$.
Decision boundaries will appear linear in the feature space, but would correspond to nonlinear boundaries in the original input space. 
Classes that are linearly separable in the feature space may not be linearly separable in the original input space. 
![](./assets/nonlinear_boundaries.png)
### [[Artificial Neural Networks & Deep Learning#Perceptron Learning Algorithm|The Perceptron Algorithm]]
The [[Artificial Neural Networks & Deep Learning#The perceptron|perceptron]] is an example of linear discriminant model. It is an online linear classification algorithm. 
It corresponds to a two-class model:
$$
y(x) = f(w^T\phi(x))
$$
$$
f(a) = 
\begin{cases}
	+1 \ \ \ \text{if} \ a \ge 0\\
	-1 \ \ \ \text{otherwise}
\end{cases}
$$

The algorithm finds the **separating hyperplane** by **minimizing the distance of misclassified points to the decision boundary**.
Using the number of misclassified point as loss function is not effective since it is a piece wise constant function (we cannot optimize it using gradient). 

We are seeking a vector $w$ such that $w^T \phi(x_n) > 0$ when $x_n \in \mathcal{C}_1$ and $w^T \phi(x_n) < 0$ otherwise.
The perceptron criterion assigns **zero error to correct classification** and $w^T \phi(x_n)t_n$ (non zero error) to misclassified patterns $x_n$: in this way the error is proportional to the distance to the decision boundary. 
The loss function to be minimized is the distance of misclassified points in $\{(x_n, t_n)\}_{n=1}^N$ with $t_n \in \{-1, 1\}$:
$$
L_P(w) = - \sum_{n \in \mathcal{M}} w^T \phi(x_n)t_n
$$
where $\mathcal{M} = \{n \in \{1,..,N\}:t_n\neq y(x_n)\}$.
Note that $w^T \phi(x_n)t_n$ is always negative, since it is a missclassification, so we adjust the sign by putting a minus before the summary. 
The perceptron update rule is similar to a [[Machine Learning#Gradient Optimization|SDG]] step, but it does not minimize a continuous loss function. Instead, it updates weights whenever a **missclassification** occurs (**online**):
$$
w^{(k+1)} = w^{(k)} - \alpha \nabla L_P(w) = w^{(k)} + \alpha \phi(x_n)t_n
$$
Since the perceptron function does not change if $w$ is multiplied by a positive constant (this because it only cares about the sign of $w^T\phi(x_n)$, not its magnitude), the **learning rate** $\alpha$ can be set to an **arbitrary** value (except for 0, generally it is set to 1). The learning rate only scales the weight updates but does not affect the number of iterations needed for convergence. 
The perceptron updates are **step-based corrections** rather than gradient-based optimizations. 

![500](./assets/perceptron_training_without_bias.gif)

The effect of a single update is to reduce the error due to misclassified pattern. However, this does not imply that the loss is reduced at each stage.  

> [!THEOREM] Perceptron Convergence theorem
> If the training data set is linearly separable in the feature space $\Phi$, then the perceptron learning algorithm is guaranteed to find an exact solution in a finite number of steps. 

- The number of steps before convergence may be substantial. There is no bound on this number, so it has no practical use. 
- We are not able to distinguish between non separable problems and slowly converging ones. 
- If multiple solutions exists, the one found depends by the initialization of the parameters and the order of presentation of the data points. 
- It can be employed for K multi-class classification by training K one-versus-the-rest classifiers. 
### K-nearest neighbors
The idea is to look at the nearby points to predict the target of a new point. It can be used for both regression and classification:
Given a test sample $x$:
1. **Compute Distance**: Calculate the distance between $x$ and all training samples using a distance metric (e.g., Euclidean, Manhattan, or cosine similarity).
2. **Find Nearest Neighbors**: Identify the $k$ closest training samples.
3. **Predict Output**:    
    - **For classification**: Assign the most frequent class label among the $k$ neighbors (**majority voting**).
    - **For regression**: Take the **average** (or weighted average) of the neighbor values.

The number of neighbors $k$ is an hyperparameter that must be chosen accordingly: a small $k$ make the algorithm sensitive to noise, while a large $k$ lead to smooth decision boundary. it can be set using [[Machine Learning#Validation approach|cross validation]].
The choice of metric affects the performance, depending on the data distribution. 

It is computationally expensive for large datasets, sensitive to irrelevant or redundant features and the performance depends on a good choice of $k$. 
The hyperparameter $k$ can be used as a regularization hyperparameter: the larger the value of $k$, the more the model is regularized. A smaller $k$ is going to give a more complex model, which is associated with smaller bias and higher variance.

KNN naturally supports multi-class classification using **majority voting** among neighbors. - In case of ties, a **tie-breaking rule** is needed (e.g., prioritize the closest neighbor or use weighted voting).
## Probabilistic Discriminative Models
### Logistic Regression
#### Binary problem
Consider a two-class classification problem. The posterior probability of class $\mathcal{C}_1$ can be written as a logistic sigmoid function:
$$
p(\mathcal{C_1}|\phi) = { 1 \over 1 + \exp(-w^T \phi)} = \sigma(w^T \phi)
$$
where (for brevity) $\phi = \phi(x)$ is the transformed input vector and $p(\mathcal{C}_2|\phi) = 1 - p(\mathcal{C}_1 | \phi)$.
The bias term is omitted for clarity.
This model is known as logistic regression. Differently from generative models, here we model $p(\mathcal{C}_k | \phi)$ directly. 
##### Maximum Likelihood for Logistic Regression
Given a dataset $\mathcal{D} = \{x_n, t_n\}$ with $n = 1, .., N$ and $t_n \in \{0, 1\}$.
We want to maximize the probability of getting the right label:
$$
p(t|X, w) = \prod_{n=1}^N y_n^{t_n}(1 - y_n)^{1-t_n}
$$
where $y_n = \sigma (w^T \phi_n)$ is the logistic prediction. 
By taking the **negative log** of the **likelihood**, we define the loss function to be minimized:
$$
L(w) = -\ln p(t|X, w) = - \sum_{n=1}^N (t_n \ln y_n + (1-t_n)\ln(1-y_n)) = \sum_{n=1}^N L_n
$$
Then, to minimize it, we differentiate by applying the chain rule on $L_n$:
$$
{\partial L_n \over \partial y_n} = -{t_n \over y_n} -(1-t_n){1 \over 1 - y_n} =  { y_n - t_n \over y_n(1-y_n)}
$$
$$
{\partial y_n \over \partial w} = {\partial y_n \over \partial z} \cdot {\partial z \over \partial w}  = y_n(1-y_n)\phi_n
$$
where $z = w^t \phi_n$ and ${\partial \sigma(z) \over \partial z} = \sigma(z)(1-\sigma(z))$.
$$
{\partial L_n \over \partial w} = {\partial L_n \over \partial y_n} {\partial y_n \over \partial w} = \underbrace{(y_n - t_n)}_\text{error}\overbrace{\phi_n}^{\text{feature}} 
$$
The gradient of the loss function is:
$$
\nabla L(w) = \sum_{n=1}^N (y_n - t_n)\phi_n
$$
- It has the same form as the gradient of the SSE function for linear regression. 
- There is NO closed form solution, due to non linearity of the logistic sigmoid function. 
- The error function is convex and can be optimized by standard gradient optimization techniques. 
- Easy to adapt to the online learning setting.
- The gradient has a meaningful interpretation: $y_n -t$ represent the difference between predicted probability and true label (prediction error): it means that the gradient updates weights in the direction that reduces the difference between predicted and actual values. 

> [!TIP] Relation between cross-entropy and negative log likelihood
> The general concept of **NLL loss** applies to **any probabilistic model** where we maximize the likelihood of observed data. It is defined as: $L_{NLL​}=−\sum_i ​\log P(y_i​∣x_i​)$.
> 
> Cross-entropy loss is **a specific case** of NLL when dealing with **categorical distributions** (e.g., classification problems with softmax outputs).
> For a classification problem with a categorical target $y$ and model predictions $\hat{y}$, it is defined as: $L_{CE} =-\sum_i y_i \log⁡ \hat{y_i}$.
> 
> In classification problems, the NLL and CE coincides. This is because classification problems typically model the probability of each class using a categorical distribution (for multiclass classification) or a Bernoulli distribution (for binary classification).

By applying the logit function $logit(y) = \log{y \over 1-y}$ to the output of logistic regression, we "unpack" the targets: $logit(y(x)) = w^Tx = w_0 + x_1 w_1 + ..$
Then, we can perform hypothesis testing on the significance of the parameters which linearly influence the log-odds of the output. 
#### Multi class problem
For the multiclass case, we represent posterior probabilities by a **softmax transformation** of linear functions of feature variables:
$$
p(\mathcal{C}_k|\phi) = y_k(\phi) = {\exp(w_k^T\phi) \over \sum_j \exp(w_j^T\phi)}
$$
Differently from generative models, here we will use maximum likelihood to determine parameters of this discriminative model directly:
$$
p(T|\phi, w_1, .., w_K) = \prod^N_{n=1} (\underbrace{\prod^K_{k=1} p(\mathcal{C}_k |\phi_n)^{t_{nk}})}_{\text{Only one term} \atop \text{corresponding to correct class}} = \prod^N_{n=1} (\prod^K_{k=1} y_{nk}^{t_{nk}})
$$
where $y_{nk} = p(\mathcal{C}_k | \phi_n) = {\exp(w_k^T\phi_n) \over \sum_j \exp (w_j^T\phi_n)}$.
Taking the negative logarithm gives the cross-entropy function for multiclass classification problem:
$$
L(w_1, .., w_K) = -\ln p(T|\phi, w_1, .., w_K) = - \sum_{n=1}^N (\sum_{k=1}^K t_{nk} \ln y_{nk})
$$
Taking the gradient:
$$
\nabla L_{w_j} (w_1, ..., w_K) = \sum_{n=1}^N (y_{nj} - t_{nj})\phi_n
$$
The gradient is computed for each set of weight for each class. 
#### Connection between Logistic Regression and Perceptron Algorithm
If we replace the logistic function with a step function, both algorithms use the same update rule: $w \leftarrow w - \alpha(y(x_n, w) - t_n)\phi_n$.
![](./assets/logistic_regression_step_function.png)
#### Decision Boundaries
If we properly train logistic regression, the classification error should **increase** when we use a **sub-optimal decision boundary**. This happens because:
- Logistic regression finds the optimal decision boundary that minimizes classification error based on maximum likelihood estimation (MLE).
- If we manually shift or modify this boundary without retraining the model, we move away from the optimal solution.
- A sub-optimal boundary misclassifies more points, leading to a higher classification error.
## Probabilistic Generative Models
### Naive Bayes
The naive assumption is that, given the class $C_k$, the input features $x_1, .., x_M$ are conditionally independent. So, the posterior probability can be simplifies to:
$$
\begin{align}
	p(C_k|x) &= {p(C_k)p(x|C_k) \over p(x)} \propto p(x_1, .., x_M, C_k) \\
	&= p(x_1|x_2, .., x_M, C_k)p(x_2,..,x_M,C_k) \\
	&= p(x_1|x_2, .., x_M, C_k)p(x_2|x_3, .., x_M, C_k)p(x_3,..,x_M,C_k) \\
	&= p(x_1|x_2, .., x_M, C_k) .. p(x_M|C_k)p(C_k) \\
	&= p(x_1|C_k)..p(x_M|C_k)p(C_k) = p(C_k)\prod^M_{j=1}p(x_j|C_k)	
\end{align}
$$
Given a prior $p(C_k)$, we maximize the MAP (maximum a posteriori) probability: 
$$
y(x) = arg \max_k p(C_k) \prod^M_{j=1}p(x_j|C_k)
$$
As loss function we use the log likelihood for fitting both the priors $p(C_k)$ and the likelihoods $p(x_j|C_k)$. We optimize with the MLE (maximum likelihood estimation).
Note that the naive Bayes is not a Bayesian method, since the priors are estimated from data and not updated using likelihoods. 
Thanks to the generative abilities of the naive Bayes classifier, we are able to generate a dataset which resembles the original one. 
## Evaluating the results
To evaluate the performance of a classifier, we can compute the confusion matrix which tells us the number of points which have been correctly classified and those which have misclassified.
From the confusion matrix, we can derive some useful metrics:
- **Accuracy**: fraction of the samples correctly classified in the dataset $Acc = {TP + TN \over N}$.
- **Precision**: fraction of samples correctly classified in the positive class among the ones classified in the positive class $Pre = {TP \over TP + FP}$.
- **Recall**: fraction of samples correctly classified in the positive class among the ones belonging to the positive class $Rec = {TP \over TP + FN}$.
- **F1 score**: harmonic mean of precision and recall $F1 = {2 \cdot Pre \cdot Rec \over Pre + Rec}$.
Ideally, we want them all to be as close to 1 as possible.
These performance metrics are not symmetric, but they depend on the class we selected as positive. Depending on the application, one might switch the classes to have measures which better evaluate the predictive power.
# Model Selection

> [!NOTE] No free lunch theorem
> For any learner L, given any distribution P over $x$ and training set size $N$ it holds that:
> $${1 \over |\mathcal{F}|} \sum_{\mathcal{F}} Acc_G(L) = {1 \over 2}$$
> where $\mathcal{F}$ is the set of all possible concepts $y=f(x)$ and $Acc_G(\mathcal{L})$ is the generalization accuracy of the learned measured on non-training examples. 

It means that the average accuracy of any learner is purely random guessing.
Moreover, for any two learners $L_1, L_2$:
- if $\exists$ learning problem s.t. $Acc_G(L_1) > Acc_G(L_2)$
- then $\exists$ a learning problem s.t. $Acc_G(L_2) > Acc_G(L_1)$
In practice, we so not expect a favorite learner to always be best, we have to try different approaches and compare them.
## Bias-Variance decomposition
Assume that we have a dataset $\mathcal{D}$ with $N$ samples obtained by a function $t_i = f(x_i) + \epsilon$ with $\mathbb{E}[\epsilon] = 0$ and $Var[\epsilon]=\sigma^2$.
We want to find a model $y(x)$ that approximates $f$ as well as possible. Let's consider the expected square error on an unseen sample $x$:
$$
\begin{align}
\mathbb{E}[(t-y(x))^2] &= \mathbb{E}[t^2 + y(x)^2 - 2ty(x)] \\ &= \mathbb{E}[t^2] + \mathbb{E}[y(x)^2] - \mathbb{E}[2ty(x)] \\ &= \mathbb{E}[t^2] \pm \mathbb{E}[t^2] + \mathbb{E}[y(x)^2] \pm \mathbb{E}[y(x)^2] -2f(x)\mathbb{E}[y(x)] \\ &= Var[t] + Var[y(x)] + (f(x) - \mathbb{E}[y(x)])^2 \\ &= \underbrace{Var[t]}_{\sigma^2} + \underbrace{Var[y(x)]}_{\text{Variance}} + \underbrace{\mathbb{E}[f(x) - y(x)]^2}_{\text{Bias}^2}
 \end{align} 
$$
where $\sigma^2$ is the irreducible error.
In the following graph, each blue dot represent a predicted model: ideally we would like to have low bias and variance.
![](./assets/bias-variance.png)

|               | Low Variance                                                                                                           | High Variance                                                                                                                |
| ------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Low Bias**  | Ideal case, since the main goal is to reduce the error which is the sum of bias and variance.                          | **Overfitting**.<br>Too large hypothesis space.<br>Solutions are to reduce the hypothesis space or increase the sample size. |
| **High Bias** | **Underfitting**.<br>Too small hypothesis space.<br>Solutions are to increase the hypothesis space (model complexity). | All wrong predictions.                                                                                                       |
### Bias
When we train a model multiple times on different datasets $\mathcal{D}$ sampled from the same distribution, we obtain different learned functions $y(x)$. The expected hypothesis, denoted as $\mathbb{E}[y(x)]$, represents the average prediction our model would make if we could train it on infinitely many datasets.

Bias quantifies the error introduced by the model’s assumptions, measuring how far the expected prediction is from the true function $f(x)$:
$$
bias^2 = \int (f(x) - \mathbb{E}[y(x)])^2p(x)dx
$$
High bias mean that the model is too simple to capture the underlying patterns, so **bias decreases when model complexity increases**. 
Since we are not able to compute $f(x), \mathbb{E}[y(x)]$ or the true data distribution $p(x)$, the previous formula is purely theoretical.
### Variance
It measure how much the learned function $y(x)$ varies when trained on different datasets. It quantifies the difference between what we learn from a particular dataset and the expected hypothesis:
$$
variance = \int \mathbb{E}[(y(x) - \overline{y}(x))^2]p(x)dx
$$
where $\overline{y}(x) = \mathbb{E}[y(x)]$ is the expected hypothesis.
Statistically speaking, increasing the data is always beneficial since it yields better performance, while computationally speaking this is not always the case. 
It **increases with simpler models** and/or **more samples**. 

![500](./assets/bias-variance(1).png)

As we can see from the graph, the balance between bias and variance determines the model's generalization ability. 
Bias and variance directly influence **training error** and **prediction error**:
- While **training error** decreases as model complexity increases, it is not a reliable measure of generalization since it behaves similarly to bias (large with simpler models and improves with higher hypothesis space).
- However, training error is an optimistically biased estimate of the prediction error, meaning it tends to underestimate the true error on unseen data. The **prediction error** accounts for both bias and variance, making it a better indicator of how well the model generalizes.

In practice, we randomly divide the dataset into test an train: we use training data to optimize parameters and test data to evaluate the prediction error.
For the test set to provide an **unbiased estimate** of the prediction error, it must not be used during training, including hyperparameter tuning or model selection. If the test set influences model choices, it effectively becomes part of the training process, leading to an optimistically biased evaluation that does not reflect real-world performance.

| ![](./assets/test_error.png) | ![](./assets/test_error(1).png) |
| ---------------------------- | ------------------------------- |
![](./assets/bias-variance_tradeoff.jpg)
The bias-variance trade-off can be managed using different techniques:
- Model selection:
	- **Feature selection**: identifies a subset of input features that are most related to the output.
	- **Regularization**: all the input features are used, but the estimated coefficients are shrunken towards zero, thus reducing the variance.
	- **Dimension reduction**: the input variables are projected into a lower-dimensional subspace. 
- Model ensemble: **bagging**, **boosting**.
### Curse of Dimensionality
It refers to the exponential increase in volume of the input space as the number of dimensions (features) grows. 
This makes working with high dimensional data challenging, due to several reasons:
- As dimensions increases, the variance also increases which may lead to overfitting.
- Need for more samples to cover the space effectively and avoid overfitting. 
- High computational cost. 

Common pitfall:
- If we can't solve a problem with a few features, adding more features seems like a good idea. 
- However, the number of samples usually stay the same.
- Despite adding more features, the model may perform worse due to the increased complexity, not better as expected. This is because the added features often do not provide meaningful information and may lead to overfitting.
## Feature Selection
1. Let $\mathcal{M}_0$ denote the null model, which contains no input feature (it simply predicts the sample mean for each observation).
2. For $k = 1, .., M$:
	1. Fit all $\binom{M}{k}$ models that contains exactly $k$ features.
	2. Pick the best among these $\binom{M}{k}$ models, having the smallest $RSS$ or equivalently largest $R^2$, and call it $\mathcal{M}_k$.
3. Select a single best model from $\mathcal{M}_0, .., \mathcal{M}_M$ using some criterion (cross-validation, AIC, BIC). 

Advantages:
- Since all possibile subsets are evaluated, it guarantees finding the best one.
Disadvantages:
- It is computationally expensive for dataset with many features. 
- If not done carefully, evaluating subsets on the training set could lead to overfitting. 

To solve the drawbacks, three main metaheuristics for **feature selection** can be applied:
- **Filter**: ranks features according to a simple criteria and select the best ones. It assumes that the features are independent and it might not find the best subset (e.g. [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)). 
- **Embedded** (built-in): the learning algorithm exploits its own variable selection technique (e.g. lasso, decision trees, auto-encoding).
- **Wrapper**: evaluate only some subsets of features.
	- **Forward Selection**: starts from an empty model (no features) and add features one-at-a-time.
	- **Backward Elimination**: starts with all the features and removes the least useful feature based on model performance, one-at-a-time.
	- **Brute force**: for each $k\in\{1, .., M\}$ number of feature, learn all the possible $\binom{M}{k}$ models with $k$ inputs and select the model with the smallest loss. In the end we select the number of features $k$ providing the model with the smallest loss. 
	  If M is large enough, the computation of all the models is unfeasible (combinatorial complexity).
	It can be computationally expensive, as it requires training several models.

The model containing all the features will always have the smallest training error, because adding more features increases the model's flexibility, allowing it to fit the training data better.
However, we want to choose a model with **low test error**, not a low training error. 
Therefore, $RSS$ and $R^2$ are not suitable for selecting the best model, since they both focus on training error and do not account for overfitting. 
There are two approaches to estimate the test error:
- **Direct estimation** using a validation approach.
  To do so, it is necessary to have another independent set of data, the **validation data**, which is used to tune hyperparameters and select the best model.
- Making an **adjustment to the training error** to account for model complexity.
  We can use methods like Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), and Cross-validation adjust for model complexity to avoid overfitting.
### Validation approach
#### Leave-One-Out Cross Validation (LOOCV)
Consider a validation set with one example $n$. 
We learn the model with dataset $\mathcal{D}\backslash\{n\}$ and evaluate its performance on the left-out point. The process is repeated for each data point, and the final error estimate is the average of all individual errors:
$$
L_{LOO} = {1 \over N} \sum_{n=1}^N (t_n - y_{\mathcal{D}\backslash\{n\}}(x_n))^2
$$

LOO is almost unbiased and slightly pessimistic. 
However, it has an high computational costs, which make its application infeasible for large dataset. 
#### k-fold Cross Validation
Randomly divide the training data into $k$ equal parts $\mathcal{D}_1, .., \mathcal{D}_k$ (folds). 
For each $i$, we learn the model using datapoints except $\mathcal{D}_i$ then its performance is estimated using $\mathcal{D}_i$ as validation set.
The final error estimate is the average over all data splits:
$$
L_{k-fold} = {1 \over k} \sum_{i=1}^k L_{\mathcal{D}_i}
$$
k-fold is much faster to compute rather than LOO, since common values for $k$ are 5 or 10, but it is more pessimistically biased. We generally use $k=10$ for a balanced bias-variance. 
### Adjustment Techniques
#### Mallows’ $C_p$ Criterion
It adjusts the residual sum of squares (RSS) to include a penalty for model complexity:
$$
C_p = {1 \over N}(RSS + 2d\tilde{\sigma^2})
$$
where $d$ is the total number of parameters and $\tilde{\sigma^2}$ is an estimate of the variance of noise $\epsilon$.
It favors model with a low RSS while penalizing those with more parameters to prevent overfitting.
#### Akaike Information Criterion (AIC)
It is used to compare models based on likelihood estimation:
$$
AIC = -2\log L + 2d
$$
where $L$ is the maximized value of likelihood function for the estimated model.
A lower AIC value indicates a better model.
#### Bayesian Information Criterion (BIC)
$$
BIC = {1 \over N}(RSS + \log(N)d\tilde{\sigma^2})
$$
BIC replaces the $2d\tilde{\sigma^2}$ of $C_p$ with $\log(N)d\tilde{\sigma^2}$ term. Since $\log N > 2$ for any $n > 7$, BIC selects smaller models.
#### Adjusted $R^2$
The "original" coefficient of determination is $R^2 = 1 - {RSS(w) \over TSS}$. It tells us how the fraction of the variance of the data is explained by the model (how much better we are doing w.r.t. just using the mean of the target).
It modifies the traditional $R^2$ to penalize excessive model complexity:
$$
AdjustedR^2 = 1 - {RSS / (N-d-1) \over TSS / (N-1)} = 1 - (1-R^2){N-2 \over N-M}
$$
where $TSS$ is the total sum of squares and $N-M$ are the degree of freedom.
Differently from other criteria, here a large value indicates a model with small test error (better predictive performance).
## Regularization
It can be used as an **embedded feature selection** method.
We have already seen regularization approaches applied to linear models ([[Machine Learning#Ridge|Ridge regression]] and [[Machine Learning#Lasso|Lasso]]). Such methods shrink the parameters towards zero. It may not be immediately obvious why such a constraint should improve the fit, but it turns out that shrinking coefficients estimates can significantly reduce the variance. 

As for subset selection, for Ridge regression and Lasso we require a method to determine which of the models under consideration is best. So, we need a method for selecting the tuning **parameter $\lambda$.**
**Cross-validation** provides a simple way to tackle this problem: 
1. Choose a grid of $\lambda$ values.
2. Compute the cross validation error rate for each value of $\lambda$, we then select the tuning parameter value for which the cross-validation error is smallest.
3. Finally, the model is **re-fit using all of the available observations** and the selected value of the tuning parameter. 
## Feature Extraction
The previous approaches operate on the original features.
**Dimension reduction** methods transform the original features and then the model is learned on the transformed variables. The idea is to avoid useless parameters. It is an unsupervised learning techniques.

There are many techniques: Principal Component Analysis (PCA), Independent Component Analysis (ICA), Self-Organizing Maps (SOM), [[Artificial Neural Networks & Deep Learning#Autoencoders (AE)|Autoencoders]].
### Principal Component Analysis (PCA)
It is a **feature extraction** method since it creates new features by transforming or combining existing ones. It is an unsupervised technique which is deterministic (as it does not have any random component). 
The idea is to project data onto a lower-dimensional **orthonormal subspace** while preserving as much variance as possible.
A conceptual algorithm is:
1. Find a direction (line) such that when the data is projected onto that line, it has the maximum variance.
2. Find another direction, orthogonal to the first, that has maximum projected variance.
3. Repeat until $m$ principal components (PCs) have been identified.
4. Project the dataset onto these PCs. 

A more rigorous algorithm:
1. Mean center the data. 
   Since the PC identify the directions where the most of the variance of the data is present (direction is defined as a vector with tail in the origin) we should remove the mean values for each component. 
2. (Optional but recommended) standardize the data to unit variance.
3. Compute the empirical covariance matrix S.
4. Calculate eigenvalues and eigenvectors of $S$: $S = {1 \over N-1} \sum_{n=1}^N(x_n - \overline{x})(x_n - \overline{x})^T$. 
	- The eigenvector $e_k$ with largest non-negative eigenvalue $\lambda_k$ is the $k$-th principal component (PC). 
	- Moreover, $\lambda_k / \sum_i \lambda_i$ is the proportion of variance captured by the $k$-th PC. 
5. Transform the data by projecting onto the top $k$ PCs.

The set of PCs form an orthonormal basis for feature space, whose axes are aligned with the maximum variances of original data. 
The projection of original data onto first $k$ PCs gives a reduced dimensionality reconstruction of the original data, while keeping most variance. In fact, $k$ must be chosen based on how much variance we would like to keep. Moreover, $k\le d$ ($k=d$ means NO reduction).
Reconstruction will have some error, but it often is acceptable given the other benefits of dimensionality reduction. 

There are a few methods to determine how many feature to choose:
- Keep all the principal components until we have a cumulative variance of 90-95%: $$ \text{cumulative variance with k components} = {\sum_{i=1}^k \lambda_i \over \sum_{i=1}^M \lambda_i}  $$
- Keep all the principal components which have more than 5% variance (discard only those which have lower variance).
- Find the elbow in the cumulative variance function (after it stops increasing in a significant way). 

Advantages:
- Help reduce overall computational complexity.
- Improves generalization in supervised learning: reduced dimension gives a simpler hypothesis space and less risk of overfitting. 
- It can be seen as noise reduction, since lower-variance directions may corresponds to noise.
Disadvantages:
- It fails when data consists of multiple clusters. 
- The direction of greatest variance may not be the most informative.
- Computationally expensive in high dimensions. 
- Assumes linearity since it computes linear combinations of features. If data lies on a nonlinear manifold kernel PCA can help.

> [!TIP] Core property of PCA
> Once we project our mean-normalized data onto the principal components (obtaining scores $t_i$), we can reconstruct the original data using the loadings matrix $W$  (i.e. the eigenvectors of the covariance matrix).
> - If we keep ALL principal components, we can reconstruct the original mean-normalized data exactly: $\tilde{x_i} = W t_i$. 
>   The reconstruction is perfect since $W$ is orthogonal: $W^T=W^{-1}$.
> - To reconstruct the original data, we also need to store the mean vector $\mu$ (since PCA is applied on centered data).
> - If we use only the top $k < d$ PC, reconstruction is approximate.

PCA can be used for:
- **Feature extraction**: reduce the dimensionality of the dataset by selecting only the number of PCs retaining information about the problem.
- **Compression**: keep the first $k$ PCs and get $T_k = \tilde{X}W_k$. The linear transformation $W_k$ minimizes the reconstruction error: $$ \min_{W_k \in \mathbb{R}^{M \times k}} ||TW_k^T - \tilde{X}||^2_2$$
- **Data visualization**: reduce the dimensionality of the input dataset with $k=2,3$ to be able to visualize the data. 
## Model Ensembles
The methods seen so far can reduce bias by increasing variance or viceversa. However, ensemble learning can reduce one without significantly increasing the other.
Bagging and boosting are meta-algorithms based on the idea of learning several models and combine their results. It typically improves accuracy by a lot. 
### Bagging
It reduces the variance without increasing the bias.
The key idea is that averaging reduces variance as $Var(\overline{x}) = Var(x) / N$. The problem is that we have one training set.
#### Bootstrap Aggregation
1. Generate $B$ **bootstrap samples** of the training data using **random sampling with replacement**. 
2. Train a classifier or a regression function using each bootstrap sample. 
	- For classification we take the majority vote.
	- For regression we take the average on the predicted results.

Advantages:
- Reduces variance thanks to averaging. 
- Improves performance for unstable learners which vary significantly with small changes in the dataset (low bias and high variance).
- Works particularly well with decision trees.
Disadvantages:
- It does not help much when there is high bias (model robust to change in the training data.)

> [!TIP] Bootstrap
> A **bootstrap sample** is a new dataset created by randomly sampling from the original dataset **"with replacement"** which means that **each time we pick a data point, we put it back**,  so it can be picked again. 
> This means that some points may appear multiple times, while others might not appear at all in a single bootstrap sample.
### Boosting
The idea is to sequentially train **weak learners** (a model which performance is slightly better than chance prediction), then combine them into a strong classifier. 
1. Weight all train samples equally.
2. Train a weak model.
3. Compute training error.
4. **Increase weights for misclassified samples** (so the next model focuses on learning them better).
5. Train new model on re-weighted train set.
6. Repeat steps 3-5 for multiple iterations.
7. Combine all weak models into a **final weighted prediction**.

Advantages:
- It reduces bias.
- It might still help with sable models. 
Disadvantages:
- It might hurt performance on noisy datasets. 
- Weights grow exponentially.

|                    | Bagging                         | Boosting                    |
| ------------------ | ------------------------------- | --------------------------- |
| Bias               | No reduction                    | Reduced                     |
| Variace            | Reduced                         | No reduction                |
| Works best with    | High-variance models (unstable) | High-bias models (stable)   |
| Robust to noise?   | Yes                             | No (can overfit noisy data) |
| Computational cost | Parallelizable (faster)         | Sequential (slower)         |
On average, boosting helps more than bagging, but it is also more common for boosting to hurt performance.

# PAC (Probably Approximately Correct) Learning
PAC learning is a formal framework to study the learnability of functions from data. It defines when a **ML algorithm** can **learn a concept** with **high probability** (confidence) and **low error** (accuracy). 
Overfitting happens because the training error is a bad estimate of the generalization error. It happens when the learner does not see "enough" examples to be able to generalize well. 

Let's define the components of the PAC learning framework:
- $X$: a set of possible instances (input space).
- $H$: the hypothesis space, which is the set of functions $h:X \rightarrow \{0,1\}$ that the learner can choose from. The possible outputs of the learning algorithm. 
- $C$: a set of possible target concepts, where each concept $c: X \rightarrow \{0,1\}$ is a boolean function (e.g. binary classifier).
- $\mathcal{D}$: training instances generated by a unknown probability distribution $\mathcal{P}$ over $X$.
The learner sees labelled data points from $\mathcal{D}$, generated according to the target concept $c \in C$, and must output a hypothesis $h$ estimating $c$.

- **Population Risk Minimization**:
  If we know $\mathcal{P}$, the learned hypothesis $h$ is evaluated according to its true error: $$L_{true} = Pr_{x \in \mathcal{P}} [c(x) \neq h(x)]$$However, since $\mathcal{P}$ and $c$ are unknown, we cannot compute $L_{ture}$ directly, so we want to bound $L_{true}$ given $L_{train}$.
- **Empirical Risk Minimization**:
  Given the training dataset $\mathcal{D}$ of i.i.d. samples drawn from $\mathcal{P}$, the learned hypothesis $h$ is evaluated according to the training error: $$L_{train} = {1 \over N} \sum_{n=1}^N l(h(x_n), t_n)$$
Note that $L_{train}$ is a negatively biased estimator for $L_{true}$. On the other hand, $L_{test}$ is an unbiased estimator of $L_{true}$.

A **version space** $VS_{H, \mathcal{D}}$ is a subset of hypotheses in $H$ consistent with training data $\mathcal{D}$: 
$$
VS_{H, \mathcal{D}} = \{h \in H | h(x) = y \ \  \forall (x, y) \in \mathcal{D}\}
$$
So it is the subset of $H$ that agrees with all training examples. 

First, we consider when $L_{train}(h) = 0$ (perfect classification).

> [!THEOREM] PAC bound for finite hypothesis spaces
> If the hypothesis space $H$ is finite and $\mathcal{D}$ is a sequence of $N \ge 1$ independent examples of some target concept $c$, then for any $0 \le \epsilon \le 1$, the probability that $VS_{H, \mathcal{D}}$ contains a hypothesis error greater than $\epsilon$ is less than $|H|e^{-\epsilon N}$.
> $$Pr(\exists h \in H : L_{train}(h) = 0 \land L_{true}(h) \ge \epsilon) \le |H|e^{-\epsilon N}$$

Proof sketch: we want to bound the probability that any "bad" hypothesis (one with true error at least $\epsilon$) is still consistent with the training data.
$$
\begin{align}
	&Pr((L_{train}(h_1) \land L_{true}(h) \ge \epsilon)) \lor ... \lor (L_{train}(h_{|H|}  = 0 \land L_{true}(h_{H}) \ge \epsilon)) \\ (\text{union bound})  
	&\le \sum_{h \in H} Pr(L_{train}(h) = 0 \land L_{true}(h) \ge \epsilon) \\ (\text{using Bayes' rule})
	&\le \sum_{h \in H} Pr(L_{train}(h) = 0 | L_{ture}(h) \ge \epsilon) \\ (\text{bound on individual } h_i s)
	&\le \sum_{h \in H_{bad}} (1-\epsilon)^N \\ (|H_{bad}| \le |H|)
	&\le |H|(1-\epsilon)^N \\ (1 - \epsilon \le e^{-\epsilon}, for \ 0 \le \epsilon \le 1)
	&\le |H|e^{-\epsilon N}
\end{align}
$$

If we want $|H|e^{-\epsilon N} \le \delta$, then we can:
- pick $\epsilon$, $\delta$ and compute N: $$ N \ge {1 \over \epsilon}(\ln|H| + \ln({1 \over \delta})) $$ Note that the number of samples $N$ grows with the logarithm of the hypothesis space $H$ (we are overfitting).
- pick $N$, $\delta$ and compute $\epsilon$: $$ \epsilon \ge {1 \over N} (\ln|H| + \ln({1 \over \delta})) $$So more data or smaller hypothesis space both lead to tighter bounds on error. 

Note that if instances are described by $M$ binary features, then the number of possible M-ary boolean functions is $|C| = 2^{2^M}$. This means that:
- The hypothesis space becomes exponentially large in M: the bounds have an exponential dependency on the number of feature M.
- The required number of samples $N$ becomes infeasibly large.
- Overfitting id more likely without regularization or restriction on $H$.
## PAC Learnability
Consider a class $C$ of possible target concepts defined over a set of instances $X$ (of length $n$), and a learner $L$ using hypothesis space $H$.

> [!Definition] PAC-learnable
> $C$ is **PAC-learnable** if there exists an algorithm $L$ such that;
> - for every $f \in C$,
> - for any distribution $\mathcal{P}$, 
> - for any $0 \le \epsilon \le {1 \over 2}$ and $0 \le \delta \le {1 \over 2}$
> 
> the algorithm $L$, with probability at least $(1 - \delta)$, outputs a hypothesis $h$ such that 
> $$L_{true}(h) \le \epsilon$$ 
> using a number of samples that is polynomial of $1 \over \epsilon$ and $1 \over \delta$.

> [!Definition] Efficiently PAC-learnable
> C is **efficiently PAC-learnable** by $L$ using $H$ if and only if for all $c \in C$, distributions $\mathcal{P}$ over $X$, $0 \le \epsilon \le {1 \over 2}$ and $o \le \delta \le {1 \over 2}$, learner $L$ will with probability at least $(1-\delta)$ output a hypothesis $h \in H$ such that 
> $$L_{true}(h) \le \epsilon$$
> in time that is polynomial in $1 \over \epsilon$, $1 \over \delta$ and $size(c)$.

## Agnostic Learning
In practice, the train error is not equal to 0, so the version space may be empty. When we have inconsistent hypothesis, we have to bound the gap between training and true errors: 
$$
L_{true}(h) \le L_{train}(h) + \epsilon
$$
> [!TIP] Agnostic definition
> In philosophy, **agnosticism** means “not knowing”, not committing to a belief either way. In **learning theory**, it means that we don’t assume the data was generated by a perfect function in our hypothesis space.

Using the Hoeffding bound: 
For $N$ i.i.d. coin flips $X_1, .., X_N$ where $X_i \in \{0,1\}$ and $0 \le \epsilon \le 1$, we define the empirical mean $\overline{X} = {1 \over N}(X_1 + .. + X_N)$, obtaining the following bound: 
$$
Pr(\mathbb{E}[\overline{X}] - \overline{X} > \epsilon) < e^{-2N\epsilon^2}
$$
We obtain a variation of the previous theorem:

> [!THEOREM] $L_{train}$ bound
> Let $H$ be a finite hypothesis space, and $\mathcal{D}$ a dataset with $N$ i.i.d. samples. Then for any $0 \le \epsilon \le 1$, it holds:
> $$
> Pr(\exists h \in H | L_{true}(h) - L_{train}(h) > \epsilon) \le |H|e^{-2N\epsilon^2}
> $$

> [!THEOREM] $L_{test}$ bound 
> Let $M$ i.i.d. samples to form a test set. Then, for any hypothesis $h \in H$ and any $\epsilon > 0$, with probability at least $1 - \delta$, the test error is bounded by: $$L_{test}(h) \le L_{train}(h) + \sqrt{\ln({2 \over \delta}) \over 2M}$$​​

> [!ATTENTION] Generalization of the Hoeffding bound
> In general $X_i \in [a, b]$, thus: $$Pr(\mathbb{E}[\overline{X}] - \overline{X} > \epsilon) < e^{-2N\epsilon^2 \over (b-a)^2}$$
> 
> So, if the loss we consider is bounded $l(y(x_i),t_i) \in [0, L]$, we have to account for the scaling factor $L$. The PAC bound becomes: $$ Pr(\exists h \in H | L_{true}(h) - L_{train}(h) > \epsilon) \le |H|e^{-2N\epsilon^2 \over L^2} $$
>

### PAC bound & Bias-Variance trade off
The generalization bound can also be written as:
$$
L_{true}(h) \le \underbrace{L_{train}(h)}_{\text{Bias}} + \underbrace{\sqrt{\ln|H| + \ln {1 \over \delta} \over 2N}}_{\text{Variance}}
$$
- For large $|H|$: potentially low bias (assuming we can find a good $h$) but high variance (looser bound).
- For small $|H|$: high bias but lower variance (tighter bound).
To ensure generalization with error at most $\epsilon$ and confidence $1-\delta$, $N$ should be at least:
$$
N \ge {1 \over 2 \epsilon^2}(\ln|H| + \ln {1 \over \delta})
$$
# Vapinik-Chervonenkis (VC) dimension
A **dichotomy** of a set $S$ is a partition of $S$ into two disjoint subset. 
A set of instances $S$ is **shattered** by hypothesis space $H$ if and only if for every possible labeling (dichotomy) of $S$ there exists some hypothesis in $H$ that classifies all elements in $S$ correctly according to that labeling. 

The VC dimension of hypothesis space $H$ defined over instance space $X$ is the **size of the largest finite subset of $X$ shattered by $H$**. If arbitrarily large finite sets of $X$ can be shattered by $H$, then $VC(H) \equiv \infty$.
Informally, the VC dimension of a hypothesis space is a measure of its capacity, that is how complex or expressive it is. It tells us how many data points the hypothesis can perfectly fit (shatter) regardless of the labeling.

A linear classifier in $\mathbb{R}^M$ can classify at most $M+1$ points in general position. As a rule of thumb the number of parameters in a model often approximates the maximum number or points that it can classify (but in general it is not guaranteed). 
It is possible to:
- Have models with infinite parameters and finite VC dimension.
- Have a model with one parameter but infinite VC dimension.

To guarantee that the true error is at most $\epsilon$ with probability at least $1-\delta$, the number of training samples must satisfy:
$$
N \ge {1 \over \epsilon}(4 \log_2({2 \over \delta}) + 8VC(H)\log_2({13 \over \epsilon}))
$$
## PAC bounding using VC dimension
If $h$ is a hypothesis returned by a learner based on $N$ i.i.d. training examples:
$$
L_{true}(h) \le L_{train}(h) + \sqrt{VC(H)(\ln{2N \over VC(H)} +1) +\ln{4 \over \delta} \over N}
$$
which shows how the VC dimension directly influences the gap between train and test error and whether a model generalizes well.
Same bias-variance trade off as always.
**Structural Risk Minimization** (SRM) is a principle where we choose the hypothesis space $H$ to minimize the generalization bound: instead of minimizing just the training error we take into account the complexity of the model (trough VC dimension). We prefer simpler models if they performs comparably to complex ones.

The VC dimension of a **finite hypothesis space** is **upper bounded**, in fact $VC(H) \le \log_2(|H|)$. If $VC(H) = d$, then there exists at least $2^d$ functions in $H$, since there are at least $2^d$ possible labelings of $d$ points: $|H| \ge 2^d$.
Moreover, concept class $C$ with $VC(C) = \infty$ is not PAC-learnable: the idea is that no finite sample size can guarantee generalization for all target concepts in $C$.

> [!TIP] PAC-learning and VC dimension
> - Finite VC dimension -> PAC learnable.
> - Infinite VC dimension -> not PAC learnable.
> This is why VC dimension is often called the combinatorial measure of learnability. 
# Kernel Methods
Kernel methods are memory-based (e.g. K-NN): they need to keep the training data since it is used during the prediction phase.
They are fast to train, but slow to predict: the bottleneck is the number of samples, so we can increase the number of features without "paying" computationally. 
They require a metric to be defined. 

In the case the model we are considering is not performing well even by tuning properly the parameters (e.g. cross-validation), we have two opposite options: simplify the model or increase its complexity. In the second case, one might see the problem in a more complex space: the kernel space.

Kernels makes linear models work in non-linear settings:
- by **mapping** data to higher dimensions where it exhibits linear patterns. They change the feature space representation. Mapping can be expensive but kernels give them for (almost) free.
- by applying the linear model to the new input space.
## Feature Mapping
Given a feature space $x = \{x_1, .., x_M\}$, consider the following mapping: 
$$
\phi : x \rightarrow \{x_1^2, .., x_M^2, x_1 x_2, .., x_1 x_M, .., x_{M-1} x_M\}
$$
It is an example of **quadratic mapping**: each new feature uses a pair of the original features.
Some problems arises: 
- Mapping usually lead to the number of features to blow up.
- Computing the mapping itself can be inefficient.
- Using the mapped representation could be inefficient too.
However, kernels help avoid both these issues as the mapping does not have to be explicitly computed and the computations with the mapped features remain efficient. 
## Kernel functions
Many linear parametric models can be re-cast into equivalent dual representations where predictions are based on a kernel function evaluated at training points. 
A kernel function is given by the **scalar product** of: 
$$
k(x, x') = \phi(x)^T \phi(x')
$$
where $\phi(x)$ is a fixed nonlinear feature space mapping (basis function). 
It is a **symmetric** function of its arguments: $k(x, x') = k(x', x)$ (as the scalar product).
It can be interpreted as similarity of $x$ and $x'$.

- The simplest kernel function is the identity mapping in the feature space: $\phi(x) = x$. It is called linear kernel and $k(x, x') = k(x', x)$.
- Function difference between arguments: $k(x, x') = k(x-x')$. It is called stationary kernel since it is invariant to translation in space.
- Homogeneous kernel, also known as radial basis functions: $k(x, x') = k(||x-x'||)$.
Note that the kernel function is a scalar value while $x$ is an M-dimensional vector. 

The **kernel trick** is an inner product that allows extending well-known algorithms. 
Idea: if an input vector $x$ appears only in the form of scalar products then we can replace scalar products with some other choice of kernel. 
## Dual Representation - kernelized Ridge regression
Many linear models for regression and classification can be reformulated in terms of dual representation in which the kernel function arises naturally. This plays an important role in [[Machine Learning#Support Vector Machines (SVM)|SMV]].

Consider a linear regression model, in particular **Ridge regression**, the parameters are obtained by minimizing the regularized sum-of-squares error function:  $$ L_w = {1 \over 2} \sum_{n=1}^N (w^T \phi(x_n) -t_n)^2 + {\lambda \over 2} w^T w $$then, to get the dual formulation, we set the gradient of $L_w$ with respect to $w$ equal to zero: 
$$
w = -{1 \over \lambda} \sum_{n=1}^N (w^T \phi(x_n) - t_n) \phi(x_n) = \sum_{n=1}^N a_n \phi(x_n) = \Phi^T a
$$
where $\Phi$ is the design matrix whose $n^{th}$ row is $\phi(x_n)^T$. 
The coefficients $a_n$ are functions of $w$:
$$
a_n = -{1 \over \lambda} (w^T \phi(x_n) -t_N)
$$
We define the **Gram matrix** $K = \Phi \times \Phi^T$ as the $N \times N$ matrix, with elements: 
$$
K_{nm} = \phi(x_n)^T \phi(x_m) = k(x_n, x_m)
$$
Given $N$ vectors, the Gram matrix is the matrix of all **inner products**:
$$
K =
\begin{bmatrix}
k(x_1, x_1) & ... & k(x_1, .., x_N) \\
.. & ... & ..\\
k(x_N, x_1) & ... & k(x_N, x_N)
\end{bmatrix}
$$
Note that:
- $\Phi$ is $N \times M$ and $K$ is $N \times N$.
- $K$ is a matrix of similarities of pairs of samples, therefore it is symmetric. 

It is possible to write the error function in terms of the Gram matrix of kernel. 
Substituting $w = \Phi^T a$ into $L_w$ gives: 
$$
L_w = {1 \over 2} a^T \Phi \Phi^T \Phi \Phi^T a - a^T \Phi \Phi^T t + {1 \over 2}t^T t + {\lambda \over 2} a^T \Phi \Phi^T a
$$
where $t = (t_1, .., t_N)^T$.
The sum of squares error function written in terms of Gram matrix is: 
$$
L_a = {1 \over 2} a^T K K a - a^T K t + {1 \over 2}t^Tt + {\lambda \over 2}a^T K a
$$
Solving for $a$ by combining $w=\Phi^T a$ and $a_n = -{1 \over \lambda} (w^T phi(x_n)- t_n)$ gives:
$$
a = (K + \lambda I_N)^{-1}t
$$
The solution for $a$ can be expressed as a linear combination of elements of $\phi(x)$ whose coefficients are entirely in terms of kernel $k(x, x')$ from which we can recover the original formulation in terms of parameters $w$.

To predict for a new input $x$ we can substitute $a$ back into the linear regression model, obtaining: 
$$
y(x) = w^T \phi(x) = a^T \Phi \phi(x) = k(x)^T (K + \lambda I_N)^{-1}t
$$
where $k(x)$ has elements $k_n(x) = k(x_n, x)$.
The prediction is a linear combination of the target values from the training set, with weights according to the similarity.

Thanks to the dual representation, the solution for $a$ is expressed entirely in terms of the kernel function $k(x, x')$. 
Once we get $a$ we can recover $w$ as linear combination of elements of $\phi(x)$ using $w=\Phi^T a$.
In parametric formulation, the solution is $w_{ML} = (\Phi^T \Phi)^{-1} \Phi^T t$: instead of inverting an $M \times M$ matrix, we are inverting an $N \times N$ matrix (an apparent disadvantage). 

The main advantage of the dual formulation is that we can work with the kernel function $k(x, x')$ and therefore avoid working with a feature vector and problems associated with high or infinite dimensionality of x. 
Moreover, the kernel functions can be defined not only over simply vectors of real numbers, but also **over objects** (e.g. graphs, sets, string, text documents) as we just have to define a metric for similarity.

Let $D = \{ (x_i, y_i) \}_{i=1}$​ be the training dataset and $x^*$ be a test point.
We define:
- $K_{X, X}$ as the **Gram matrix** of the training points (size $N \times N$).
- $K_{x^*, X}$​ as the vector of kernel evaluations between the test point and each training point (size $1 \times N$).
- $K_{X, x^*} = K_{x^*, X}^T$​ as the transposed version (size $N \times 1$).
- $K_{x^*, x^*}$​ as the kernel evaluation at the test point (a scalar).
#### Posterior Mean
$$\mu^* (x^*) = K_{x^*, X} \cdot (K_{X, X} + \sigma_n^2 I)^{-1} \cdot y$$
where:
- $\sigma_n^2$​ is the noise variance in the observations (which is added to the kernel matrix to stabilize the inversion). 
- $y$ is the vector of training outputs (size $N \times 1$).
It depends on the output sample.
#### Posterior Variance
$$\sigma^{*2}(x^*) = K_{x^*, x^*} - K_{x^*, X} \cdot (K_{X, X} + \sigma_n^2 I)^{-1} \cdot K_{X, x^*}$$
It does not depends on the output sample. 

In both cases, the overall complexity is **dominated by the cubic complexity** of the matrix inversion step.
## Constructing Kernels
To exploit kernel substitution, we need valid kernel functions.

The first method is to choose a feature space mapping $\phi(x)$ and use it to find a corresponding kernel: 
$$
k(x, x') = \phi(x)^T \phi(x) = \sum_{i=1}^M \phi_i(x) \phi_i(x')
$$
where $\phi(x)$ are basis functions such as polynomial and for each $i$ we choose $\phi_i(x) = x^i$. 

The second method is a direct construction: the function we choose has to correspond to a scalar product in some (perhaps infinite dimensional) space.
Without having to construct the function $\phi(x)$ explicitly, a **necessary and sufficient condition** for a function to be a kernel is that the **Gram matrix** $K$, whose elements are given by $k(x_n, x_m)$ is **positive semi-definite** for all possible choice of the set $\{x_n\}$.

> [!THEOREM] Mercer's theorem
> **Any** continuous, symmetric, positive semi-definite kernel function $k(x, x')$ can be expressed as a **dot product** in a high-dimensional space.

New kernels can be constructed from simpler kernels as building blocks.
Given valid kernels $k_1(x, x')$ and $k_2(x, x')$, the following new kernels will be valid:
$$
\begin{align}
1) \ k(x, x') &= ck_1(x, x')\\
2) \ k(x, x') &= f(x)k_1(x,x')f(x') & \text{where } f(\cdot) \text{ is any function}\\
3) \ k(x, x') &= q(k_1(x, x')) & \text{where } q(\cdot) \text{ is a polynomial with non-negative coefficients}\\
4) \ k(x, x') &= exp(k_1(x, x'))\\
5) \ k(x, x') &= k_1(x, x') + k_2(x, x')\\
6) \ k(x, x') &= k_1(x, x') \cdot k_2(x, x')\\
7) \ k(x, x') &= k_3(\phi(x), \phi(x')) & \text{where } \phi(x) \text{ is a function from } x \text{ to } \mathbb{R}^M\\
8) \ k(x, x') &=  x^T A x' & \text{where } A \text{ is a symmetric postive semi-definite matrix}\\
9) \ k(x, x') &=  k_a(x_a, x_a') + k_b(x_b, x_b') & \text{where } x_a, x_b \text{ are variables with } x = (x_a, x_b)\\
10) \ k(x, x') &=  k_a(x_a, x_a') \cdot k_b(x_b, x_b')\\
\end{align}
$$
### Gaussian Kernel - Radial Basis Function (RBF)
A commonly used kernel is the Gaussian  or Radial Basis Function (RBF) kernel:
$$
k(x, x') = \exp({-||x-x'||^2 \over 2 \sigma^2})
$$
It is valid: by expanding the square we get $||x-x'||^2 = x^Tx + x'^Tx' - 2x^Tx'$ so that $k(x, x') = \exp(-{x^Tx \over 2 \sigma^2})\exp(-{x'^Tx' \over 2\sigma^2})\exp({x^Tx' \over \sigma^2})$, then from kernel construction rules 2 and 4 together with the validity of linear kernel we have proved it to be valid.
It can be extended to non-Euclidean distances:
$$
k(x, x') = \exp(-{1 \over 2 \sigma^2}(\kappa(x, x) + \kappa(x', x') -2\kappa(x, x')))
$$
Kernels can be extended to inputs that are symbolic, rather than simply vectors of real numbers. 

Given a generative model $p(x)$ we define a kernel by $k(x, x') = p(x)p(x')$. 
It is valid since it is an inner product in the one-dimensional feature space defined by the mapping $p(x)$. 
The idea is that two inputs $x$ and $x'$ are similar if they have high probabilities of being generated in a certain context. 

Radial basis functions are function that depend only on the radial distance (typically Euclidean) of a point $x$ from a center $\mu_i$: 
$$
\phi_j (x) = h(\|x-\mu_j|_2)
$$
RBF can be used for exact interpolation: 
$$
f(x) = \sum_{n=1}^N w_ h(\|x-x_n\|_2)
$$
However, since the data in ML are generally noisy, exact interpolation is not very useful. Exact interpolation can lead to overfitting, where the model tries to perfectly fit noisy training data, reducing generalization to new data points.

To mitigate issues related to regions of low basis function activation, we can use normalized radial basis functions. 
Normalization is sometimes used in practice as it avoids having regions of input space where all basis functions takes small values, which would necessarily lead to predictions in such regions that are either small or controlled purely bu the bias parameter. 
### Automatic Relevance Determination (ARD) kernel
In many datasets, some input features may be more relevant than others for making predictions.
The ARD kernel is a variation of the standard RBF kernel that assigns a separate length scale $\sigma_d$​ to each input dimension $d$. This allows the model to learn the relative importance of each feature, effectively performing feature selection. It is defined as:
$$
k(x, x') = \sigma^2 exp(-\sum_d \frac{\|x_d-x_d'\|^2}{2l_d^2})
$$
where $\sigma^2$ is the signal variance, $l_d$ is the length-scale for dimension $d$ (learned from data):
- a small $l_d$ means that the output changes rapidly with $x_d$, so the dimension is important. 
- a large $l_d$ means that the output is insensitive to $x_d$, so the dimension is irrelevant. 
### Gaussian Processes 
A Gaussian process is a **probability distribution over functions** $y(x)$ such that the values of the function at any set of points $x_1, .., x_N$ jointly have a Gaussian distribution. 
A GP is completely specified by (second-order statistics):
- A **mean** function $\mu(x)$.
- A **covariance** function (kernel) $K(x_i, x_j)$.
#### Prior distribution
Usually we do not have any prior information about the mean of $y(x)$, so we take it to be zero while the covariance is given by the kernel function:
$$
Cov[y(x_i), y(x_j)|x_i, x_j] = \mathcal{E}[y(x_i)y(x_j)|x_i, x_j] = K(x_i, x_j)
$$
#### Posterior distribution
To compute the posterior **mean**, the following formula is used:
$$
\mu(x) = k(x, x_t)(K_t + \sigma^2I)^{-1} y_t
$$
where $x_t$ is the test point, $K_t$ is the covariance matrix of the training data, $\sigma^2$ is the noise variance and $I$ is the identity matrix. 

To compute the posterior **variance** we use:
$$
s^2(x) = k(x, x) - k(x, x_t)^T(K_t + \sigma^2I)^{-1}k(x, x_t)
$$

With this formulation, GP are kernel methods that can be applied to solve regression problems. 
# Support Vector Machines (SVM)
It is one of the best method for classification. They can also be used for regression, ranking, feature selection, clustering and semi-supervised learning. 
A SVM consist of:
- **Support vectors**: a subset of training examples $x$ that are most influential in defining the decision boundary.
- **Weights vector** ($a$): coefficients that represents the importance of each support vector in determining the decision boundary.
- **Kernel function**: a similarity function $K(x, x')$ that implicitly maps data into a higher-dimensional space to make it linearly separable. 

For a given input $x_q$, the SVM outputs a class prediction $t_i \in \{-1, 1\}$ based on the following decision function:
$$f(x_q) = sign(\underbrace{\sum_{m \in \mathcal{S}} \alpha_m t_m k(x_q, x_m)}_{\text{linear combination}}+ b)$$
where $\mathcal{S}$ is the set of indices of the support vectors and $b$ is the bias term.
SVMs are often considered a more sophisticated extension of the [[Artificial Neural Networks & Deep Learning#The perceptron|perceptron]], leveraging the concept of margin maximization and kernel methods to handle complex, non-linear data.
## From Perceptron to Kernel Methods
The key idea is that the perceptron decision function can be reformulated in a way that resembles the weighted k-NN decision rule: 
- Take the perceptron.
- Replace the dot product with an arbitrary similarity function: it is still a dot product but in a transformed space.
- As result, we have a more powerful learner that can handle complex, non-linear patterns while maintaining convex optimization, meaning there are no local minima, only a single global optimum.
The perceptron decision function can be written as:
$$
f(x_q) = sign(\sum^M_{j=1} w_j \phi_j(x_q))
$$
where $w_j$ are the weights for each feature and $\phi_j(x_q)$ is the feature representation of the input $x_q$.
However, the weights $w_j$​ can also be expressed as a linear combination of the training examples:
$$
w_j = \sum_{n=1}^N \alpha_n t_n \phi_j(x_n)
$$
Substituting this in the original function gives:
$$
\begin{align}
	f(x_q) &= sign(\sum_{j=1}^M \sum_{n=1}^N \alpha_n t_n \phi_j(x_n) \phi_j(x_q)) \\ 
	&= sign(\sum_{n=1}^N \alpha_n t_n \sum_{j=1}^M \phi_j(x_n) \phi_j(x_q)).
\end{align}
$$
Notice that the inner sum is the dot product $\phi(x_q) \cdot \phi(x_n) = \sum_{j=1}^M \phi_j(x_q) \phi_j(x_n)$. Thus, the decision function can be written as:
$$
f(x_q) = sign(\sum_{n=1}^N \alpha_n t_n (\phi(x_q) \cdot \phi(x_n)))
$$
In conclusion:
- The perceptron decision function can be expressed as a weighted sum over training samples, where the weights are given by $\alpha_n t_n$.​
- The similarity function is the dot product $\phi(x_q) \cdot \phi(x_n)$.
- Thus, the perceptron can be viewed as an **instance-based learner** where the similarity function is the dot product in the feature space.
This formulation directly connects the perceptron to kernel methods and provides the foundation for the SVM.
## Learning SVMs
The goal is to **maximize the margin**, which is the distance between the decision boundary (hyperplane) and the closest data points from either class, the support vectors: 
$$
\min_n t_n (w^T \phi(x_n) + b)
$$
Maximizing this margin helps improve generalization by ensuring the classifier has a **robust separation between classes**, which leads to **better performance on unseen data**.
It is a weight optimization problem. 

![](./assets/SVM_2.png)
### Primal problem
To maximize the margin, we have to find the optimal $w^*$ and $b^*$ by solving:
$$
w^* = \arg \max_{w, b} (\frac{1}{\|w\|_2} \min_n (t_n (w^T \phi(x_n) +b)))
$$
Since the direct solution is complex, we need to consider an equivalent problem which is easier to be solved: we fix the margin to 1 and minimize weights.
$$
\begin{align}
& \text{Minimize} & \frac{1}{2} \|w\|_2^2\\
& \text{Subject to} & t_n(w^T \phi(x_n) + b) \ge 1 \\ & &\text{forall } n\\
\end{align}h_i(w) = 0
$$
- The objective $\frac{1}{2} \|w\|_2^2$​ is chosen to make the optimization problem **convex and differentiable** (a quadratic programming problem). 
- The margin constraint ensures that **all training points are correctly classified with a margin of at least 1**.
It as a constrained optimization problem (in this general form):
$$
\begin{align}
& \text{Minimize} & f(w)\\
& \text{Subject to} & h_i(w) = 0, \ \forall i\\
\end{align}
$$
if $f$ and $h_i$ are linear we have linear programming, but in this case we have a quadratic problem: it possible to apply optimization techniques such as **Lagrangian multipliers** or and **KKT conditions** (Karush-Kuhn-Tucker).

To solve the constrained optimization problem, we construct the Lagrangian function: 
$$
L(w, \lambda) = f(w) + \sum_i \lambda_i h_i (w)
$$
where $\lambda_i$ are the Lagrangian multipliers (one for each constraint, $\ge 0$ to ensure that the constraint are respected). 
We solve $\nabla L (w^*, \lambda^*) = 0$, to obtain the optimal solution. 

- The solution $w^*$ lies in the span of the support vectors.
- The optimal weight vector is expressed as a **linear combination of the support vectors**, each weighted by $\lambda_n t_n$​.
#### Inequality constraints
When dealing with inequality constraints, the optimization problem can be formulated as: 
$$
\begin{align}
& \text{Minimize} & f(w)\\
& \text{Subject to} & g_i(w) \le 0, \  \forall i\\
& & h_i(w) =0, \ \forall i
\end{align}
$$
We introduce Lagrangian multipliers $\alpha_i \ge 0$ for the inequality constraints and $\lambda_i$ for the equality constraints. 
The Lagrangian function becomes:
$$
L(w, \alpha, \lambda) = f(w) + \sum_i \alpha_i g_i(w) + \sum_i \lambda_i h_i(w)
$$
The inequality constraints in SVMs are given by:
$$
t_n (w^T \phi(x_n) + b) - 1 \geq 0
$$
- If $\alpha_i > 0$, the corresponding constraint is **active**, meaning the sample is a **support vector**.
- If $\alpha_i = 0$, the sample is **not a support vector** and does not influence the final decision boundary.

The **KKT conditions** provide the **necessary conditions for optimality in constrained optimization problems with inequality constraints**:
$$
\begin{align}
& \nabla L(w^*, \alpha^*, \lambda^*) = 0 & \text{sationarity} \\
& h_i(w^*) = 0 \\
& g_i(w^*) \le 0 &\text {primal feasibility} \\
& \alpha_i^* \ge 0 & \text{dual feasibility} \\
& \alpha_i^∗​ g_i​(w^∗) = 0 & \text{complementary slackness}
\end{align}
$$
According to the complementary slackness, either the constraint $g_i(w^*) = 0$ is active or its multiplier $\alpha_i^*$ is zero.
### Dual Problem
In optimization, we can often approach the same problem from two perspectives:
- **Primal Problem:** focuses on the **feature space** and the weight vector $w$.
- **Dual Problem:** re-frames the optimization in terms of the Lagrange multipliers $\alpha$, operating in the **instance space**.
The dual problem can be derived by applying the Lagrangian formulation and eliminating $w$ through the KKT conditions:
$$
\begin{align}
& \text{Minimize} & \sum_{n=1}^N \alpha_n - \frac{1}{2} \sum_{n=1}^N \sum^N_{m=1} \alpha_n \alpha_m t_n t_m k(x_n, x_m)\\
& \text{Subject to} & 0 \le \alpha_n,\ \forall n\\
& & \sum_{n=1}^N \alpha_n t_n = 0
\end{align}
$$
- The dual problem is now a **quadratic programming problem** involving only the Lagrange multipliers $\alpha_n$.
- The kernel function $k(x_n, x_m)$ allows for **non-linear decision boundaries** without explicitly transforming the feature space.    
- Most $\alpha_n$​ values will be zero. The non-zero $\alpha_n$ correspond to the **support vectors**, the points that define the margin.

When solving the SVM optimization problem, standard quadratic programming solvers can become impractical due to the large number of constraints (memory and time complexity grow **quadratically with the number of samples**). 
Specialized algorithm are preferred to handle large dataset efficiently (e.g. Sequential Minimal Optimization).
## SVMs prediction
After training the SVM and obtaining the optimal $\alpha_n$ and $b$, we can classify a new point $x$ using the following decision function:
$$
y(x) = sign(\sum^N_{n=1} \alpha_n t_n k(x, x_n) + b)
$$
The bias term can be computed as follow: 
$$
b = \frac{1}{N_{\mathcal{S}}} \sum_{n \in \mathcal{S}} (t_n - \sum_{m \in \mathcal{S}} \alpha_m t_m k(x_n, x_m))
$$
where $\mathcal{S}$ is the set of support vectors and $N_{\mathcal{S}}$ is the number of support vectors.
Notice that $N_{\mathcal{S}}$ is usually much smaller than $N$, leading to a sparse representation of the model. 
## Curse of dimensionality
As the number of dimensions (features) increases, several issues arises in SVMs:
- **Increase in support vectors**: as in higher dimensions data points tends to be more sparse, and more support vectors are required to define the separating hyperplane.
- **Scalability** issues: as the **computational cost** of evaluating the kernel function grows with the number of support vectors. Moreover, **memory requirement** increases as each support vector needs to be stored and processes during prediction. 
- Impact on **generalization**: with more support vectors the SVMs may overfit.
### Bounds: SVMs generalization capacity
#### Margin bound
The **bound on the VC dimension decreases as the margin increases**: a larger margin reduces the model's capacity to overfit, effectively lowering the VC dimension. 
However, the margin bound is generally considered **loose**, meaning it may not provide a tight estimate of the actual generalization error. 
#### Leave-One-Out Bound (LOO Bound)
An upper bound on the leave-one-out error can be easily computed using the number of support vectors:
$$
L_h \le \frac{E[N_s]}{N}
$$
where $N_s$ is the number of support vectors and $N$ is the total number of training samples. 
This bound does not need to retrain the SVM multiple times. It provides a direct measure of generalization based on the number of support vectors. 
## Handling noisy data: soft-margin SVM
Real-world data is often noisy, making it challenging to find a perfectly separable hyperplane. 
To address this, **soft-margin SVMs** introduce slack variables $\xi_i \ge 0$ which allows some data points to violate the margin constraint, account for missclassification or noisy data. 

The objective function now includes a **penalty term** for these violations, controlling the trade-off between maximizing the margin and minimizing classification errors.
$$
\begin{align}
& \text{Minimize} & \|w\|_2^2 + C \sum_i \xi_i \\
& \text{Subject to} & t_i(w^Tx_i + b) \ge 1 - \xi_i, \  \forall i\\
& & \xi_i \ge 0, \ \forall i
\end{align}
$$
where $C$ is a regularization parameter (hyperparameter), controlling the **trade-off between margin width and misclassification tolerance**:
- a larger $C$ results in a stricter margin, prioritizing classification accuracy over margin size. 
- a smaller $C$ results in a wider margin, allowing more misclassifications to prevent overfitting.
The value of $C$ is typically chosen through cross-validation, balancing the bias-variance trade-off.

Now, the dual formulation accounts for the $C$ term:
$$
\begin{align}
& \text{Minimize} & \sum_{n=1}^N \alpha_n - \frac{1}{2} \sum_{n=1}^N \sum^N_{m=1} \alpha_n \alpha_m t_n t_m k(x_n, x_m)\\
& \text{Subject to} & 0 \le \alpha_n \le C,\ \forall n\\
& & \sum_{n=1}^N \alpha_n t_n = 0
\end{align}
$$
where support vectors are points associated to an $\alpha_n \ge 0$:
- if $\alpha_n < C$ the points lies on the margin.
- if $\alpha_n = C$ the point lies inside the margin, and it can be either correctly classifies ($\xi_i \le 1$) or misclassified ($\xi_i > 1$).