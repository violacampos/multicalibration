import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
import pandas as pd
import matplotlib.pyplot as plt

test_alpha = 0.5
test_beta = 0.75

x = np.random.rand(100)
logit_x = expit(test_alpha + test_beta * logit(x))


# mse function to optimize for alpha and beta
def mse(params):
    alpha, beta = params
    transformed = expit(alpha + beta * logit(x))  # LS[f](x)
    return np.mean((transformed - logit_x) ** 2)  # calculate the MSE

# minimize for the mse and get alpha and beta values
result = minimize(mse, x0=[0, 1.0])  
alpha_star, beta_star = result.x

print(alpha_star, beta_star)