import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
import pandas as pd
import matplotlib.pyplot as plt


# mse function to optimize for alpha and beta
def mse(params):
    alpha, beta = params
    transformed = expit(alpha + beta * x)  # LS[f](x)
    print(np.mean((transformed - y) ** 2))
    return np.mean((transformed - y) ** 2)  # calculate the MSE

# minimize for the mse and get alpha and beta values
result = minimize(mse, x0=[0, 1.0])  
alpha_star, beta_star = result.x

print(alpha_star, beta_star)

