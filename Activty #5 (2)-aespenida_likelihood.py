"""
Created on Sat Mar  9 19:52:27 2024

@author: aespenida 
"""
import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(1.65, 1.8, num=50)
test = np.linspace(0, 2)
uniform_dist = sts.uniform.pdf(mu) + 1
uniform_dist = uniform_dist / uniform_dist.sum()
beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.2)
beta_dist = beta_dist / beta_dist.sum()

def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)
    return likelihood_out / likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability Density/Likelihood")
plt.xlabel("Value of $\mu$")
plt.show()


