# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:14:52 2024

@author: acer
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the prior distribution (uniform)
def prior_distribution(min_val, max_val, num_points):
    return np.ones(num_points) / num_points

# Define the likelihood function
def likelihood(waiting_time, observed_time):
    # For simplicity, assume a normal distribution with fixed variance
    variance = 20
    return np.exp(-0.5 * ((waiting_time - observed_time) / np.sqrt(variance))**2)

# Perform Bayesian inference
def bayesian_inference(prior, observed_time):
    likelihoods = likelihood(np.linspace(0, 60, len(prior)), observed_time)
    posterior = prior * likelihoods
    posterior /= np.sum(posterior)  # Normalize the posterior distribution
    return posterior

# Parameters
min_waiting_time = 0
max_waiting_time = 60
num_points = 1000
observed_time = 30  # Observed waiting time in minutes

# Generate the prior distribution
prior = prior_distribution(min_waiting_time, max_waiting_time, num_points)

# Perform Bayesian inference
posterior = bayesian_inference(prior, observed_time)

# Calculate the mean of the posterior distribution (estimated waiting time)
estimated_waiting_time = np.sum(np.linspace(0, 60, len(posterior)) * posterior)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 60, len(prior)), prior, label='Prior', linestyle='-')
plt.plot(np.linspace(0, 60, len(posterior)), posterior, label='Posterior')
plt.axvline(x=estimated_waiting_time, color='r', linestyle='-', label='Estimated Waiting Time')
plt.title('Bayesian Inference of Waiting Time Distribution')
plt.xlabel('Waiting Time (minutes)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

print("Estimated waiting time:", estimated_waiting_time, "minutes")