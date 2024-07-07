import numpy as np
from scipy.optimize import minimize, root_scalar

# Configurable parameters
m_values = [number_of_samples_per_pool] * number_of_pools_with_same_number + [different_number_of_samples]  # Sample counts configuration
total_samples = sum(m_values)  # Total number of samples
n = total_number_of_pools  # Total number of pools
k = number_of_positive_pools  # Number of positive pools

# Calculate MIR (Minimum Infection Rate)
mir = (k / total_samples) * 1000

# Likelihood function for calculating IRMLE (Infection Rate per 1000 Samples by the Maximum Likelihood Estimate) and CI (Confidence Interval)
def likelihood(p):
    return p**k * (1 - p)**(total_samples - k)

# Maximize the likelihood function to find IRMLE
result = minimize(lambda p: -likelihood(p), 0.01, bounds=[(0, 1)])
p_hat = result.x[0]

# Calculate IRMLE
irmle = p_hat * 1000

# Define the log-likelihood function for calculating confidence intervals
def log_likelihood(p):
    return k * np.log(p) + (total_samples - k) * np.log(1 - p)

# Use the difference of the log-likelihood function to find confidence intervals
L_hat = log_likelihood(p_hat)
def ci_function(p):
    return log_likelihood(p) - (L_hat - 1.92)

# Use numerical methods to find the limits of the confidence interval
ci_lower_bound = root_scalar(ci_function, bracket=[0, p_hat]).root
ci_upper_bound = root_scalar(ci_function, bracket=[p_hat, 1]).root

# Output results
print("MIR:", mir)
print("IRMLE:", irmle)
print("95% Confidence Interval for IRMLE:", ci_lower_bound * 1000, "to", ci_upper_bound * 1000)
