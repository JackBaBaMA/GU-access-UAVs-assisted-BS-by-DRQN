import numpy as np
import scipy.integrate as spi
from scipy.stats import norm

# Define the Q-function (complementary CDF of the standard normal distribution)
def Q(x):
    return 0.5 * (1.0 - norm.cdf(x))

# Define the PDF of SINR (simplified Gaussian distribution for this example)
def pdf_sinr(gamma, m):
    return gamma**m * np.exp(-gamma)

# Define the integrand for the probability calculation
def integrand(gamma):
    x = np.sqrt(2 * 5 * (1e-9 * 100**2)**2 / (1 * 1 * 1)) * np.sqrt(gamma)  # Substitute values
    return Q(x) * pdf_sinr(gamma, 2)

# Perform the integration
result, _ = spi.quad(integrand, 0, np.inf)
print(result)
