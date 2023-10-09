import numpy as np
from numpy.random import gamma
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from scipy.stats import kde, gaussian_kde, beta, invgamma
from scipy.stats import multivariate_normal as mvnrnd
from scipy.optimize import fsolve, minimize
from scipy.special import gammaln, betaln
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import large_bvar as bvar
import sys

sys.path.append('/Users/sudikshajoshi/Desktop/Fall 2022/ECON527 Macroeconometrics/'
                'BVAR of US Economy/AdditionalFunctions')

############################### Test for logMLVAR_formcmc_covid function #########################################
#
# Initialize simulation parameters
np.random.seed(42)  # For reproducibility

T = 50  # Number of time periods
n = 4  # Number of variables
lags = 2  # Number of lags
k = n * lags + 1  # Total number of explanatory variables
Tcovid = 40  # Time of Covid

# Initialize y and x matrices with random values
y = np.random.rand(T, n)
x = np.hstack([np.ones((T, 1)), np.random.rand(T, k - 1)])

# Initialize other parameters
b = np.eye(k, n)
SS = np.random.rand(n, 1)
Vc = 1000
pos = []
mn = {'alpha': 0}
sur = 1
noc = 1
y0 = np.random.rand(1, n)
draw = 1
hyperpriors = 1

# Initialize MIN and MAX dicts
MIN = {'lambda': 0.000001, 'theta': 0.00001, 'miu': 0.00001, 'alpha': 0.1, 'eta': [1, 1, 1, 0.005]}
MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5, 'eta': [500, 500, 500, 0.995]}

# Initialize priorcoef dict
priorcoef = {
    'lambda': {'k': 1.64, 'theta': 0.3123},
    'miu': {'k': 2.618, 'theta': 0.618},
    'theta': {'k': 2.618, 'theta': 0.618},
    'eta4': {'alpha': 3.0347, 'beta': 1.5089}
}

# Initialization
logML = -1e16
while logML == -1e16:
    # Randomly generate initial parameters within bounds
    par = np.array([
        np.random.rand() * (MAX['lambda'] - MIN['lambda']) + MIN['lambda'],
        np.random.rand() * (MAX['theta'] - MIN['theta']) + MIN['theta'],
        np.random.rand() * (MAX['miu'] - MIN['miu']) + MIN['miu'],
        np.random.rand() * (MAX['alpha'] - MIN['alpha']) + MIN['alpha'],
        np.random.rand() * (MAX['eta'][3] - MIN['eta'][3]) + MIN['eta'][3],
        np.random.rand(),
        np.random.rand()
    ])  # Additional parameters, adjust as needed

    # Call the function
    logML, betadraw, drawSIGMA = bvar.logMLVAR_formcmc_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc,
                                                             pos, mn, sur, noc, y0, draw, hyperpriors, priorcoef,
                                                             Tcovid)

# Display the results
print('LogML:', logML)
print('Betadraw:', betadraw)
print('DrawSIGMA:', drawSIGMA)

############################### Test for csminwel function #########################################

# # Initialize input parameters
# # Set random seed for reproducibility
# np.random.seed(42)
#
# # Initial parameters for the function
# x0 = np.random.rand(7, 1)  # 7x1 initial point
# H0 = np.diag([1, 1, 1, 1, 1, 1, 1])  # 7x7 initial Hessian
# crit = 0.0001  # Convergence criterion
# nit = 1000  # Number of iterations
#
# # Initialize MIN and MAX dicts based on your example
# MIN = {'lambda': 0.2, 'alpha': 0.5, 'theta': 0.5, 'miu': 0.5, 'eta': np.array([1, 1, 1, 0.005])}
# MAX = {'lambda': 5, 'alpha': 5, 'theta': 50, 'miu': 50, 'eta': np.array([500, 500, 500, 0.995])}
#
# # Simulation parameters
# T = 50  # Number of time periods
# n = 4  # Number of variables
# lags = 2  # Number of lags
# k = n * lags + 1  # Total number of explanatory variables
# Tcovid = 40  # Time of Covid
#
# # Initialize y and x matrices with random values
# y = np.random.rand(T, n)
# x = np.random.rand(T, k)
#
# # Initialize other parameters based on your example
# b = np.random.randint(0, 2, (k, n))  # Initialize b matrix with random 0s and 1s
# SS = np.random.rand(n, 1)  # Prior scale matrix
# Vc = 1000  # Prior variance for the constant
# pos = []  # Positions of variables without a constant
# mn = {'alpha': 0}  # Minnesota prior
# sur = 1  # Dummy for the sum-of-coefficients prior
# noc = 1  # Dummy for the no-cointegration prior
# y0 = np.random.rand(1, n)  # Initial values for the variables
# hyperpriors = 1  # Hyperpriors on the VAR coefficients
#
# priorcoef = {'lambda': {'k': 1.64, 'theta': 0.3123},
#              'miu': {'k': 2.618, 'theta': 0.618},
#              'theta': {'k': 2.618, 'theta': 0.618},
#              'eta4': {'alpha': 3.0347, 'beta': 1.5089}}
#
# # Assemble varargin list
# varargin = [y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid]
#
# # Call csminwel function
# fhat, xhat, grad, Hessian, itct, fcount, retcode = bvar.csminwel(bvar.logMLVAR_formin_covid, x0, H0, None, crit,
#                                                            nit, *varargin)
#
# # Display the results
# print("Best function value:")
# print(fhat)
# print("Best parameter estimates:")
# print(xhat)


############################### Test for csminit function #########################################

# # Initial parameters based on your example
# np.random.seed(0)
# x0 = np.random.rand(7, 1)
# f0 = 10.5
# g0 = np.random.rand(7, 1)
# badg = 0
# H0 = np.diag([0, 0, 0, 0, 0, 0, 0])
#
# # Initialize MIN and MAX dicts
# MIN = {'lambda': 0.2, 'alpha': 0.5, 'theta': 0.5, 'miu': 0.5, 'eta': np.array([1, 1, 1, 0.005])}
# MAX = {'lambda': 5, 'alpha': 5, 'theta': 50, 'miu': 50, 'eta': np.array([500, 500, 500, 0.995])}
#
# # Other parameters
# T = 50
# n = 4
# lags = 2
# k = n * lags + 1  # Total number of explanatory variables
#
# # Initialize b matrix with random 0s and 1s
# b = np.random.randint(0, 2, (k, n))
# SS = np.random.rand(n, 1)
# Vc = 10000
# pos = []
# mn = {'alpha': 0}
# sur = 1
# noc = 1
# y0 = np.random.rand(1, n)
# hyperpriors = 1
# y = np.random.rand(T, n)
# x = np.hstack([np.ones((T, 1)), np.random.rand(T, k - 1)])  # matrix with the first column as a vector of 1s
#
# priorcoef = {
#     'lambda': {'k': 1.64, 'theta': 0.3123},
#     'miu': {'k': 2.618, 'theta': 0.618},
#     'theta': {'k': 2.618, 'theta': 0.618},
#     'eta4': {'alpha': 3.0347, 'beta': 1.5089}
# }
# Tcovid = 40
#
# # Create varargin list
# varargin = [y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid]
#
# # Call csminit function
# fhat, xhat, fcount, retcode = bvar.csminit(bvar.logMLVAR_formin_covid, x0, f0, g0, badg, H0, *varargin)
#
# # Display the results
# print("Best function value:", fhat)
# print("Corresponding point:", xhat.T)
# print("Number of function evaluations:", fcount)
# print("Return code:", retcode)


############################### Test for bfgsi function #########################################
#
# # Set random seed for reproducibility
# np.random.seed(42)
#
# # Generate a 7x7 diagonal matrix for H0
# H0 = np.diag(np.random.randint(1, 11, 7))
#
# # Generate 7x1 column vectors for dg and dx
# dg = np.random.rand(7, 1)
# dx = np.random.rand(7, 1)
#
# # Call the bfgsi function
# H_updated = bvar.bfgsi(H0, dg, dx)
#
# # Display the original and updated inverse Hessians
# print("Original H0:")
# print(H0)
# print("Updated H:")
# print(H_updated)

############################### Test for numgrad function #########################################
# #
# # Initialize input parameters
# T = 50
# n = 4
# k = 9  # Number of lags * number of endogenous variables + 1
# lags = 2
# Tcovid = 40  # The time of Covid change, just for this example
#
# # Random data and initial parameter estimates
# np.random.seed(1234)
# par = np.random.rand(7, 1)
# y = np.random.randn(T, n)
# x = np.hstack([np.ones((T, 1)), np.random.randn(T, k - 1)])
# b = np.random.randint(0, 2, size=(k, n)).astype(float)
#
# # Initialize MIN and MAX dictionaries for hyperparameter bounds
# MIN = {'lambda': 0.1, 'theta': 0.1, 'miu': 0.1, 'eta': np.array([1, 1, 1, 0.005]), 'alpha': 0.1}
# MAX = {'lambda': 5, 'theta': 50, 'miu': 50, 'eta': np.array([500, 500, 500, 0.995]), 'alpha': 5}
#
# # Other parameters
# SS = np.ones((n, 1)) * 0.5
# Vc = 10000
# pos = None
# mn = {'alpha': 0}
# sur = 1
# noc = 1
# y0 = np.random.rand(1, n)
# hyperpriors = 1
#
# # Initialize priorcoef dictionary
# priorcoef = {
#     'lambda': {'k': 1.64, 'theta': 0.3123},
#     'theta': {'k': 2.618, 'theta': 0.618},
#     'miu': {'k': 2.618, 'theta': 0.618},
#     'eta4': {'alpha': 3.0347, 'beta': 1.5089}
# }
#
# # Package additional arguments into a tuple (analogous to varargin in MATLAB)
# varargin = (y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid)
#
# # Call numgrad function
# g, badg = bvar.numgrad(bvar.logMLVAR_formin_covid, par, *varargin)
#
# # Display the results
# print('Numerical Gradient:')
# print(g)
# print('Bad Gradient Flag:')
# print(badg)

############################### Test for logMLVAR_formin_covid function #########################################

# # Your example inputs
# T = 50  # Number of time points
# n = 4  # Number of endogenous variables
# k = 9  # Number of lags * number of endogenous variables + 1
# Tcovid = 40  # The time of Covid change, just for this example
# np.random.seed(1234)
# y = np.random.randn(T, n)
# x = np.random.randn(T, k)
# lags = 2
# b = np.random.randn(k, n)
# MIN = {'lambda': 0.1, 'theta': 0.1, 'miu': 0.1, 'eta': np.array([0.1, 0.2, 0.3, 0.4]), 'alpha': 0.1}
# MAX = {'lambda': 1, 'theta': 1, 'miu': 1, 'eta': np.array([1, 1, 1, 1]), 'alpha': 1}
# SS = np.ones((n, 1)) * 0.5
# Vc = 10000
# pos = []
# mn = {'alpha': 0}
# sur = 1
# noc = 1
# y0 = np.ones((1, n))
# hyperpriors = 1
# priorcoef = {'lambda': {'k': 1, 'theta': 1},
#              'theta': {'k': 1, 'theta': 1},
#              'miu': {'k': 1, 'theta': 1},
#              'eta4': {'alpha': 1, 'beta': 1}}
# par = np.ones((7, 1))
#
# # Call the function
# logML, betahat, sigmahat = bvar.logMLVAR_formin_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc,
#                                                       pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid)
#
# # Display the results
# print('Log of the marginal likelihood:', logML)
# print('Posterior mode of the VAR coefficients:', betahat)
# print('Posterior mode of the covariance matrix of the residuals:', sigmahat)

############################### Test for write_tex_sidewaystable function #########################################

# Example usage of write_tex_sidewaystable
# with open('example_table.tex', 'w') as fid:
#     header = ['Header 1', 'Header 2', 'Header 3']
#     style = 'l|c|r'
#     table_body = [
#         ['Row 1, Col 1', 1.23, 'Row 1, Col 3'],
#         ['Row 2, Col 1', 4.56, 'Row 2, Col 3'],
#         ['Row 3, Col 1', 7.89, 'Row 3, Col 3']
#     ]
#     above_tabular = 'This is a sample table.'
#     below_tabular = 'Table notes go here.'
#
#     bvar.write_tex_sidewaystable(fid, header, style, table_body, above_tabular, below_tabular)
