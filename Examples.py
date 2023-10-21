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

# Define the common directory path
dir_path = '/Users/sudikshajoshi/Desktop/Fall 2022/ECON527 Macroeconometrics/BVAR of US Economy/AdditionalFunctions'

# Add the path to your Python environment
sys.path.append(dir_path)

# TODO: Debug and test the bvarGLP_covid function after debugging the hessian function
############################### Test for bvarGLP_covid function #########################################
#
# # Load Ylog matrix from the specified location
# file_path = f'{dir_path}/Ylog.mat'
# Ylog = scipy.io.loadmat(file_path)['Ylog']
#
# # Simulation parameters
# lags = 2  # Number of lags
#
# # Parameters for the bvarGLP_covid function
# mcmc = 1
# MCMCconst = 1
# MNpsi = 0  # Some hypothetical value for MNpsi
# sur = 1  # Dummy for the sum-of-coefficients prior
# noc = 1  # Dummy for the no-cointegration prior
# Ndraws = 2000  # Number of draws
# hyperpriors = 1  # Hyperpriors on the VAR coefficients
# Tcovid = 376  # Time of Covid
#
# # Create a dictionary to hold name-value pairs
# params = {
#     'mcmc': mcmc,
#     'MCMCconst': MCMCconst,
#     'MNpsi': MNpsi,
#     'sur': sur,
#     'noc': noc,
#     'Ndraws': Ndraws,
#     'hyperpriors': hyperpriors,
#     'Tcovid': Tcovid
# }
#
# # Call bvarGLP_covid function (which you would define elsewhere)
# result_struct = bvarGLP_covid(Ylog, lags, **params)
#
# # Display the results (This part would depend on what bvarGLP_covid returns)
# print('Results from bvarGLP_covid:')
# print(result_struct)



## TODO: Debug and test the hessian function, after debugging the hessdiag function
############################### Test for hessian function #########################################

# # Set the random seed for reproducibility
# np.random.seed(42)
#
# # Define initial parameter estimates
# par = 0.5 + np.random.rand(7, 1)  # 7x1
# T = 50
# n = 4
# k = 9  # Number of lags * number of endogenous variables + 1
# b = np.random.rand(k, n) > 0.5  # 1s and 0s
# y = np.random.rand(T, n)
# x = np.hstack((np.ones((T, 1)), np.random.rand(T, k - 1)))  # T x k
# lags = 2
#
# # Initialize MIN and MAX dicts
# MIN = {'lambda': 1e-4, 'miu': 1e-4, 'theta': 1e-4, 'alpha': 0.1, 'eta': [1, 1, 1, 0.005]}
# MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5, 'eta': [500, 500, 500, 0.995]}
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
# Tcovid = 40
#
# # Initialize priorcoef dict
# priorcoef = {
#     'lambda': {'k': 1.6404, 'theta': 0.3123},
#     'theta': {'k': 2.618, 'theta': 0.618},
#     'miu': {'k': 2.618, 'theta': 0.618},
#     'eta': {'alpha': 3.0357, 'beta': 1.5089}
# }
#
# # Create varargin list
# varargin = [y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, 0, hyperpriors, priorcoef, Tcovid]
#
# # Create function handle for the objective function
# fun = lambda params: bvar.logMLVAR_formcmc_covid(params, *varargin)
#
# # Compute the Hessian matrix using the hessian function
# hess, err = bvar.hessian(fun, par)
#
# # Display the results
# print('Hessian matrix:')
# print(hess)
# print('Error estimates:')
# print(err)

## TODO: Debug and test the hessdiag function
############################### Test for hessdiag function #########################################

# Set the random seed for reproducibility
np.random.seed(42)

# Initialize parameters
y = np.random.rand(50, 4)  # Example values
x = np.random.rand(50, 9)  # Example values
lags = 2  # Example value
T = 50  # Example value
n = 4  # Example value
b = np.random.rand(9, 4)  # Example values
MIN = {'lambda': 1e-4, 'miu': 1e-4, 'theta': 1e-4, 'alpha': 0.1, 'eta': [1, 1, 1, 0.005]}
MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5, 'eta': [500, 500, 500, 0.995]}
SS = np.ones((4, 1)) * 0.5  # Example value
Vc = 10000  # Example value
pos = []  # Example value
mn = {'alpha': 0}  # Example value
sur = 1  # Example value
noc = 1  # Example value
y0 = np.random.rand(1, 4)  # Example values
hyperpriors = 1  # Example value
priorcoef = {'lambda': {'k': 1.6404, 'theta': 0.3123},
             'theta': {'k': 2.618, 'theta': 0.618},
             'miu': {'k': 2.618, 'theta': 0.618},
             'eta4': {'alpha': 3.0357, 'beta': 1.5089}}
Tcovid = 40  # Example value

# Function handle
fun_handle = lambda par: bvar.logMLVAR_formcmc_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, 0,
                                         hyperpriors, priorcoef, Tcovid)

# Initial point (7x1 array)
x0 = 0.5 + np.random.rand(7, 1)

# Call hessdiag function
HD, err, finaldelta = bvar.hessdiag(fun_handle, x0)

# Display the results
print('Diagonal elements of the Hessian matrix:')
print(HD)
print('Error estimates:')
print(err)
print('Final delta:')
print(finaldelta)

############################## Test for gradest function #########################################

#
# # Set the random seed for reproducibility
# np.random.seed(42)
#
# # Initialize parameters
# y = np.random.rand(50, 4)  # Example values
# x = np.random.rand(50, 9)  # Example values
# lags = 2  # Example value
# T = 50  # Example value
# n = 4  # Example value
# b = np.random.rand(9, 4)  # Example values
# MIN = {'lambda': 1e-4, 'miu': 1e-4, 'theta': 1e-4, 'alpha': 0.1, 'eta': [1, 1, 1, 0.005]}
# MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5, 'eta': [500, 500, 500, 0.995]}
# SS = np.ones((4, 1)) * 0.5  # Example value
# Vc = 10000  # Example value
# pos = []  # Example value
# mn = {'alpha': 0}  # Example value
# sur = 1  # Example value
# noc = 1  # Example value
# y0 = np.random.rand(1, 4)  # Example values
# hyperpriors = 1  # Example value
# priorcoef = {
#     'lambda': {'k': 1.6404, 'theta': 0.3123},
#     'theta': {'k': 2.618, 'theta': 0.618},
#     'miu': {'k': 2.618, 'theta': 0.618},
#     'eta4': {'alpha': 3.0357, 'beta': 1.5089}
# }
# Tcovid = 40  # Example value
#
# # Function handle
# fun_handle = lambda par: bvar.logMLVAR_formcmc_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur,
#                                                      noc, y0, 0, hyperpriors, priorcoef, Tcovid)
#
# # Initial point (7x1 array)
# x0 = 0.5 + np.random.rand(7, 1)
#
# # Call gradest to compute the gradient, error, and final delta
# grad, err, finaldelta = bvar.gradest(fun_handle, x0)
#
# # Display the results
# print("Gradient:")
# print(grad)
# print("Error estimates:")
# print(err)
# print("Final delta:")
# print(finaldelta)

############################### Test for derivest function #########################################

# # Define x0 and ind
# x0 = np.array([1, 2, 3, 4, 5])
# ind = 2  # Note: Python is 0-based indexing
#
# # Define the function to differentiate using the swapelement function
# # compute element-wise squared difference between original array x0 and the modified array
# # then sum these squared differences
# fun_handle = lambda xi: np.sum((x0 - np.array(bvar.swapelement(x0, ind, xi))) ** 2)
#
# # Define the point at which to evaluate the derivative
# xi_val = 0.875
#
# # Define the additional arguments
# varargin = ['deriv', 2, 'vectorized', 'no']
#
# # Call derivest
# der, errest, finaldelta = bvar.derivest(fun_handle, xi_val, varargin)
#
# # Display the results
# print(f"Estimated derivative: {der}")
# print(f"Estimated error: {errest}")
# print(f"Final delta: {finaldelta}")


############################### Test for swap2 function #########################################

# # Initialize a 7x1 vector of zeros
# initial_vec = np.zeros((7, 1))
#
# # Define indices and values to be swapped
# ind1 = 1  # Python uses 0-based indexing, so ind1 = 1 corresponds to MATLAB's ind1 = 2
# val1 = 0.0089
# ind2 = 0  # Python uses 0-based indexing, so ind2 = 0 corresponds to MATLAB's ind2 = 1
# val2 = 43.72
#
# # Call the swap2 function on a copy of initial_vec
# swapped_vec = bvar.swap2(initial_vec.copy(), ind1, val1, ind2, val2)
#
# # Display the original and swapped vectors
# print("Original vector:")
# print(initial_vec)
# print("Swapped vector:")
# print(swapped_vec)


############################### Test for fdamat function #########################################

# sr = 2  # Scalar for step ratio
# parity = 2  # Scalar for parity
# nterms = 4  # Number of terms
#
# # Call the fdamat function
# result_mat = bvar.fdamat(sr, parity, nterms)
#
# # Display the results
# print("Resulting Matrix:")
# print(result_mat)


############################### Test for check_params function #########################################
#
# # Original par dictionary with missing or empty fields
# par = {
#     'DerivativeOrder': None,  # Missing, should default to 1
#     'MethodOrder': None,      # Missing, should default to 2
#     'RombergTerms': None,     # Missing, should default to 2
#     'MaxStep': None,          # Missing, should default to 10
#     'StepRatio': 2,
#     'NominalStep': None,
#     'Vectorized': None,       # Missing, should default to 'yes'
#     'FixedStep': None,
#     'Style': None             # Missing, should default to 'central'
# }
#
# # Display the original par dictionary
# print("Original par dictionary:")
# print(par)
#
# # Assume check_params is a function that modifies the par dictionary
# par_modified = bvar.check_params(par)
#
# # Display the modified par dictionary
# print("Modified par dictionary:")
# print(par_modified)


############################### Test for parse_pv_pairs function #########################################

# # Define default parameters as a dictionary
# default_params = {
#     'DerivativeOrder': 1,
#     'MethodOrder': 4,
#     'RombergTerms': 2,
#     'MaxStep': 100,
#     'StepRatio': 2,
#     'NominalStep': None,  # Using None for MATLAB's []
#     'Vectorized': 'yes',
#     'FixedStep': None,  # Using None for MATLAB's []
#     'Style': 'central'
# }
#
# # Define a list of property-value pairs to override default settings
# pv_pairs = ['deriv', 2, 'vectorized', 'no']
#
# # Call the parse_pv_pairs function (you would need to implement this)
# updated_params = bvar.parse_pv_pairs(default_params, pv_pairs)
#
# # Display the original and updated parameters
# print('Default parameters:')
# print(default_params)
# print('Updated parameters:')
# print(updated_params)


############################### Test for rombextrap function #########################################


# np.random.seed(42)  # For reproducibility
# # Initialize parameters
# StepRatio = 2  # Scalar
# der_init = np.random.rand(23, 1)  # 23x1 numpy array, random for testing
# rombexpon = [4, 6]  # List
#
# # Call the rombextrap function
# der_romb, errest = bvar.rombextrap(StepRatio, der_init, rombexpon)
#
# # Display the results
# print('Derivative estimates returned:')
# print(der_romb)
# print('Error estimates:')
# print(errest)


############################### Test for vec2mat function #########################################

# np.random.seed(42)  # For reproducibility
# # Create a hypothetical vector vec of size 26x1
# # Populated with random numbers
# vec = np.random.rand(26, 1)
#
# # Define n and m based on the comments
# n = 23
# m = 2
#
# # Call the vec2mat function
# result_mat = bvar.vec2mat(vec, n, m)
#
# # Display the result
# print("Resulting matrix:")
# print(result_mat)



############################### Test for swapelement function #########################################

# # Create a 7x1 numpy array
# vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
#
# # Display the original vector
# print("Original vector:")
# print(vec)
#
# # Choose an index and a value based on the example
# ind = 0  # Note that Python uses zero-based indexing
# val = 0.8745
#
# # Call the swapelement function (assuming it's already defined)
# vec_modified = bvar.swapelement(vec, ind, val)
#
# # Display the modified vector
# print("Modified vector:")
# print(vec_modified)

############################### Test for logMLVAR_formcmc_covid function #########################################

# Initialize simulation parameters
# np.random.seed(42)  # For reproducibility
#
# T = 50  # Number of time periods
# n = 4  # Number of variables
# lags = 2  # Number of lags
# k = n * lags + 1  # Total number of explanatory variables
# Tcovid = 40  # Time of Covid
#
# # Initialize y and x matrices with random values
# y = np.random.rand(T, n)
# x = np.hstack([np.ones((T, 1)), np.random.rand(T, k - 1)])
#
# # Initialize other parameters
# b = np.eye(k, n)
# SS = np.random.rand(n, 1)
# Vc = 1000
# pos = []
# mn = {'alpha': 0}
# sur = 1
# noc = 1
# y0 = np.random.rand(1, n)
# draw = 1
# hyperpriors = 1
#
# # Initialize MIN and MAX dicts
# MIN = {'lambda': 0.000001, 'theta': 0.00001, 'miu': 0.00001, 'alpha': 0.1, 'eta': [1, 1, 1, 0.005]}
# MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5, 'eta': [500, 500, 500, 0.995]}
#
# # Initialize priorcoef dict
# priorcoef = {
#     'lambda': {'k': 1.64, 'theta': 0.3123},
#     'miu': {'k': 2.618, 'theta': 0.618},
#     'theta': {'k': 2.618, 'theta': 0.618},
#     'eta4': {'alpha': 3.0347, 'beta': 1.5089}
# }
#
# # Initialization
# logML = -1e16
# while logML == -1e16:
#     # Randomly generate initial parameters within bounds
#     par = np.array([
#         np.random.rand() * (MAX['lambda'] - MIN['lambda']) + MIN['lambda'],
#         np.random.rand() * (MAX['theta'] - MIN['theta']) + MIN['theta'],
#         np.random.rand() * (MAX['miu'] - MIN['miu']) + MIN['miu'],
#         np.random.rand() * (MAX['alpha'] - MIN['alpha']) + MIN['alpha'],
#         np.random.rand() * (MAX['eta'][3] - MIN['eta'][3]) + MIN['eta'][3],
#         np.random.rand(),
#         np.random.rand()
#     ])  # Additional parameters, adjust as needed
#
#     # Call the function
#     logML, betadraw, drawSIGMA = bvar.logMLVAR_formcmc_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc,
#                                                              pos, mn, sur, noc, y0, draw, hyperpriors, priorcoef,
#                                                              Tcovid)
#
# # Display the results
# print('LogML:', logML)
# print('Betadraw:', betadraw)
# print('DrawSIGMA:', drawSIGMA)


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
