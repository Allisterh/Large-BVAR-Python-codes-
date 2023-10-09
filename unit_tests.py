# Importing necessary libraries
import os
import numpy as np
import pandas as pd
from numpy.random import gamma
from numpy.linalg import pinv
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from scipy.stats import kde, gaussian_kde, beta, invgamma
from scipy.stats import multivariate_normal as mvnrnd
from scipy.optimize import fsolve
from scipy.special import gammaln, betaln
from mpl_toolkits.mplot3d import Axes3D
import io
import contextlib
import tempfile
import unittest
from unittest.mock import patch
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import sys

sys.path.append('/Users/sudikshajoshi/Desktop/Fall 2022/ECON527 Macroeconometrics/'
                'BVAR of US Economy/AdditionalFunctions')
import large_bvar as bvar


class TestBetaCoef(unittest.TestCase):

    def test_beta_coef(self):
        # Define the alpha, beta, mode, and standard deviation values
        x = [2, 5]
        mosd = [0.2, 0.1]

        # Expected results based on the given parameters
        expected_r1 = 0.0  # Expected result for r1
        expected_r2 = -0.05971914124998498  # Expected result for r2

        # Run the function to test
        result = bvar.beta_coef(x, mosd)

        # Check if the results are as expected
        self.assertAlmostEqual(result[0], expected_r1, places=8)
        self.assertAlmostEqual(result[1], expected_r2, places=8)

    def tearDown(self):
        if self.currentResult.wasSuccessful():
            print("Unit test for beta_coef successful")

    def run(self, result=None):
        self.currentResult = result  # remember result for use in tearDown
        unittest.TestCase.run(self, result)  # call superclass run method

    # Define the beta_coef function
    def beta_coef(x, mosd):
        al = x[0]  # alpha parameter
        bet = x[1]  # beta parameter
        mode = mosd[0]
        sd = mosd[1]
        r1 = mode - (al - 1) / (al + bet - 2)
        r2 = sd - (al * bet / ((al + bet) ** 2 * (al + bet + 1))) ** 0.5
        return [r1, r2]


class TestCsminit(unittest.TestCase):

    def test_csminit(self):
        # Initial parameters based on your example
        np.random.seed(0)
        x0 = np.random.rand(7, 1)
        f0 = 10.5
        g0 = np.random.rand(7, 1)
        badg = 0
        H0 = np.diag([0, 0, 0, 0, 0, 0, 0])

        # Initialize MIN and MAX dicts
        MIN = {'lambda': 0.2, 'alpha': 0.5, 'theta': 0.5, 'miu': 0.5, 'eta': np.array([1, 1, 1, 0.005])}
        MAX = {'lambda': 5, 'alpha': 5, 'theta': 50, 'miu': 50, 'eta': np.array([500, 500, 500, 0.995])}

        # Other parameters
        T = 50
        n = 4
        lags = 2
        k = n * lags + 1  # Total number of explanatory variables

        # Initialize b matrix with random 0s and 1s
        b = np.random.randint(0, 2, (k, n))
        SS = np.random.rand(n, 1)
        Vc = 10000
        pos = []
        mn = {'alpha': 0}
        sur = 1
        noc = 1
        y0 = np.random.rand(1, n)
        hyperpriors = 1
        y = np.random.rand(T, n)
        x = np.hstack([np.ones((T, 1)), np.random.rand(T, k - 1)])  # matrix with the first column as a vector of 1s

        priorcoef = {
            'lambda': {'k': 1.64, 'theta': 0.3123},
            'miu': {'k': 2.618, 'theta': 0.618},
            'theta': {'k': 2.618, 'theta': 0.618},
            'eta4': {'alpha': 3.0347, 'beta': 1.5089}
        }
        Tcovid = 40

        # Create varargin list
        varargin = [y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid]
        # Call csminit function
        fhat, xhat, fcount, retcode = bvar.csminit(bvar.logMLVAR_formin_covid, x0, f0, g0, badg, H0, *varargin)

        # Compare against expected results (replace these with your expected results)
        expected_fhat = 10.5  # Replace with your expected value
        expected_xhat = np.array([0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548, 0.64589411, 0.43758721])
        expected_fcount = 25  # Replace with your expected value
        expected_retcode = 6  # Replace with your expected value

        np.testing.assert_allclose(fhat, expected_fhat, rtol=1e-5)
        np.testing.assert_allclose(xhat.reshape(-1), expected_xhat, rtol=1e-5)
        self.assertEqual(fcount, expected_fcount)
        self.assertEqual(retcode, expected_retcode)


class TestCsminwel(unittest.TestCase):

    def test_csminwel(self):
        # Initialize input parameters
        # Set random seed for reproducibility
        np.random.seed(42)

        # Initial parameters for the function
        x0 = np.random.rand(7, 1)  # 7x1 initial point
        H0 = np.diag([1, 1, 1, 1, 1, 1, 1])  # 7x7 initial Hessian
        crit = 0.0001  # Convergence criterion
        nit = 1000  # Number of iterations

        # Initialize MIN and MAX dicts based on your example
        MIN = {'lambda': 0.2, 'alpha': 0.5, 'theta': 0.5, 'miu': 0.5, 'eta': np.array([1, 1, 1, 0.005])}
        MAX = {'lambda': 5, 'alpha': 5, 'theta': 50, 'miu': 50, 'eta': np.array([500, 500, 500, 0.995])}

        # Simulation parameters
        T = 50  # Number of time periods
        n = 4  # Number of variables
        lags = 2  # Number of lags
        k = n * lags + 1  # Total number of explanatory variables
        Tcovid = 40  # Time of Covid

        # Initialize y and x matrices with random values
        y = np.random.rand(T, n)
        x = np.random.rand(T, k)

        # Initialize other parameters based on your example
        b = np.random.randint(0, 2, (k, n))  # Initialize b matrix with random 0s and 1s
        SS = np.random.rand(n, 1)  # Prior scale matrix
        Vc = 1000  # Prior variance for the constant
        pos = []  # Positions of variables without a constant
        mn = {'alpha': 0}  # Minnesota prior
        sur = 1  # Dummy for the sum-of-coefficients prior
        noc = 1  # Dummy for the no-cointegration prior
        y0 = np.random.rand(1, n)  # Initial values for the variables
        hyperpriors = 1  # Hyperpriors on the VAR coefficients

        priorcoef = {'lambda': {'k': 1.64, 'theta': 0.3123},
                     'miu': {'k': 2.618, 'theta': 0.618},
                     'theta': {'k': 2.618, 'theta': 0.618},
                     'eta4': {'alpha': 3.0347, 'beta': 1.5089}}

        # Assemble varargin list
        varargin = [y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid]

        # Call csminwel function
        fhat, xhat, grad, Hessian, itct, fcount, retcode = bvar.csminwel(bvar.logMLVAR_formin_covid, x0, H0, None, crit,
                                                                         nit, *varargin)


        # Assertions
        self.assertIsInstance(fhat, float, "fhat should be a float")
        self.assertIsInstance(xhat, np.ndarray, "xhat should be a numpy array")
        self.assertIsInstance(grad, np.ndarray, "grad should be a numpy array")
        self.assertIsInstance(Hessian, np.ndarray, "Hessian should be a numpy array")
        self.assertIsInstance(itct, int, "itct should be an integer")
        self.assertIsInstance(fcount, int, "fcount should be an integer")
        self.assertIsInstance(retcode, int, "retcode should be an integer")


class TestBFGSI(unittest.TestCase):

    def test_bfgsi_specific_case(self):
        """Test that bfgsi produces the expected output for a specific input."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate a 7x7 diagonal matrix for H0
        H0 = np.diag(np.random.randint(1, 11, 7))

        # Generate 7x1 column vectors for dg and dx
        dg = np.random.rand(7, 1)
        dx = np.random.rand(7, 1)

        # Call the bfgsi function
        H_updated = bvar.bfgsi(H0, dg, dx)

        # Expected output matrix
        expected_H = np.array([
            [13.14000197, -1.85083459, 4.54069634, 3.7829455, -0.14228596, -0.51855766, -2.02075975],
            [-1.85083459, 3.99692431, -1.95826863, -1.21789215, -1.20962392, -0.01440345, -0.04728123],
            [4.54069634, -1.95826863, 10.73845063, 2.71607823, -1.42145471, -0.56288859, -2.18421487],
            [3.7829455, -1.21789215, 2.71607823, 7.32000363, -0.26084473, -0.34309535, -1.33577993],
            [-0.14228596, -1.20962392, -1.42145471, -0.26084473, 4.20682098, -0.36914089, -1.41874631],
            [-0.51855766, -0.01440345, -0.56288859, -0.34309535, -0.36914089, 9.99184358, -0.02909227],
            [-2.02075975, -0.04728123, -2.18421487, -1.33577993, -1.41874631, -0.02909227, 2.89698313]
        ])

        # Check if the output matches the expected output within a certain tolerance
        self.assertTrue(np.allclose(H_updated, expected_H, atol=1e-6))


class TestBvarFcst(unittest.TestCase):

    def test_bvarFcst(self):
        # Define mock data for y
        y = np.array([[1, 2],
                      [2, 3],
                      [4, 3],
                      [3, 2],
                      [2, 1],
                      [1, 2],
                      [2, 3],
                      [3, 4],
                      [5, 4],
                      [4, 3]])

        # Define mock coefficients for beta
        beta = np.array([[0.1, 0.2],
                         [0.3, 0.4],
                         [0.5, 0.6]])

        # Define horizons for the forecast
        hz = [1, 2]

        # Expected forecasted values based on the test case
        expected_forecast = np.array([[2.8, 3.6],
                                      [2.74, 3.48]])

        # Run the function to test
        forecast = bvar.bvarFcst(y, beta, hz)

        # Check if the results are as expected
        self.assertTrue(np.allclose(forecast, expected_forecast, atol=1e-8))

    def tearDown(self):
        if self.currentResult.wasSuccessful():
            print("Unit test for bvarFcst successful")

    def run(self, result=None):
        self.currentResult = result  # remember result for use in tearDown
        unittest.TestCase.run(self, result)  # call superclass run method


class TestBvarIrfs(unittest.TestCase):

    def test_bvarIrfs(self):
        # Define the hypothetical data
        beta = np.array([
            [0.2, 0.3],
            [0.1, 0.4],
            [0.5, 0.1],
            [0.3, 0.2],
            [0.1, 0.3],
            [0.4, 0.1]
        ])
        sigma = np.array([[0.5, 0.1], [0.1, 0.6]])
        nshock = 1
        hmax = 5

        # Expected Impulse Response Functions at different horizons
        expected_irf = np.array([
            [0.70710678, 0.0],
            [0.07071068, 0.28284271],
            [0.36062446, 0.1979899],
            [0.18455487, 0.26304372],
            [0.27796368, 0.23164818]
        ])

        # Run the function to test
        result = bvar.bvarIrfs(beta, sigma, nshock, hmax)

        # Check if the results are as expected
        self.assertTrue(np.allclose(result, expected_irf, atol=1e-8))

    def tearDown(self):
        print("Unit test for bvarIrfs successful")


class TestCholredFunction(unittest.TestCase):

    def test_cholred(self):
        S = np.array([
            [4, 2, 0.6],
            [2, 3, 0.4],
            [0.6, 0.4, 0.25]
        ])
        expected_result = np.array([
            [1.86083605, 0.73126173, -0.05045273],
            [1.4478077, -0.95052528, -0.01882953],
            [0.31358677, 0.04917183, 0.38632301]
        ])
        computed_result = bvar.cholred(S)
        # Assert that the returned Cholesky decomposition is almost equal to the expected one
        np.testing.assert_almost_equal(computed_result, expected_result, decimal=6)
        print("Unit test for cholred() was successful.")


# Writing the unit test
class TestColsFunction(unittest.TestCase):

    def test_cols(self):
        # Test with a 3x2 matrix
        x1 = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(bvar.cols(x1), 2)

        # Test with a 4x4 matrix
        x2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        self.assertEqual(bvar.cols(x2), 4)

        # Test with a 2x5 matrix
        x3 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        self.assertEqual(bvar.cols(x3), 5)

        print("The unit test for the cols() function was successful.")


class TestNumGrad(unittest.TestCase):

    def test_gradient(self):
        # Your settings here (including the function logMLVAR_formin_covid and the actual parameters)

        # Initialize input parameters
        T = 50
        n = 4
        k = 9  # Number of lags * number of endogenous variables + 1
        lags = 2
        Tcovid = 40  # The time of Covid change, just for this example

        # Random data and initial parameter estimates
        np.random.seed(1234)
        par = np.random.rand(7, 1)
        y = np.random.randn(T, n)
        x = np.hstack([np.ones((T, 1)), np.random.randn(T, k - 1)])
        b = np.random.randint(0, 2, size=(k, n)).astype(float)

        # Initialize MIN and MAX dictionaries for hyperparameter bounds
        MIN = {'lambda': 0.1, 'theta': 0.1, 'miu': 0.1, 'eta': np.array([1, 1, 1, 0.005]), 'alpha': 0.1}
        MAX = {'lambda': 5, 'theta': 50, 'miu': 50, 'eta': np.array([500, 500, 500, 0.995]), 'alpha': 5}

        # Other parameters
        SS = np.ones((n, 1)) * 0.5
        Vc = 10000
        pos = None
        mn = {'alpha': 0}
        sur = 1
        noc = 1
        y0 = np.random.rand(1, n)
        hyperpriors = 1

        # Initialize priorcoef dictionary
        priorcoef = {
            'lambda': {'k': 1.64, 'theta': 0.3123},
            'theta': {'k': 2.618, 'theta': 0.618},
            'miu': {'k': 2.618, 'theta': 0.618},
            'eta4': {'alpha': 3.0347, 'beta': 1.5089}
        }

        # Package additional arguments into a tuple (analogous to varargin in MATLAB)
        varargin = (y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid)

        # Call the numgrad function (assuming it's available in the current scope)
        computed_grad, badg = bvar.numgrad(bvar.logMLVAR_formin_covid, par,
                                           *varargin)  # Using your real function and parameters

        # Expected gradient and bad flag based on your previous Python execution
        expected_grad = np.array([
            [15.0291105],
            [2.08934614],
            [2.34590073],
            [10.48148954],
            [34.37632154],
            [20.84238349],
            [19.25045297]
        ])
        expected_badg = 0

        # Validate results
        np.testing.assert_array_almost_equal(computed_grad, expected_grad, decimal=6)
        self.assertEqual(badg, expected_badg)


class TestDrsnbrck(unittest.TestCase):

    def test_derivative(self):
        x = np.array([[1], [1]])
        expected_derivative = np.array([[0], [0]])
        expected_badg = 0

        derivative, badg = bvar.drsnbrck(x)

        # Check if the derivative is as expected
        np.testing.assert_array_almost_equal(derivative, expected_derivative, decimal=5)

        # Check if the bad gradient flag is as expected
        self.assertEqual(badg, expected_badg)


# Define the unit test
class TestMissData(unittest.TestCase):

    def test_miss_data(self):
        # Input data
        y = np.array([1, 2, np.nan, 4, 5])
        C = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        R = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
        c1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Expected output
        expected_y = np.array([1., 2., 4., 5.])
        expected_C = np.array([[1, 2], [3, 4], [7, 8], [9, 10]])
        expected_R = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        expected_c1 = np.array([0.1, 0.2, 0.4, 0.5])

        # Call the function to test
        updated_y, updated_C, updated_R, updated_c1 = bvar.MissData(y, C, R, c1)

        # Check if the function output matches the expected output
        np.testing.assert_array_almost_equal(updated_y, expected_y, decimal=6)
        np.testing.assert_array_almost_equal(updated_C, expected_C, decimal=6)
        np.testing.assert_array_almost_equal(updated_R, expected_R, decimal=6)
        np.testing.assert_array_almost_equal(updated_c1, expected_c1, decimal=6)


## TODO: CHANGE example and test again
class TestFIS(unittest.TestCase):

    def test_FIS(self):
        np.random.seed(0)  # Seed random number generator for reproducibility

        # Initialize parameters
        nobs = 4  # Number of observations
        m = 2  # Dimension of state vector

        # Create dummy input data
        Y = np.random.randn(nobs, 1)
        Z = np.eye(m)
        R = np.eye(m)
        T = np.eye(m)
        Q = np.eye(m)

        # Create a dictionary S for Kalman Filter estimates
        S = {
            'Am': np.random.randn(m, nobs),
            'Pm': np.random.randn(m, m, nobs),
            'AmU': np.random.randn(m, nobs + 1),
            'PmU': np.random.randn(m, m, nobs + 1),
            'KZ': np.eye(m)
        }

        # Run the FIS function
        S_out = bvar.FIS(Y, Z, R, T, S)

        # Assert the outputs for AmT
        expected_AmT = np.array([
            [4.58221489, 0.18209962, 1.22159736, -2.86382298, 0.],
            [-2.96339772, -0.12885061, -0.79667026, 1.84082651, 0.]
        ])
        np.testing.assert_array_almost_equal(S_out['AmT'], expected_AmT, decimal=6)

        # Assert the outputs for PmT
        expected_PmT = np.array([
            [
                [-1.71662233e+01, 4.12672694e+00, -5.66298624e+00, 2.86811031e+00, 0.],
                [6.61956669e+00, -3.46254950e+00, 5.51177176e-01, -1.84023334e+00, 0.]
            ],
            [
                [4.35089765e-03, -2.06647117e+00, -5.51311728e-01, -1.37805335e-01, 0.],
                [-1.56284611e+00, 4.72847099e-01, 4.94577985e-01, 8.73935618e-02, 0.]
            ]
        ])
        np.testing.assert_array_almost_equal(S_out['PmT'], expected_PmT, decimal=6)

        # Assert the outputs for PmT_1
        expected_PmT_1 = np.array([
            [
                [-1.8754928, -1.77414095, -1.42817858, 0.],
                [-1.07970382, 1.45101503, 0.07619422, 0.]
            ],
            [
                [-0.81237915, 0.20275672, 0.93241186, 0.],
                [0.12331498, -0.31890925, -0.0508983, 0.]
            ]
        ])
        np.testing.assert_array_almost_equal(S_out['PmT_1'], expected_PmT_1, decimal=6)


class TestFormCompanionMatrices(unittest.TestCase):

    def Test_form_companion_matrices(self):
        # Set seed for reproducibility
        np.random.seed(0)

        # Inputs for FormCompanionMatrices
        n = 3
        lags = 2
        TTfcst = 10
        tstar = 3
        etapar = np.array([0.5, 0.6, 0.7, 0.8])
        betadraw = np.random.rand(1 + n * lags, n)
        G = np.random.rand(n, n)

        # Call your function
        varc, varZ, varG, varC, varT, varH = form_companion_matrices(betadraw, G, etapar, tstar, n, lags, TTfcst)

        # Expected outputs (replace these with the actual expected outputs)
        expected_varc = np.zeros((n, TTfcst))  # Your expected varc here
        expected_varZ = np.zeros((n, n * lags, TTfcst))  # Your expected varZ here
        expected_varG = np.zeros((n, n, TTfcst))  # Your expected varG here
        expected_varC = np.array([0.5488135, 0.71518937, 0.60276338, 0., 0., 0.])  # Your expected varC here
        expected_varT = np.array([
            [0.54488318, 0.43758721, 0.38344152, 0.56804456, 0.0871293, 0.77815675],
            [0.4236548, 0.891773, 0.79172504, 0.92559664, 0.0202184, 0.87001215],
            [0.64589411, 0.96366276, 0.52889492, 0.07103606, 0.83261985, 0.97861834],
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.]
        ])  # Your expected varT here
        expected_varH = np.zeros((n * lags, n, TTfcst))

        # Assertions
        assert_array_almost_equal(varc, expected_varc, decimal=6)
        assert_array_almost_equal(varZ, expected_varZ, decimal=6)
        assert_array_almost_equal(varG, expected_varG, decimal=6)
        assert_array_almost_equal(varC, expected_varC, decimal=6)
        assert_array_almost_equal(varT, expected_varT, decimal=6)
        assert_array_almost_equal(varH, expected_varH, decimal=6)


class TestGammaCoef(unittest.TestCase):

    def test_gamma_coef(self):
        # Define hypothetical data
        mode = 4.0
        sd = 1.5

        # Expected values (These are hypothetical and should be replaced by actual expected values)
        expected_k = 9.0  # Replace with actual expected value
        expected_theta = 0.5  # Replace with actual expected value

        # Call the function
        output = bvar.gamma_coef(mode, sd, 0)

        # Assertions to check if the output matches the expected output
        assert_almost_equal(output['k'], expected_k, decimal=6)
        assert_almost_equal(output['theta'], expected_theta, decimal=6)

    @patch('matplotlib.pyplot.show')
    def test_gamma_coef_plot(self, mock_show):
        # Define hypothetical data
        mode = 4.0
        sd = 1.5
        plotit = 1

        # Call the function
        bvar.gamma_coef(mode, sd, plotit)

        # Assertion to check if plt.show() is called
        mock_show.assert_called_once()


class TestKFilterConst(unittest.TestCase):

    def test_kfilter_const(self):
        np.random.seed(123)  # Seed for reproducibility

        # Define parameters
        n = 3  # Dimension of observation
        m = 2  # Dimension of state

        y = np.random.randn(n, 1)
        c = 0.5
        Z = np.random.randn(n, m)
        G = np.eye(n)
        C = 0.2
        T = np.random.randn(m, m)
        H = np.eye(m)
        shat = np.random.randn(m, 1)
        sig = np.eye(m)

        # Expected values (These should be calculated and filled in based on the known output)
        expected_shatnew = np.array([[0.85107116], [0.30028426]])
        expected_signew = np.array([[0.2398133, 0.10354184], [0.10354184, 0.16698419]])
        expected_v = np.array([[-0.22861471], [-2.219862], [0.74400761]])
        expected_k = np.array([[-0.42113884, 0.14477362, 0.02821841], [-0.25258161, -0.2342243, 0.16698094]])
        expected_sigmainv = np.array([[0.21949701, 0.0825495, 0.13912046], [0.0825495, 0.19252829, 0.35860827],
                                      [0.13912046, 0.35860827, 0.800716]])

        # Call the function
        shatnew, signew, v, k, sigmainv = bvar.kfilter_const(y, c, Z, G, C, T, H, shat, sig)

        # Assertions to check if the output matches the expected output
        assert_array_almost_equal(shatnew, expected_shatnew, decimal=6)
        assert_array_almost_equal(signew, expected_signew, decimal=6)
        assert_array_almost_equal(v, expected_v, decimal=6)
        assert_array_almost_equal(k, expected_k, decimal=6)
        assert_array_almost_equal(sigmainv, expected_sigmainv, decimal=6)


class TestLagFunction(unittest.TestCase):

    def test_default_lag(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_output = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(bvar.lag(x), expected_output)

    def test_custom_lag(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        expected_output = np.array([[0, 0], [0, 0], [1, 2]])
        np.testing.assert_array_equal(bvar.lag(x, n=2), expected_output)

    def test_custom_initial_values(self):
        x = np.array([[1], [2], [3]])
        expected_output = np.array([[999], [1], [2]])
        np.testing.assert_array_equal(bvar.lag(x, n=1, v=999), expected_output)

    def test_empty_output(self):
        x = np.array([[1, 2], [3, 4]])
        expected_output = np.array([])
        np.testing.assert_array_equal(bvar.lag(x, n=0), expected_output)


class TestLagMatrix(unittest.TestCase):

    def test_lagmatrix(self):
        # Define the input matrix and lags
        Y = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
        lags = [-1, 0, 1]

        # Expected output matrix
        expected_YLag = np.array([
            [2., 7., 1., 6., np.nan, np.nan],
            [3., 8., 2., 7., 1., 6.],
            [4., 9., 3., 8., 2., 7.],
            [5., 10., 4., 9., 3., 8.],
            [np.nan, np.nan, 5., 10., 4., 9.]
        ])

        # Call the function
        output_YLag = bvar.lag_matrix(Y, lags)

        # Assert that the output matches the expected output
        np.testing.assert_array_almost_equal(output_YLag, expected_YLag, decimal=6)


class TestLogBetaPDF(unittest.TestCase):

    def test_log_beta_pdf(self):
        # Define parameters and sample values for testing
        alpha = 2
        beta_param = 5
        x_values = np.array([0.1, 0.3, 0.5])

        # Loop through each sample value and compare custom and scipy results
        for x_value in x_values:
            log_pdf_custom = bvar.log_beta_pdf(x_value, alpha, beta_param)
            log_pdf_scipy = np.log(beta.pdf(x_value, alpha, beta_param))

            # Use assertAlmostEqual because we're comparing floats
            self.assertAlmostEqual(log_pdf_custom, log_pdf_scipy, places=8)


class TestLogGammaPdf(unittest.TestCase):

    def test_scalar_input(self):
        # Test with scalar input
        self.assertAlmostEqual(bvar.log_gamma_pdf(0.2, 1.64, 0.3123), 0.34503764832403605, places=10)

    def test_array_input(self):
        # Test with array input
        x_values = np.array([0.1, 0.2, 0.3])
        expected_results = bvar.log_gamma_pdf(x_values, 1.64, 0.3123)
        calculated_results = np.array([bvar.log_gamma_pdf(0.1, 1.64, 0.3123),
                                       bvar.log_gamma_pdf(0.2, 1.64, 0.3123),
                                       bvar.log_gamma_pdf(0.3, 1.64, 0.3123)])
        np.testing.assert_almost_equal(expected_results, calculated_results, decimal=10)


class TestLogIG2PDF(unittest.TestCase):

    def test_log_ig2pdf(self):
        # Define some parameters and sample values for testing
        alpha = 3
        beta = 2
        x_values = np.array([1.0, 2.0, 3.0])

        # Expected log PDF values based on your earlier test output
        expected_values = np.array([-0.6137056388801095, -2.386294361119891, -3.6748214602192153])

        # Loop through each sample value and compare it to the expected log PDF
        for i, x_value in enumerate(x_values):
            log_pdf_custom = bvar.log_ig2pdf(x_value, alpha, beta)

            # Use the assertAlmostEqual method to check if the log PDF is almost equal to the expected value
            self.assertAlmostEqual(log_pdf_custom, expected_values[i], places=6)


class TestLogMLVARFormCMCCovid(unittest.TestCase):

    def test_output(self):
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
                                                                     pos, mn, sur, noc, y0, draw, hyperpriors,
                                                                     priorcoef,
                                                                     Tcovid)

        expected_logML = np.array([169.47591782])
        expected_betadraw = np.array([
            [ 0.00381148,  0.4324629 ,  0.10954747,  0.39794204],
            [ 0.27133489,  0.0180622 ,  0.55758251,  0.12022486],
            [-0.18997244,  0.36405837, -0.26705004, -0.43654857],
            [-0.17664508, -0.00162889, -0.19330479,  0.44670608],
            [ 0.06185378,  0.06884097,  0.02745698,  0.10760423],
            [ 0.13965301,  0.02756273,  0.35610567, -0.3048596 ],
            [ 0.03515588, -0.20739113,  0.03140875, -0.13332706],
            [ 0.42290286,  0.00727132,  0.11728262, -0.07167937],
            [-0.13869773,  0.07932549, -0.21052211,  0.05544635]
        ])
        expected_drawSIGMA = np.array([
            [ 0.07675985,  0.00410223, -0.00521122,  0.0270709 ],
            [ 0.00410223,  0.04625054,  0.00930697,  0.02281862],
            [-0.00521122,  0.00930697,  0.08293307,  0.01153445],
            [ 0.0270709 ,  0.02281862,  0.01153445,  0.06159708]
        ])

        # Check if the output matches the expected output
        np.testing.assert_almost_equal(logML, expected_logML, decimal=5)
        np.testing.assert_almost_equal(betadraw, expected_betadraw, decimal=5)
        np.testing.assert_almost_equal(drawSIGMA, expected_drawSIGMA, decimal=5)


class TestLogMLVARForminCovid(unittest.TestCase):

    def test_function(self):
        # Expected values based on your provided output
        expected_logML = np.array([349.36491477])
        expected_betahat = np.array([
            [-0.37440111, -0.18501195, 0.16084043, -0.03305045],
            [-0.01014004, 0.03388156, -0.11240791, -0.06888523],
            [0.21679144, -0.25879377, -0.06927085, 0.46864471],
            [0.05384616, -0.24812184, 0.00990955, 0.05254996],
            [0.09398046, 0.12333702, 0.20203256, 0.07769862],
            [0.13290688, 0.16555262, 0.13637155, -0.22952501],
            [-0.02884597, 0.43255212, 0.1187389, -0.06559368],
            [-0.20614557, 0.33888547, -0.06914737, -0.01906166],
            [-0.01082166, 0.09945283, 0.04550154, -0.14814634]
        ])
        expected_sigmahat = np.array([
            [0.84724503, -0.0771215, 0.06963778, -0.1097004],
            [-0.0771215, 1.13463036, -0.10124885, -0.03313186],
            [0.06963778, -0.10124885, 0.90616263, -0.43505537],
            [-0.1097004, -0.03313186, -0.43505537, 1.14668981]
        ])

        # Your example inputs
        T = 50  # Number of time points
        n = 4  # Number of endogenous variables
        k = 9  # Number of lags * number of endogenous variables + 1
        Tcovid = 40  # The time of Covid change, just for this example
        np.random.seed(1234)
        y = np.random.randn(T, n)
        x = np.random.randn(T, k)
        lags = 2
        b = np.random.randn(k, n)
        MIN = {'lambda': 0.1, 'theta': 0.1, 'miu': 0.1, 'eta': np.array([0.1, 0.2, 0.3, 0.4]), 'alpha': 0.1}
        MAX = {'lambda': 1, 'theta': 1, 'miu': 1, 'eta': np.array([1, 1, 1, 1]), 'alpha': 1}
        SS = np.ones((n, 1)) * 0.5
        Vc = 10000
        pos = []
        mn = {'alpha': 0}
        sur = 1
        noc = 1
        y0 = np.ones((1, n))
        hyperpriors = 1
        priorcoef = {'lambda': {'k': 1, 'theta': 1},
                     'theta': {'k': 1, 'theta': 1},
                     'miu': {'k': 1, 'theta': 1},
                     'eta4': {'alpha': 1, 'beta': 1}}
        par = np.ones((7, 1))

        # Call the function
        logML, betahat, sigmahat = bvar.logMLVAR_formin_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc,
                                                              pos, mn, sur, noc, y0, hyperpriors, priorcoef, Tcovid)

        # Check if the function output matches the expected output
        np.testing.assert_almost_equal(logML, expected_logML, decimal=8)
        np.testing.assert_almost_equal(betahat, expected_betahat, decimal=8)
        np.testing.assert_almost_equal(sigmahat, expected_sigmahat, decimal=8)


class TestOls1(unittest.TestCase):

    def test_ols1(self):
        # Given data
        y = np.array([[2.5], [3.6], [4.2], [4.8], [6.1]])
        x = np.array([
            [1, 2.1, 1.5],
            [1, 2.8, 2.1],
            [1, 3.3, 2.9],
            [1, 3.7, 3.2],
            [1, 4.4, 3.8]
        ])

        # Expected output
        expected_nobs = 5
        expected_nvar = 3
        expected_bhatols = np.array([[-1.2767002], [2.34004024], [-0.78215962]])
        expected_yhatols = np.array([[2.46414487], [3.63287726], [4.17716968], [4.87853789], [6.04727029]])
        expected_resols = np.array([[0.03585513], [-0.03287726], [0.02283032], [-0.07853789], [0.05272971]])
        expected_sig2hatols = np.array([[0.00591818]])
        expected_sigbhatols = np.array([[0.07531206, -0.08942756, 0.08052053], [-0.08942756, 0.13098578, -0.12503188],
                                        [0.08052053, -0.12503188, 0.121142]])
        expected_R2 = 0.9983587976370011

        # Run function
        result = bvar.ols1(y, x)

        # Assertions
        self.assertEqual(result['nobs'], expected_nobs)
        self.assertEqual(result['nvar'], expected_nvar)
        np.testing.assert_array_almost_equal(result['bhatols'], expected_bhatols, decimal=6)
        np.testing.assert_array_almost_equal(result['yhatols'], expected_yhatols, decimal=6)
        np.testing.assert_array_almost_equal(result['resols'], expected_resols, decimal=6)
        np.testing.assert_array_almost_equal(result['sig2hatols'], expected_sig2hatols, decimal=6)
        np.testing.assert_array_almost_equal(result['sigbhatols'], expected_sigbhatols, decimal=6)
        self.assertAlmostEqual(result['R2'], expected_R2, places=6)


class TestPlotJointMarginal(unittest.TestCase):

    def setUp(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        self.N = 500
        self.Y1 = np.random.normal(0, 1, self.N)
        self.Y2 = 0.5 * self.Y1 + np.random.normal(0, 1, self.N)
        self.YY = np.column_stack((self.Y1, self.Y2))
        self.Y1CondLim = [-1, 1]
        self.xlab = 'Variable 1'
        self.ylab = 'Variable 2'

    @patch('matplotlib.pyplot.show')
    def test_plot_joint_marginal(self, mock_show):
        # Test if the function calls plt.show() when vis=True
        bvar.plot_joint_marginal(self.YY, self.Y1CondLim, self.xlab, self.ylab, vis=True)
        mock_show.assert_called_once()


class TestPlotJointMarginal2(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)  # Setting the seed for reproducibility
        self.YY = np.random.randn(100, 2)  # Generate some random data
        self.idx = np.random.choice([0, 1], size=100)  # Generate random indices
        self.xlab = 'X-axis'
        self.ylab = 'Y-axis'

    def test_plot_joint_marginal2(self):
        e = None  # initialize e to None so that e has a value regardless of whether try block succeeds or fails
        try:
            bvar.plot_joint_marginal2(self.YY, self.idx, self.xlab, self.ylab, vis='off', LW=1.5)
            run_status = True
        except Exception as e:
            print(f"An exception occurred: {e}")
            run_status = False

        self.assertTrue(run_status, f"Function did not run successfully due to: {e}")


class TestPlotJointMarginal3(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)  # Setting the seed for reproducibility
        self.YYa = np.random.randn(100, 2)  # Generate some random data for YYa
        self.YYb = 2 + np.random.randn(100, 2)  # Generate some random data for YYb
        self.xlab = 'X-axis'
        self.ylab = 'Y-axis'

    def test_plot_joint_marginal3(self):
        e = None  # Initialize 'e' to None so it has a value regardless of whether the try block succeeds or fails
        try:
            bvar.plot_joint_marginal3(self.YYa, self.YYb, self.xlab, self.ylab, vis='off', LW=1.5)
            run_status = True
        except Exception as e:
            print(f"An exception occurred: {e}")
            run_status = False

        self.assertTrue(run_status, f"Function did not run successfully due to: {e}")


class TestPlotJointMarginal4(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)  # Setting the seed for reproducibility
        self.YYa = np.random.randn(100, 2)  # Generate some random data for YYa
        self.YYb = 2 + np.random.randn(100, 2)  # Generate some random data for YYb
        self.xlab = 'X-axis'
        self.ylab = 'Y-axis'

    def test_plot_joint_marginal4(self):
        e = None  # Initialize 'e' to None so it has a value regardless of whether the try block succeeds or fails
        try:
            bvar.plot_joint_marginal4(self.YYa, self.YYb, self.xlab, self.ylab, vis='off', LW=1.5)
            run_status = True
        except Exception as e:
            print(f"An exception occurred: {e}")
            run_status = False

        self.assertTrue(run_status, f"Function did not run successfully due to: {e}")


class TestPrintPDF(unittest.TestCase):

    def setUp(self):
        # Initialize a figure for testing
        self.fig, self.ax = plt.subplots()
        self.ax.plot([1, 2, 3], [1, 2, 3])
        self.outfilename = "test_output.pdf"

    def tearDown(self):
        # Remove the test file if it exists
        if os.path.exists(self.outfilename):
            os.remove(self.outfilename)

    def test_printpdf(self):
        print("Current Working Directory:", os.getcwd())
        # Run the function
        bvar.printpdf(self.fig, self.outfilename)

        # Check if the file has been created
        self.assertTrue(os.path.exists(self.outfilename), "PDF file was not created")


class TestQuantilePlot(unittest.TestCase):  # recheck this unit test as test_plot shows axes but no graph

    def setUp(self):
        self.Time = np.linspace(0, 10, 100)
        self.Center = np.sin(self.Time)
        self.InnerBot = self.Center - 0.1
        self.InnerTop = self.Center + 0.1
        self.Quantiles_5 = np.column_stack([self.InnerBot, self.InnerBot, self.Center, self.InnerTop, self.InnerTop])

    def test_quantilePlot_five_quantiles(self):
        try:
            # Create a plot using the function
            print("Before running bvar.quantile_plot")  # Debugging print
            bvar.quantile_plot(self.Time, self.Quantiles_5)  # Assuming bvar is imported or defined elsewhere
            print("After running bvar.quantile_plot")  # Debugging print

            fig1 = plt.gcf()  # Get the current figure
            ax1 = plt.gca()  # Get the current axes
            plot_path = os.path.join(os.getcwd(), 'test_plot.png')
            print(f"Saving test plot to {plot_path}")  # Debugging print
            fig1.savefig(plot_path)
            plt.close(fig1)

            # Create an expected plot
            fig2, ax2 = plt.subplots()
            expected_path = os.path.join(os.getcwd(), 'expected_plot.png')
            plt.plot(self.Time, self.Center, linewidth=2, color=np.array([44, 127, 184]) / 255)
            plt.fill_between(self.Time, self.InnerBot, self.InnerTop, color=np.array([44, 127, 184]) / 255, alpha=0.4)
            print(f"Saving expected plot to {expected_path}")  # Debugging print
            fig2.savefig(expected_path)
            plt.close(fig2)

            # Compare the two plots
            result = compare_images(expected_path, plot_path, tol=50)
            self.assertIsNone(result, f"Quantile plot did not match expected. Difference was {result}")

        except Exception as e:
            self.fail(f"An exception occurred: {e}")


class TestRsnbrckFunction(unittest.TestCase):

    def test_rsnbrck_at_origin(self):
        self.assertEqual(bvar.rsnbrck([0, 0]), 1)

    def test_rsnbrck_at_one_one(self):
        self.assertEqual(bvar.rsnbrck([1, 1]), 0)

    def test_rsnbrck_at_negative_one_two(self):
        self.assertEqual(bvar.rsnbrck([-1, 2]), 109)


class TestSetPriors(unittest.TestCase):

    def test_default_values(self):
        result = bvar.set_priors()
        expected = {
            'hyperpriors': 1,
            'Vc': 10e6,
            'pos': [],
            'MNalpha': 0,
            'MNpsi': 1,
            'sur': 1,
            'noc': 1,
            'Fcast': 1,
            'hz': [1, 2, 3, 4, 5, 6, 7, 8],
            'mcmc': 0,
            'Ndraws': 20000,
            'Ndrawsdiscard': 10000,
            'MCMCconst': 1,
            'MCMCfcast': 1,
            'MCMCstorecoeff': 1
        }
        self.assertEqual(result, expected)

    def test_custom_values(self):
        result = bvar.set_priors('Vc', 5e6, 'MNalpha', 1, 'hz', [1, 2, 3, 4])
        expected = {
            'hyperpriors': 1,
            'Vc': 5e6,
            'pos': [],
            'MNalpha': 1,
            'MNpsi': 1,
            'sur': 1,
            'noc': 1,
            'Fcast': 1,
            'hz': [1, 2, 3, 4],
            'mcmc': 0,
            'Ndraws': 20000,
            'Ndrawsdiscard': 10000,
            'MCMCconst': 1,
            'MCMCfcast': 1,
            'MCMCstorecoeff': 1
        }
        self.assertEqual(result, expected)


class TestSetPriorsCovid(unittest.TestCase):

    def test_default_values(self):
        r, mode, sd, priorcoef, MIN, MAX, var_info = bvar.set_priors_covid()

        # Assertions for 'r'
        self.assertEqual(r['hyperpriors'], 1)
        self.assertEqual(r['Vc'], 10e6)
        self.assertEqual(r['pos'], [])
        self.assertEqual(r['MNalpha'], 0)
        self.assertEqual(r['Tcovid'], [])
        self.assertEqual(r['sur'], 1)
        self.assertEqual(r['noc'], 1)
        self.assertEqual(r['Fcast'], 1)
        self.assertEqual(r['hz'], list(range(1, 9)))
        self.assertEqual(r['mcmc'], 0)
        self.assertEqual(r['Ndraws'], 20000)
        self.assertEqual(r['Ndrawsdiscard'], 10000)
        self.assertEqual(r['MCMCconst'], 1)
        self.assertEqual(r['MCMCfcast'], 1)
        self.assertEqual(r['MCMCstorecoeff'], 1)

        # Assertions for 'mode' and 'sd'
        self.assertEqual(mode, {'lambda': 0.2, 'miu': 1, 'theta': 1, 'eta': [0.8]})
        self.assertEqual(sd, {'lambda': 0.4, 'miu': 1, 'theta': 1, 'eta': [0.2]})

    def test_custom_values(self):
        custom_values = {
            'hyperpriors': 0,
            'Vc': 1e6,
            'pos': [1, 2, 3],
            'MNalpha': 1,
            'Tcovid': [100],
            'sur': 0,
            'noc': 0,
            'Fcast': 0,
            'hz': [1, 2, 3],
            'mcmc': 1,
            'Ndraws': 10000,
            'Ndrawsdiscard': 5000,
            'MCMCconst': 0,
            'MCMCfcast': 0,
            'MCMCstorecoeff': 0
        }

        r, mode, sd, priorcoef, MIN, MAX, var_info = bvar.set_priors_covid(**custom_values)

        # Assertions for 'r'
        self.assertEqual(r['hyperpriors'], 0)
        self.assertEqual(r['Vc'], 1e6)
        self.assertEqual(r['pos'], [1, 2, 3])
        self.assertEqual(r['MNalpha'], 1)
        self.assertEqual(r['Tcovid'], [100])
        self.assertEqual(r['sur'], 0)
        self.assertEqual(r['noc'], 0)
        self.assertEqual(r['Fcast'], 0)
        self.assertEqual(r['hz'], [1, 2, 3])
        self.assertEqual(r['mcmc'], 1)
        self.assertEqual(r['Ndraws'], 10000)
        self.assertEqual(r['Ndrawsdiscard'], 5000)
        self.assertEqual(r['MCMCconst'], 0)
        self.assertEqual(r['MCMCfcast'], 0)
        self.assertEqual(r['MCMCstorecoeff'], 0)

        # Assertions for 'mode' and 'sd'
        self.assertEqual(mode, {})
        self.assertEqual(sd, {})


class TestSKF(unittest.TestCase):

    def test_skf(self):
        Y = np.array([[1, 2, 3], [4, 5, 6]])
        Z = np.array([[0.5, 0.3], [0.1, 0.7]])
        R = np.array([[0.1, 0.0], [0.0, 0.2]])
        T = np.array([[0.9, 0.1], [0.2, 0.8]])
        Q = np.array([[0.05, 0.01], [0.01, 0.04]])
        A_0 = np.array([0.5, 0.5])
        P_0 = np.array([[0.1, 0.02], [0.02, 0.09]])
        c1 = np.array([0.1, 0.1])
        c2 = np.array([0.0, 0.0])

        results = bvar.SKF(Y, Z, R, T, Q, A_0, P_0, c1, c2)

        # Test shapes
        self.assertEqual(results['Am'].shape, (2, 3))
        self.assertEqual(results['Pm'].shape, (2, 2, 3))
        self.assertEqual(results['AmU'].shape, (2, 3))
        self.assertEqual(results['PmU'].shape, (2, 2, 3))

        # Test specific values based on the output you provided
        expected_Am = np.array([[0.5, 1.12292747, 1.95187608],
                                [0.5, 1.4452939, 2.42326301]])
        np.testing.assert_array_almost_equal(results['Am'], expected_Am, decimal=8)

        expected_Pm = np.array([[[0.1355, 0.12625128, 0.123198],
                                 [0.05, 0.04234192, 0.04032892]],
                                [[0.05, 0.04234192, 0.04032892],
                                 [0.108, 0.09361368, 0.08897237]]])
        np.testing.assert_array_almost_equal(results['Pm'], expected_Pm, decimal=8)

        expected_AmU = np.array([[1.07687512, 1.88453509, 2.85861695],
                                 [1.53739859, 2.55794499, 3.623489]])
        np.testing.assert_array_almost_equal(results['AmU'], expected_AmU, decimal=8)

        expected_PmU = np.array([[[0.09012704, 0.08670167, 0.08537893],
                                  [0.0140971, 0.01290655, 0.01274555]],
                                 [[0.0140971, 0.01290655, 0.01274555],
                                  [0.07108988, 0.0646472, 0.06233398]]])
        np.testing.assert_array_almost_equal(results['PmU'], expected_PmU, decimal=8)


class TestRunKF_DK(unittest.TestCase):

    def test_values(self):
        np.random.seed(0)  # Set random seed for reproducibility

        # Create synthetic data (observable variables)
        y = np.random.randn(4, 4)

        # Define system matrices
        A = np.array([[0.5, 0.1],
                      [0.2, 0.7]])

        C = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [0.5, 0.5],
                      [0.2, 0.3]])

        Q = np.array([[0.1, 0.05],
                      [0.05, 0.2]])

        R = np.array([[0.2, 0.1, 0.05, 0.02],
                      [0.1, 0.3, 0.04, 0.01],
                      [0.05, 0.04, 0.25, 0.03],
                      [0.02, 0.01, 0.03, 0.15]])

        x_0 = np.array([0.0, 0.0])
        Sig_0 = np.array([[1.0, 0.0],
                          [0.0, 1.0]])

        c1 = np.array([0.0, 0.0, 0.1, 0.1])
        c2 = np.array([0.0, 0.0])

        # Call the function
        result = bvar.runKF_DK(y, A, C, Q, R, x_0, Sig_0, c1, c2)

        # Expected result
        expected_result = np.array([[0.82188748, 0.56534059, 0.60389242, 0.87646068, 0.],
                                    [0.57814567, 0.00330498, 0.26098285, 0.04477614, 0.]])

        # Check if the result matches the expected result
        np.testing.assert_array_almost_equal(result, expected_result, decimal=8)


class TestTransformData(unittest.TestCase):

    def test_transform_data(self):
        # Sample raw data
        data_raw = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 6.0, 9.0, 12.0]
        ])

        # Transformation specifications
        spec = {
            'Transformation': ['log', 'lin', 'log', 'lin']
        }

        # Expected result
        expected_result = np.array([
            [0., 2., 109.86122887, 4.],
            [69.31471806, 4., 179.17594692, 8.],
            [109.86122887, 6., 219.72245773, 12.]
        ])

        # Call the function
        result = bvar.transform_data(spec, data_raw)

        # Check if the result matches the expected result
        np.testing.assert_array_almost_equal(result, expected_result, decimal=8)


class TestTrimrFunction(unittest.TestCase):

    def test_trimr(self):
        # Create a sample matrix
        x = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])

        # Specify the number of rows to trim from the top and bottom
        n1 = 1
        n2 = 1

        # Expected result
        expected_result = np.array([
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        # Call the function
        result = bvar.trimr(x, n1, n2)

        # Assert that the result matches the expected result
        np.testing.assert_array_equal(result, expected_result)


class TestWriteTexSidewaystable(unittest.TestCase):

    def test_write_tex_sidewaystable(self):
        # Prepare arguments for the function
        header = ['Header 1', 'Header 2', 'Header 3']
        style = 'l|c|r'
        table_body = [
            ['Row 1, Col 1', 1.23, 'Row 1, Col 3'],
            ['Row 2, Col 1', 4.56, 'Row 2, Col 3'],
            ['Row 3, Col 1', 7.89, 'Row 3, Col 3']
        ]
        above_tabular = 'This is a sample table.'
        below_tabular = 'Table notes go here.'

        # Redirect the output to a string
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            bvar.write_tex_sidewaystable(None, header, style, table_body, above_tabular, below_tabular)

            # Fetch the generated LaTeX code and strip leading/trailing white spaces from each line
            latex_output = '\n'.join(line.strip() for line in f.getvalue().split('\n')).strip()

            # Prepare the expected LaTeX code and strip leading/trailing white spaces from each line
            expected_output = '''\\begin{sidewaystable}[htpb!]
                                        This is a sample table.
                                        \\centering
                                        \\begin{tabular}{l|c|r}
                                        Header 1 & Header 2 & Header 3 \\\\
                                        Row 1, Col 1 & 1.23 & Row 1, Col 3 \\\\
                                        Row 2, Col 1 & 4.56 & Row 2, Col 3 \\\\
                                        Row 3, Col 1 & 7.89 & Row 3, Col 3 \\\\
                                        \\hline
                                        \\end{tabular}
                                        Table notes go here.
                                        \\end{sidewaystable}
                                        '''
            expected_output = '\n'.join(line.strip() for line in expected_output.split('\n')).strip()

            # Assert if the generated LaTeX code matches the expected code
            self.assertEqual(latex_output, expected_output)


class TestVARcfDKcksFunction(unittest.TestCase):

    def test_VARcf_DKcks(self):
        # Set random seed for reproducibility
        np.random.seed(1234)

        # Create synthetic data for observable variables X (50, 4)
        X = np.random.randn(50, 4)
        X[-6:, :] = np.nan  # Add NaNs to the end to simulate missing observations

        # Define number of lags in VAR (p = 2)
        p = 2

        # Create synthetic coefficients for the VAR (9, 4)
        beta = np.random.randn(9, 4)

        # Create synthetic covariance matrix of the VAR (4, 4)
        Su = np.cov(np.random.randn(100, 4), rowvar=False)

        # Number of draws (optional, nDraws = 1)
        nDraws = 1

        # Run your function
        Xcond = bvar.VARcf_DKcks(X, p, beta, Su, nDraws)

        # Read the expected output from Excel
        expected_output = pd.read_excel('Xcond_VARcf_DKcks.xlsx').values

        # Compare actual and expected outputs
        np.testing.assert_allclose(Xcond, expected_output, rtol=1e-5, atol=1e-8)

        print("Unit test for VARcf_DKcks() was successful.")


# Run the tests
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLogMLVARFormCMCCovid)
    unittest.TextTestRunner().run(suite)
