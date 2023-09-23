# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde, gaussian_kde
from scipy.optimize import fsolve
from scipy.special import gammaln, betaln
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import gamma
from matplotlib import cm
import unittest

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


# Define the unit test class, inheriting from unittest.TestCase
class TestCsminit(unittest.TestCase):

    def test_csminit(self):
        # Define a simple quadratic objective function for testing
        def objective_function(x):
            x = x.ravel()  # Convert x to a 1D array
            Q = np.array([[4, 2], [2, 2]])
            return np.dot(x.T, np.dot(Q, x)), np.dot(Q, x)

        # Initial point (x0)
        x0 = np.array([1.0, 1.0])

        # Function value at the initial point (f0)
        f0, _ = objective_function(x0)

        # Gradient at the initial point (g0)
        _, g0 = objective_function(x0)

        # Flag indicating if the gradient is bad (badg)
        badg = False

        # Approximate inverse Hessian or Hessian matrix at the initial point (H0)
        H0 = np.eye(2)

        # Run csminit function and get the results
        fhat, xhat, fcount, retcode = bvar.csminit(objective_function, x0, f0, g0, badg, H0)

        # Assertions to check if the function behaves as expected
        self.assertAlmostEqual(fhat, 0.21059324284113823, places=8)
        self.assertTrue(np.allclose(xhat, np.array([[-0.28878803, 0.14080798]]), atol=1e-8))
        self.assertIsInstance(fcount, int)
        self.assertIsInstance(retcode, int)


class TestBFGSI(unittest.TestCase):

    def test_bfgsi(self):
        # Initial inverse Hessian (identity matrix for simplicity)
        H0 = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Previous changes in gradient and x
        dg = np.array([0.1, -0.2])
        dx = np.array([-0.3, 0.4])

        # Expected updated inverse Hessian (based on your manual test)
        expected_H_new = np.array([[0.00826446, 1.50413223],
                                   [1.50413223, -1.24793388]])

        # Run the bfgsi function
        H_new = bvar.bfgsi(H0, dg, dx)

        # Check if the results are as expected
        self.assertTrue(np.allclose(H_new, expected_H_new, atol=1e-8))

    def tearDown(self):
        if self.currentResult.wasSuccessful():
            print("Unit test for bfgsi successful")

    def run(self, result=None):
        self.currentResult = result  # remember result for use in tearDown
        unittest.TestCase.run(self, result)  # call superclass run method


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


# Define the unit test class
class TestNumGrad(unittest.TestCase):
    def test_numgrad(self):
        # Define a test function f(x) = x1^2 + x2^2
        def test_function(x):
            return x[0] ** 2 + x[1] ** 2

        # Define a point at which to evaluate the gradient
        x_test = np.array([1, 1])

        # Compute the numerical gradient at the test point
        g, badg = bvar.numgrad(test_function, x_test)

        # Analytical gradient for comparison
        analytical_gradient = np.array([2 * x_test[0], 2 * x_test[1]])

        # Perform the tests
        np.testing.assert_almost_equal(g, analytical_gradient, decimal=5,
                                       err_msg="Numerical gradient not close to analytical gradient")
        self.assertEqual(badg, 0, "Bad gradient flag should be 0")


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

# Run the tests
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMissData)
    unittest.TextTestRunner().run(suite)


