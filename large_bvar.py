import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde, gaussian_kde
from scipy.optimize import fsolve
from scipy.special import gammaln, betaln
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import gamma
from matplotlib import cm


def beta_coef(x, mosd):
    """
    Computes the coefficients for the Beta distribution.

    Args:
        x (list): Contains alpha and beta values for the Beta distribution
        mosd (list): Contains mode and standard deviation values.

    Returns:
        list: List with two results that represent two coefficients
    """

    al = x[0]  # alpha parameter
    bet = x[1]  # beta parameter

    # mode and standard deviation of the beta distribution
    mode = mosd[0]
    sd = mosd[1]

    # compute the first and second results based on the parameters
    r1 = mode - (al - 1) / (al + bet - 2)
    r2 = sd - (al * bet / ((al + bet) ** 2 * (al + bet + 1))) ** 0.5

    return [r1, r2]


def bfgsi(H0, dg, dx):
    """
    BFGS update of the inverse Hessian matrix.

    The function uses the Broyden-Fletcher-Goldfarb-Shanno (BFGS) formula
    to compute the new inverse Hessian based on the previous changes in
    gradient and variable x. If the update fails, a warning is displayed,
    and the original inverse Hessian is returned.

    Args:
        H0 (array_like): Initial inverse Hessian matrix. Must be 2D square matrix.
        dg (array_like): Previous change in gradient. Can be 1D or 2D column vector.
        dx (array_like): Previous change in x. Can be 1D or 2D column vector.

    Returns:
        H (array_like): Updated inverse Hessian matrix.
    """
    # Ensure dg and dx are column vectors by reshaping if they are 1D
    if len(dg.shape) == 1:
        dg = dg.reshape(-1, 1)
    if len(dx.shape) == 1:
        dx = dx.reshape(-1, 1)

    # Compute the product of H0 and dg
    Hdg = H0 @ dg
    # Compute the dot product of dg and dx
    dgdx = dg.T @ dx

    # Check if dgdx is not too small to avoid division by zero
    if abs(dgdx) > 1e-12:
        # BFGS update formula for the inverse Hessian
        H = H0 + (1 + (dg.T @ Hdg) / dgdx) * (dx @ dx.T) / dgdx - (dx @ Hdg.T + Hdg @ dx.T) / dgdx
    else:
        # Display a warning if the BFGS update fails
        print('bfgs update failed.')
        print('|dg| =', np.sqrt(dg.T @ dg), '|dx| =', np.sqrt(dx.T @ dx))
        print("dg'*dx =", dgdx)
        print('|H*dg| =', Hdg.T @ Hdg)
        # Return the original inverse Hessian if the update fails
        H = H0

    return H


def bvarFcst(y, beta, hz):
    """
    Computes the forecasts for y at the horizons specified in hz using the coefficients beta.

    Args:
        y (array_like): Observed data matrix (T x n).
        beta (array_like): Coefficients for the VAR model (k x n).
        hz (array_like): Horizons for the forecast.

    Returns:
        forecast (array_like): Forecasted values at the specified horizons (len(hz) x n).
    """
    k, n = beta.shape
    lags = (k - 1) // n
    T = y.shape[0]

    # Initialize forecast matrix with zeros at the end
    Y = np.vstack([y, np.zeros((max(hz), n))])

    # Compute the forecasts
    for tau in range(1, max(hz) + 1):
        xT = np.hstack([1, Y[T + tau - lags - 1:T + tau - 1, :].flatten()])[np.newaxis, :]
        Y[T + tau - 1, :] = xT @ beta

    # Extract the forecasts at the specified horizons
    forecast = Y[T + np.array(hz) - 1, :]

    return forecast


def bvarIrfs(beta, sigma, nshock, hmax):
    """
    Computes IRFs using Cholesky ordering to shock in position nshock
    up to horizon hmax based on beta and sigma.

    Parameters:
        beta (array_like): Coefficient matrix (k x n)
        sigma (array_like): Covariance matrix (n x n)
        nshock (int): Position of the shock
        hmax (int): Maximum horizon for the impulse response

    Returns:
        irf (array_like): Impulse response functions at different horizons
    """

    k, n = beta.shape
    lags = (k - 1) // n
    cholVCM = np.linalg.cholesky(sigma).T

    # Initialize Y with zeros, adding an extra row to avoid out-of-bounds indexing
    Y = np.zeros((lags + hmax + 1, n))

    in_ = lags
    vecshock = np.zeros(n)
    vecshock[nshock - 1] = 1

    for tau in range(1, hmax + 1):
        Y_slice = Y[in_ + tau - 1:in_ + tau - lags - 1:-1, :]
        xT = Y_slice.T.flatten(order='F')  # Flatten column-wise

        # Handle shape discrepancy by adding zeros if needed
        if xT.shape[0] < k - 1:
            xT = np.pad(xT, (0, k - 1 - xT.shape[0]), 'constant')

        Y[in_ + tau, :] = xT @ beta[1:, :] + (tau == 1) * (cholVCM @ vecshock).T

    # Remove the extra row that was added to avoid out-of-bounds indexing
    irf = Y[in_ + 1:in_ + hmax + 1, :]
    return irf


def cholred(S):
    """
    Compute the reduced Cholesky decomposition of a matrix.

    Args:
        S (array_like): Input matrix (n x n).

    Returns:
        array_like: Reduced Cholesky decomposition (n x n).
    """
    # Compute the eigenvalues and eigenvectors
    v, D = np.linalg.eig((S + S.T) / 2)

    # Take only the real parts of the eigenvalues
    d = np.real(v)

    # Compute the scaling factor
    scale = np.mean(np.diag(S)) * 1e-12

    # Identify the significant eigenvalues
    J = d > scale

    # Initialize the resulting matrix
    C = np.zeros_like(S)

    # Compute the columns of C based on the significant eigenvalues
    for i in range(len(J)):
        if J[i]:
            C[:, i] = np.dot(D[:, i], np.sqrt(d[i]))

    return C


def cols(x):
    """
    Return the number of columns in a matrix x.

    Args:
        x (array_like): Input matrix.

    Returns:
        int: Number of columns in x.
    """
    return x.shape[1]


def csminit(fcn, x0, f0, g0, badg, H0, *args):
    """
    Performs a line search to find a suitable step size for optimization.

    This function iteratively adjusts the step size lambda in the direction of
    descent. It accounts for two separate cases: the objective function returning
    either a scalar or a tuple.

    Args:
        fcn (function): Objective function to minimize. The function may return either a scalar
            value or a tuple containing the value and gradient.
        x0 (array): Initial point in the optimization space, shape (n, 1).
        f0 (float): Function value at the initial point, f0 = fcn(x0).
        g0 (array): Gradient at the initial point, shape (n, 1).
        badg (bool): Flag indicating if the gradient is bad (potentially inaccurate).
        H0 (matrix): Approximate inverse Hessian or Hessian matrix at the initial point, shape (n, n).
        *args: Additional arguments passed to the target function.

    Returns:
        fhat (float): Best function value found during the line search.
        xhat (array): Point corresponding to the best function value, shape (n, 1).
        fcount (int): Number of function evaluations.
        retcode (int): Return code indicating the termination condition.

    Raises:
        ValueError: If H0 is not a square matrix.
    """
    # Constants
    ANGLE = 0.005
    THETA = 0.3
    FCHANGE = 1000
    MINLAMB = 1e-9
    MINDFAC = 0.01

    # Initialization
    fcount = 0
    lambda_ = 1
    xhat = x0
    f = f0
    fhat = f0
    fPeak = f0
    g = g0
    gnorm = np.linalg.norm(g)
    retcode = 0

    # Ensure x0 is a 2D array (row vector if originally a 1D array)
    if len(x0.shape) == 1:
        x0 = x0.reshape(1, -1)

    # Ensure g0 is a 2D column vector
    if len(g0.shape) == 1:
        g0 = g0.reshape(-1, 1)

    # Check if H0 is a square matrix
    if H0.shape[0] != H0.shape[1]:
        raise ValueError("H0 must be a square matrix")

    # Check for convergence based on gradient norm
    if gnorm < 1e-12 and not badg:
        retcode = 1
        dxnorm = 0
    else:
        # Calculate direction of descent
        dx = -H0.dot(g)
        dxnorm = np.linalg.norm(dx)
        if dxnorm > 1e12:
            print('Near-singular H problem.')
            dx *= FCHANGE / dxnorm
        dfhat = dx.T.dot(g0)

        # Check if dfhat is a 2D array with shape (1, 1)
        if dfhat.shape == (1, 1) or dfhat.shape == (1,):
            # If dfhat is a 2D array with shape (1, 1), extract the scalar value
            dfhat_scalar = dfhat[0]
        else:
            # If dfhat is already a scalar, use it as-is
            dfhat_scalar = dfhat

        # Print the predicted improvement using the scalar value
        print(f'Predicted improvement: {-dfhat_scalar / 2:.9f}')

        # Correct for low angle if gradient is not bad
        if not badg:
            a = -dfhat_scalar / (gnorm * dxnorm)
            if a < ANGLE:
                dx -= (ANGLE * dxnorm / gnorm + dfhat_scalar / (gnorm * gnorm)) * g
                dx *= dxnorm / np.linalg.norm(dx)
                dfhat = dx.T.dot(g)

                # Check if dfhat is a 2D array with shape (1, 1) after recomputation
                if dfhat.shape == (1, 1):
                    dfhat_scalar = dfhat[0, 0]
                else:
                    dfhat_scalar = dfhat

                print(f'Correct for low angle: {a}')
                print(f'Predicted improvement: {-dfhat_scalar / 2:.9f}')

        print(f'Predicted improvement: {-dfhat_scalar / 2:.9f}')

        # Variables for adjusting step size lambda
        done = False
        factor = 3
        shrink = True
        lambdaMax = np.inf
        lambdaPeak = 0

        # Loop to adjust step size lambda
        while not done:
            dxtest = x0 + (dx.T * lambda_) if x0.shape[1] > 1 else x0 + dx * lambda_
            result = fcn(dxtest, *args)
            f = result if np.isscalar(result) else result[0]
            print(f'lambda = {lambda_:10.5g}; f = {f:20.7f}')

            # Update best function value and corresponding x
            if f < fhat:
                fhat = f
                xhat = dxtest

            fcount += 1

            # Determine shrink or grow signals
            shrinkSignal = (~badg & (f0 - f < max([-THETA * dfhat * lambda_, 0]))) or (badg & (f0 - f < 0))
            growSignal = ~badg & ((lambda_ > 0) & (f0 - f > -(1 - THETA) * dfhat * lambda_))

            # Conditions to shrink lambda
            if shrinkSignal and ((lambda_ > lambdaPeak) or (lambda_ < 0)):
                if lambda_ > 0 and ((not shrink) or (lambda_ / factor <= lambdaPeak)):
                    shrink = True
                    factor **= 0.6
                    while lambda_ / factor <= lambdaPeak:
                        factor **= 0.6
                    if abs(factor - 1) < MINDFAC:
                        retcode = 2 if abs(lambda_) < 4 else 7
                        done = True
                if (lambda_ < lambdaMax) and (lambda_ > lambdaPeak):
                    lambdaMax = lambda_
                lambda_ /= factor
                if abs(lambda_) < MINLAMB:
                    if (lambda_ > 0) and (f0 <= fhat):
                        # try going against gradient, which may be inaccurate
                        lambda_ = -lambda_ * factor ** 6
                    else:
                        retcode = 6 if lambda_ < 0 else 3
                        done = True

            # Conditions to grow lambda
            elif (growSignal and lambda_ > 0) or (shrinkSignal and ((lambda_ <= lambdaPeak) and (lambda_ > 0))):
                if shrink:
                    shrink = False
                    factor **= 0.6
                    if abs(factor - 1) < MINDFAC:
                        retcode = 4 if abs(lambda_) < 4 else 7
                        done = True
                if (f < fPeak) and (lambda_ > 0):
                    fPeak = f
                    lambdaPeak = lambda_
                    if lambdaMax <= lambdaPeak:
                        lambdaMax = lambdaPeak * factor * factor
                lambda_ *= factor
                if abs(lambda_) > 1e20:
                    retcode = 5
                    done = True

            else:
                done = True
                retcode = 7 if factor < 1.2 else 0

        print(f'Norm of dx {float(dxnorm):10.5g}')
    # Ensure xhat is reshaped to match the shape of x0
    xhat = xhat.reshape(x0.shape)

    return fhat, xhat, fcount, retcode


def csminwel(fcn, x0, H0, grad=None, crit=1e-6, nit=1000, *args):
    """
        Minimizes a given function using a quasi-Newton method.

        This function attempts to find the minimum of a given function using a quasi-Newton
        method, starting from an initial guess and following an iterative process. It can handle
        special cases during optimization like hitting a "wall" or a "cliff," and it provides
        debugging information to inform the user of the status and any special conditions that
        occurred during the optimization.

        Args:
            fcn (callable): The objective function to be minimized.
            x0 (array-like): Initial guess for the parameter vector.
            H0 (array-like): Initial value for the inverse Hessian matrix; must be positive definite.
            grad (callable or array-like): Either a callable that calculates the gradient, or an
                array representing the gradient. If None, the program calculates a numerical gradient.
            crit (float): Convergence criterion. Iteration will cease when it proves impossible
                to improve the function value by more than `crit`.
            nit (int): Maximum number of iterations.
            *args: Additional parameters that get handed off to `fcn` each time it is called.

        Returns:
            tuple:
                fh (float): The value of the function at the minimum.
                xh (array-like): The value of the parameters that minimize the function.
                gh (array-like): The gradient of the function at the minimum.
                H (array-like): The estimated inverse Hessian at the minimum.
                itct (int): The total number of iterations performed.
                fcount (int): The total number of function evaluations.
                retcodeh (int): Return code that provides information about why the algorithm terminated.

        Notes:
            If the program ends abnormally, it is possible to retrieve the current `x`, `f`, and `H`
            from the files `g1.mat`, `g2.mat`, `g3.mat`, and `H.mat` that are written at each iteration
            and at each Hessian update, respectively. This is a feature present in the original MATLAB
            implementation and would need additional code to implement in Python.
        """
    x0 = np.asarray(x0).reshape(-1, 1)  # Ensure x0 is a column vector
    nx, no = x0.shape
    nx = max(nx, no)
    Verbose = 1
    NumGrad = grad is None

    done = 0
    itct = 0
    fcount = 0
    snit = 100

    # Only assign the function value to f0, and ignore the gradient
    f0, _ = fcn(x0, *args)
    if f0 > 1e50:
        print('Bad initial parameter.')
        return None

    if NumGrad:
        g, badg = numgrad(fcn, x0, True, *args)
        g = g.reshape(-1, 1)  # ensure that g is a column vector
    else:
        g, badg = grad(x0, *args) if callable(grad) else grad, any(grad == 0)
        g = g.reshape(-1, 1)  # ensure that g is a column vector

    retcode3 = 101
    x = x0
    f = f0
    H = H0
    cliff = 0

    # Initialize variables with default values before entering the while loop
    f3 = badg3 = badg2 = badg1 = 0  # Assuming 0 is a sensible default
    fh = xh = retcodeh = None  # Assuming None is a sensible default, you may adjust as needed

    while not done:
        # initialize empty gradient vectors
        g1 = g2 = g3 = []
        # Save current state to files
        np.save('g1.npy', g)  # Save current gradient
        np.save('x.npy', x)  # Save current parameters
        np.save('f.npy', f)  # Save current function value
        np.save('H.npy', H)  # Save current Hessian
        print("-----------------")
        print("-----------------")
        print(f'f at the beginning of new iteration, {f:20.10f}')
        print(f'x = {x.ravel()}')  # print as row vector

        itct += 1
        f1, x1, fc, retcode1 = csminit(fcn, x, f, g, badg, H, *args)
        x1 = x1.reshape(-1, 1)  # ensure x1 is a column vector
        fcount += fc

        if retcode1 != 1:
            if retcode1 == 2 or retcode1 == 4:
                wall1 = 1
                badg1 = 1
            else:
                g1, badg1 = ((numgrad(fcn, x1, *args) if NumGrad else grad(x1, *args))
                             if callable(grad) else grad, any(grad == 0))
                wall1 = badg1
            if wall1 and len(H) > 1:
                Hcliff = H + np.diag(np.diag(H) * np.random.rand(nx, 1))
                print('Cliff. Perturbing search direction.')
                f2, x2, fc, retcode2 = csminit(fcn, x, f, g, badg, Hcliff, *args)
                fcount += fc
                if f2 < f:
                    if retcode2 == 2 or retcode2 == 4:
                        wall2 = 1
                        badg2 = 1
                    else:
                        g2, badg2 = ((numgrad(fcn, x2, *args) if NumGrad else grad(x2, *args))
                                     if callable(grad) else grad, any(grad == 0))
                        wall2 = badg2

                        if wall2:
                            print('Cliff again. Try traversing')
                            if np.linalg.norm(x2 - x1) < 1e-13:
                                f3, x3, badg3, retcode3 = f, x, 1, 101
                            else:
                                gcliff = ((f2 - f1) / ((np.linalg.norm(x2 - x1)) ** 2)) * (x2 - x1)
                                if x0.shape[1] > 1:
                                    gcliff = gcliff.T
                                gcliff = gcliff.reshape(-1, 1)  # ensure gcliff is a column vector
                                f3, x3, fc, retcode3 = csminit(fcn, x, f, gcliff, 0, np.eye(nx), *args)
                                fcount += fc
                                if retcode3 == 2 or retcode3 == 4:
                                    wall3 = 1
                                    badg3 = 1
                                else:
                                    g3, badg3 = ((numgrad(fcn, x3, *args) if NumGrad else grad(x3, *args)) if
                                                 callable(grad) else grad, any(grad == 0))
                                    wall3 = badg3
                else:
                    f3, x3, badg3, retcode3 = f, x, 1, 101
            else:
                f2, f3, badg2, badg3, retcode2, retcode3 = f, f, 1, 1, 101, 101
        else:
            f2, f3, f1, retcode2, retcode3 = f, f, f, retcode1, retcode1

        if f3 < f - crit and badg3 == 0:
            ih = 3
            fh = f3
            xh = x3.reshape(-1, 1)  # ensure that xh is a column vector
            gh = g3.reshape(-1, 1)  # ensure that gh is a column vector
            badgh = badg3
            retcodeh = retcode3
        elif f2 < f - crit and badg2 == 0:
            ih = 2
            fh = f2
            xh = x2
            gh = g2
            badgh = badg2
            retcodeh = retcode2
        elif f1 < f - crit and badg1 == 0:
            ih = 1
            fh = f1
            xh = x1
            gh = g1
            badgh = badg1
            retcodeh = retcode1
        else:
            fh, ih = min((f1, 1), (f2, 2), (f3, 3))
            xh = [x1, x2, x3][ih - 1]
            retcodei = [retcode1, retcode2, retcode3]
            retcodeh = retcodei[ih - 1]

            # Inside the loop, before determining optimal values:
            if not hasattr(locals(), 'gh') or gh is None:
                nogh = True
            else:
                nogh = False
            if nogh:
                gh, badgh = ((numgrad(fcn, xh, *args) if NumGrad else grad(xh, *args)) if callable(grad)
                             else grad, any(grad == 0))
            badgh = 1

        stuck = abs(fh - f) < crit
        if not badg and not badgh and not stuck:
            H = bfgsi(H, gh - g, xh - x)
            np.save('H.npy', H)  # Save updated Hessian

        if Verbose:
            print('----')
            print(f'Improvement on iteration {itct} = {f - fh:18.9f}')
            if itct > nit:
                print('iteration count termination')
                done = 1
            elif stuck:
                print('improvement < crit termination')
                done = 1
            rc = retcodeh
            print('Return code:', rc)

        f = fh
        x = xh
        g = gh
        badg = badgh

    return fh, xh, gh, H, itct, fcount, retcodeh


def csolve(FUN, x, gradfun, crit, itmax, *varargin):
    """
        Finds the solution to a system of nonlinear equations using iterative methods.

        Args:
            FUN (function): The function representing the system of equations. Accepts a vector or matrix `x`.
            x (list or np.array): Initial guess for the solution.
            gradfun (function or None): Function to evaluate the gradient matrix. If None, a numerical gradient is used.
            crit (float): Tolerance level; if the sum of absolute values returned by FUN is less than this, the equation is considered solved.
            itmax (int): Maximum number of iterations.
            *varargin: Additional arguments passed on to FUN and gradfun.

        Returns:
            tuple: A tuple containing:
                - x (list or np.array): Solution to the system of equations.
                - rc (int): Return code indicating the status of the solution.
                    - 0: Normal solution.
                    - 1, 3: No solution (likely a numerical problem or discontinuity).
                    - 4: Termination due to reaching the maximum number of iterations.

        Example:
            def fun(x):
                return [x[0]**2 + x[1] - 3, x[0] + x[1]**2 - 3]

            x0 = [1, 1]
            crit = 1e-6
            itmax = 100

            x, rc = csolve(fun, x0, None, crit, itmax)
        """
    delta = 1e-6
    alpha = 1e-3
    verbose = 1
    analyticg = 1 if gradfun else 0
    nv = len(x)
    tvec = delta * np.eye(nv)
    done = 0
    f0 = FUN(x, *varargin) if varargin else FUN(x)
    af0 = np.sum(np.abs(f0))
    af00 = af0
    itct = 0
    while not done:
        if itct > 3 and af00 - af0 < crit * max(1, af0) and itct % 2 == 1:
            randomize = 1
        else:
            if not analyticg:
                grad = ((FUN(x * np.ones((1, nv)) + tvec, *varargin) - f0 * np.ones((1, nv))) /
                        delta) if varargin else (FUN(x * np.ones((1, nv)) + tvec) - f0 * np.ones((1, nv))) / delta
            else:
                grad = gradfun(x, *varargin)
            if np.linalg.cond(grad) < 1e-12:
                grad += tvec
            dx0 = -np.linalg.solve(grad, f0)
            randomize = 0
        if randomize:
            if verbose:
                print("\n Random Search")
            dx0 = np.linalg.norm(x) / np.random.randn(*x.shape)
        lambda_ = 1
        lambdamin = 1
        fmin = f0
        xmin = x
        afmin = af0
        dxSize = np.linalg.norm(dx0)
        factor = 0.6
        shrink = 1
        subDone = 0
        while not subDone:
            dx = lambda_ * dx0
            f = FUN(x + dx, *varargin) if varargin else FUN(x + dx)
            af = np.sum(np.abs(f))
            if af < afmin:
                afmin = af
                fmin = f
                lambdamin = lambda_
                xmin = x + dx
            if ((lambda_ > 0) and (af0 - af < alpha * lambda_ * af0)) or ((lambda_ < 0) and (af0 - af < 0)):
                if not shrink:
                    factor = factor ** 0.6
                    shrink = 1
                if abs(lambda_ * (1 - factor)) * dxSize > 0.1 * delta:
                    lambda_ = factor * lambda_
                elif (lambda_ > 0) and (factor == 0.6):
                    lambda_ = -0.3
                else:
                    subDone = 1
                    if lambda_ > 0:
                        if factor == 0.6:
                            rc = 2
                        else:
                            rc = 1
                    else:
                        rc = 3
            elif (lambda_ > 0) and (af - af0 > (1 - alpha) * lambda_ * af0):
                if shrink:
                    factor = factor ** 0.6
                    shrink = 0
                lambda_ = lambda_ / factor
            else:
                subDone = 1
                rc = 0
        itct += 1
        if verbose:
            print(f'\nitct {itct}, af {afmin}, lambda {lambdamin}, rc {rc}')
            print(f'   x  {xmin}')
            print(f'   f  {fmin}')
        x = xmin
        f0 = fmin
        af00 = af0
        af0 = afmin
        if itct >= itmax:
            done = 1
            rc = 4
        elif af0 < crit:
            done = 1
            rc = 0
    return x, rc


def drsnbrck(x):
    """
    Compute the derivative for the Rosenbrock problem.

    Parameters:
        x (numpy.array): Input vector of shape (2, 1)

    Returns:
        dr (numpy.array): Derivative of shape (2, 1)
        badg (int): Flag indicating if the gradient is bad (always 0)
    """
    dr = np.zeros((2, 1))
    dr[0, 0] = 2 * (x[0] - 1) - 8 * 105 * x[0] * (x[1] - x[0] ** 2) ** 3
    dr[1, 0] = 4 * 105 * (x[1] - x[0] ** 2) ** 3
    badg = 0

    return dr, badg


def FIS(Y, Z, R, T, S):
    """
    Fixed interval smoother (see Durbin and Koopman, 2001, p. 64-71).
    FIS returns the smoothed state vector AmT and its covariance matrix PmT.

    Args:
        Y (numpy.ndarray): Data (nobs x n).
        Z (numpy.ndarray): Matrix Z.
        R (numpy.ndarray): Matrix R.
        T (numpy.ndarray): Matrix T.
        S (dict): Estimates from Kalman filter SKF.

    Returns:
        dict: Smoothed estimates added to above.
    """
    m, nobs = S['Am'].shape
    S['AmT'] = np.zeros((m, nobs))
    S['PmT'] = np.zeros((m, m, nobs))
    S['AmT'][:, nobs - 1] = S['AmU'][:, nobs - 1]
    r = np.zeros(m)

    for t in range(nobs - 1, -1, -1):
        y_t, Z_t = MissData(Y[:, t], Z, R, np.zeros(len(Y[:, t])))
        r = S['ZF'][t].dot(S['V'][t]) + (T.dot(np.eye(m) - S['Pm'][:, :, t].dot(S['ZF'][t]).dot(Z_t))).T.dot(r)
        S['AmT'][:, t] = S['Am'][:, t] + S['Pm'][:, :, t].dot(r)

    return S




def form_companion_matrices(betadraw, G, etapar, tstar, n, lags, TTfcst):
    """
    Forms the matrices of the VAR companion form.

    Args:
        betadraw (array_like): Coefficient matrix.
        G (array_like): Matrix G.
        etapar (array_like): Eta parameters.
        tstar (int): Parameter tstar.
        n (int): Number of variables.
        lags (int): Number of lags.
        TTfcst (int): Forecast periods.

    Returns:
        tuple: varc, varZ, varG, varC, varT, varH matrices.
    """
    # Matrices of observation equation
    varc = np.zeros((n, TTfcst))
    varZ = np.zeros((n, n * lags))
    varZ[:, :n] = np.eye(n)
    varZ = np.tile(varZ[:, :, np.newaxis], (1, 1, TTfcst))
    varG = np.tile(np.zeros((n, n)), (1, 1, TTfcst))

    # Matrices of state equation
    B = betadraw
    varC = np.zeros((n * lags, 1))
    varC[:n] = B[0, :].T
    varT = np.vstack((B[1:, :].T, np.hstack((np.eye(n * (lags - 1)), np.zeros((n * (lags - 1), n))))))
    varH = np.zeros((n * lags, n, TTfcst))

    for t in range(TTfcst):
        if t < tstar:
            varH[:n, :, t] = G
        elif t == tstar:
            varH[:n, :, t] = G * etapar[0]
        elif t == tstar + 1:
            varH[:n, :, t] = G * etapar[1]
        elif t == tstar + 2:
            varH[:n, :, t] = G * etapar[2]
        elif t > tstar + 2:
            varH[:n, :, t] = G * (1 + (etapar[2] - 1) * etapar[3] ** (t - tstar - 2))

    return varc, varZ, varG, varC, varT, varH


def gamma_coef(mode, sd, plotit):
    """
    Computes the coefficients of Gamma distribution coefficients and makes plots, if requested
    The parameters of the Gamma distribution are
    k = shape parameter: affects the PDF of the Gamma distribution, including skewness and mode
    theta = scale parameter: affects the spread of the distribution i.e. it shrinks or stretches the
    distribution along the x-axis.

    Args:
        mode (float): Mode of the Gamma distribution.
        sd (float): Standard deviation of the Gamma distribution.
        plotit (int): Flag to determine if the plot should be shown (1) or not (0).

    Returns:
        dict: Dictionary containing the 'k' and 'theta' parameters of the Gamma distribution.
    """
    # compute the k and theta parameters of the gamma distribution
    r_k = (2 + mode ** 2 / sd ** 2 + np.sqrt((4 + mode ** 2 / sd ** 2) * mode ** 2 / sd ** 2)) / 2
    r_theta = np.sqrt(sd ** 2 / r_k)

    if plotit == 1:  # if we request to make plot
        xxx = np.arange(0, mode + 5 * sd, 0.000001)
        # plot and show the Gamma distribution
        plt.plot(xxx, (xxx ** (r_k - 1) * np.exp(-xxx / r_theta) * r_theta ** -r_k) / gamma(r_k), 'k--', linewidth=2)
        plt.show()
    # display the computed k and theta parameters
    return {'k': r_k, 'theta': r_theta}


def kfilter_const(y, c, Z, G, C, T, H, shat, sig):
    """
    Kalman filter with constant variance for the state-space model.

    Args:
        y (np.array): Observation at time t.
        c (float): Constant term in observation equation.
        Z (np.array): Observation loading matrix.
        G (np.array): Observation noise loading matrix.
        C (float): Constant term in state equation.
        T (np.array): State transition matrix.
        H (np.array): State noise loading matrix.
        shat (float): Prior/initial state estimate.
        sig (np.array): Prior/initial state covariance matrix.

    Returns:
        shatnew (float): Updated state estimate.
        signew (np.array): Updated state covariance matrix.
        v (float): Prediction error.
        k (np.array): Kalman gain.
        sigmainv (np.array): Inverse of the innovation covariance.
    """
    n = len(y)
    omega = T @ sig @ T.T + H @ H.T
    sigmainv = np.linalg.inv(Z @ omega @ Z.T + G @ G.T)
    k = omega @ Z.T @ sigmainv
    v = y - c - Z @ (C + T @ shat)
    shatnew = C + T @ shat + k @ v
    signew = omega - k @ Z @ omega

    return shatnew, signew, v, k, sigmainv


def lag(x, n=1, v=0):
    """
    Creates a matrix or vector of lagged values.

    Args:
        x (array_like): Input matrix or vector, (nobs x k).
        n (int, optional): Order of lag. Defaults to 1.
        v (optional): Initial values. Defaults to 0.

    Returns:
        z (array_like): Matrix or vector of lags (nobs x k).

    Notes:
        If n <= 0, an empty array is returned.
    """
    if n < 1:
        return np.array([])

    cols_x = x.shape[1] if len(x.shape) > 1 else 1
    zt = np.ones((n, cols_x)) * v
    z = np.vstack([zt, x[:-n]])

    return z


def lagmatrix(Y, lags):
    """
    Create a matrix of lagged (time-shifted) series. Positive lags correspond
    to delays; negative lags correspond to leads.

    Args:
        Y (array_like): Time series data, a vector or a matrix.
        lags (array_like): Vector of integer delays or leads.

    Returns:
        YLag (array_like): numObs-by-(numSeries*numLags) matrix of lagged versions of the series in Y.
    """
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]  # Ensure a column vector

    # Ensure lags is a vector of integers
    lags = np.array(lags)
    non_integer_lags = lags != np.round(lags)
    if np.any(non_integer_lags):
        raise ValueError("Non-integer values in lags.")

    # Preallocate with NaN values
    num_lags = len(lags)
    num_obs, num_series = Y.shape
    YLag = np.nan * np.ones((num_obs, num_series * num_lags))

    for c in range(num_lags):
        L = lags[c]
        columns = slice(num_series * c, num_series * (c + 1))  # Columns to fill for this lag

        if L > 0:  # Time delays
            YLag[L:, columns] = Y[:-L]
        elif L < 0:  # Time leads
            YLag[:L, columns] = Y[-L:]
        else:  # No shifts
            YLag[:, columns] = Y

    return YLag


def logMLVAR_formcmc_covid(par, y, x, lags, T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, draw, hyperpriors,
                           priorcoef, Tcovid=None):
    """
    Compute the log marginal likelihood (logML) for a Bayesian VAR model with potential COVID adjustments.

    Args:
        par (array_like): Parameter vector including lambda, eta, theta, miu, alpha, omega, psi.
        y (array_like): Observed data matrix (T x n).
        x (array_like): Explanatory variables matrix (T x k).
        lags (int): Number of lags in the VAR model.
        T (int): Number of time periods.
        n (int): Number of variables.
        b (array_like): Prior mean for the regression coefficients (k x 1).
        MIN (dict): Lower bounds for the hyperparameters.
        MAX (dict): Upper bounds for the hyperparameters.
        SS (array_like): Scale matrix for the inverse-Wishart prior (n x n).
        Vc (float): Prior degrees of freedom for the covariance matrix.
        pos (list): Positions of the variables to include in the no-cointegration prior.
        mn (dict): Prior means for the hyperparameters.
        sur (int): Dummy for SUR prior.
        noc (int): Dummy for no-cointegration prior.
        y0 (array_like): Initial values for the dependent variables (n x 1).
        draw (int): Indicator for drawing from the posterior of beta and SIGMA.
        hyperpriors (int): Indicator for hyperpriors on the hyperparameters.
        priorcoef (dict): Prior distribution parameters for the hyperparameters.
        Tcovid (array_like, optional): Indicator variable for COVID adjustments.

    Returns:
        logML (float): Log marginal likelihood.
        betadraw (array_like): Draw from the posterior of the regression coefficients (k x 1).
        drawSIGMA (array_like): Draw from the posterior of the covariance matrix (n x n).
    """
    # Extraction of parameters
    lambda_ = par[0]
    eta = par[1:5]
    theta = par[5]
    miu = par[6]
    alpha = par[7]
    omega = par[8]
    psi = par[9]

    # Dummy observations
    Td = 0  # Initialize Td
    xdsur = []
    ydsur = []
    xdnoc = []
    ydnoc = []
    if sur == 1:
        xdsur = [1 / theta] + [(1 / theta) * np.tile(y0, (1, lags))]
        ydsur = (1 / theta) * y0
        y = np.vstack([y, ydsur])
        x = np.vstack([x, xdsur])
        Td = 1
    if noc == 1:
        ydnoc = (1 / miu) * np.diag(y0)
        ydnoc[pos, pos] = 0
        xdnoc = np.hstack([np.zeros((n, 1)), (1 / miu) * np.tile(np.diag(y0), (1, lags))])
        y = np.vstack([y, ydnoc])
        x = np.vstack([x, xdnoc])
        Td += n

        # COVID adjustments
        if Tcovid is not None:
            x = x[:-Tcovid]
            y = y[:-Tcovid]
            T = T - Tcovid
            Td = 0

    # Setting up matrices
    xxi = np.linalg.inv(x.T @ x + np.eye(x.shape[1]) * 1e-10)
    yx = y.T @ x
    SSR = np.sum((y - x @ xxi @ yx.T) ** 2)
    SIGMAols = SSR / (T + Td - 1) * np.eye(n)
    SIGMAdraw = SIGMAols
    SIGMAmean = SIGMAols
    V = np.kron(SIGMAols, xxi)
    iV = np.kron(np.linalg.inv(SIGMAols), np.linalg.inv(xxi))
    Sb = np.linalg.inv(x.T @ x / lambda_ + iV)  # Prior covariance for beta

    # Prior mean for beta
    b = np.concatenate([b, np.zeros((lags * n, 1))])
    b_prior = Sb @ (x.T @ x / lambda_ @ b + iV @ yx.T.flatten(order='F').reshape(-1, 1))

    # Hyperparameter priors
    logprior = 0
    if hyperpriors == 1:
        logprior += priorcoef['lambda']['a'] * np.log(lambda_) - priorcoef['lambda']['b'] * lambda_
        logprior += sum(priorcoef['eta']['a'] * np.log(eta) - priorcoef['eta']['b'] * eta)
        logprior += priorcoef['theta']['a'] * np.log(theta) - priorcoef['theta']['b'] * theta
        logprior += priorcoef['miu']['a'] * np.log(miu) - priorcoef['miu']['b'] * miu
        logprior += priorcoef['alpha']['a'] * np.log(alpha) - priorcoef['alpha']['b'] * alpha
        logprior += priorcoef['omega']['a'] * np.log(omega) - priorcoef['omega']['b'] * omega
        logprior += priorcoef['psi']['a'] * np.log(psi) - priorcoef['psi']['b'] * psi

    # Restrictions
    if lambda_ < MIN['lambda'] or lambda_ > MAX['lambda']:
        logML = -np.inf
        return logML, [], []
    if any(eta < MIN['eta']) or any(eta > MAX['eta']):
        logML = -np.inf
        return logML, [], []
    if theta < MIN['theta'] or theta > MAX['theta']:
        logML = -np.inf
        return logML, [], []
    if miu < MIN['miu'] or miu > MAX['miu']:
        logML = -np.inf
        return logML, [], []
    if alpha < MIN['alpha'] or alpha > MAX['alpha']:
        logML = -np.inf
        return logML, [], []
    if omega < MIN['omega'] or omega > MAX['omega']:
        logML = -np.inf
        return logML, [], []
    if psi < MIN['psi'] or psi > MAX['psi']:
        logML = -np.inf
        return logML, [], []

    # Computing the log marginal likelihood
    # Part 1: The constant term
    c = -(T + Td) * n / 2 * np.log(2 * np.pi)

    # Part 2: The log-prior for beta
    b_posterior = Sb @ (yx.T.flatten(order='F').reshape(-1, 1) + iV @ b_prior)
    Q_posterior = yx @ b_posterior - b_prior.T @ iV @ b_posterior
    log_prior_beta = (-1 / 2 * b_prior.T @ iV @ b_prior +
                      1 / 2 * b_posterior.T @ iV @ b_posterior +
                      1 / 2 * np.log(np.linalg.det(Sb)) -
                      1 / 2 * np.log(np.linalg.det(iV)) -
                      k / 2 * np.log(lambda_) +
                      logprior)

    # Part 3: The log-likelihood for beta|SIGMA
    log_likelihood_beta_SIGMA = -(T + Td) / 2 * np.log(np.linalg.det(SIGMAols)) - 1 / 2 * Q_posterior

    # Part 4: The log-likelihood for SIGMA
    log_likelihood_SIGMA = -(Vc + T + Td) / 2 * np.log(np.linalg.det(SIGMAols)) + n / 2 * np.log(
        np.linalg.det(Vc * SS))

    # Combine all parts
    logML = c + log_prior_beta + log_likelihood_beta_SIGMA + log_likelihood_SIGMA

    # Drawing from the posterior of beta and SIGMA
    betadraw = []
    drawSIGMA = []
    if draw == 1:
        betadraw = np.random.multivariate_normal(b_posterior.flatten(), Sb * SIGMAols)
        drawSIGMA = invwishart.rvs(df=Vc + T + Td - 1, scale=inv(Vc * SS + SSR), size=1)

    return logML, betadraw, drawSIGMA


def log_betapdf(x, al, bet):
    """
    Compute the log probability density function (PDF) of the Beta distribution.

    Args:
        x (float): Value at which to evaluate the PDF.
        al (float): Alpha parameter of the Beta distribution.
        bet (float): Beta parameter of the Beta distribution.

    Returns:
        float: Log PDF of the Beta distribution.
    """

    return (al - 1) * np.log(x) + (bet - 1) * np.log(1 - x) - betaln(al, bet)


def MissData(y, C, R, c1):
    """
    Eliminates the rows in y & matrices C, R, and vector c1 that correspond to missing data (NaN) in y.

    Args:
        y (numpy.ndarray): Vector of data.
        C (numpy.ndarray): Matrix C.
        R (numpy.ndarray): Matrix R.
        c1 (numpy.ndarray): Vector c1.

    Returns:
        tuple: Tuple containing the updated y, C, R, c1 after eliminating rows corresponding to missing data.
    """
    ix = ~np.isnan(y)
    index_array = np.where(ix)[0]  # Convert boolean array to integer array

    y = y[ix]
    c1 = c1[ix]
    C = C[ix, :]
    R = R[np.ix_(index_array, index_array)]  # Use integer array for slicing R

    return y, C, R, c1


def numgrad(fcn, x, returns_tuple=False, *args):
    """
    Computes the numerical gradient of a given function at a specific point.

    Args:
        fcn (callable): Function handle to the target function.
        x (array-like): Point at which the gradient is to be computed.
        returns_tuple (bool, optional): Flag indicating whether the target function returns a tuple containing both the
                                          function value and the gradient. Defaults to False.
        *args: Additional arguments passed to the target function.

    Returns:
        tuple: A tuple containing:
            - g (array): Numerical gradient at point x.
            - badg (int): Flag indicating if any component of the gradient is bad (0 if good, 1 if bad).

    Example:
        def f(x): return x[0]**2 + x[1]**2
        x = np.array([1, 1])
        g, badg = numgrad(f, x)
    """

    # Ensure x is treated as a column vector
    x = np.reshape(x, (-1, 1))

    # Define perturbation value for finite difference calculation
    delta = 1e-6

    # Get the length of the input x
    n = len(x)

    # Create a matrix with delta along the diagonal, for perturbing each variable
    tvec = delta * np.eye(n)

    # Initialize the gradient vector
    g = np.zeros(n)

    # Evaluate the function at the initial point x
    f0 = fcn(x, *args)
    if returns_tuple:
        f0, _ = f0  # Unpack the tuple if the target function returns a tuple

    # Flag to indicate if a bad gradient component is encountered
    badg = 0

    # Loop over each dimension to calculate the gradient
    for i in range(n):
        # Scaling factor for perturbation
        scale = 1

        # Select the appropriate perturbation vector
        tvecv = tvec[:, i]

        # Reshape the perturbation vector to match the shape of x
        tvecv_reshaped = np.reshape(tvecv, (-1, 1))

        # Compute the gradient for the i-th component using central difference
        f1 = fcn(x + scale * tvecv_reshaped, *args)
        if returns_tuple:
            f1, _ = f1  # Unpack the tuple if the target function returns a tuple
        g0 = (f1 - f0) / (scale * delta)

        # Check if the gradient component is within acceptable limits
        if abs(g0) < 1e15:
            g[i] = g0
        else:
            # If gradient component is bad, set it to 0 and flag the occurrence
            g[i] = 0
            badg = 1

    return g, badg


def log_ig2pdf(x, alpha, beta):
    """
    Compute the log probability density function (PDF) of the Inverse Gamma distribution.

    Args:
        x (float): Value at which to evaluate the PDF.
        alpha (float): Shape parameter of the Inverse Gamma distribution.
        beta (float): Scale parameter of the Inverse Gamma distribution.

    Returns:
        float: Log PDF of the Inverse Gamma distribution.
    """

    return alpha * np.log(beta) - (alpha + 1) * np.log(x) - beta / x - gammaln(alpha)


def ols1(y, x):
    if y.shape[0] != x.shape[0]:
        raise ValueError("x and y must have the same number of observations")

    nobs, nvar = x.shape
    result = {"nobs": nobs, "nvar": nvar, "bhatols": np.linalg.lstsq(x.T @ x, x.T @ y, rcond=None)[0]}

    # Compute the OLS coefficients
    result["yhatols"] = x @ result["bhatols"]
    result["resols"] = y - result["yhatols"]

    # Compute the estimated variance
    result["sig2hatols"] = (result["resols"].T @ result["resols"]) / (nobs - nvar)
    result["sigbhatols"] = result["sig2hatols"] * np.linalg.inv(x.T @ x)
    result["XX"] = x.T @ x

    # Compute R-squared
    result["R2"] = np.var(result["yhatols"]) / np.var(y)

    return result


def plot_joint_marginal(YY, Y1CondLim, xlab, ylab, vis=False, LW=1.5):
    """
        Plots the joint distribution in the center, with marginals on the side.
        This version also plots a version conditioning on a set of limits on the first variable.

        Args:
            YY (array_like): Matrix containing the data to be plotted.
            Y1CondLim (array_like): Limits for the first variable's conditioning.
            xlab (str): Label for the x-axis.
            ylab (str): Label for the y-axis.
            vis (str, optional): Figure visibility ('on' or 'off'). Defaults to 'off'.
            LW (float, optional): Line width. Defaults to 1.5.

        Returns:
            None: The function plots the joint distribution, marginals, and conditionals.
        """
    plotCond = True if Y1CondLim else False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Y1Lim = np.quantile(YY[:, 0], [0, 1])
    Y2Lim = np.quantile(YY[:, 1], [0, 1])

    gridY1 = np.linspace(Y1Lim[0], Y1Lim[1], 100)
    gridY2 = np.linspace(Y2Lim[0], Y2Lim[1], 100)

    # Plot data
    ax.scatter(YY[:, 0], YY[:, 1], .001 * np.ones(YY[:, 0].shape), s=2.5, c='k', alpha=.35)
    ax.set_xlim(Y1Lim)
    ax.set_ylim(Y2Lim)

    # Plot unconditional contour
    X, Y = np.meshgrid(gridY1, gridY2)
    values = np.vstack([X.ravel(), Y.ravel()])
    kernel = gaussian_kde(YY.T)
    Z = kernel(values)
    Z = Z.reshape(X.shape)
    ax.contour(X, Y, Z, colors='k')

    # Plot conditionals
    if plotCond:
        YYCond = YY[(YY[:, 0] >= Y1CondLim[0]) & (YY[:, 0] <= Y1CondLim[1]), :]
        ax.scatter(YYCond[:, 0], YYCond[:, 1], .001 * np.ones(YYCond[:, 0].shape), s=2.5, c='r', alpha=.35)

        # Plot conditional marginals
        kde_Y1Cond = gaussian_kde(YYCond[:, 0])
        kde_Y2Cond = gaussian_kde(YYCond[:, 1])
        ax.plot(gridY1, Y2Lim[1] * np.ones_like(gridY1), kde_Y1Cond(gridY1) / max(kde_Y1Cond(gridY1)), lw=LW, color='r')
        ax.plot(Y1Lim[0] * np.ones_like(gridY2), gridY2, kde_Y2Cond(gridY2) / max(kde_Y2Cond(gridY2)), lw=LW, color='r')

    # Plot unconditional marginals
    kde_Y1 = gaussian_kde(YY[:, 0])
    kde_Y2 = gaussian_kde(YY[:, 1])
    ax.plot(gridY1, Y2Lim[1] * np.ones_like(gridY1), kde_Y1(gridY1) / max(kde_Y1(gridY1)), lw=LW, color='k')
    ax.plot(Y1Lim[0] * np.ones_like(gridY2), gridY2, kde_Y2(gridY2) / max(kde_Y2(gridY2)), lw=LW, color='k')

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel('Normalized')

    if not vis:
        plt.close()

    plt.show()


def plot_joint_marginal2(YY, idx, xlab, ylab, vis='off', LW=1.5):
    """
    Plots joint distribution in center, with marginals on the side.
    This version also plots a version conditioning on a set of limits on the first variable.

    Args:
        YY (numpy.ndarray): Data to plot.
        idx (numpy.ndarray): Index for conditional data.
        xlab (str): Label for the x-axis.
        ylab (str): Label for the y-axis.
        vis (str, optional): Plot visibility ('on' or 'off'). Defaults to 'off'.
        LW (float, optional): Line width. Defaults to 1.5.
    """

    # Implementation details
    plot_cond = True if idx.any() else False

    Y1_lim = np.quantile(YY[:, 0], [0, 1])
    Y2_lim = np.quantile(YY[:, 1], [0, 1])

    grid_Y1 = np.linspace(Y1_lim[0], Y1_lim[1], 100)
    grid_Y2 = np.linspace(Y2_lim[0], Y2_lim[1], 100)

    grid_Y1, grid_Y2 = np.meshgrid(grid_Y1, grid_Y2)
    grid = np.c_[grid_Y1.ravel(), grid_Y2.ravel()]

    ax_angle = 45
    el = 30

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.scatter(YY[:, 0], YY[:, 1], .001 * np.ones(YY[:, 0].shape), s=2.5, c='k', alpha=.25)
    ax3d.set_xlim(Y1_lim)
    ax3d.set_ylim(Y2_lim)

    ax3d.plot([Y1_lim[0], Y1_lim[0]], Y2_lim, [0, 0], color='k', linewidth=1)
    ax3d.plot(Y1_lim, [Y2_lim[1], Y2_lim[1]], [0, 0], color='k', linewidth=1)

    density = kde.gaussian_kde(YY.T)
    f = density(grid.T)
    F = f.reshape(grid_Y1.shape)
    cont_unc = ax3d.contour(grid_Y1, grid_Y2, F, colors='k')
    for line in cont_unc.collections:
        line.set_linewidth(.75)

    if plot_cond:
        YY_cond = YY[idx == 1, :]
        ax3d.scatter(YY_cond[:, 0], YY_cond[:, 1], .001 * np.ones(YY_cond[:, 0].shape), s=2.5, c='r', alpha=.25)
        density_cond = kde.gaussian_kde(YY_cond.T)
        f_cond = density_cond(grid.T)
        F_cond = f_cond.reshape(grid_Y1.shape)
        cont_cond = ax3d.contour(grid_Y1, grid_Y2, F_cond, colors='r')
        for line in cont_cond.collections:
            line.set_linewidth(.75)

    Y1_i = np.linspace(Y1_lim[0], Y1_lim[1], 400)
    f_Y1 = kde.gaussian_kde(YY[:, 0])(Y1_i)
    Y2_i = np.linspace(Y2_lim[0], Y2_lim[1], 400)
    f_Y2 = kde.gaussian_kde(YY[:, 1])(Y2_i)

    if plot_cond:
        f_Y1_cond = kde.gaussian_kde(YY_cond[:, 0])(Y1_i)
        f_Y2_cond = kde.gaussian_kde(YY_cond[:, 1])(Y2_i)
        f_Y1_cond /= max(f_Y1_cond)
        f_Y2_cond /= max(f_Y2_cond)
        ax3d.plot(Y1_i, Y2_lim[1] * np.ones(Y1_i.shape), f_Y1_cond, linewidth=LW, color='r')
        ax3d.plot(Y1_lim[0] * np.ones(Y2_i.shape), Y2_i, f_Y2_cond, linewidth=LW, color='r')

    f_Y1 /= max(f_Y1)
    f_Y2 /= max(f_Y2)
    ax3d.plot(Y1_i, Y2_lim[1] * np.ones(Y1_i.shape), f_Y1, linewidth=LW, color='k')
    ax3d.plot(Y1_lim[0] * np.ones(Y2_i.shape), Y2_i, f_Y2, linewidth=LW, color='k')

    ax3d.set_xlim(Y1_lim)
    ax3d.set_ylim(Y2_lim)
    ax3d.view_init(ax_angle, el)
    plt.grid()
    plt.box(True)
    plt.xlabel(xlab, rotation=-el, fontsize=10)
    plt.ylabel(ylab, rotation=el, fontsize=10)
    ax3d.set_zlabel('Normalized')
    plt.show()


def plot_joint_marginal3(YYa, YYb, xlab, ylab, vis='off', LW=1.5):
    # Create figure and axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Limits for both datasets
    Y1aLim = np.quantile(YYa[:, 0], [0, 1])
    Y2aLim = np.quantile(YYa[:, 1], [0, 1])
    Y1bLim = np.quantile(YYb[:, 0], [0, 1])
    Y2bLim = np.quantile(YYb[:, 1], [0, 1])

    # Global limits
    Y1Lim = [min(Y1aLim[0], Y1bLim[0]), max(Y1aLim[1], Y1bLim[1])]
    Y2Lim = [min(Y2aLim[0], Y2bLim[0]), max(Y2aLim[1], Y2bLim[1])]
    gridY1 = np.linspace(Y1Lim[0], Y1Lim[1], 100)
    gridY2 = np.linspace(Y2Lim[0], Y2Lim[1], 100)

    GridY1, GridY2 = np.meshgrid(gridY1, gridY2)
    Grid = np.column_stack([GridY1.ravel(), GridY2.ravel()])

    # Plot data
    ax.scatter3D(YYa[:, 0], YYa[:, 1], 0.001 * np.ones_like(YYa[:, 0]), c='k', s=2.5, alpha=0.15)
    ax.scatter3D(YYb[:, 0], YYb[:, 1], 0.001 * np.ones_like(YYb[:, 0]), c='r', s=2.5, alpha=0.15)

    # Plot reference lines for marginals
    ax.plot3D([Y1Lim[0], Y1Lim[0]], Y2Lim, [0, 0], color='k', linewidth=1)
    ax.plot3D(Y1Lim, [Y2Lim[1], Y2Lim[1]], [0, 0], color='k', linewidth=1)

    # Plot contour a
    kde_a = gaussian_kde(YYa.T)
    F_a = kde_a.evaluate(Grid.T)
    F_a = F_a.reshape(GridY1.shape)
    ax.contour3D(GridY1, GridY2, F_a, cmap=cm.Greys, linewidths=0.75)

    # Plot contour b
    kde_b = gaussian_kde(YYb.T)
    F_b = kde_b.evaluate(Grid.T)
    F_b = F_b.reshape(GridY1.shape)
    ax.contour3D(GridY1, GridY2, F_b, cmap=cm.Reds, linewidths=0.75)

    # Create marginals (a)
    kde_Y1a = gaussian_kde(YYa[:, 0])
    kde_Y2a = gaussian_kde(YYa[:, 1])
    fY1a = kde_Y1a.evaluate(gridY1) / max(kde_Y1a.evaluate(gridY1))
    fY2a = kde_Y2a.evaluate(gridY2) / max(kde_Y2a.evaluate(gridY2))

    # Create marginals (b)
    kde_Y1b = gaussian_kde(YYb[:, 0])
    kde_Y2b = gaussian_kde(YYb[:, 1])
    fY1b = kde_Y1b.evaluate(gridY1) / max(kde_Y1b.evaluate(gridY1))
    fY2b = kde_Y2b.evaluate(gridY2) / max(kde_Y2b.evaluate(gridY2))

    # Plot marginals (a)
    ax.plot3D(gridY1, Y2Lim[1] * np.ones_like(gridY1), fY1a, linewidth=LW, color='k')
    ax.plot3D(Y1Lim[0] * np.ones_like(gridY2), gridY2, fY2a, linewidth=LW, color='k')
    ax.plot3D(gridY1, Y2Lim[1] * np.ones_like(gridY1), fY1b, linewidth=LW, color='r')
    ax.plot3D(Y1Lim[0] * np.ones_like(gridY2), gridY2, fY2b, linewidth=LW, color='r')

    ax.set_xlim(Y1Lim)
    ax.set_ylim(Y2Lim)
    ax.view_init(45, 30)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_zlabel('Normalized')

    if vis == 'off':
        plt.close(fig)
    else:
        plt.show()


def plot_joint_marginal4(YYa, YYb, xlab, ylab, vis='off', LW=1.5):
    """
    Plots the joint marginal distribution of two datasets in 2D space.

    Args:
        YYa (np.ndarray): The first dataset, an array of shape (n, 2).
        YYb (np.ndarray): The second dataset, an array of shape (n, 2).
        xlab (str): Label for the x-axis.
        ylab (str): Label for the y-axis.
        vis (str, optional): Visibility of the plot, either 'on' or 'off'. Defaults to 'off'.
        LW (float, optional): Line width for the plot. Defaults to 1.5.

    Returns:
        None: Displays the plot if vis is 'on'; otherwise, the plot is closed.
    """
    if vis == 'on':
        plt.figure()

    # Limits for both datasets
    Y1aLim = np.quantile(YYa[:, 0], [0, 1])
    Y2aLim = np.quantile(YYa[:, 1], [0, 1])
    Y1bLim = np.quantile(YYb[:, 0], [0, 1])
    Y2bLim = np.quantile(YYb[:, 1], [0, 1])

    # Global limits
    Y1Lim = [min(Y1aLim[0], Y1bLim[0]), max(Y1aLim[1], Y1bLim[1])]
    Y2Lim = [min(Y2aLim[0], Y2bLim[0]), max(Y2aLim[1], Y2bLim[1])]

    # Plotting data
    plt.plot(YYa[:, 0], YYa[:, 1], 'k.', markersize=4)
    plt.plot(YYb[:, 0], YYb[:, 1], 'r.', markersize=4)
    plt.xlim(Y1Lim)
    plt.ylim(Y2Lim)
    plt.xlabel(xlab, fontsize=10)
    plt.ylabel(ylab, fontsize=10)
    plt.gca().tick_params(labelsize=10)
    plt.gca().spines['top'].set_linewidth(LW)
    plt.gca().spines['right'].set_linewidth(LW)
    plt.gca().spines['bottom'].set_linewidth(LW)
    plt.gca().spines['left'].set_linewidth(LW)

    if vis == 'off':
        plt.close()
    else:
        plt.show()


def printpdf(h, outfilename):
    """
    Saves the given figure as a PDF file with specified dimensions.

    Args:
        h (matplotlib.figure.Figure): The figure object to be saved.
        outfilename (str): The path and name of the output PDF file.

    Example:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 1])
        printpdf(fig, "output.pdf")
    """
    # Set the figure's paper size to its current size
    h.set_size_inches(h.get_figwidth(), h.get_figheight())
    h.savefig(outfilename, format='pdf', bbox_inches='tight')


def quantilePlot(Time, Quantiles, baseColor=None):
    """
    Plots a line chart with filled quantile bands.

    Args:
        Time (array-like): Time values for the x-axis.
        Quantiles (array-like): Quantiles to be plotted, either a 5 or 7 column matrix.
        baseColor (tuple, optional): RGB color value for the plot. Defaults to blue.

    Raises:
        ValueError: If the quantile matrix is not of size 5 or 7.

    Example:
        Time = np.arange(0, 10, 0.1)
        Quantiles = np.column_stack([np.sin(Time) * i for i in [0.5, 0.75, 1.0, 0.75, 0.5]])
        quantilePlot(Time, Quantiles)
    """
    if baseColor is None:
        baseColor = np.array([44, 127, 184]) / 255

    if Quantiles.shape[1] == 5:
        OuterBot, InnerBot, Center, InnerTop, OuterTop = Quantiles.T
    elif Quantiles.shape[1] == 7:
        OuterBot, MiddleBot, InnerBot, Center, InnerTop, MiddleTop, OuterTop = Quantiles.T
    else:
        raise ValueError('Enter a valid quantile matrix')

    plt.plot(Time, Center, linewidth=2, color=baseColor)

    InnerIdx = ~np.isnan(InnerBot + InnerTop)
    plt.fill(np.concatenate([Time[InnerIdx], Time[InnerIdx][::-1]]),
             np.concatenate([InnerBot[InnerIdx], InnerTop[InnerIdx][::-1]]), baseColor,
             linestyle='none',
             alpha=0.4)

    if Quantiles.shape[1] == 7:
        MiddleIdx = ~np.isnan(MiddleBot + MiddleTop)
        plt.fill(np.concatenate([Time[MiddleIdx], Time[MiddleIdx][::-1]]),
                 np.concatenate([MiddleBot[MiddleIdx], MiddleTop[MiddleIdx][::-1]]), baseColor,
                 linestyle='none',
                 alpha=0.25)

    OuterIdx = ~np.isnan(OuterBot + OuterTop)
    plt.fill(np.concatenate([Time[OuterIdx], Time[OuterIdx][::-1]]),
             np.concatenate([OuterBot[OuterIdx], OuterTop[OuterIdx][::-1]]), baseColor,
             linestyle='none',
             alpha=0.15)

    plt.show()


def runKF_DK(y, A, C, Q, R, x_0, Sig_0, c1, c2):
    """
    Runs Kalman filter and smoother.

    Args:
        y (numpy.ndarray): Matrix of observable variables.
        A (numpy.ndarray): Transition matrix.
        C (numpy.ndarray): Measurement matrix.
        Q (numpy.ndarray): Covariance matrix Q.
        R (numpy.ndarray): Covariance matrix R.
        x_0 (numpy.ndarray): Initial state vector.
        Sig_0 (numpy.ndarray): Initial covariance matrix.
        c1 (numpy.ndarray): Constant vector c1.
        c2 (numpy.ndarray): Constant vector c2.

    Returns:
        numpy.ndarray: Smoothed state vector.
    """
    # Run the filter
    S = SKF(y, C, R, A, Q, x_0, Sig_0, c1, c2)
    # Run the smoother
    S = FIS(y, C, R, A, S)

    return S['AmT']


def rsnbrck(x):
    """
    Rosenbrock function.

    Args:
        x (list or numpy.ndarray): A vector of length 2.

    Returns:
        float: The result of the Rosenbrock function evaluated at the given vector `x`.
    """
    y = (1 - x[0]) ** 2 + 105 * (x[1] - x[0] ** 2) ** 4
    return y


def set_priors(*args):
    """
    This function sets up the default choices for the priors of the BVAR of Giannone, Lenza, and Primiceri (2012).

    Optional Parameters:
        hyperpriors: int (default 1)
            - 0: No priors on hyperparameters
            - 1: Reference priors on hyperparameters
        Vc: float (default 10e6)
            - Prior variance in the MN prior for the coefficients multiplying the constant term
        pos: list (default [])
            - Position of the variables that enter the VAR in first differences
        MNpsi: int (default 1)
            - Treatment of diagonal elements of the scale matrix of the IW prior on the covariance of the residuals
        MNalpha: int (default 0)
            - Lag-decaying parameter of the MN prior treated as hyperparameter
        sur: int (default 1)
            - Single-unit-root prior
        noc: int (default 1)
            - No-cointegration (sum-of coefficients) prior
        Fcast: int (default 1)
            - Generates forecasts at the posterior mode
        hz: int (default 8)
            - Longest horizon at which the code generates forecasts
        mcmc: int (default 0)
            - Runs the MCMC after the maximization
        Ndraws: int (default 20000)
            - Number of draws in the MCMC
        Ndrawsdiscard: int (default Ndraws/2)
            - Number of draws initially discarded to allow convergence in the MCMC
        MCMCconst: int (default 1)
            - Scaling constant for the MCMC
        MCMCfcast: int (default 1)
            - Generates forecasts while running the MCMC
        MCMCstorecoeff: int (default 1)
            - Stores the MCMC draws of the VAR coefficients and residual covariance matrix

    Returns:
        dict: Dictionary containing the set default choices for the priors.
    """

    # Default values
    r = {
        'hyperpriors': 1,
        'Vc': 10e6,
        'pos': [],
        'MNalpha': 0,
        'MNpsi': 1,
        'sur': 1,
        'noc': 1,
        'Fcast': 1,
        'hz': list(range(1, 9)),
        'mcmc': 0,
        'Ndraws': 20000,
        'Ndrawsdiscard': 10000,
        'MCMCconst': 1,
        'MCMCfcast': 1,
        'MCMCstorecoeff': 1
    }

    # Update values based on input arguments
    for i in range(0, len(args), 2):
        r[args[i]] = args[i + 1]

    # Additional options (not exposed to the user)
    if r['hyperpriors'] == 1:
        mode = {'lambda': 0.2, 'miu': 1, 'theta': 1}
        sd = {'lambda': 0.4, 'miu': 1, 'theta': 1}
        scalePSI = 0.02 ** 2
        priorcoef = {
            'lambda': gamma_coef(mode['lambda'], sd['lambda'], 0),
            'miu': gamma_coef(mode['miu'], sd['miu'], 0),
            'theta': gamma_coef(mode['theta'], sd['theta'], 0),
            'alpha.PSI': scalePSI,
            'beta.PSI': scalePSI
        }

    return r


def set_priors_covid(**kwargs):
    """
    This function sets up the default choices for the priors of the BVAR of
    Giannone, Lenza and Primiceri (2015), augmented with a change in
    volatility at the time of Covid (March 2020).

    Args:
        kwargs (dict): Keyword arguments for various options (see the script for details).
                       The optional keywords customize priors

    Returns:
        dict: Dictionary containing the set priors and options.
    """
    # Main options
    r = {
        'hyperpriors': kwargs.get('hyperpriors', 1),
        'Vc': kwargs.get('Vc', 10e6),
        'pos': kwargs.get('pos', []),
        'MNalpha': kwargs.get('MNalpha', 0),
        'Tcovid': kwargs.get('Tcovid', []),
        'sur': kwargs.get('sur', 1),
        'noc': kwargs.get('noc', 1),
        'Fcast': kwargs.get('Fcast', 1),
        'hz': kwargs.get('hz', list(range(1, 9))),
        'mcmc': kwargs.get('mcmc', 0),
        'Ndraws': kwargs.get('Ndraws', 20000),
        'Ndrawsdiscard': kwargs.get('Ndrawsdiscard', 10000),
        'MCMCconst': kwargs.get('MCMCconst', 1),
        'MCMCfcast': kwargs.get('MCMCfcast', 1),
        'MCMCstorecoeff': kwargs.get('MCMCstorecoeff', 1)
    }

    # Other options
    if r['hyperpriors'] == 1:
        mode = {'lambda': 0.2, 'miu': 1, 'theta': 1}
        sd = {'lambda': 0.4, 'miu': 1, 'theta': 1}
        scalePSI = 0.02 ** 2
        priorcoef = {
            'lambda': gamma_coef(mode['lambda'], sd['lambda'], 0),
            'miu': gamma_coef(mode['miu'], sd['miu'], 0),
            'theta': gamma_coef(mode['theta'], sd['theta'], 0)
        }
        mode['eta'] = [0.8]
        sd['eta'] = [0.2]
        # Additional computations for priorcoef.eta4 can be added here

    else:
        priorcoef = {}

    # Bounds for maximization
    MIN = {'lambda': 0.0001, 'alpha': 0.1, 'theta': 0.0001, 'miu': 0.0001, 'eta': [1, 1, 1, 0.005]}
    MAX = {'lambda': 5, 'miu': 50, 'theta': 50, 'alpha': 5, 'eta': [500, 500, 500, 0.995]}

    # Additional computations and settings can be added here as needed

    return r


def SKF(Y, Z, R, T, Q, A_0, P_0, c1, c2):
    """
    Kalman filter for stationary systems with time-varying system matrices and missing data.

    Args:
        Y (numpy.ndarray): Data (nobs x n).
        Z (numpy.ndarray): Matrix Z.
        R (numpy.ndarray): Matrix R.
        T (numpy.ndarray): Matrix T.
        Q (numpy.ndarray): Covariance matrix Q.
        A_0 (numpy.ndarray): Initial state vector.
        P_0 (numpy.ndarray): Initial covariance matrix.
        c1 (numpy.ndarray): Constant vector c1.
        c2 (numpy.ndarray): Constant vector c2.

    Returns:
        dict: Dictionary containing predicted and filtered state vectors and covariances.
    """
    # Output structure & dimensions
    n, m = Z.shape
    nobs = Y.shape[1]

    S = {'Am': np.nan * np.zeros((m, nobs)), 'Pm': np.nan * np.zeros((m, m, nobs)),
         'AmU': np.nan * np.zeros((m, nobs)), 'PmU': np.nan * np.zeros((m, m, nobs)),
         'ZF': [None] * nobs, 'V': [None] * nobs}

    Au = A_0  # A_0|0
    Pu = P_0  # P_0|0

    for t in range(nobs):
        # A = A_t|t-1 & P = P_t|t-1
        A = T.dot(Au) + c2
        P = T.dot(Pu).dot(T.T) + Q
        P = 0.5 * (P + P.T)

        # Handling the missing data
        y_t, Z_t, R_t, c1_t = MissData(Y[:, t], Z, R, c1)

        if y_t.size == 0:
            Au = A
            Pu = P
            ZF = np.zeros((m, 0))
            V = np.zeros((0, 1))
        else:
            PZ = P.dot(Z_t.T)
            F = (Z_t.dot(PZ) + R_t)
            ZF = Z_t.T.dot(np.linalg.inv(F))
            PZF = P.dot(ZF)

            V = y_t - Z_t.dot(A) - c1_t
            Au = A + PZF.dot(V)
            Pu = P - PZF.dot(PZ)
            Pu = 0.5 * (Pu + Pu.T)

        S['ZF'][t] = ZF
        S['Am'][:, t] = A
        S['Pm'][:, :, t] = P
        S['V'][t] = V

        S['AmU'][:, t] = Au
        S['PmU'][:, :, t] = Pu

    return S


def transform_data(spec, data_raw):
    """
    Transforms the raw data based on the specified transformation.

    Args:
        spec (dict): A dictionary containing a field called 'Transformation' that specifies
            the transformation to apply to each column of data in data_raw. The
            'Transformation' field should be a list of strings, where each
            string is either 'log' (indicating a logarithmic transformation) or
            'lin' (indicating a linear transformation).
        data_raw (numpy.ndarray): A matrix containing the raw data to be transformed.

    Returns:
        numpy.ndarray: A matrix containing the transformed data.
    """
    # Initialize DataTrans to be a matrix of the same size as DataRaw but with
    # all elements set to NaN. This creates a matrix that will be filled with
    # transformed data.
    data_trans = np.full(data_raw.shape, np.nan)

    # Loop through each transformation specified in trans.
    for i, transformation in enumerate(spec['Transformation']):
        # Check if the ith transformation is a logarithmic transformation.
        if transformation == 'log':
            # If it is logarithmic, transform the ith column of DataRaw using a
            # logarithmic transformation and store the result in the ith column
            # of DataTrans.
            data_trans[:, i] = 100 * np.log(data_raw[:, i])
        # Check if the ith transformation is a linear transformation.
        elif transformation == 'lin':
            # If it is linear, simply copy the ith column of DataRaw into the
            # ith column of DataTrans.
            data_trans[:, i] = data_raw[:, i]
        # If the ith transformation is neither logarithmic nor linear, generate
        # an error.
        else:
            raise ValueError('Enter valid transformation')

    return data_trans


def trimr(x, n1, n2):
    """
    Return a matrix (or vector) x stripped of the specified rows.

    Parameters:
        x (numpy.array): Input matrix (or vector) (n x k)
        n1 (int): First n1 rows to strip
        n2 (int): Last n2 rows to strip

    Returns:
        z (numpy.array): x with the first n1 and last n2 rows removed
    """
    n, _ = x.shape
    if (n1 + n2) >= n:
        raise ValueError("Attempting to trim too much in trimr")

    h1 = n1
    h2 = n - n2

    z = x[h1:h2, :]

    return z


def VARcf_DKcksV2(X, p, beta, Su, nDraws=0, LinComb=None):
    """
    Computes conditional forecasts for the missing observations in X using a
    VAR (Vector Autoregression) and Kalman filter and smoother.

    Args:
        X (numpy.ndarray): Matrix of observable variables.
        p (int): Number of lags in VAR.
        beta (numpy.ndarray): Coefficients of the VAR.
        Su (numpy.ndarray): Covariance matrix of the VAR.
        nDraws (int, optional): Number of draws. If == 0, then simple Kalman smoother is run,
                                otherwise nDraws draws of the states are done. Defaults to 0.
        LinComb (numpy.ndarray, optional): Linear combination matrix. Defaults to None.

    Returns:
        numpy.ndarray: Nans from X are replaced by the conditional forecasts.
    """

    N = Su.shape[1]
    T = X.shape[0]

    if LinComb is not None:
        q = LinComb.shape[1]
        CCadd = np.zeros((q, N * p))
        CCadd[:, :N] = LinComb.T
    else:
        q = 0
        CCadd = []

    idxNaN = np.any(np.isnan(X), axis=1)
    idxNaNcs = np.cumsum(idxNaN[::-1])
    nNaNs = np.sum(idxNaNcs == np.arange(1, T + 1))

    # end part with missing observations
    Xub = X[-nNaNs:, :]
    # balanced part
    X = X[:-nNaNs, :]
    Xinit = X[:, q:]

    # State space representation

    # transition equation
    AA = np.zeros((N * p, N * p))
    AA[:N, :N * p] = beta[:-1, :].T
    AA[N:N * p, :N * (p - 1)] = np.eye(N * (p - 1))
    c2 = np.concatenate([beta[-1, :].reshape(-1, 1), np.zeros((N * (p - 1), 1))], axis=0)

    # measurement equation
    CC = np.zeros((N, N * p))
    CC[:, :N] = np.eye(N)
    if len(CCadd) > 0:
        CC = np.vstack([CCadd, CC])
    QQ = np.zeros((N * p, N * p))
    QQ[:N, :N] = Su
    c1 = np.zeros((N + q, 1))

    # initialization of the Kalman filter
    initx = lagmatrix(Xinit, maxlag=p - 1)[-1, :].reshape(-1, 1)
    initV = np.eye(len(initx)) * 1e-7

    # Conditional forecasts
    yinput = Xub[1:, :]

    Tub = yinput.shape[0]

    if nDraws == 0:  # point forecast
        # Kalman filter and smoother
        xsmooth = runKF_DK(yinput.T, AA, CC, QQ, np.diag(np.ones(N + q) * 1e-12), initx, initV, c1, c2)
        Xcond = np.vstack([Xinit, xsmooth[:N, :].T]) @ CC[:, :N].T
    else:
        Xcond = np.nan * np.ones((T, N + q, nDraws))
        # Durbin and Koopman simulation smoother
        for kg in range(nDraws):
            aplus = np.nan * np.ones((N * p, Tub))
            yplus = np.nan * np.ones((N + q, Tub))
            for t in range(Tub):
                aplus[:, t:t + 1] = AA @ initx + np.concatenate([mvnrnd(mean=np.zeros(N), cov=Su).reshape(-1, 1),
                                                                 np.zeros((N * (p - 1), 1))], axis=0) + c2
                initx = aplus[:, t:t + 1]
                yplus[:, t:t + 1] = CC @ aplus[:, t:t + 1] + c1
            ystar = yinput.T - yplus
            ahatstar = runKF_DK(ystar, AA, CC, QQ, np.diag(np.ones(N + q) * 1e-12), np.zeros(initx.shape), initV,
                                np.zeros((N, 1)), np.zeros(initx.shape))
            atilda = ahatstar + aplus
            Xcond[:, :, kg] = np.vstack([Xinit, atilda[:N, :].T]) @ CC[:, :N].T

    return Xcond


import numpy as np
from scipy.stats import multivariate_normal as mvnrnd


def VARcf_DKcksV3(X, p, beta, Su, nDraws=0, LinCombLong=None):
    """
    Computes conditional forecasts for the missing observations in X using a
    VAR and Kalman filter and smoother.

    Args:
        X (numpy.ndarray): Matrix of observable variables.
        p (int): Number of lags in VAR.
        beta (numpy.ndarray): Coefficients of the VAR.
        Su (numpy.ndarray): Covariance matrix of the VAR.
        nDraws (int, optional): Number of draws. If == 0, then simple Kalman smoother is run,
                                otherwise nDraws draws of the states are done. Defaults to 0.
        LinCombLong (numpy.ndarray, optional): Linear combination matrix. Defaults to None.

    Returns:
        numpy.ndarray: Nans from X are replaced by the conditional forecasts.
    """

    N = Su.shape[1]
    T = X.shape[0]

    if LinCombLong is not None:
        q = LinCombLong.shape[1]
        CCadd = LinCombLong.T
    else:
        q = 0
        CCadd = []

    idxNaN = np.any(np.isnan(X), axis=1)
    idxNaNcs = np.cumsum(idxNaN[::-1])
    nNaNs = np.sum(idxNaNcs == np.arange(1, T + 1))

    # end part with missing observations
    Xub = X[-nNaNs:, :]
    # balanced part
    X = X[:-nNaNs, :]
    Xinit = X[:, q:]

    # State space representation

    # transition equation
    AA = np.zeros((N * p, N * p))
    AA[:N, :N * p] = beta[:-1, :].T
    AA[N:N * p, :N * (p - 1)] = np.eye(N * (p - 1))
    c2 = np.concatenate([beta[-1, :].reshape(-1, 1), np.zeros((N * (p - 1), 1))], axis=0)

    # measurement equation
    CC = np.zeros((N, N * p))
    CC[:, :N] = np.eye(N)
    if len(CCadd) > 0:
        CC = np.vstack([CCadd, CC])
    QQ = np.zeros((N * p, N * p))
    QQ[:N, :N] = Su
    c1 = np.zeros((N + q, 1))

    # initialization of the Kalman filter
    initx = lagmatrix(Xinit, range(p))
    initx = initx[-1, :].reshape(-1, 1)
    initV = np.eye(len(initx)) * 1e-7

    # Conditional forecasts
    yinput = Xub[1:, :]

    Tub = yinput.shape[0]

    if nDraws == 0:  # point forecast
        # Kalman filter and smoother
        xsmooth = runKF_DK(yinput.T, AA, CC, QQ, np.diag(np.ones(N + q) * 1e-12), initx, initV, c1, c2)
        Xcond = np.vstack([Xinit, xsmooth[:N, :].T]) @ CC[q:, :N].T
    else:
        Xcond = np.nan * np.ones((T, N, nDraws))
        # Durbin and Koopman simulation smoother
        for kg in range(nDraws):
            aplus = np.nan * np.ones((N * p, Tub))
            yplus = np.nan * np.ones((N + q, Tub))
            for t in range(Tub):
                aplus[:, t:t + 1] = AA @ initx + np.concatenate([mvnrnd(mean=np.zeros(N), cov=Su).reshape(-1, 1),
                                                                 np.zeros((N * (p - 1), 1))], axis=0) + c2
                initx = aplus[:, t:t + 1]
                yplus[:, t:t + 1] = CC @ aplus[:, t] + c1
            ystar = yinput.T - yplus
            ahatstar = runKF_DK(ystar, AA, CC, QQ,
                                np.diag(np.ones(N + q) * 1e-12), np.zeros(initx.shape), initV, np.zeros((N + q, 1)),
                                np.zeros(initx.shape))
            atilda = ahatstar + aplus
            Xcond[:, :, kg] = np.vstack([Xinit, atilda[:N, :].T]) @ CC[q:, :N].T

    return Xcond


# Escaping the curly braces for the style formatting

def write_tex_sidewaystable(fid, header, style, table_body, above_tabular, below_tabular=None):
    """
        Writes a LaTeX sidewaystable to the given file identifier or prints it to the screen.

        Args:
            fid (file object): File identifier obtained by opening the file in write mode. Can be None to print to the screen.
            header (list of str): List of column titles/headers for the table.
            style (str): LaTeX style for the columns, such as 'r|cccc' or 'l|rr|rr|'.
            table_body (list of list): Nested list or cell matrix of table content to write. Can include text, numbers, NaN, or empty values.
            above_tabular (str or list of str): Text to write between '\\begin{table}' and '\\begin{tabular}{style}'. Can be a string or a list of strings.
            below_tabular (str or list of str, optional): Text to write between '\\end{tabular}' and '\\end{table}'. Can be a string or a list of strings.

        Example:
            with open('file.tex', 'w') as fid:
                row_names = ['row1', 'row2', 'row3']
                data = [[1.23, 4.56, 7.89], [0.12, 3.45, 6.78], [9.01, 2.34, 5.67]]
                header = ['col1', 'col2', 'col3']
                table_data = [row_names, data]
                style = 'r|ccc'
                above_tabular = 'Random Numbers'
                write_tex_sidewaystable(fid, header, style, table_data, above_tabular)
        """

    def fmt(x):
        if isinstance(x, (int, float)):
            if abs(x) < 1:
                return f"{x:.3f}"
            elif abs(x) < 10:
                return f"{x:.2f}"
            else:
                return f"{x:.1f}"
        elif x == "":
            return ""
        else:
            return str(x).replace('&', '\\&')

    writeline = lambda s: print(s, file=fid)
    writerow = lambda rowcell: writeline(' & '.join(rowcell) + ' \\\\')
    write_cell = lambda to_write: [writeline(fmt(txt)) for txt in to_write]
    write_string_cell = lambda to_write: write_cell([to_write]) if isinstance(to_write, str) else write_cell(to_write)

    writeline('\\begin{sidewaystable}[htpb!]')
    if above_tabular:
        write_string_cell(above_tabular)
    writeline('\\centering')
    writeline(f'\\begin{{tabular}}{{{style}}}')  # Escaping the curly braces

    writerow(header)
    for row in table_body:
        writerow([fmt(cell) for cell in row])

    writeline('\\hline')
    writeline('\\end{tabular}')
    if below_tabular:
        write_string_cell(below_tabular)
    writeline('\\end{sidewaystable}')
    writeline('')
