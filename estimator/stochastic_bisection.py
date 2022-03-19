import numpy as np
"""
    Stochastic bisection algorithm, 
    as described in
    "Probabilistic bisection converges almost 
     as quickly as stochastic approximation,
     Peter I. Frazier, Shane G. Henderson, Rolf Waeber"
"""


def stochastic_bisection(measure,
                         gamma=0.9,
                         maxiter=100,
                         miniter=10,
                         maxdrift=500,
                         tol=1e-3,
                         alpha=0.0,
                         verbose=0,
                         lbx=0.0,
                         ubx=1.0,
                         increasing=False,
                         return_stats=False):
    """

    Parameters
    ----------
    measure : Callable[float, float]
        The 1-D stochastic function to seek the root.
    gamma : float, optional
        Gamma factor for drift test. The default is 0.9.
    maxiter : int, optional
        Maximum number of iterations. The default is 100.
    miniter : int, optional
        Minimum number of iterations before checking tolerance. The default is 10.
    maxdrift : int, optional
        Maximum number of iterations for each drift test. The default is 500.
    tol : float, optional
        Error tolerance. The default is 1e-3.
    alpha : float, optional
        Memory for tolerance. The default is 0.0.
    verbose : int, optional
        Logging frequency. The default is 0.
    lbx : float, optional
        Lower bound of function. The default is 0.0.
    ubx : float, optional
        Upper bound of function. The default is 1.0.
    increasing : bool, optional
        Whether it is an increasing or decreasing function. The default is False.
    return_stats : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    float or (float, return_stats)
        The estimated root and (if return_stats = True) the search statistics.

    """
    p0 = 1.0 - gamma/2
    points = [0.0, 1.0]
    values = [0.0, 1.0]
    x_m = 0.5 #Query point
    x_r0 = x_m #Running mean of queries
    running_alpha = (1-alpha) #Forgetting for running mean
    stats = dict()
    if verbose == 0: #There won't be any verbose
        verbose = maxiter+1
    for n in range(maxiter):
        sign_func = _signify(measure, x_m, lbx, ubx, increasing) #Noisy sign function
        z_m = _drift_test(sign_func, gamma, maxdrift)
        if z_m == -1:
            p_update = p0
        elif z_m == 1:
            p_update = 1-p0
        else:
            continue
        points, values = _update_cdf(x_m, p_update, points, values)
        x_m = _get_median(points, values)
        x_r = x_r0 + running_alpha*(x_m-x_r0)
        if np.abs(x_r-x_r0) <= tol:
            break
        else:
            x_r0 = x_r
        if (n+1) % verbose == 0:
            print(x_r, x_m)
    x = (1-x_r)*lbx + x_r*ubx
    if verbose != maxiter + 1:
        print("Finished")
    if return_stats:
        stats['points'] = points
        stats['values'] = values
        return x, stats
    else:
        return x


def _signify(measure, x, lbx, ubx, increasing):
    def sign_func():
        scale = 1.0 if not increasing else -1.0;
        return scale*np.sign(measure((1-x)*lbx + x*ubx))
    return sign_func


def _drift_test(sign_func, gamma, maxiter):
    s0 = 0.0
    m = 1
    while True:
        k = np.sqrt(2*m*np.log(m+1)-np.log(gamma))
        s0 += sign_func()
        if s0 >= k:
            return 1
        elif s0 <= -k:
            return -1
        elif m >= maxiter:
            return 0
        m += 1


def _location_ordered_list(x, L):
    # L : [p0,p1,p2,...,pN], where it's ordered
    # returns : i, where i is the index where p[i-1] <= x < p[i]
    #          if empty list, return 0.
    #          if x < p[0], return 0. x >= p[-1], return len(p)
    N = len(L)
    if N == 0:
        return 0
    if N == 1:
        return 1 if x >= L[0] else 0
    if x >= L[-1]:
        return N
    elif x < L[0]:
        return 0
    else:
        i = (N-1)//2
        if L[i+1] <= x:
            return i+1 + _location_ordered_list(x, L[i+1:])
        else:
            if L[i] <= x:
                return i+1
            else:
                return _location_ordered_list(x, L[:i+1])


def _update_cdf(x, p, points, values):
    # x \in (0,1). F_n(x) = 1/2
    # p \in (0,1)
    # points : [x0=0,x1,x2,...,xN=1]
    # values : [y0=0,y1,y2,...,yN=1]
    # Those represent the CDF of an uniform by parts.
    # Let the PDF of if be f_N
    # returns: 
    # CDF of the distribution with density
    # f_{N+1}(y) = p*f_N(y) if y < x else (1-p)*f_N(y)
    q = 1-p
    p_ = 2*p
    q_ = 2*q
    ind = _location_ordered_list(x, points)
    points_low = points[:ind]
    points_high = points[ind:]
    values_low = values[:ind]
    values_high = values[ind:]
    values_low = list(p_*np.array(values_low))
    values_high = list((p_-q_)/2+q_*np.array(values_high))
    points_new = points_low + [x] + points_high
    values_new = values_low + [p_/2] + values_high
    return points_new, values_new


def _get_median(points, values):
    # get the median of a CDF defined by points and values
    # points : [x0=0,x1,x2,...,xN=1]
    # values : [y0=0,y1,y2,...,yN=1]
    y_median = 0.5
    i = _location_ordered_list(y_median, values)
    xlow, xhigh, ylow, yhigh = points[i-1], points[i], values[i-1], values[i]
    x_median = ((xhigh-xlow)*y_median-(ylow*xhigh-yhigh*xlow))/(yhigh-ylow)
    return x_median