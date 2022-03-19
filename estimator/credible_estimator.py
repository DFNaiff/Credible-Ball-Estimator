#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    A simple estimator for credible balls, for distributions
    where sampling is readily available. Uses stochastic bisection
    algorithm described in
    "Probabilistic bisection converges almost 
     as quickly as stochastic approximation,
     Peter I. Frazier, Shane G. Henderson, Rolf Waeber"
    Example usage can be found in the notebook
"""
import numpy as np

from . import stochastic_bisection


def credible_ball_estimator(sampler, p, x0,
                            nsamples=1000,
                            delta_step=1.0,
                            maxiter_warming=100,
                            maxiter=1000,
                            miniter=10,
                            tol=1e-2,
                            logify=False,
                            maxdrift=1000,
                            gamma=0.9,
                            alpha=0.0,
                            verbose=0):
    """

    Parameters
    ----------
    sampler : Callable[np.ndarray, np.ndarray]
        Sampler accepting a 2d array of size (nsamples, D).
    p : float
        The credibility such that P(||X|| < r) = p.
    x0 : np.ndarray
        Ball center.
    nsamples : int, optional
        Number of samples. The default is 1000.
    delta_step : float, optional
        Step size for warming. The default is 1.0.
    maxiter_warming : int, optional
        Maximum warming. The default is 100.
    maxiter : int, optional
        Maximum number of iterations. The default is 1000.
    miniter : int, optional
        Minimum number of iterations before checking tolerance. The default is 10.
    tol : float, optional
        Error tolerance. The default is 1e-2.
    logify : boo, optional
        Whether to try to increase stability by finding the 
        ball by finding the associated root of the log.
        Somewhat risky.
    maxdrift : int, optional
        Maximum number of iterations for each drift test. The default is 1000.
    gamma : float, optional
        Gamma factor for drift test. The default is 0.9.
    alpha : TYPE, optional
        Memory for tolerance. The default is 0.0.
    verbose : TYPE, optional
        Logging frequency. The default is 0.

    Raises
    ------
    ValueError
        If ball is greater than maxiter_warming*delta_step.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    k = delta_step
    warm_steps = 1
    print("Beginning warming...")
    while True:
        if warm_steps > maxiter_warming:
            raise ValueError("Too many warm steps. Increase delta step")
        p_test = np.mean((sampler(nsamples)**2).sum(axis=-1) < k**2)
        if p_test > 0.999:
            break
        else:
            k += delta_step
            warm_steps += 1
    print("k = %f" % k)
    if not logify:
        def f(r):
            y0 = np.mean((sampler(nsamples)**2).sum(axis=-1) < r**2)/p
            return 1 - y0
    else:
        def f(r):
            y0 = np.mean((sampler(nsamples)**2).sum(axis=-1) < r**2)/p
            if y0 == 0.0:
                raise RuntimeError
            return -np.log(y0)
    print("Beginning calculation...")
    r = stochastic_bisection.stochastic_bisection(f,
                                                  gamma=gamma,
                                                  maxiter=maxiter,
                                                  maxdrift=maxdrift,
                                                  ubx=k,
                                                  tol=tol,
                                                  alpha=alpha,
                                                  verbose=verbose)
    return r
