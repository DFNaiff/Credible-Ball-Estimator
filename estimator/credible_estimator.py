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


def credible_ball_estimator(sampler, p, x0, N=1000, delta_step=1.0, maxiter_warming=100,
                            maxiter=1000, tol=1e-2, maxdrift=1000, gamma=0.9, alpha=0.0,
                            verbose=0):
    """
        sampler(N) : returns (N,D) samples
        p : probability that we want to P(||X - x0|| < r) equals to. Must be less than 0.99
        x0 : x0 in the above descriptio
        N : number of samples per step
        delta_step: step increase in warming
        maxiter_warming: maximum iterations in warming
        gamma : gamma factor for drift test, as described in the article
        maxiter : maximum number of iterations of algorithm
        tol : tolerance (NOT IMPLEMENTED YET)
        maxdrift : maximum number of iterations for each drift test
        verbose : frequency of printings of x_m
    """
    # Pre warming
    k = delta_step
    warm_steps = 1
    print("Beginning warming...")
    while True:
        if warm_steps > maxiter_warming:
            raise ValueError("Too many warm steps. Increase delta step")
        p_test = np.mean((sampler(N)**2).sum(axis=-1) < k**2)
        if p_test > 0.999:
            break
        else:
            k += delta_step
            warm_steps += 1
    print("k = %f" % k)

    def f(r):
        return p - np.mean((sampler(N)**2).sum(axis=-1) < r**2)
    print("Beginning calculation...")
    r = stochastic_bisection.stochastic_bisection(f,
                                                  gamma=gamma,
                                                  maxiter=maxiter,
                                                  maxdrift=maxdrift,
                                                  ubx=k,
                                                  tol=tol,
                                                  alpha=alpha,
                                                  verbose=verbose)
    return r, f
