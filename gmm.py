import numpy as np
from scipy.stats import norm
from random import gauss

# Class to represent a Gaussian mixture model.
# Instances are callable (they can be treated as a probability density function),
# and indexable (square brackets retrieve a component Gaussian)
class MixedGaussian :
    def __init__(self, pi, mu, sigma):
        self.k = len(pi)
        assert abs(1.0-sum(pi)) < .001
        assert self.k == len(mu)
        assert self.k == len(sigma)
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
   
    def __getitem__(self, j):
        return lambda x : self.pi[j] * norm.pdf(x, self.mu[j], self.sigma[j])
   
    def __call__(self, x):
        return  sum(self[j](x) for j in range(self.k))
   
    def __str__(self):
        return " + ".join([str(self.pi[j]) + "N(" + str(self.mu[j]) + "," + str(self.sigma[j])+")" for j in range(self.k)])

# Function to produce a function that acts as a Gaussian mixture model.
# This is simpler than the class above and can be used just as well
# as a callable function, but lacks indexing or the facility to
# retrieve the parameters
def mixed_gaussian(pi, mu, sigma):
    k = len(pi)
    assert k == len(mu)
    assert k == len(sigma)
    return lambda x : sum([pi[i] * norm.pdf(x, mu[i], sigma[i]) for i in range(k)])

# Compute the log likelihood of the data for a current model
def log_likelihood(p, X):
    return sum([np.log(p(x_i)) for x_i in X])

# Compute the average log likelihood, included just because I like
# saying "average log likelihood".
def ave_log_like(p, X):
    return log_likelihood(p, X)/len(X)

DEBUG_LOTS = False
DEBUG_SOME = False

# Train a Gaussian mixture model on the given data
# X -- An array of observations
# k -- The number of components
# pi_initial -- An array of initial settings for the weights (optional)
# mu_initial -- An array of initial settings for the means (optional)
# sigma_initial -- An array of initial settings for the st deviations (optional)
# epsilon -- The tolerance (optional)
def train(X, k, pi_initial=None, mu_initial=None, sigma_initial = None, epsilon = 0.0001):

    N = len(X)

   
    # --- Initial parameters ---
    # pi
    if pi_initial :  # If pi_initial is given, use it        
        assert len(pi_initial) == k
        assert abs(1-sum(pi_initial)) < epsilon
        pi = pi_initial
    else:   # Otherwise, default to all weights initially being equal
        pi = [1/k for i in range(k)]

    # mu
    if mu_initial :  # if mu_initial is given, use it
        assert len(mu_initial) == k
        mu = mu_initial
    else :  # Otherwise, randomly generate mu
        mu = np.array([min(X) + (max(X)-min(X))/(k + 1) * (i + 1) for i in range(k)]) 
        #Maybe I want to change this to random at some point.

    # sigma
    if sigma_initial:  # if sigma_initial is given, use it
        assert len(sigma_initial) == k
        sigma = sigma_initial
    else :
        sigma = [(max(X) - min(X)) / (k ** 2) for i in range(k)]

    # Initial gaussian mixture and log likelihood
    mg = MixedGaussian(pi, mu, sigma)
    if DEBUG_LOTS :
        print(mg)
    loglike = log_likelihood(mg, X)
    prev_loglike = None
    if DEBUG_SOME :
        print("Initial loglike: " + str(loglike))
    iterations = 0

    # Repeat until good enough    
    while (not prev_loglike) or (abs(loglike - prev_loglike) > epsilon):
        # E step -- compute responsibilities
        r = np.array([[mg[i](X[n]) / sum([mg[j](X[n]) for j in range(k)]) for i in range(k)] for n in range(N)])
        # r is a list of lists, each of which contains the responsibilities for a single observation

        # M step -- compute new mu, sigma, and pi
        Ni = [sum([r[n][i] for n in range(N)]) for i in range(k)]
        mu = [sum([r[n][i] * X[n] for n in range(N)]) / Ni[i] for i in range(k)]
        sigma = [np.sqrt(sum([r[n][i] * ((X[n] - mu[i]) ** 2) for n in range(N)]) / Ni[i]) for i in range(k)]
        pi = [(Ni[i] / N) for i in range(k)]
     
        # new gaussian mixture
        mg = MixedGaussian(pi, mu, sigma)
        if DEBUG_LOTS :
            print(str(mg))

        # recalculated log likelihood
        prev_loglike = loglike
        loglike = log_likelihood(mg, X)
        iterations += 1
        if DEBUG_LOTS or (DEBUG_SOME and iterations // 10 == 0):
            print("loglike: " + str(loglike))
            print("change in loglike: " + str(abs(loglike - prev_loglike)))
         
    if DEBUG_SOME :
        print(str(iterations) + " iterations")
    return mg