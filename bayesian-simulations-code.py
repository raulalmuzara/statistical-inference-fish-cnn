import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

#Choose parameters
obs= #observed (TD)
fish= #mu
wrong= #lambda
prob= #p

#Define Bayesian model
with pm.Model() as model:
    A = pm.Poisson("A", fish)
    TD = pm.Normal("TD",
                   mu = A*prob+wrong,
                   sigma = np.sqrt(A*prob*(1-prob)+wrong),
                   observed = obs)

#Sample
with model:
    trace = pm.sample(100000)

#Display results
pm.plot_posterior(trace[5000:])
plt.show()

pm.summary(trace[5000:])