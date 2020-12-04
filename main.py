from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from pytest import approx

uniform = lambda x:1 # a constant pdf equal to 1 corresponds to the uniform prior

def laplace(trials,failures):
	successes = trials - failures
	virtual_sucesses = 1
	virtual_failures = 1
	next_trial_p = (virtual_sucesses+successes)/(trials+virtual_sucesses+virtual_failures)
	return next_trial_p

def forecast_laplace(failures=2020-1956,forecast_years=16):
	p_failure_by_target = 1
	for i in range(forecast_years):
		p_failure = 1-laplace(failures,failures)
		p_failure_by_target = p_failure_by_target*p_failure
		failures += 1
	p_success_by_target = 1-p_failure_by_target
	return p_success_by_target

def _noAIupdate_bayes_single(p_density):
	p_density_unnormalized = lambda x: p_density(x)*(1-x)
	normalization_constant = integrate.quad(p_density_unnormalized,0,1)[0]
	return lambda x: p_density_unnormalized(x)/normalization_constant

def noAIupdate_bayes(failures,prior):
	p_density = prior
	plot_domain = np.linspace(0, 1, 50)
	for i in range(failures):
		p_density = _noAIupdate_bayes_single(p_density)
		# plt.plot(plot_domain,p_density(plot_domain))
		# plt.show()
	return p_density

def forecast_bayes(failures,prior,forecast_years):
	p_failure_by_target = 1
	for i in range(forecast_years):
		density = noAIupdate_bayes(failures,prior)
		p_failure = 1-integrate.quad(lambda x: density(x) * x, 0, 1)[0]
		p_failure_by_target = p_failure_by_target*p_failure
		failures += 1
	p_success_by_target = 1-p_failure_by_target
	return p_success_by_target

assert approx(forecast_bayes(failures=64,prior=uniform,forecast_years=16)) == forecast_laplace(failures=64,forecast_years=16)