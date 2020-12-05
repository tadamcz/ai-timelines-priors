from scipy import integrate, stats
import numpy as np
import matplotlib.pyplot as plt
from pytest import approx

uniform = lambda x: 1  # a constant PDF equal to 1 corresponds to the uniform prior
jeffreys = stats.beta(a=1/2, b=1/2).pdf  # Jeffreys prior PDF is a beta(1/2,1/2). See https://en.wikipedia.org/wiki/Jeffreys_prior#Bernoulli_trial

def generalized_laplace(trials, failures, virtual_successes=1, virtual_failures=1):
	successes = trials - failures
	next_trial_p = (virtual_successes + successes) / (trials + virtual_successes + virtual_failures)
	return next_trial_p


def forecast_generalized_laplace(failures, forecast_years, virtual_successes=1, virtual_failures=1):
	p_failure_by_target = 1
	for i in range(forecast_years):
		p_failure = 1 - generalized_laplace(failures, failures, virtual_successes, virtual_failures)
		p_failure_by_target = p_failure_by_target * p_failure
		failures += 1
	p_success_by_target = 1 - p_failure_by_target
	return p_success_by_target


def _noAIupdate_bayes_single(p_density):
	p_density_unnormalized = lambda x: p_density(x) * (1 - x)
	normalization_constant = integrate.quad(p_density_unnormalized, 0, 1)[0]
	return lambda x: p_density_unnormalized(x) / normalization_constant


def noAIupdate_bayes(failures, prior):
	p_density = prior
	plot_domain = np.linspace(0, 1, 50)
	for i in range(failures):
		p_density = _noAIupdate_bayes_single(p_density)
	# plt.plot(plot_domain,p_density(plot_domain))
	# plt.show()
	return p_density


def forecast_bayes(failures, prior, forecast_years):
	p_failure_by_target = 1
	for i in range(forecast_years):
		density = noAIupdate_bayes(failures, prior)
		p_failure = 1 - integrate.quad(lambda x: density(x) * x, 0, 1)[0]
		p_failure_by_target = p_failure_by_target * p_failure
		failures += 1
	p_success_by_target = 1 - p_failure_by_target
	return p_success_by_target

# Verify that Bayes with uniform prior corresponds to Laplace
assert approx(
	forecast_bayes(failures=5, prior=uniform, forecast_years=3)) == \
	forecast_generalized_laplace(failures=5, forecast_years=3)

# Verify that Bayes with Jeffrey's prior corresponds to modified laplace with 0.5 virtual successes and failures
assert approx(
	forecast_bayes(failures=5, prior=jeffreys, forecast_years=3)) == \
	forecast_generalized_laplace(failures=5, forecast_years=3, virtual_failures=0.5, virtual_successes=0.5)

# Verify the more general proposition, from Appendix 2.2, that virtual_succeses == alpha, etc.
# (Appendix 2.2: "The parameterisation of Beta distributions used in the semi-informative priors framework").
alpha = 0.5
beta = 1.23
beta_distribution = stats.beta(a=alpha,b=beta).pdf

assert approx(
	forecast_bayes(failures=5,prior=beta_distribution,forecast_years=3)) == \
	forecast_generalized_laplace(failures=5,forecast_years=3,virtual_successes=alpha,virtual_failures=beta)
