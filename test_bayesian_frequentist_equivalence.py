from pytest import approx
from scipy import stats

import functions

uniform = lambda x: 1  # a constant PDF equal to 1 corresponds to the uniform prior
jeffreys = stats.beta(a=1/2, b=1/2).pdf  # Jeffreys prior PDF is a beta(1/2,1/2). See https://en.wikipedia.org/wiki/Jeffreys_prior#Bernoulli_trial


# Verify that Bayes with uniform prior corresponds to Laplace
assert approx(
	functions.forecast_bayes(failures=5, prior=uniform, forecast_years=3)) == \
	   functions.forecast_generalized_laplace(failures=5, forecast_years=3)

# Verify that Bayes with Jeffrey's prior corresponds to modified laplace with 0.5 virtual successes and failures
assert approx(
	functions.forecast_bayes(failures=5, prior=jeffreys, forecast_years=3)) == \
	   functions.forecast_generalized_laplace(failures=5, forecast_years=3, virtual_failures=0.5, virtual_successes=0.5)

# Verify the more general proposition, from Appendix 2.2, that virtual_succeses == alpha, etc.
# (Appendix 2.2: "The parameterisation of Beta distributions used in the semi-informative priors framework").
alpha = 0.5
beta = 1.23
beta_distribution = stats.beta(a=alpha,b=beta).pdf

assert approx(
	functions.forecast_bayes(failures=5, prior=beta_distribution, forecast_years=3)) == \
	   functions.forecast_generalized_laplace(failures=5, forecast_years=3, virtual_successes=alpha, virtual_failures=beta)