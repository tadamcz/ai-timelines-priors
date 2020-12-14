from scipy import integrate
import numpy as np

def generalized_laplace(trials, failures, virtual_successes, virtual_failures=None, ftp=None):
	if ftp is not None:
		if virtual_failures is not None and virtual_successes is not None:
			raise ValueError("Provide exactly two of virtual_failures, virtual_successes, and ftp")
		if virtual_failures is None:
			virtual_failures = (virtual_successes/ftp)-virtual_successes

	successes = trials - failures
	next_trial_p = (virtual_successes + successes) / (trials + virtual_successes + virtual_failures)
	return next_trial_p


def forecast_generalized_laplace(failures, forecast_years, virtual_successes, virtual_failures=None, ftp=None):
	kwargs = {}
	if virtual_failures is not None:
		kwargs['virtual_failures'] = virtual_failures
	if ftp is not None:
		kwargs['ftp'] = ftp
	p_failure_by_target = 1
	for i in range(forecast_years):
		p_failure = 1 - generalized_laplace(failures, failures, virtual_successes, **kwargs)
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

def fourParamFramework(ftp,regime_start=1956,forecast_from=2020,forecast_to=2036,virtual_successes=1):
	virtual_failures = (virtual_successes/ftp)-virtual_successes
	failures = forecast_from-regime_start
	return forecast_generalized_laplace(failures=failures,forecast_years=forecast_to-forecast_from,virtual_successes=virtual_successes,virtual_failures=virtual_failures)
