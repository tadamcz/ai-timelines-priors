from scipy import integrate, optimize
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

def fourParamFrameworkCalendar(ftp, regime_start=1956, forecast_from=2020, forecast_to=2036, virtual_successes=1):
	virtual_failures = (virtual_successes/ftp)-virtual_successes
	failures = forecast_from-regime_start
	return forecast_generalized_laplace(failures=failures,forecast_years=forecast_to-forecast_from,virtual_successes=virtual_successes,virtual_failures=virtual_failures)

def solveFor_ftp_res(g_exp=4.3/100,ftp_cal=1/300):
	'''
	To determine ftp_res, solve for ftp_res:
	PrAgi_ResModel(ftp_res,g_exp) = PrAgi_CalModel(ftp_cal)
	Where g_exp and ftp_cal are constants
	'''
	PrAgi_ResModel = lambda ftp_res: fourParamFrameworkResearcher(g_act=g_exp,ftp_res=ftp_res)
	PrAgi_CalModel = fourParamFrameworkCalendar(ftp_cal)

	# we solve f_to_solve=0
	f_to_solve = lambda ftp_res: PrAgi_ResModel(ftp_res)-PrAgi_CalModel

	bound = 1e-9
	sol_leftbound,sol_rightbound = bound,1-bound
	ftp_res_solution = optimize.brentq(f_to_solve,sol_leftbound,sol_rightbound)

	return ftp_res_solution

def fourParamFrameworkResearcher(g_act, ftp_res=None, ftp_cal_equiv=None, g_exp=None, regime_start=1956, forecast_from=2020, forecast_to=2036, virtual_successes=1):
	if ftp_cal_equiv is not None and g_exp is not None:
		method = 'indirect'
	if ftp_res is not None:
		method = 'direct'

	if ftp_res is not None and (ftp_cal_equiv is not None or g_exp is not None):
		raise ValueError("Supply either (ftp_cal_equiv and g_exp) or supply ftp_res")

	if method == 'indirect':
		ftp_res = solveFor_ftp_res(g_exp,ftp_cal_equiv)

	n_trials_per_year = g_act*100 # because 1 trial = 1% increase

	failures = int((forecast_from - regime_start)*n_trials_per_year)
	n_trials_forecast = int((forecast_to - forecast_from)*n_trials_per_year)

	return forecast_generalized_laplace(failures=failures,
										   forecast_years=n_trials_forecast,
										   virtual_successes=virtual_successes,
										   ftp=ftp_res)