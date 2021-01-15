from scipy import integrate, optimize
import numpy as np
from collections import OrderedDict

# Global variables
probability_solution_leftbound, probability_solution_rightbound = 1e-9, 1 - 1e-9
trial_increment = 1/100

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

	ftp_res_solution = optimize.brentq(f_to_solve,probability_solution_leftbound,probability_solution_rightbound)

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

	n_trials_per_year = g_act*(1/trial_increment)

	failures = int((forecast_from - regime_start)*n_trials_per_year)
	n_trials_forecast = int((forecast_to - forecast_from)*n_trials_per_year)

	return forecast_generalized_laplace(failures=failures,
										   forecast_years=n_trials_forecast,
										   virtual_successes=virtual_successes,
										   ftp=ftp_res)

def solveFor_ftp_comp(ftp_res,relative_impact_research_compute):
	res_trials = 1   # arbitrary
	comp_trials = relative_impact_research_compute*res_trials


	# Number of trials that occur in both models, before the `res_trials` of the research model are compared to `comp_trials` of
	# the computation model. This is arbitrary and does not affect the outcome
	n = 5

	PrAGI_ResModel = forecast_generalized_laplace(failures=n,virtual_successes=1,ftp=ftp_res, forecast_years=res_trials)
	PrAGI_CompModel = lambda ftp_comp: forecast_generalized_laplace(failures=n, virtual_successes=1, ftp=ftp_comp, forecast_years=comp_trials)

	# we solve f_to_solve=0
	f_to_solve = lambda ftp_comp: PrAGI_CompModel(ftp_comp) - PrAGI_ResModel

	ftp_comp_solution = optimize.brentq(f_to_solve, probability_solution_leftbound, probability_solution_rightbound)

	return ftp_comp_solution

def solveFor_ftp_comp_indirect(ftp_cal,relative_impact_research_compute,g_exp):
	ftp_res = solveFor_ftp_res(g_exp=g_exp,ftp_cal=ftp_cal)
	return solveFor_ftp_comp(ftp_res, relative_impact_research_compute)

def NumberIncreaseQuantity(start,end,increment=trial_increment):
	return int(np.log(end/start)/np.log(1+increment))

# Dictionaries copied directly from Tom's code without checking
compute_prices = {
	1800: 10**-6,
	1956: 10**-6,
	1970: 10**-7,
	2000: 10**-12,
	2008: 10**-16,
	2020: 10**-17,
	2021: 10**-17.2,
	2022: 10**-17.4,
	2023: 10**-17.6,
	2024: 10**-17.8,
	2025: 10**-18,
	2026: 10**-18.1,
	2027: 10**-18.2,
	2028: 10**-18.3,
	2029: 10**-18.4,
	2030: 10**-18.5,
	2031: 10**-18.6,
	2032: 10**-18.7,
	2033: 10**-18.8,
	2034: 10**-18.9,
	2035: 10**-18.95,
	2036: 10**-19,
}
biggest_spends_conservative = {
	1956: 10**1,
	2020: 10**6.7,
	2021: 10**6.78,
	2022: 10**6.86,
	2023: 10**6.94,
	2024: 10**7.02,
	2025: 10**7.1,
	2026: 10**7.18,
	2027: 10**7.26,
	2028: 10**7.35,
	2029: 10**7.43,
	2030: 10**7.51,
	2031: 10**7.59,
	2032: 10**7.67,
	2033: 10**7.75,
	2034: 10**7.83,
	2035: 10**7.91,
	2036: 10**8,
}
biggest_spends_aggressive = {
	1956: 10**1,
	2020: 10**6.7,
	2021: 10**7.0,
	2022: 10**7.25,
	2023: 10**7.5,
	2024: 10**7.8,
	2025: 10**8.05,
	2026: 10**8.3,
	2027: 10**8.6,
	2028: 10**8.85,
	2029: 10**9.1,
	2030: 10**9.4,
	2031: 10**9.65,
	2032: 10**9.9,
	2033: 10**10.2,
	2034: 10**10.45,
	2035: 10**10.7,
	2036: 10**11,
}

def fourParamFrameworkComp(relative_impact_research_compute=None,
						   forecast_to_year=2036,
						   regime_start_year=1956,
						   biggest_spends_method=None,  # 'aggressive' or 'conservative'
						   computation_at_regime_start=None,
						   computation_at_forecasted_time=None,
						   ftp_cal_equiv=None,
						   virtual_successes=1,
						   ftp_comp=None):
	'''
	The most indirect method, from biggest_spend_end + ftp_cal_equiv, g_exp, and relative_impact_research_compute.
	The most direct method: from biggest_spend_end + ftp_comp.
	'''

	if ftp_comp is None:
		ftp_comp = solveFor_ftp_comp_indirect(ftp_cal_equiv,relative_impact_research_compute,g_exp=4.3/100)


	computation2020 = getComputationAmountForYear(2020, biggest_spends_method)

	if computation_at_regime_start is None and computation_at_forecasted_time is None:
		computation_at_regime_start = getComputationAmountForYear(regime_start_year, biggest_spends_method)
		computation_at_forecasted_time = getComputationAmountForYear(forecast_to_year, biggest_spends_method)

	n_failures_regime_start_to_now = NumberIncreaseQuantity(computation_at_regime_start,computation2020)
	n_trials_forecast = NumberIncreaseQuantity(computation2020, computation_at_forecasted_time)


	return forecast_generalized_laplace(failures=n_failures_regime_start_to_now,
										forecast_years=n_trials_forecast,
										virtual_successes=virtual_successes,
										ftp=ftp_comp)


def getYearForComputationAmount(c,biggest_spends_method):
	if biggest_spends_method == 'aggressive':
		biggest_spends = biggest_spends_aggressive
	if biggest_spends_method == 'conservative':
		biggest_spends = biggest_spends_conservative

	computation_to_year = OrderedDict()
	year_to_computation = OrderedDict()
	for year,price in compute_prices.items():
		try:
			computation_to_year[biggest_spends[year]/price] = year
			year_to_computation[year] = biggest_spends[year]/price
		except KeyError:
			pass

	# otherwise, return the first year the computation was greater than c
	for computation,year in computation_to_year.items():
		if computation>c:
			return year

def getComputationAmountForYear(y, biggest_spends_method):
	if biggest_spends_method == 'aggressive':
		biggest_spends = biggest_spends_aggressive
	if biggest_spends_method == 'conservative':
		biggest_spends = biggest_spends_conservative

	year_to_computation = OrderedDict()
	for year, price in compute_prices.items():
		try:
			year_to_computation[year] = biggest_spends[year] / price
		except KeyError:
			pass

	return year_to_computation[y]


def evolutionaryAnchor(biggest_spends_method,virtual_successes=1):

	c_initial = 1e21
	c_evolution = 1e41

	n_trials_initial_to_evolution = NumberIncreaseQuantity(c_initial,c_evolution)

	PrAGI_CompModel = lambda ftp_comp: forecast_generalized_laplace(failures=0,
																	virtual_successes=virtual_successes,
																	ftp=ftp_comp,
																	forecast_years=n_trials_initial_to_evolution)

	f_to_solve = lambda ftp_comp: PrAGI_CompModel(ftp_comp) - .5

	ftp_comp_solution = optimize.brentq(f_to_solve, probability_solution_leftbound, probability_solution_rightbound)

	c_brain_debug = 1e21

	computation_at_forecasted_time = getComputationAmountForYear(2036, biggest_spends_method)

	return fourParamFrameworkComp(computation_at_regime_start=c_brain_debug,
								  computation_at_forecasted_time=computation_at_forecasted_time,
								  biggest_spends_method=biggest_spends_method,
								  ftp_comp=ftp_comp_solution,
								  virtual_successes=virtual_successes)

def lifetimeAnchor(biggest_spends_method,virtual_successes=1):
	c_initial = getComputationAmountForYear(1956, biggest_spends_method)
	c_lifetime = 1e24

	n_trials_1956_to_lifetime = NumberIncreaseQuantity(c_initial, c_lifetime)

	PrAGI_CompModel = lambda ftp_comp: forecast_generalized_laplace(failures=0,
																	virtual_successes=virtual_successes,
																	ftp=ftp_comp,
																	forecast_years=n_trials_1956_to_lifetime)

	f_to_solve = lambda ftp_comp: PrAGI_CompModel(ftp_comp) - .5

	ftp_comp_solution = optimize.brentq(f_to_solve, probability_solution_leftbound, probability_solution_rightbound)

	return fourParamFrameworkComp(forecast_to_year=2036,
						   			regime_start_year=1956,
								  	biggest_spends_method=biggest_spends_method,
								  	ftp_comp=ftp_comp_solution,
								  	virtual_successes=virtual_successes)

def logUniform(biggest_spends_method):
	p_agi_by_evo_c = .8
	OOMs_2020_to_evo_c = 41 - np.log10(getComputationAmountForYear(2020,biggest_spends_method))

	p_agi_per_OOM = p_agi_by_evo_c/OOMs_2020_to_evo_c

	OOMs_2020_to_2036 = np.log10(getComputationAmountForYear(2036,biggest_spends_method)) - np.log10(getComputationAmountForYear(2020,biggest_spends_method))

	return p_agi_per_OOM*OOMs_2020_to_2036