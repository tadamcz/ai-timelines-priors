from scipy import integrate, optimize
import numpy as np
from collections import OrderedDict

# Global variables
probability_solution_leftbound, probability_solution_rightbound = 1e-9, 1 - 1e-9
trial_increment = 1 / 100


def generalized_laplace(trials, failures, virtual_successes, virtual_failures=None, ftp=None):
	if ftp == 0:
		return 0
	if ftp is not None:
		if virtual_failures is not None and virtual_successes is not None:
			raise ValueError("Provide exactly two of virtual_failures, virtual_successes, and ftp")
		if virtual_failures is None:
			virtual_failures = (virtual_successes / ftp) - virtual_successes

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


def _no_AI_update_bayes_single(p_density):
	p_density_unnormalized = lambda x: p_density(x) * (1 - x)
	normalization_constant = integrate.quad(p_density_unnormalized, 0, 1)[0]
	return lambda x: p_density_unnormalized(x) / normalization_constant


def no_AI_update_bayes(failures, prior):
	p_density = prior
	for i in range(failures):
		p_density = _no_AI_update_bayes_single(p_density)
	return p_density


def forecast_bayes(failures, prior, forecast_years):
	p_failure_by_target = 1
	for i in range(forecast_years):
		density = no_AI_update_bayes(failures, prior)
		p_failure = 1 - integrate.quad(lambda x: density(x) * x, 0, 1)[0]
		p_failure_by_target = p_failure_by_target * p_failure
		failures += 1
	p_success_by_target = 1 - p_failure_by_target
	return p_success_by_target


def four_param_framework_calendar(ftp, regime_start=1956, forecast_from=2020, forecast_to=2036, virtual_successes=1):
	if ftp == 0:
		return 0
	virtual_failures = (virtual_successes / ftp) - virtual_successes
	failures = forecast_from - regime_start
	return forecast_generalized_laplace(failures=failures, forecast_years=forecast_to - forecast_from, virtual_successes=virtual_successes, virtual_failures=virtual_failures)


def solveFor_ftp_res(g_exp=4.3 / 100, ftp_cal=1 / 300):
	"""
	To determine ftp_res, solve for ftp_res:
	p_AGI_res_model(ftp_res,g_exp) = p_AGI_cal_model(ftp_cal)
	Where g_exp and ftp_cal are constants
	"""
	p_AGI_res_model = lambda ftp_res: four_param_framework_researcher(g_act=g_exp, ftp_res=ftp_res)
	p_AGI_cal_model = four_param_framework_calendar(ftp_cal)

	# we solve f_to_solve=0
	f_to_solve = lambda ftp_res: p_AGI_res_model(ftp_res) - p_AGI_cal_model

	ftp_res_solution = optimize.brentq(f_to_solve, probability_solution_leftbound, probability_solution_rightbound)

	return ftp_res_solution


def four_param_framework_researcher(g_act, ftp_res=None, ftp_cal_equiv=None, g_exp=None, regime_start=1956, forecast_from=2020, forecast_to=2036, virtual_successes=1):
	if ftp_cal_equiv is not None and g_exp is not None:
		method = 'indirect'
	if ftp_res is not None:
		method = 'direct'

	if ftp_res is not None and (ftp_cal_equiv is not None or g_exp is not None):
		raise ValueError("Supply either (ftp_cal_equiv and g_exp) or supply ftp_res")

	if method == 'indirect':
		ftp_res = solveFor_ftp_res(g_exp, ftp_cal_equiv)

	n_trials_per_year = g_act * (1 / trial_increment)

	failures = int((forecast_from - regime_start) * n_trials_per_year)
	n_trials_forecast = int((forecast_to - forecast_from) * n_trials_per_year)

	return forecast_generalized_laplace(
		failures=failures,
		forecast_years=n_trials_forecast,
		virtual_successes=virtual_successes,
		ftp=ftp_res)


def solve_for_ftp_comp(ftp_res, rel_imp_res_comp):
	res_trials = 1  # arbitrary
	comp_trials = rel_imp_res_comp * res_trials

	# Number of trials that occur in both models, before the `res_trials` of the research model are compared to `comp_trials` of
	# the computation model. This is arbitrary and does not affect the outcome
	n = 5

	p_agi_res_model = forecast_generalized_laplace(failures=n, virtual_successes=1, ftp=ftp_res, forecast_years=res_trials)
	p_agi_comp_model = lambda ftp_comp: forecast_generalized_laplace(failures=n, virtual_successes=1, ftp=ftp_comp, forecast_years=comp_trials)

	# we solve f_to_solve=0
	f_to_solve = lambda ftp_comp: p_agi_comp_model(ftp_comp) - p_agi_res_model

	ftp_comp_solution = optimize.brentq(f_to_solve, probability_solution_leftbound, probability_solution_rightbound)

	return ftp_comp_solution


def solve_for_ftp_comp_indirect(ftp_cal, rel_imp_res_comp, g_exp):
	ftp_res = solveFor_ftp_res(g_exp=g_exp, ftp_cal=ftp_cal)
	return solve_for_ftp_comp(ftp_res, rel_imp_res_comp)


def number_geometric_increases(start, end, increment=trial_increment):
	return int(np.log(end / start) / np.log(1 + increment))


# Dictionaries compute_prices, biggest_spends_conservative and biggest_spends_aggressive copied directly from client's code without checking
compute_prices = {
	1800: 10 ** -6,
	1956: 10 ** -6,
	1970: 10 ** -7,
	2000: 10 ** -12,
	2008: 10 ** -16,
	2020: 10 ** -17,
	2021: 10 ** -17.2,
	2022: 10 ** -17.4,
	2023: 10 ** -17.6,
	2024: 10 ** -17.8,
	2025: 10 ** -18,
	2026: 10 ** -18.1,
	2027: 10 ** -18.2,
	2028: 10 ** -18.3,
	2029: 10 ** -18.4,
	2030: 10 ** -18.5,
	2031: 10 ** -18.6,
	2032: 10 ** -18.7,
	2033: 10 ** -18.8,
	2034: 10 ** -18.9,
	2035: 10 ** -18.95,
	2036: 10 ** -19,
}
biggest_spends_conservative = {
	1956: 10 ** 1,
	2020: 10 ** 6.7,
	2021: 10 ** 6.78,
	2022: 10 ** 6.86,
	2023: 10 ** 6.94,
	2024: 10 ** 7.02,
	2025: 10 ** 7.1,
	2026: 10 ** 7.18,
	2027: 10 ** 7.26,
	2028: 10 ** 7.35,
	2029: 10 ** 7.43,
	2030: 10 ** 7.51,
	2031: 10 ** 7.59,
	2032: 10 ** 7.67,
	2033: 10 ** 7.75,
	2034: 10 ** 7.83,
	2035: 10 ** 7.91,
	2036: 10 ** 8,
}
biggest_spends_aggressive = {
	1956: 10 ** 1,
	2020: 10 ** 6.7,
	2021: 10 ** 7.0,
	2022: 10 ** 7.25,
	2023: 10 ** 7.5,
	2024: 10 ** 7.8,
	2025: 10 ** 8.05,
	2026: 10 ** 8.3,
	2027: 10 ** 8.6,
	2028: 10 ** 8.85,
	2029: 10 ** 9.1,
	2030: 10 ** 9.4,
	2031: 10 ** 9.65,
	2032: 10 ** 9.9,
	2033: 10 ** 10.2,
	2034: 10 ** 10.45,
	2035: 10 ** 10.7,
	2036: 10 ** 11,
}
biggest_spends_central = {
	2036: 10 ** 9,
}

for k in biggest_spends_conservative.keys():
	# biggest_spends_central: geometric interpolation between conservative and aggressive, anchoring off the value for 2036

	conservative = np.log10(biggest_spends_conservative[k])
	aggressive = np.log10(biggest_spends_aggressive[k])

	factor = (np.log10(biggest_spends_central[2036]) - np.log10(biggest_spends_conservative[2036])) / (np.log10(biggest_spends_aggressive[2036]) - np.log10(biggest_spends_conservative[2036]))
	biggest_spends_central[k] = 10 ** (conservative + (aggressive - conservative) * factor)


def four_param_framework_comp(
		rel_imp_res_comp=None,
		forecast_from_year=2020,
		forecast_to_year=2036,
		regime_start_year=1956,
		biggest_spends_method=None,
		computation_at_regime_start=None,
		computation_at_forecasted_time=None,
		ftp_cal_equiv=None,
		virtual_successes=1,
		ftp_comp=None,
		g_exp=4.3 / 100):
	"""
	The most indirect method, from biggest_spend_end + ftp_cal_equiv, g_exp, and rel_imp_res_comp.
	The most direct method: from biggest_spend_end + ftp_comp.
	"""

	if ftp_comp is None:
		ftp_comp = solve_for_ftp_comp_indirect(ftp_cal_equiv, rel_imp_res_comp, g_exp=g_exp)

	computation2020 = get_computation_amount_for_year(forecast_from_year, biggest_spends_method)

	if computation_at_regime_start is None and computation_at_forecasted_time is None:
		computation_at_regime_start = get_computation_amount_for_year(regime_start_year, biggest_spends_method)
		computation_at_forecasted_time = get_computation_amount_for_year(forecast_to_year, biggest_spends_method)

	n_failures_regime_start_to_now = number_geometric_increases(computation_at_regime_start, computation2020)
	n_trials_forecast = number_geometric_increases(computation2020, computation_at_forecasted_time)

	return forecast_generalized_laplace(
		failures=n_failures_regime_start_to_now,
		forecast_years=n_trials_forecast,
		virtual_successes=virtual_successes,
		ftp=ftp_comp)


def get_year_for_computation_amount(c, biggest_spends_method):
	if biggest_spends_method == 'aggressive':
		biggest_spends = biggest_spends_aggressive
	if biggest_spends_method == 'conservative':
		biggest_spends = biggest_spends_conservative
	if biggest_spends_method == 'central':
		biggest_spends = biggest_spends_central

	computation_to_year = OrderedDict()
	year_to_computation = OrderedDict()
	for year, price in compute_prices.items():
		try:
			computation_to_year[biggest_spends[year] / price] = year
			year_to_computation[year] = biggest_spends[year] / price
		except KeyError:
			pass

	# otherwise, return the first year the computation was greater than c
	for computation, year in computation_to_year.items():
		if computation > c:
			return year


def get_computation_amount_for_year(y, biggest_spends_method):
	if biggest_spends_method == 'aggressive':
		biggest_spends = biggest_spends_aggressive
	if biggest_spends_method == 'conservative':
		biggest_spends = biggest_spends_conservative
	if biggest_spends_method == 'central':
		biggest_spends = biggest_spends_central

	year_to_computation = OrderedDict()
	for year, price in compute_prices.items():
		try:
			year_to_computation[year] = biggest_spends[year] / price
		except KeyError:
			pass

	return year_to_computation[y]


def evolutionary_anchor(biggest_spends_method, virtual_successes=1, forecast_to_year=2036):
	c_initial = 1e21
	c_evolution = 1e41

	n_trials_initial_to_evolution = number_geometric_increases(c_initial, c_evolution)

	p_agi_comp_model = lambda ftp_comp: forecast_generalized_laplace(
		failures=0,
		virtual_successes=virtual_successes,
		ftp=ftp_comp,
		forecast_years=n_trials_initial_to_evolution)

	f_to_solve = lambda ftp_comp: p_agi_comp_model(ftp_comp) - .5

	ftp_comp_solution = optimize.brentq(f_to_solve, probability_solution_leftbound, probability_solution_rightbound)

	c_brain_debug = 1e21

	computation_at_forecasted_time = get_computation_amount_for_year(forecast_to_year, biggest_spends_method)

	return four_param_framework_comp(
		computation_at_regime_start=c_brain_debug,
		computation_at_forecasted_time=computation_at_forecasted_time,
		biggest_spends_method=biggest_spends_method,
		ftp_comp=ftp_comp_solution,
		virtual_successes=virtual_successes)


def lifetime_anchor(biggest_spends_method, virtual_successes=1, regime_start_year=1956, forecast_from_year=2020, forecast_to_year=2036):
	c_initial = get_computation_amount_for_year(regime_start_year, biggest_spends_method)
	c_lifetime = 1e24

	n_trials_reg_start_to_lifetime = number_geometric_increases(c_initial, c_lifetime)

	p_agi_comp_model = lambda ftp_comp: forecast_generalized_laplace(
															failures=0,
															virtual_successes=virtual_successes,
															ftp=ftp_comp,
															forecast_years=n_trials_reg_start_to_lifetime)

	f_to_solve = lambda ftp_comp: p_agi_comp_model(ftp_comp) - .5

	ftp_comp_solution = optimize.brentq(f_to_solve, probability_solution_leftbound, probability_solution_rightbound)

	return four_param_framework_comp(
		forecast_to_year=forecast_to_year,
		forecast_from_year=forecast_from_year,
		regime_start_year=regime_start_year,
		biggest_spends_method=biggest_spends_method,
		ftp_comp=ftp_comp_solution,
		virtual_successes=virtual_successes)


def log_uniform(biggest_spends_method, forecast_from=2020, forecast_to=2036):
	OOMs_evo_c = 41
	OOMs_brain_debug = 21

	# Unconditional probability mass between brain debug and evolution
	p_agi_debug_to_evo = .8

	OOMs_start = np.log10(get_computation_amount_for_year(forecast_from, biggest_spends_method))
	OOMs_end = np.log10(get_computation_amount_for_year(forecast_to, biggest_spends_method))

	# Number of orders of magnitude in the intersection of (brain debug,evolution) and (start,end)
	# It is the number of orders of magnitude between start and evolution that have non-zero probability mass.
	OOMs_in_brain_to_evo_and_start_to_end = min(OOMs_evo_c, OOMs_end) - max(OOMs_brain_debug, OOMs_start)

	# Number of orders of magnitude in the intersection of (brain debug,evolution) and (start,evolution)
	OOMs_in_brain_to_evo_and_start_to_evo = OOMs_evo_c - max(OOMs_brain_debug, OOMs_start)

	# Probability per order of magnitude in the intersection of (brain debug,evolution) and (start,evolution), conditional on no AGI by start.
	# It is the probability per order of magnitude remaining until evolution, once we have reached start.
	p_per_OOM_after_start = p_agi_debug_to_evo / OOMs_in_brain_to_evo_and_start_to_evo

	return p_per_OOM_after_start * OOMs_in_brain_to_evo_and_start_to_end


def hyper_prior(rules: list, initial_weights: list) -> dict:
	ps_AGI_2036 = []
	ps_no_AGI_2020 = []

	for rule in rules:
		if 'virtual_successes' not in rule:
			rule['virtual_successes'] = 1  # Default value

		if rule['name'] == 'calendar':
			p_no_AGI_2020 = 1 - four_param_framework_calendar(
				ftp=rule['ftp'],
				regime_start=rule['regime_start'],
				forecast_from=rule['regime_start'],
				forecast_to=2020,
				virtual_successes=rule['virtual_successes'])

			p_AGI_2036_static = four_param_framework_calendar(
				ftp=rule['ftp'],
				regime_start=rule['regime_start'],
				virtual_successes=rule['virtual_successes'])

		if rule['name'] == 'res-year':
			p_no_AGI_2020 = 1 - four_param_framework_researcher(
				g_exp=rule['g_exp'],
				g_act=rule['g_act'],
				ftp_cal_equiv=rule['ftp_cal_equiv'],
				regime_start=rule['regime_start'],
				forecast_from=rule['regime_start'],
				forecast_to=2020,
				virtual_successes=rule['virtual_successes'])

			p_AGI_2036_static = four_param_framework_researcher(
				g_exp=rule['g_exp'],
				g_act=rule['g_act'],
				ftp_cal_equiv=rule['ftp_cal_equiv'],
				regime_start=rule['regime_start'],
				virtual_successes=rule['virtual_successes'])

		if rule['name'] == 'computation':
			if 'biohypothesis' not in rule:
				p_no_AGI_2020 = 1 - four_param_framework_comp(
					g_exp=rule['g_exp'],
					ftp_cal_equiv=rule['ftp_cal_equiv'],
					rel_imp_res_comp=rule['rel_imp_res_comp'],
					regime_start_year=rule['regime_start'],
					forecast_from_year=rule['regime_start'],
					forecast_to_year=2020,
					biggest_spends_method=rule['biggest_spends_method'],
					virtual_successes=rule['virtual_successes'])

				p_AGI_2036_static = four_param_framework_comp(
					g_exp=rule['g_exp'],
					ftp_cal_equiv=rule['ftp_cal_equiv'],
					rel_imp_res_comp=rule['rel_imp_res_comp'],
					regime_start_year=rule['regime_start'],
					biggest_spends_method=rule['biggest_spends_method'],
					virtual_successes=rule['virtual_successes'])

			else:
				if rule['biohypothesis'] == 'lifetime':
					p_no_AGI_2020 = 1 - lifetime_anchor(
						biggest_spends_method=rule['biggest_spends_method'],
						regime_start_year=rule['regime_start'],
						forecast_from_year=rule['regime_start'],
						forecast_to_year=2020,
						virtual_successes=rule['virtual_successes'])

					p_AGI_2036_static = lifetime_anchor(
						biggest_spends_method=rule['biggest_spends_method'],
						regime_start_year=rule['regime_start'],
						virtual_successes=rule['virtual_successes'])

				if rule['biohypothesis'] == 'evolution':
					p_no_AGI_2020 = 1 - evolutionary_anchor(
						biggest_spends_method='aggressive',
						forecast_to_year=2020,
						virtual_successes=rule['virtual_successes'])

					p_AGI_2036_static = evolutionary_anchor(
						biggest_spends_method='aggressive',
						virtual_successes=rule['virtual_successes'])

		if rule['name'] == 'computation-loguniform':
			# Forecast-from should really be 'the beginning of time', but any year before brain debugging compute is achieved will give the same result.
			# I take the earliest year that will not result in a KeyError.
			p_no_AGI_2020 = 1 - log_uniform(rule['biggest_spends_method'], forecast_from=1956, forecast_to=2020)

			p_AGI_2036_static = log_uniform(rule['biggest_spends_method'], forecast_from=2020, forecast_to=2036)

		if rule['name'] == 'impossible':
			p_no_AGI_2020 = 1
			p_AGI_2036_static = 0

		ps_no_AGI_2020.append(p_no_AGI_2020)
		ps_AGI_2036.append(p_AGI_2036_static)

	final_weights_unnormalized = np.asarray(initial_weights) * np.asarray(ps_no_AGI_2020)

	# Not necessary for computation, but useful additional information for display or debugging
	normalization_constant = sum(final_weights_unnormalized)
	final_weights = final_weights_unnormalized / normalization_constant

	return {
		'pr2036hyper': np.average(ps_AGI_2036, weights=final_weights),
		'pr2036static': np.average(ps_AGI_2036, weights=initial_weights),
		'wts2020': final_weights}
