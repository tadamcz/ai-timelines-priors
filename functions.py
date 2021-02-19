from scipy import integrate, optimize
import numpy as np
from collections import OrderedDict
import research_computation_dictionaries
from functools import lru_cache

# Note: To understand much of this file, the report 'Semi-informative priors over AI timelines' is required reading.
# The comments and docstrings assume that you are (very) familiar with the models described in the report; information
# is generally not repeated.

# Global variables
probability_solution_leftbound, probability_solution_rightbound = 1e-9, 1 - 1e-9

# To very slightly improve precision at a cost to performance, set this to a smaller number.
# Setting it to for example 1/10,000 changes some of the results by 1 on the second significant digit.
trial_increment = 1 / 100

def number_geometric_increases(start, end, increment=trial_increment):
	return int(np.log(end / start) / np.log(1 + increment))

@lru_cache(maxsize=int(1e6))
def generalized_laplace(trials, failures, virtual_successes, virtual_failures=None, ftp=None):
	"""
	Generalization of Laplace's rule:
	- to an arbitrary number of trials and failures (hence an arbitrary number of successes, which here will generally be 0).
	- to allow setting 'virtual' successes and failures, or a first-trial probability

	You must provide exactly two of virtual failures, virtual successes, and the first-trial probability.
	"""
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

@lru_cache(maxsize=int(1e4))
def forecast_generalized_laplace(failures, forecast_years, virtual_successes, virtual_failures=None, ftp=None):
	"""
	Use the generalized Laplace's rule to make forecasts about the probability of at least one success within the next
	`forecast_years` years.
	"""
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

# These next three Bayesian functions are very computationally intensive, and are only used to check agreement
# of the frequentist and Bayesian formulations of Laplace's rule.
def _no_AI_update_bayes_single(p_density):
	"""
	Bayesian update on a single failure.
	"""
	p_density_unnormalized = lambda x: p_density(x) * (1 - x)
	normalization_constant = integrate.quad(p_density_unnormalized, 0, 1)[0]
	return lambda x: p_density_unnormalized(x) / normalization_constant


def no_AI_update_bayes(failures, prior):
	"""
	Bayesian update on multiple failures, with a prior probabilty density given as a callable.
	"""
	p_density = prior
	for i in range(failures):
		p_density = _no_AI_update_bayes_single(p_density)
	return p_density


def forecast_bayes(failures, prior, forecast_years):
	"""
	Conduct a series of Bayesian updates to make forecasts about the probability of at least one success within the next
	`forecast_years` years.
	"""
	p_failure_by_target = 1
	for i in range(forecast_years):
		density = no_AI_update_bayes(failures, prior)
		p_failure = 1 - integrate.quad(lambda x: density(x) * x, 0, 1)[0]
		p_failure_by_target = p_failure_by_target * p_failure
		failures += 1
	p_success_by_target = 1 - p_failure_by_target
	return p_success_by_target


def four_param_framework_calendar(ftp, regime_start=1956, forecast_from=2020, forecast_to=2036, virtual_successes=1):
	"""
	The case where the trial definition is one calendar-year.
	"""
	if ftp == 0:
		return 0
	virtual_failures = (virtual_successes / ftp) - virtual_successes
	failures = forecast_from - regime_start
	return forecast_generalized_laplace(failures=failures, forecast_years=forecast_to - forecast_from, virtual_successes=virtual_successes, virtual_failures=virtual_failures)


def solve_for_ftp_res(g_exp=4.3 / 100, ftp_cal=1 / 300):
	"""
	Numerical equation solver used in some formulations of the researcher-year trial definition and the computation trail definition.
	"""
	p_AGI_res_model = lambda ftp_res: four_param_framework_researcher(g_act=g_exp, ftp_res=ftp_res)
	p_AGI_cal_model = four_param_framework_calendar(ftp_cal)

	# we solve f_to_solve=0
	f_to_solve = lambda ftp_res: p_AGI_res_model(ftp_res) - p_AGI_cal_model

	ftp_res_solution = optimize.brentq(f_to_solve, probability_solution_leftbound, probability_solution_rightbound)

	return ftp_res_solution


def four_param_framework_researcher(g_act, ftp_res=None, ftp_cal_equiv=None, g_exp=None, regime_start=1956, forecast_from=2020, forecast_to=2036, virtual_successes=1, g_act_after_2036=None):
	"""
	Used in the researcher-year trial definition
	"""
	if ftp_cal_equiv is not None and g_exp is not None:
		method = 'indirect'
	if ftp_res is not None:
		method = 'direct'
	if g_act_after_2036 is None:
		if g_exp is not None:
			g_act_after_2036 = g_exp
		else:
			g_act_after_2036 = 0  # Set some other default value if g_exp not provided. This should have no effect on the final result.

	if ftp_res is not None and (ftp_cal_equiv is not None or g_exp is not None):
		raise ValueError("Supply either (ftp_cal_equiv and g_exp) or supply ftp_res")

	if method == 'indirect':
		# To determine ftp_res, solve for ftp_res:
		# 	p_AGI_res_model(ftp_res,g_exp) = p_AGI_cal_model(ftp_cal)
		# Where g_exp and ftp_cal are constants
		ftp_res = solve_for_ftp_res(g_exp, ftp_cal_equiv)

	researcher_years = research_computation_dictionaries.generate_cumulative_researcher_years_dict(growth_to_2036=g_act, growth_after_2036=g_act_after_2036, regime_start=regime_start)
	failures = number_geometric_increases(researcher_years[regime_start], researcher_years[forecast_from])
	n_trials_forecast = number_geometric_increases(researcher_years[forecast_from], researcher_years[forecast_to])

	return forecast_generalized_laplace(
		failures=failures,
		forecast_years=n_trials_forecast,
		virtual_successes=virtual_successes,
		ftp=ftp_res)


def solve_for_ftp_comp(ftp_res, rel_imp_res_comp):
	"""
	Numerical equation solver used in some formulations of the computation trail definition.
	"""
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
	ftp_res = solve_for_ftp_res(g_exp=g_exp, ftp_cal=ftp_cal)
	return solve_for_ftp_comp(ftp_res, rel_imp_res_comp)

def four_param_framework_comp(
		rel_imp_res_comp=None,
		forecast_from=2020,
		forecast_to=2036,
		regime_start=1956,
		spend2036=None,
		computation_at_regime_start=None,
		computation_at_forecasted_time=None,
		ftp_cal_equiv=None,
		virtual_successes=1,
		ftp_comp=None,
		g_exp=4.3 / 100):
	"""
	Used in the computation trial definition.

	:param rel_imp_res_comp: The relative importance of research and computation. 1 doubling of the total researcher-years
	so far will have the same probability of producing AGI as `rel_imp_res_comp` doublings in
	the largest amount of computation used to develop an AI system.
	:param ftp_cal_equiv: called ftp_cal in the report, here given a more explicit name
	"""

	if ftp_comp is None:
		ftp_comp = solve_for_ftp_comp_indirect(ftp_cal_equiv, rel_imp_res_comp, g_exp=g_exp)

	computation2020 = get_computation_amount_for_year(forecast_from, spend2036)

	if computation_at_regime_start is None and computation_at_forecasted_time is None:
		computation_at_regime_start = get_computation_amount_for_year(regime_start, spend2036)
		computation_at_forecasted_time = get_computation_amount_for_year(forecast_to, spend2036)

	n_failures_regime_start_to_now = number_geometric_increases(computation_at_regime_start, computation2020)
	n_trials_forecast = number_geometric_increases(computation2020, computation_at_forecasted_time)

	return forecast_generalized_laplace(
		failures=n_failures_regime_start_to_now,
		forecast_years=n_trials_forecast,
		virtual_successes=virtual_successes,
		ftp=ftp_comp)


def get_year_for_computation_amount(c, spend2036):
	"""
	Currently not used anywhere, but is a useful function for possible future use
	"""
	if spend2036 in ['conservative','central','aggressive']:
		biggest_spends = research_computation_dictionaries.generate_named_spending_dict(spend2036)
	else:
		biggest_spends = research_computation_dictionaries.generate_spending_dict(spend2036)

	computation_to_year = OrderedDict()
	year_to_computation = OrderedDict()
	for year, price in research_computation_dictionaries.computation_prices.items():
		try:
			computation_to_year[biggest_spends[year] / price] = year
			year_to_computation[year] = biggest_spends[year] / price
		except KeyError:
			pass

	# otherwise, return the first year the computation was greater than c
	for computation, year in computation_to_year.items():
		if computation > c:
			return year


def get_computation_amount_for_year(y, spend2036):
	if spend2036 in ['conservative','central','aggressive']:
		biggest_spends = research_computation_dictionaries.generate_named_spending_dict(spend2036)
	else:
		biggest_spends = research_computation_dictionaries.generate_spending_dict(spend2036)

	year_to_computation = OrderedDict()
	for year, price in research_computation_dictionaries.computation_prices.items():
		try:
			year_to_computation[year] = biggest_spends[year] / price
		except KeyError:
			pass

	return year_to_computation[y]

@lru_cache()
def evolutionary_anchor(spend2036, virtual_successes=1, forecast_from=2020, forecast_to=2036):
	"""
	The "evolutionary anchor" biological hypothesis can be used within the computation trial definition.

	Note that there is no regime start year, as this hypothesis assumes that the regime started when
	'brain debugging computation' was achieved.
	"""
	c_initial = 1e21  # Brain debugging computation
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

	computation_at_forecasted_time = get_computation_amount_for_year(forecast_to, spend2036)

	return four_param_framework_comp(
		computation_at_regime_start=c_brain_debug,
		computation_at_forecasted_time=computation_at_forecasted_time,
		spend2036=spend2036,
		ftp_comp=ftp_comp_solution,
		forecast_from=forecast_from,
		virtual_successes=virtual_successes)

@lru_cache()
def lifetime_anchor(spend2036, virtual_successes=1, regime_start=1956, forecast_from=2020, forecast_to=2036):
	"""
	The "lifetime anchor" biological hypothesis can be used within the computation trial definition.
	"""
	c_initial = get_computation_amount_for_year(regime_start, spend2036)
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
		forecast_to=forecast_to,
		forecast_from=forecast_from,
		regime_start=regime_start,
		spend2036=spend2036,
		ftp_comp=ftp_comp_solution,
		virtual_successes=virtual_successes)


def log_uniform(spend2036, forecast_from=2020, forecast_to=2036):
	"""
	Not part of the semi-informative priors framework, this function predicts the probability of AGI
	based on the assumption that the probability of achieving AGI at a given amount of computation
	has a log-uniform distribution over the amount of computation between brain debugging-equivalent computation and
	evolution-equivalent computation. This distribution is updated as new failures are observed.
	"""
	OOMs_evo_c = 41
	OOMs_brain_debug = 21

	# Unconditional probability mass between brain debug and evolution, the rest of the mass is on AGI being impossible
	p_agi_debug_to_evo = .8

	OOMs_start = np.log10(get_computation_amount_for_year(forecast_from, spend2036))
	OOMs_end = np.log10(get_computation_amount_for_year(forecast_to, spend2036))

	# Number of orders of magnitude in the intersection of (brain debug,evolution) and (start,end)
	# It is the number of orders of magnitude between start and evolution that have non-zero probability mass.
	OOMs_in_brain_to_evo_and_start_to_end = min(OOMs_evo_c, OOMs_end) - max(OOMs_brain_debug, OOMs_start)

	# Number of orders of magnitude in the intersection of (brain debug,evolution) and (start,evolution)
	OOMs_in_brain_to_evo_and_start_to_evo = OOMs_evo_c - max(OOMs_brain_debug, OOMs_start)

	# Probability per order of magnitude in the intersection of (brain debug,evolution) and (start,evolution), conditional on no AGI by start.
	# It is the probability per order of magnitude remaining until evolution, once we have reached start.
	p_per_OOM_after_start = p_agi_debug_to_evo / OOMs_in_brain_to_evo_and_start_to_evo

	return p_per_OOM_after_start * OOMs_in_brain_to_evo_and_start_to_end


def hyper_prior_single_update(rules, initial_weights, forecast_to=2036, forecast_from=None) -> dict:
	"""
	This function will usually be called from within hyper_prior rather than by itself.

	This function updates the update rule weights from their initial values `initial_weights` to their values
	in the year `forecast_from`. If `forecast_from` is not provided, it is set to just one year before `forecast_to`,
	ensuring that the weights reflect all the information available up until that point (all the observed failures).
	This function assumes that AGI has not happened by `forecast_from`.

	For forecasts far into the future, we need to perform these yearly updates many times in a loop, which can become
	computationally heavy. To gain speed, we can make an approximation by updating the rule weights less frequently than once
	per year; this means that we will be computing 'p_forecast_to_hyper' with outdated weights. To do so, we will set `forecast_from`
	to something other than `None`.

	:param rules: list of dictionaries, each describing an update rule.
	:param initial_weights: initial weight of each update rule
	:return:
	"""
	if not forecast_from:
		# By default we do a single update on a single year of observed AGI failures, this gives the most accurate answer and is technically correct
		forecast_from = forecast_to - 1
	else:
		# We can override the default to conduct just a single update even when multiple years of AGI failure have been observed.
		# This will cause forecasts to be weighted at the weights in `forecast_from`.
		# These weights are outdated when `forecast_from` is not equal to `forecast_to - 1`.
		forecast_from = forecast_from

	ps_AGI_forecast_to = []
	ps_no_AGI_forecast_from = []

	for rule in rules:
		if 'virtual_successes' not in rule:
			rule['virtual_successes'] = 1  # Default value

		if rule['name'] == 'calendar':
			p_no_AGI_forecast_from = 1 - four_param_framework_calendar(
				ftp=rule['ftp'],
				regime_start=rule['regime_start'],
				forecast_from=rule['regime_start'],
				forecast_to=forecast_from,
				virtual_successes=rule['virtual_successes'])

			p_AGI_forecast_to_static = four_param_framework_calendar(
				ftp=rule['ftp'],
				regime_start=rule['regime_start'],
				virtual_successes=rule['virtual_successes'],
				forecast_from=forecast_from,
				forecast_to=forecast_to
			)

		if rule['name'] == 'res-year':
			p_no_AGI_forecast_from = 1 - four_param_framework_researcher(
				g_exp=rule['g_exp'],
				g_act=rule['g_act'],
				ftp_cal_equiv=rule['ftp_cal_equiv'],
				regime_start=rule['regime_start'],
				forecast_from=rule['regime_start'],
				forecast_to=forecast_from,
				virtual_successes=rule['virtual_successes'])

			p_AGI_forecast_to_static = four_param_framework_researcher(
				g_exp=rule['g_exp'],
				g_act=rule['g_act'],
				ftp_cal_equiv=rule['ftp_cal_equiv'],
				regime_start=rule['regime_start'],
				virtual_successes=rule['virtual_successes'],
				forecast_from=forecast_from,
				forecast_to=forecast_to
			)

		if rule['name'] == 'computation':
			if 'biohypothesis' not in rule:
				p_no_AGI_forecast_from = 1 - four_param_framework_comp(
					g_exp=rule['g_exp'],
					ftp_cal_equiv=rule['ftp_cal_equiv'],
					rel_imp_res_comp=rule['rel_imp_res_comp'],
					regime_start=rule['regime_start'],
					forecast_from=rule['regime_start'],
					forecast_to=forecast_from,
					spend2036=rule['spend2036'],
					virtual_successes=rule['virtual_successes'])

				p_AGI_forecast_to_static = four_param_framework_comp(
					g_exp=rule['g_exp'],
					ftp_cal_equiv=rule['ftp_cal_equiv'],
					rel_imp_res_comp=rule['rel_imp_res_comp'],
					regime_start=rule['regime_start'],
					spend2036=rule['spend2036'],
					virtual_successes=rule['virtual_successes'],
					forecast_from=forecast_from,
					forecast_to=forecast_to
				)

			else:
				if rule['biohypothesis'] == 'lifetime':
					p_no_AGI_forecast_from = 1 - lifetime_anchor(
						spend2036=rule['spend2036'],
						regime_start=rule['regime_start'],
						forecast_from=rule['regime_start'],
						forecast_to=forecast_from,
						virtual_successes=rule['virtual_successes'])

					p_AGI_forecast_to_static = lifetime_anchor(
						spend2036=rule['spend2036'],
						regime_start=rule['regime_start'],
						virtual_successes=rule['virtual_successes'],
						forecast_from=forecast_from,
						forecast_to=forecast_to,
					)

				if rule['biohypothesis'] == 'evolution':
					p_no_AGI_forecast_from = 1 - evolutionary_anchor(
						spend2036=rule['spend2036'],
						forecast_to=forecast_from,
						virtual_successes=rule['virtual_successes'],
					)

					p_AGI_forecast_to_static = evolutionary_anchor(
						spend2036=rule['spend2036'],
						virtual_successes=rule['virtual_successes'],
						forecast_from=forecast_from,
						forecast_to=forecast_to,
					)

		if rule['name'] == 'computation-loguniform':
			# Forecast-from should really be 'the beginning of time', but any year before brain debugging compute is achieved will give the same result.
			# I take the earliest year that will not result in a KeyError.
			p_no_AGI_forecast_from = 1 - log_uniform(rule['spend2036'], forecast_from=1956, forecast_to=forecast_from)

			p_AGI_forecast_to_static = log_uniform(rule['spend2036'], forecast_from=forecast_from, forecast_to=forecast_to)

		if rule['name'] == 'impossible':
			p_no_AGI_forecast_from = 1
			p_AGI_forecast_to_static = 0

		ps_no_AGI_forecast_from.append(p_no_AGI_forecast_from)
		ps_AGI_forecast_to.append(p_AGI_forecast_to_static)

	final_weights_unnormalized = np.asarray(initial_weights) * np.asarray(ps_no_AGI_forecast_from)

	# Not necessary for computation, but useful additional information for display or debugging
	normalization_constant = sum(final_weights_unnormalized)
	weights_forecast_from = final_weights_unnormalized / normalization_constant

	return {
		'p_forecast_to_hyper': np.average(ps_AGI_forecast_to, weights=weights_forecast_from),
		'p_forecast_to_static': np.average(ps_AGI_forecast_to, weights=initial_weights),
		'wts_forecast_from': weights_forecast_from}

def hyper_prior(
		rules: list,
		initial_weights: list,
		forecast_from=2020,
		forecast_to=2036,
		return_sequence=False,
		pivot_to_coarse=None,
		forecast_years_explicit=None,
) -> dict:
	"""
	Creates a sequence of forecasts about the probability of AGI, using a weighted average of several update rules.
	The `initial_weights` of the update rules form a hyper-prior that is being updated based on the
	probability that each update rule assigned to what actually happened (or what is assumed to happen in a branch of the simulation).

	By default, the weights of the update rules are updated every year based on that year's observations. This is technically
	correct but computationally costly. We can update the weights less frequently, which is less accurate but improves performance.

	When we update the weights less frequently, what specifically happens is that we make forecasts for each year in
	`forecast_years`, a sequence of years. The probability of AGI for `forecast_year[i]` is the probability assigned by each rule
	to that year, weighted with weights corresponding to updating the hyper-prior on the evidence that AGI has
	not happened by forecast_year[i-1].

	There are two ways of specifying how to update less frequently. One is to set the `pivot_to_coarse`, which is the year by which
	we switch from updating every year to every 15 years (these constants could easily be changed). The other way is to pass in
	the sequence `forecast_years_explicit`, which directly defines `forecast_years`.

	:param return_sequence: By default, we only return the results for `forecast_to`. If True, return the entire sequence of 'p_forecast_to_hyper', from `forecast_from` to `forecast_to`.
	:param pivot_to_coarse: If different from None, update more infrequently after year `pivot_to_coarse`. Useful if predicting out to 2100 and highest precision is not required.
	:param forecast_years_explicit: Provide the array explicitly. Overrides `forecast_from`, `forecast_to`, `pivot_to_coarse`.
	"""
	if forecast_years_explicit is not None:
		forecast_from, forecast_to = forecast_years_explicit[0], forecast_years_explicit[-1]
		forecast_years = forecast_years_explicit
	else:
		fine_update_frequency = 1
		coarse_update_frequency = 15
		if pivot_to_coarse is None:
			forecast_years = range(forecast_from+1, forecast_to+1)
		else:
			forecast_years = np.concatenate((np.arange(forecast_from+1, pivot_to_coarse, fine_update_frequency), np.arange(pivot_to_coarse, forecast_to+1, coarse_update_frequency), (forecast_to,)))

	hyper_results = {}
	p_failure_by_target = 1  # stores the probability of failure by a given year as we loop through `forecast_years`
	for index,forecast_to_single in enumerate(forecast_years):
		# `forecast_from_single` and `forecast_to_single` are the dates passed to hyper_prior_single_update
		# at each iteration of the loop, and are named to distinguish them from `forecast_from` and `forecast_to`.
		if index == 0:
			forecast_from_single = forecast_years[0] - 1
		else:
			forecast_from_single = forecast_years[index-1]

		# Perform a single hyper-prior update using the weights in year `forecast_from_single`.
		hyper_results_year = hyper_prior_single_update(rules=rules, initial_weights=initial_weights, forecast_from=forecast_from_single, forecast_to=forecast_to_single)

		p_failure = 1 - hyper_results_year['p_forecast_to_hyper']
		p_failure_by_target = p_failure_by_target * p_failure
		p_success_by_target_hyper = 1 - p_failure_by_target

		hyper_results[forecast_to_single] = {
			'p_forecast_to_hyper': p_success_by_target_hyper,
			'forecast_from': forecast_from_single,
			'wts_forecast_from': hyper_results_year['wts_forecast_from']  # We include the weights that were used to produce the result
		}

	# The static calculation can be done all in one go:
	static = hyper_prior_single_update(rules=rules, initial_weights=initial_weights, forecast_to=forecast_to, forecast_from=forecast_from)
	hyper_results[forecast_to]['p_forecast_to_static'] = static['p_forecast_to_static']

	if return_sequence:
		return hyper_results
	else:
		if forecast_from == forecast_years[0] - 1:  # to avoid a ValueError
			after_forecast_from = forecast_years[0]
		else:
			after_forecast_from = forecast_years[forecast_years.index(forecast_from) + 1]
		return {
			'p_forecast_to_hyper': hyper_results[forecast_to]['p_forecast_to_hyper'],
			'p_forecast_to_static': hyper_results[forecast_to]['p_forecast_to_static'],

			# Here,'wts_forecast_from' means the weights in the year `forecast_from`,
			# not the weights 'that were used to make the forecast', as there is no single
			# such set of weights.
			'wts_forecast_from': hyper_results[after_forecast_from]['wts_forecast_from']
		}
