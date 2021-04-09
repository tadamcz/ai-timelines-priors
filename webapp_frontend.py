from flask import Flask, render_template
from wtforms import FloatField, IntegerField, SelectField, StringField
from wtforms import validators
from datetime import datetime
import fractions
import pylru

from flask_wtf import FlaskForm

import functions
from sections import to_percentage_strings, to_fraction_strings, round_sig
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import matplotlib.ticker as mtick
from collections import OrderedDict

app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False  # not needed, there are no user accounts

is_probability = validators.number_range(0, 1, message='Probability must be between 0 and 1')
is_non_negative = validators.number_range(min=0, max=None, message='Cannot be negative')
is_non_zero = validators.NoneOf((0,), message='Cannot be 0')
is_not_one = validators.NoneOf((1,), message='Cannot be 1')
reasonable_growth = validators.number_range(max=1e5, message='Such big numbers will confuse this poor computer, please try something smaller')


class HyperPriorForm(FlaskForm):
	rule_out_agi_by = IntegerField(validators=[validators.DataRequired(), validators.number_range(min=2020, max=2099)], label='We can rule out AGI being developed before the end of ', default=2020)
	first_trial_probability = StringField(default='1/300', label='First trial probability')
	virtual_successes = FloatField(validators=[validators.DataRequired(), is_non_negative, is_non_zero], label='Virtual successes', default=1)
	regime_start_year = IntegerField(validators=[validators.Optional()], label='Regime start year', default=1956)

	g_exp = FloatField(validators=[validators.Optional(), is_non_negative, is_non_zero, reasonable_growth], label='Typical annual growth for STEM researchers (%)', default=4.3)
	g_act = FloatField(validators=[validators.Optional(), is_non_negative, is_non_zero, reasonable_growth], label='Annual growth of AI researchers until 2036 (%)', default=11)

	relative_imp_res_comp = IntegerField(validators=[validators.Optional(), is_non_negative, is_non_zero, reasonable_growth], label='A 1% increase in the number of researchers is equivalent to an X% increase in computation', default=5)

	comp_spending_assumption = FloatField(validators=[validators.Optional(), validators.number_range(min=0.00502, message='Must be greater than 2020 spend of 0.00502 billion')], label='Maximum computation spend by 2036 ($ billions)', default=1)

	init_weight_calendar = FloatField(validators=[validators.Optional(), is_non_negative], label='Calendar-year trial definition', default=.3)
	init_weight_researcher = FloatField(validators=[validators.Optional(), is_non_negative], label='Researcher-year trial definition', default=.3)
	init_weight_comp_relative_res = FloatField(validators=[is_non_negative], label='Computation trial definition: relative importance of research and computation', default=.05)
	init_weight_lifetime = FloatField(validators=[validators.Optional(), is_non_negative], label='Computation trial definition: lifetime anchor', default=.1)
	init_weight_evolution = FloatField(validators=[validators.Optional(), is_non_negative], label='Computation trial definition: evolutionary anchor', default=.15)
	init_weight_agi_impossible = FloatField(validators=[validators.Optional(), is_non_negative], label='AGI is impossible', default=.1)

	def calendar_year_filled(self):
		c1 = self.first_trial_probability.data is not None
		c2 = self.regime_start_year.data is not None

		if all((c1, c2)):
			return True
		else:
			self.init_weight_calendar.errors.append("Not all inputs provided, this rule will be ignored")
			return False

	def researcher_filled(self):
		c1 = self.g_exp.data is not None
		c2 = self.g_act.data is not None

		if all((c1, c2)):
			return True
		else:
			self.init_weight_researcher.errors.append("Not all inputs provided, this rule will be ignored")
			return False

	def computation_relative_res_filled(self):
		c1 = self.relative_imp_res_comp.data is not None

		if c1:
			return True
		else:
			self.init_weight_comp_relative_res.errors.append("Not all inputs provided, this rule will be ignored")
			return False


	def initial_weights_filled(self):
		init_weight_fields = [
			self.init_weight_calendar,
			self.init_weight_researcher,
			self.init_weight_comp_relative_res,
			self.init_weight_lifetime,
			self.init_weight_evolution,
			self.init_weight_agi_impossible,
		]

		all_filled = True
		for field in init_weight_fields:
			if field.data is None:
				all_filled = False
				field.errors.append("Required for hyper-prior update")

		return all_filled

	def validate(self):
		validity = True
		if not super(HyperPriorForm, self).validate():
			validity = False

		self.check_initial_weights()

		if not self.rule_out_agi_by.errors and not self.regime_start_year.errors:
			if self.regime_start_year.data > self.rule_out_agi_by.data:
				validity = False
				self.regime_start_year.errors.append("Cannot be later than %s, the year by which we can rule out AGI" % self.rule_out_agi_by.data)

		if self.first_trial_probability.data is not None:
			try:
				self.first_trial_probability.data = fractions.Fraction(self.first_trial_probability.data)
			except (SyntaxError, ValueError, ZeroDivisionError):
				self.first_trial_probability.errors.append("Could not interpret as a decimal or fraction")
				validity = False
			else:
				if not 0 <= self.first_trial_probability.data <= 1:
					validity = False
					self.first_trial_probability.errors.append("Probability must be between 0 and 1")
				if self.first_trial_probability.data == 0 or self.first_trial_probability.data == 1:
					validity = False
					self.first_trial_probability.errors.append("Cannot be 0 or 1")


		return validity





	def check_initial_weights(self):
		init_weight_fields = [
			self.init_weight_calendar,
			self.init_weight_researcher,
			self.init_weight_comp_relative_res,
			self.init_weight_lifetime,
			self.init_weight_evolution,
			self.init_weight_agi_impossible,
		]

		for field in init_weight_fields:
			if field.data is None:
				field.data = 0
				field.raw_data = ['0.']

		# Avoid the case where all are 0
		if all(field.data == 0 for field in init_weight_fields):
			for field in init_weight_fields:
				field.data = 1
				field.raw_data = ['1.']

def plot_helper(xs,ys):
	fig, ax = plt.subplots()
	fig.set_size_inches(5, 2)
	ax.plot(xs, ys)
	ax.set_ylabel("Pr(AGI)")
	ax.set_xlabel("Year")

	# We use FixedFormatter and FixedLocator here to circumvent the limitations and unpredictable behaviour of mpld3.
	#
	# mpld3 does not always display the same number of tick marks and decimal places as matplotlib.
	# PercentFormatter and MaxNLocator have buggy implementations in mpld3 (see also https://github.com/mpld3/mpld3/issues/341)
	#
	# We define the tick mark locations and labels explicitly in Python, and hard-code them into FixedFormatter and FixedLocator.
	# This ensures that the number of tick marks is always the same (10), and let us control the number of decimal places.

	# Option 1: always display 2 significant figures
	# y_ticks_fixed = np.linspace(0,max(ys)*100,10)
	# y_ticks_fixed_str = [s+'%' for s in round_sig(y_ticks_fixed,2, type=str)]
	# ax.yaxis.set_major_locator(mtick.FixedLocator([i/100 for i in y_ticks_fixed]))
	# ax.yaxis.set_major_formatter(mtick.FixedFormatter(y_ticks_fixed_str))

	# Option 2: always display the same number of digits after the decimal point
	decimals = 1-max(int(np.log10(max(ys)))+1,0)
	y_ticks_fixed = np.linspace(0,max(ys)*100,10)
	y_ticks_fixed_str = [s+'%' for s in round_sig(y_ticks_fixed,decimals=decimals, type=str)]
	ax.yaxis.set_major_locator(mtick.FixedLocator([i/100 for i in y_ticks_fixed]))
	ax.yaxis.set_major_formatter(mtick.FixedFormatter(y_ticks_fixed_str))

	mpld3.plugins.clear(fig)  # We only need a very simple plot
	plot_html = mpld3.fig_to_html(fig)
	plt.close(fig)  # Otherwise the matplotlib object stays in memory forever
	return plot_html

def plot_helper_multiline(dict_named_x_y_pairs):
	fig, ax = plt.subplots()
	fig.set_size_inches(5, 3.5)
	for name,pair in reversed(dict_named_x_y_pairs.items()):
		xs, ys = pair
		ax.plot(xs, ys, label=name)
	ax.legend()
	ax.set_ylabel("Pr(AGI)")
	ax.set_xlabel("Year")
	ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
	mpld3.plugins.clear(fig)  # We only need a very simple plot
	plot_html = mpld3.fig_to_html(fig)
	plt.close(fig)  # Otherwise the matplotlib object stays in memory forever
	return plot_html

class UpdateRuleResult:
	def __init__(self, _callable, input_args, init_weight, rule_out_agi_by):
		self.input_args = input_args
		self.init_weight = init_weight
		self.rule_out_agi_by = rule_out_agi_by
		if _callable:
			self.p2036 = to_percentage_strings(_callable(2036))
			self.create_plot(_callable, x_from_to=(self.rule_out_agi_by,2100))


	def create_plot(self, _callable, x_from_to=(2020, 2100), is_date=True):
		xs = generate_years_to_forecast(start_end=x_from_to, force_include={2020, self.rule_out_agi_by})
		ys = [_callable(i) for i in xs]
		self.cdf_iterable = zip(xs,to_percentage_strings(ys))

		if is_date:
			xs = [datetime(x, 1, 1) for x in xs]
		self.plot = plot_helper(xs,ys)
		self.xs_plot = xs
		self.ys_plot = ys


class HyperPriorResult:
	def __init__(self, update_hyper_from):
		self.update_hyper_from = update_hyper_from
		self.researcher = None
		self.calendar = None
		self.comp_relative_res = None
		self.agi_impossible = None
		self.lifetime = None
		self.evolution = None

	def update_hyper_prior(self):

		rules_dicts = []
		weights = []

		rules_attributes = [self.agi_impossible, self.lifetime, self.evolution]

		# consider refactoring around this ugly code. would require some change to how hyper_prior takes args.
		if self.calendar:
			rules_attributes.append(self.calendar)
			self.calendar.input_args['name'] = 'calendar'

		if self.researcher:
			rules_attributes.append(self.researcher)
			self.researcher.input_args['name'] = 'res-year'

		if self.comp_relative_res:
			rules_attributes.append(self.comp_relative_res)
			self.comp_relative_res.input_args['name'] = 'computation'

		self.lifetime.input_args['name'] = 'computation'
		self.lifetime.input_args['biohypothesis'] = 'lifetime'

		self.evolution.input_args['name'] = 'computation'
		self.evolution.input_args['biohypothesis'] = 'evolution'

		if not rules_attributes:
			return

		for rule in rules_attributes:
			rules_dicts.append(rule.input_args)
			weights.append(rule.init_weight)

		init_weights_normalized = np.asarray(weights)/sum(weights)

		# We create the data here instead of passing a callable because it's far more efficient to generate the whole sequence in one call to functions.hyper_prior
		hyper_results = {}
		for year in range(2020,self.update_hyper_from + 1):
			hyper_results[year] = {'p_forecast_to_hyper': 0}
		forecast_years = generate_years_to_forecast((self.update_hyper_from,2100), fine_interval=3, coarse_interval=15)
		hyper_results.update(functions.hyper_prior(rules_dicts,weights, forecast_years_explicit=forecast_years, return_sequence=True))

		for i in range(len(rules_attributes)):
			rules_attributes[i].weight2020 = to_percentage_strings(hyper_results[self.update_hyper_from + 1]['wts_forecast_from'][i])
			rules_attributes[i].weight_init_normalized = to_percentage_strings(init_weights_normalized[i])

		self.pAGI_hyper = hyper_results
		self.pAGI_2036_hyper = to_percentage_strings(hyper_results[2036]['p_forecast_to_hyper'])

	def create_plot(self):
		xs = self.pAGI_hyper.keys()
		ys = [v['p_forecast_to_hyper'] for v in self.pAGI_hyper.values()]
		self.hyper_cdf_iterable = zip(xs,to_percentage_strings(ys))

		xs = [datetime(x, 1, 1) for x in xs]
		self.plot_hyper = plot_helper(xs,ys)

class HashableDict(dict):
	def hash_value(self, value):
		if isinstance(value, dict):
			return self.hash_dict_helper(value)
		elif isinstance(value, set):
			return frozenset(value)
		elif isinstance(value, list):
			return tuple(value)
		else:
			# Assume any other values are hashable types
			return value.__hash__()

	def hash_dict_helper(self, dictionary):
		return hash((frozenset(dictionary), tuple(self.hash_value(v) for v in dictionary.values())))

	def __hash__(self):
		return self.hash_dict_helper(self)

# We use an LRU dictionary instead of the @lru_cache decorator
# because we would need to pass in the object `form`, which is not hashable
# Each response is about 82kbytes, I set the maxsize to take up about 500mbytes.
cache = pylru.lrucache(size=int(6000))


@app.route('/', methods=['GET', 'POST'])
def show():
	form = HyperPriorForm()
	form_data_hashable = HashableDict(form.data)
	if form_data_hashable in cache:
		return cache[form_data_hashable]
	if form.validate():
		dict_x_y_pairs_for_multiline_plot = OrderedDict()  # Use for compatibility with Python version 3.7 running on Elastic Beanstalk

		result = HyperPriorResult(update_hyper_from=form.rule_out_agi_by.data)
		kwargs_all_rules = {
			'forecast_from': form.rule_out_agi_by.data,
			'virtual_successes': form.virtual_successes.data,
		}

		if form.calendar_year_filled():
			kwargs = {**kwargs_all_rules, **{
				'ftp': float(form.first_trial_probability.data),
				'regime_start': int(form.regime_start_year.data),
			}}

			def calendar_callable(year):
				return functions.four_param_framework_calendar(forecast_to=year, **kwargs)

			try:
				calendar_result = UpdateRuleResult(calendar_callable, kwargs, form.init_weight_calendar.data, form.rule_out_agi_by.data)
				dict_x_y_pairs_for_multiline_plot['Calendar-year'] = (calendar_result.xs_plot,calendar_result.ys_plot)
			except Exception:
				calendar_result = None
			result.calendar = calendar_result

		if form.researcher_filled():
			kwargs = {**kwargs_all_rules, **{
				'regime_start': form.regime_start_year.data,
				'ftp_cal_equiv': float(form.first_trial_probability.data),
				'g_exp': form.g_exp.data/100,
				'g_act': form.g_act.data/100,
			}}

			def researcher_callable(year):
				return functions.four_param_framework_researcher(forecast_to=year, **kwargs)

			try:
				researcher_result = UpdateRuleResult(researcher_callable, kwargs, form.init_weight_researcher.data, form.rule_out_agi_by.data)
				dict_x_y_pairs_for_multiline_plot['Researcher-year'] = (researcher_result.xs_plot, researcher_result.ys_plot)
			except Exception:
				researcher_result = None
			result.researcher = researcher_result



		if form.computation_relative_res_filled():
			kwargs = {**kwargs_all_rules, **{
				'spend2036': 1e9*form.comp_spending_assumption.data,
				'rel_imp_res_comp': form.relative_imp_res_comp.data,
				'g_exp': form.g_exp.data/100,
				'ftp_cal_equiv': float(form.first_trial_probability.data),
				'regime_start': 1956,
			}}

			def computation_callable(year):
				return functions.four_param_framework_comp(forecast_to=year, **kwargs)

			try:
				comp_relative_res = UpdateRuleResult(computation_callable, kwargs, form.init_weight_comp_relative_res.data, form.rule_out_agi_by.data)
				dict_x_y_pairs_for_multiline_plot['Computation vs research'] = (comp_relative_res.xs_plot,comp_relative_res.ys_plot)
			except Exception:
				comp_relative_res = None
			result.comp_relative_res = comp_relative_res



		lifetime_kwargs = {**kwargs_all_rules, **{
			'spend2036': 1e9*form.comp_spending_assumption.data,
			'regime_start': 1956}}

		evolution_kwargs = {**kwargs_all_rules, **{
			'spend2036': 1e9*form.comp_spending_assumption.data}}

		def lifetime_callable(year):
			return functions.lifetime_anchor(forecast_to=year, **lifetime_kwargs)

		def evolution_callable(year):
			return functions.evolutionary_anchor(forecast_to=year,**evolution_kwargs)

		lifetime_result = UpdateRuleResult(lifetime_callable, lifetime_kwargs, form.init_weight_lifetime.data, form.rule_out_agi_by.data)
		evolution_result = UpdateRuleResult(evolution_callable, evolution_kwargs, form.init_weight_evolution.data, form.rule_out_agi_by.data)

		result.lifetime = lifetime_result
		result.evolution = evolution_result

		dict_x_y_pairs_for_multiline_plot['Lifetime'] = (lifetime_result.xs_plot, lifetime_result.ys_plot)
		dict_x_y_pairs_for_multiline_plot['Evolution'] = (evolution_result.xs_plot, evolution_result.ys_plot)

		result.plot_multiline = plot_helper_multiline(dict_x_y_pairs_for_multiline_plot)

		agi_impossible = UpdateRuleResult(None, {'name': 'impossible'}, form.init_weight_agi_impossible.data, form.rule_out_agi_by.data)
		agi_impossible.p2036 = to_percentage_strings(0)
		result.agi_impossible = agi_impossible
		if form.initial_weights_filled():
			try:
				result.update_hyper_prior()
				result.create_plot()
			except Exception:
				pass
		response_page = render_template('index.html', form=form, result=result)
	else:
		response_page = render_template('index.html', form=form, result=None)
	cache[form_data_hashable] = response_page
	return response_page

def generate_years_to_forecast(start_end, fine_interval=3, coarse_interval=15, force_include=()):
	start, end = start_end
	forecast_years = set(force_include)

	if start<2036:
		forecast_years.add(2036)
		for i in np.arange(start+1, 2037, fine_interval):
			forecast_years.add(i)
		for i in np.arange(2037, end-1, coarse_interval):
			forecast_years.add(i)
	else:
		for i in np.arange(start+1, end-1, coarse_interval):
			forecast_years.add(i)

	forecast_years.add(end)

	forecast_years = sorted(list(forecast_years))

	return forecast_years

if __name__ == "__main__":
	app.run()
