from flask import Flask, render_template
from wtforms import FloatField, IntegerField, SelectField, StringField
from wtforms import validators
from datetime import datetime
import fractions

from flask_wtf import FlaskForm

import functions
from sections import to_percentage_strings, to_fraction_strings, round_sig
import matplotlib.pyplot as plt
import numpy as np
import mpld3

app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False  # not needed, there are no user accounts

probability_validator = validators.number_range(0, 1, message='Probability must be between 0 and 1')
positive_validator = validators.number_range(min=0, max=None, message='Cannot be a negative number')


class HyperPriorForm(FlaskForm):
	first_trial_probability = StringField()
	virtual_successes = FloatField(validators=[positive_validator], label='Virtual successes (used in all update rules)')
	regime_start_year = IntegerField(validators=[validators.Optional()], label='Regime start year (used in all update rules except the evolutionary anchor)')

	g_exp = FloatField(validators=[validators.Optional()], label='Typical growth for STEM researchers')
	g_act = FloatField(validators=[validators.Optional()], label='Growth of AI researchers')

	relative_imp_res_comp = IntegerField(validators=[validators.Optional(), positive_validator], label='One doubling in the number of researchers is equivalent to X doublings in computation')

	comp_spending_assumption = SelectField(choices=('conservative','central','aggressive'), label='Assumption about future spending on computation (used in all computation trial definitions)')

	init_weight_comp_relative_res = FloatField(validators=[positive_validator], label='Computation trial definition: relative importance of research and computation')
	init_weight_calendar = FloatField(validators=[validators.Optional(), positive_validator], label='Calendar-year trial definition')
	init_weight_researcher = FloatField(validators=[validators.Optional(), positive_validator], label='Researcher-year trial definition')
	init_weight_lifetime = FloatField(validators=[validators.Optional(), positive_validator], label='Computation trial definition: lifetime anchor')
	init_weight_evolution = FloatField(validators=[validators.Optional(), positive_validator], label='Computation trial definition: evolutionary anchor')
	init_weight_agi_impossible = FloatField(validators=[validators.Optional(), positive_validator], label='AGI is impossible')

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

		if self.first_trial_probability.data is not None:
			try:
				self.first_trial_probability.data = fractions.Fraction(self.first_trial_probability.data)
			except (SyntaxError, ValueError):
				self.first_trial_probability.errors.append("Could not interpret as a decimal or fraction")
				validity = False

		return validity





	def check_initial_weights(self):
		"""
		Doesn't work right now, not used
		"""
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

		self.validate()



class UpdateRuleResult:
	def __init__(self, _callable, input_args, init_weight):
		if _callable:
			self.p2036 = to_percentage_strings(_callable(2036))
			self.create_graph(_callable)
		self.input_args = input_args
		self.init_weight = init_weight

	def create_graph(self, _callable, x_from_to=(2020, 2036), is_date=True):
		fig, ax = plt.subplots()
		fig.set_size_inches(5, 2)
		x_from, x_to = x_from_to
		xs = np.arange(x_from, x_to, 3)  # Only every 3, because this has an effect on performance
		ys = [_callable(i) for i in xs]
		if is_date:
			xs = [datetime(x, 1, 1) for x in xs]
		ax.plot(xs, ys)
		ax.set_ylabel("Pr(AGI)")
		ax.set_xlabel("Year")
		self.plot = mpld3.fig_to_html(fig)
		plt.close(fig)  # Otherwise the matplotlib object stays in memory forever


class HyperPriorResult:
	def __init__(self):
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

		# consider refactoring around these if statements. would require some change to how hyper_prior takes args.
		if self.calendar:
			rules_attributes.append(self.calendar)
			self.calendar.input_args['name'] = 'calendar'

		if self.researcher:
			rules_attributes.append(self.researcher)
			self.researcher.input_args['name'] = 'res-year'

		if self.comp_relative_res:
			self.comp_relative_res.input_args['regime_start'] = self.comp_relative_res.input_args['regime_start_year']
			rules_attributes.append(self.comp_relative_res)
			self.comp_relative_res.input_args['name'] = 'computation'

		if not rules_attributes:
			return

		for rule in rules_attributes:
			rules_dicts.append(rule.input_args)
			weights.append(rule.init_weight)

		hyper_results = functions.hyper_prior(rules_dicts, weights)

		for i in range(len(rules_attributes)):
			rules_attributes[i].weight2020 = to_percentage_strings(hyper_results['wts2020'][i])

		self.p2036hyper = to_percentage_strings(hyper_results['pr2036hyper'])


@app.route('/', methods=['GET', 'POST'])
def show():
	form = HyperPriorForm()
	if form.validate():
		result = HyperPriorResult()

		if form.calendar_year_filled():
			kwargs = {
				'ftp': form.first_trial_probability.data,
				'regime_start': int(form.regime_start_year.data),
				'virtual_successes': form.virtual_successes.data,
			}

			def calendar_callable(year):
				return functions.four_param_framework_calendar(forecast_to=year, **kwargs)

			calendar_result = UpdateRuleResult(calendar_callable, kwargs, form.init_weight_calendar.data)
			result.calendar = calendar_result

		if form.researcher_filled():
			kwargs = {
				'regime_start': form.regime_start_year.data,
				'ftp_cal_equiv': form.first_trial_probability.data,
				'g_exp': form.g_exp.data,
				'g_act': form.g_act.data,
				'virtual_successes': form.virtual_successes.data,
			}

			def researcher_callable(year):
				return functions.four_param_framework_researcher(forecast_to=year, **kwargs)

			researcher_result = UpdateRuleResult(researcher_callable, kwargs, form.init_weight_researcher.data)
			result.researcher = researcher_result

		if form.computation_relative_res_filled():
			kwargs = {
				'biggest_spends_method': form.comp_spending_assumption.data,
				'rel_imp_res_comp': form.relative_imp_res_comp.data,
				'g_exp': form.g_exp.data,
				'ftp_cal_equiv': form.first_trial_probability.data,
				'regime_start_year': form.regime_start_year.data,
				'virtual_successes': form.virtual_successes.data,
			}

			def computation_callable(year):
				return functions.four_param_framework_comp(forecast_to_year=year, **kwargs)

			comp_relative_res = UpdateRuleResult(computation_callable, kwargs, form.init_weight_comp_relative_res.data)
			result.comp_relative_res = comp_relative_res

		def lifetime_callable(year):
			return functions.lifetime_anchor('aggressive', form.virtual_successes.data, form.regime_start_year.data, forecast_to_year=year)

		def evolution_callable(year):
			return functions.evolutionary_anchor(
				biggest_spends_method=form.comp_spending_assumption.data,
				virtual_successes=form.virtual_successes.data,
				forecast_to_year=year)

		lifetime_kwargs = {'name': 'computation',
						   'biggest_spends_method': form.comp_spending_assumption.data,
						   'biohypothesis': 'lifetime',
						   'regime_start': form.regime_start_year.data,
						   'virtual_successes': form.virtual_successes.data,
						   }

		evolution_kwargs = {'name': 'computation',
							'biggest_spends_method': form.comp_spending_assumption.data,
							'biohypothesis': 'evolution',
							'virtual_successes': form.virtual_successes.data,
							}
		result.lifetime = UpdateRuleResult(lifetime_callable, lifetime_kwargs, form.init_weight_lifetime.data)
		result.evolution = UpdateRuleResult(evolution_callable, evolution_kwargs, form.init_weight_evolution.data)

		agi_impossible = UpdateRuleResult(None, {'name': 'impossible'}, form.init_weight_agi_impossible.data)
		result.agi_impossible = agi_impossible
		if form.initial_weights_filled():
			result.update_hyper_prior()
		return render_template('index.html', form=form, result=result)
	else:
		return render_template('index.html', form=form, result=None)


if __name__ == "__main__":
	app.run()
