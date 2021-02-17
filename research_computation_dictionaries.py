import numpy as np
from functools import lru_cache

computation_prices = {
	1800: 10 ** -6,
	1956: 10 ** -6,
	1970: 10 ** -7,
	2000: 10 ** -12,
	2008: 10 ** -16,
	2020: 10 ** -17,
	2036: 10 ** -19,
}

# Interpolate computation prices between 2020 and 2036. Results are spaced evenly in log-space.
log_units_total = np.log(computation_prices[2036]) - np.log(computation_prices[2020])
log_units_per_year = log_units_total / 16
for year in range(2021, 2036):
	distance_to_2020 = year - 2020
	log_price = np.log(computation_prices[2020]) + distance_to_2020 * log_units_per_year
	computation_prices[year] = np.exp(log_price)

# Extrapolate computation prices based on a halving time of 2.5 years after 2036
for year in range(2037, 2150):
	distance_to_2036 = year - 2036
	computation_prices[year] = computation_prices[2036] / 2 ** (distance_to_2036 / 2.5)


@lru_cache()
def generate_spending_dict(spend2036, spend2020=10 ** 6.7):
	"""
	Extrapolate spending based on values for 2020 and 2036, in an inverse quadratic model.
	"""

	def quadratic(year):
		# Convert to orders of magnitude
		OOMspend2036 = np.log10(spend2036)
		OOMspend2020 = np.log10(spend2020)

		# Do maths with orders of magnitude
		b = (OOMspend2036 - OOMspend2020) / ((2036 - 2020) ** 2)
		OOMresult = OOMspend2036 - b * (2036 - year) ** 2

		return 10 ** OOMresult

	spending_dict_out = {}

	# 1956 always has the same value
	spending_dict_out[1956] = 10 ** 1

	for year in range(2020, 2037):
		spending_dict_out[year] = quadratic(year)

	for year in range(2037, 2101):
		spending_dict_out[year] = spend2036

	return spending_dict_out


@lru_cache()
def generate_named_spending_dict(name):
	if name == 'central':
		# geometric interpolation between conservative and aggressive, anchoring off the value for 2036
		aggressive = generate_spending_dict(10 ** 11)
		conservative = generate_spending_dict(10 ** 8)

		central = {}
		for key in aggressive:
			OOMconservative = np.log10(conservative[key])
			OOMaggressive = np.log10(aggressive[key])

			OOMfactor = (9 - 8) / (11 - 8)
			central[key] = 10 ** (OOMconservative + (OOMaggressive - OOMconservative) * OOMfactor)
		return central

	elif name == 'aggressive':
		return generate_spending_dict(10 ** 11)

	elif name == 'conservative':
		return generate_spending_dict(10 ** 8)

@lru_cache()
def generate_cumulative_researcher_years_dict(growth_to_2036, growth_after_2036, regime_start):
	"""
	Assume there were initially 10 researchers at the regime start-time.
	Assume there were initially 10/g_act cumulative researcher-years at the regime start-time.
	Calculate how the number of cumulative researcher-years grows over time. (Note: it doesn't grow at the same rate as the number
	of researchers. You have to calculate a running total of the cumulative researcher-years over time.)
	"""

	researchers = {regime_start: 10}

	cumulative_research_years = {regime_start: 10 / growth_to_2036}

	for year in range(regime_start + 1, 2037):
		researchers[year] = researchers[year - 1] * (1 + growth_to_2036)

	for year in range(2037, 2101):
		researchers[year] = researchers[year - 1] * (1 + growth_after_2036)

	for year in range(regime_start + 1, 2101):
		cumulative_research_years[year] = cumulative_research_years[year - 1] + researchers[year]

	return cumulative_research_years
