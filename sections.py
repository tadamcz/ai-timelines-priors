from columnar import columnar
import functions
import numpy as np
from sigfig import round as round_lib
import pandas as pd
from copy import deepcopy

pd.options.display.width = 0
pd.options.display.max_colwidth = 75

def round(x,*args,**kwargs):
	if isinstance(x,(float,int)):
		return round_lib(x,*args,**kwargs)
	if isinstance(x,(list,np.ndarray)):
		return [round_lib(i,*args,**kwargs) for i in x]
	else:
		raise TypeError("Must be numeric or list/array of numerics")


def section4_2():
	print("Section 4.2")
	print_output = []
	return_output = []

	for ftp in [1/50,1/100,1/200,1/300,1/500,1/1000,1/2000,1/3000]:
		pragi = functions.fourParamFrameworkCalendar(ftp=ftp, regime_start=1956, forecast_from=2020, forecast_to=2036, virtual_successes=1)
		pragi_perc = round(pragi * 100, 2)
		pragi_str = round(pragi*100, 2,type=str)

		return_output.append(pragi_perc)
		print_output.append(['1/'+str(int(1/ftp))] + [str(pragi_str)+'%'])

	print(columnar(print_output,no_borders=True))
	return return_output


def section5_2_2():
	print("Section 5.2.2")
	print_output = []
	return_output = []
	vs = [.1,.5,1,2,10]
	print_output.append(['Number of virtual successes']+vs)

	pr_after_50 = [functions.generalized_laplace(trials=50, failures=50, virtual_successes=i, ftp=1/100) for i in vs]
	return_output.append(pr_after_50)
	print_output.append(['After 50 failures']+['1/'+str(int(1/x)) for x in pr_after_50])

	pr_after_100 = [functions.generalized_laplace(trials=100, failures=100, virtual_successes=i, ftp=1/100) for i in vs]
	return_output.append(pr_after_100)
	print_output.append(['After 100 failures']+['1/'+str(int(1/x)) for x in pr_after_100])

	print(columnar(print_output,no_borders=True))
	return return_output



def section5_2_3_t1():
	print("Section 5.2.3, Table 1")
	print_output = []
	return_output = []
	header = ["Number of virtual successes","ftp 1/100","ftp 1/300","ftp 1/1000"]
	print_output.append(header)
	for vs in [.1,.25,.5,1,2,4,10]:
		pragi = [
			functions.fourParamFrameworkCalendar(ftp=1 / 100, virtual_successes=vs),
			functions.fourParamFrameworkCalendar(ftp=1 / 300, virtual_successes=vs),
			functions.fourParamFrameworkCalendar(ftp=1 / 1000, virtual_successes=vs)]
		pragi = [round(x*100,2) for x in pragi]
		return_output.append(pragi)

		row = [vs] + [str(x)+'%' for x in pragi]
		print_output.append(row)
	print(columnar(print_output,no_borders=True))
	return return_output

def section5_2_3_t2(virtual_successes,ftps,header):
	print_output = []
	return_output = []

	print_output.append(header)

	prAGIfirst50 = [functions.forecast_generalized_laplace(failures=0,forecast_years=50,virtual_successes=virtual_successes,ftp=ftp) for ftp in ftps]
	prAGIfirst100 = [functions.forecast_generalized_laplace(failures=0, forecast_years=100, virtual_successes=virtual_successes, ftp=ftp) for ftp in ftps]
	prAGI2036 = [functions.fourParamFrameworkCalendar(ftp, virtual_successes=virtual_successes) for ftp in ftps]

	prAGIfirst100 = [i*100 for i in prAGIfirst100]
	prAGIfirst50 =  [i*100 for i in prAGIfirst50]
	prAGI2036 = [i*100 for i in prAGI2036]

	prAGIfirst50_perc = round(prAGIfirst50,2)
	prAGIfirst100_perc = round(prAGIfirst100,2)
	prAGI2036_perc = round(prAGI2036,2)

	prAGIfirst50_str = round(prAGIfirst50, 2, type=str)
	prAGIfirst100_str = round(prAGIfirst100, 2,type=str)
	prAGI2036_str = round(prAGI2036, 2, type=str)

	return_output.append(prAGIfirst50_perc)
	return_output.append(prAGIfirst100_perc)
	return_output.append(prAGI2036_perc)

	print_output.append(["pr(AGI in first 50 years)"]+[p+'%' for p in prAGIfirst50_str])
	print_output.append(["pr(AGI in first 100 years)"]+[p+'%' for p in prAGIfirst100_str])
	print_output.append(["pr(AGI by 2036 | no AGI by 2020)"]+[p+'%' for p in prAGI2036_str])

	print(columnar(print_output,no_borders=True))
	return return_output

def section5_2_3_t2_left():
	print("Section 5.2.3, Table 2")
	print("Left side, 0.5 virtual successes")
	return section5_2_3_t2(virtual_successes=0.5,ftps=[1/50,1/100,1/300,1/1000],header=["First-trial probability","1/50","1/100","1/300","1/1000"])

def section5_2_3_t2_right():
	print("Section 5.2.3, Table 2")
	print("Right side, 1 virtual success")
	return section5_2_3_t2(virtual_successes=1,ftps=[1/100,1/300,1/1000],header=["First-trial probability","1/100","1/300","1/1000"])


def section5_3_t1():
	print("Section 5.3 Table 1")
	headers = ["First-trial probability","2000","1956","1945","1650","5000 BC"]

	print_output = [headers]
	return_output =[]

	regime_starts = [2000,1956,1945,1650,-5000]
	for ftp in [1/50,1/100,1/300,1/1000]:
		prAGI = [functions.fourParamFrameworkCalendar(ftp, regime_start=r) for r in regime_starts]
		prAGI = [i*100 for i in prAGI]

		prAGI_perc = round(prAGI,2)
		prAGI_str = round(prAGI, 2, type=str)

		print_output.append(['1/'+str(int(1/ftp))]+[str(i)+'%' for i in prAGI_str])
		return_output.append(prAGI_perc)
	print(columnar(print_output,no_borders=True))
	return return_output

def section5_3_t2():
	print("Section 5.3 Table 2")

	header1 = [""]+["Trials 5000 BC-1956"]*4
	header2 = [""]*5
	header3 = ["ftp", 168, 220,139,0]
	print_output = [header1,header2,header3]
	return_output = []

	trials_before_1956 = [168,220,139,0]
	for ftp in [1/2,1/100,1/300,1/1000]:
		prAGI = [functions.fourParamFrameworkCalendar(ftp, regime_start=1956 - i) for i in trials_before_1956]
		prAGI = [i * 100 for i in prAGI]

		prAGI_perc = round(prAGI, 2)
		prAGI_str = round(prAGI, 2, type=str)

		print_output.append(['1/' + str(int(1 / ftp))] + [str(i) + '%' for i in prAGI_str])
		return_output.append(prAGI_perc)
	print(columnar(print_output, no_borders=True))
	return return_output

def section6_1_2():
	print("Section 6.1.2")
	header = ["","g_act=3%","g_act=7%","g_act=11%","g_act=16%","g_act=21%"]
	print_output = [header]
	return_output = []

	for ftp_cal_equiv in [1/50,1/100,1/300,1/1000]:
		prAGI = [functions.fourParamFrameworkResearcher(g_act=g_act, g_exp=4.3/100, ftp_cal_equiv=ftp_cal_equiv) \
																for g_act in [3/100,7/100,11/100,16/100,21/100]]

		prAGI = [i * 100 for i in prAGI]

		prAGI_perc = round(prAGI, 2)
		prAGI_str = round(prAGI, 2, type=str)

		row_name = 'ftp_cal = 1/' + str(int(1 / ftp_cal_equiv))
		print_output.append([row_name] + [str(i) + '%' for i in prAGI_str])
		return_output.append(prAGI_perc)

	print(columnar(print_output, no_borders=True))
	return return_output

## From here on, I only create the disp output (not the return output), since Tom at OpenPhil has asked me to directly edit the document.
def section6_2_comp_from_research(biggest_spends_method):
	print("\nSection 6.2. Computation trial definition, with ftp from research, table with biggest_spends_method:",biggest_spends_method)
	columns = [1,5,10]
	df = pd.DataFrame(columns=columns)

	for ftp_cal_equiv in [1 / 50, 1 / 100, 1 / 300, 1 / 1000, 1/3000]:
		dict_comprehension = {rel_imp_res_comp:
								  functions.fourParamFrameworkComp(rel_imp_res_comp=rel_imp_res_comp,
																   regime_start_year=1956,
																   forecast_to_year=2036,
																   biggest_spends_method=biggest_spends_method,
																   ftp_cal_equiv=ftp_cal_equiv)
							  for rel_imp_res_comp in columns}
		dict_comprehension = {k:round(v*100,2,type=str)+"%" for k,v in dict_comprehension.items()}
		row_name = 'ftp_cal = 1/' + str(int(1 / ftp_cal_equiv))
		row = pd.Series(data=dict_comprehension, name=row_name)
		df = df.append(row)

	df.columns = ["X="+str(x) for x in columns]  # rename the columns
	print(df)

def section6_2_from_research():
	section6_2_comp_from_research('conservative')
	section6_2_comp_from_research('aggressive')

def section6_2_from_biology():
	print("\nSection 6.2. Computation trial definition, with ftp from biology")
	columns = ['lifetime','evolutionary']
	df = pd.DataFrame(columns=columns)

	for method in ['conservative','aggressive']:
		datadict = {'lifetime':functions.lifetimeAnchor(method), 'evolutionary':functions.evolutionaryAnchor(method)}
		datadict = {k:round(v*100,2,type=str)+"%" for k,v in datadict.items()}
		row = pd.Series(data=datadict,name=method)
		df = df.append(row)
	print(df)

def section6_2_log_uniform():
	print("\nSection 6.2: Log-uniform compute model")

	c = float(functions.logUniform('conservative'))
	a = float(functions.logUniform('aggressive'))
	print('P(AGI by 2036) with conservative spend estimate',round(c*100,2,type=str)+"%")
	print('P(AGI by 2036) with aggressive spend estimate', round(a*100,2,type=str)+"%")

def section6_2_3(virtual_successes=1, disp=True):
	if disp: print("Section 6.2.3")
	df = pd.DataFrame(columns=['lifetime','evolutionary','log-uniform','from research','eq. wt. avg.'])

	overall = []
	for method in ['conservative', 'aggressive']:
		datadict = {'lifetime':functions.lifetimeAnchor(method,virtual_successes=virtual_successes),
					'evolutionary':functions.evolutionaryAnchor(method,virtual_successes=virtual_successes),
					'log-uniform':functions.logUniform(method),
					'from research':functions.fourParamFrameworkComp(rel_imp_res_comp=5,
																	 ftp_cal_equiv=1 / 300,
																	 biggest_spends_method=method,
																	 virtual_successes=virtual_successes)
					}
		average = float(np.average([value for value in datadict.values()]))
		overall.append(average)
		datadict['eq. wt. avg.'] = average
		datadict = {k:float(v) for k,v in datadict.items()}

		datadict = {k: round(v * 100, 2, type=str) + "%" for k, v in datadict.items()}
		row = pd.Series(data=datadict, name=method)
		df = df.append(row)

	if disp: print(df)
	overall = np.average(overall)
	if disp: print("\nAll things considered average:", round(float(overall) * 100, 2, type=str) + "%")
	return overall


def section6_3_1_helper_research(g_act, ftp_cal_equiv, g_exp, rowname, df):
	left= functions.fourParamFrameworkResearcher(g_act=g_act, ftp_cal_equiv=ftp_cal_equiv, g_exp=g_exp)
	right = functions.fourParamFrameworkResearcher(g_act=g_act, ftp_cal_equiv=ftp_cal_equiv, g_exp=g_exp, virtual_successes=0.5)
	left = round(left * 100, 2, type=str) + "%"
	right = round(right * 100, 2, type=str) + "%"
	return df.append(pd.Series(name=rowname,data={'1 VS':left, '0.5 VS':right}))

def section6_3_1_helper_comp(ftp_cal_equiv, rel_imp_res_comp, biggest_spends_method, rowname, df):

	if biggest_spends_method == '50/50':
		left = 1/2 * (functions.fourParamFrameworkComp(ftp_cal_equiv=ftp_cal_equiv,
													   rel_imp_res_comp=rel_imp_res_comp,
													   biggest_spends_method='aggressive')+
					  functions.fourParamFrameworkComp(ftp_cal_equiv=ftp_cal_equiv,
													   rel_imp_res_comp=rel_imp_res_comp,
													   biggest_spends_method='conservative'))

		right = 1 / 2 * (functions.fourParamFrameworkComp(ftp_cal_equiv=ftp_cal_equiv,
														  rel_imp_res_comp=rel_imp_res_comp,
														  biggest_spends_method='aggressive',
														  virtual_successes=0.5) +
						functions.fourParamFrameworkComp(ftp_cal_equiv=ftp_cal_equiv,
														 rel_imp_res_comp=rel_imp_res_comp,
														 biggest_spends_method='conservative',
														 virtual_successes=0.5))

	else:
		left = functions.fourParamFrameworkComp(ftp_cal_equiv=ftp_cal_equiv,
												rel_imp_res_comp=rel_imp_res_comp,
												biggest_spends_method=biggest_spends_method)

		right =functions.fourParamFrameworkComp(ftp_cal_equiv=ftp_cal_equiv,
												rel_imp_res_comp=rel_imp_res_comp,
												biggest_spends_method=biggest_spends_method,
												virtual_successes=0.5)

	left = round(left * 100, 2, type=str) + "%"
	right = round(right * 100, 2, type=str) + "%"
	return df.append(pd.Series(name=rowname, data={'1 VS': left, '0.5 VS': right}))




def section6_3_1_virtual_succ():
	print("\nSection 6.3.1, virtual successes")
	df = pd.DataFrame(columns=['1 VS','0.5 VS'])
	g_exp = 4.3/100

	rowname = "Researcher-year, low"
	g_act = 7 / 100
	ftp_cal_equiv = 1 / 1000
	df = section6_3_1_helper_research(g_act, ftp_cal_equiv, g_exp, rowname, df)

	rowname = "Researcher-year, middle"
	g_act = 11 / 100
	ftp_cal_equiv = 1 / 300
	df = section6_3_1_helper_research(g_act, ftp_cal_equiv, g_exp, rowname, df)

	rowname = "Researcher-year, high"
	g_act = 16 / 100
	ftp_cal_equiv = 1 / 100
	df = section6_3_1_helper_research(g_act, ftp_cal_equiv, g_exp, rowname, df)

	rowname = 'Computation, low'
	ftp_cal_equiv = 1 / 1000
	rel_imp_res_comp = 10
	biggest_spends_method = 'conservative'
	df = section6_3_1_helper_comp(ftp_cal_equiv, rel_imp_res_comp, biggest_spends_method, rowname, df)

	rowname = 'Computation, central'
	left = 0.5*(functions.lifetimeAnchor('conservative',virtual_successes=1)+functions.lifetimeAnchor('aggressive',virtual_successes=1))
	right = 0.5 * (functions.lifetimeAnchor('conservative', virtual_successes=.5) + functions.lifetimeAnchor('aggressive', virtual_successes=.5))
	left = round(float(left) * 100, 2, type=str) + "%"
	right = round(float(right) * 100, 2, type=str) + "%"
	df = df.append(pd.Series(name=rowname, data={'1 VS': left, '0.5 VS': right}))


	rowname = 'Computation, high'
	ftp_cal_equiv = 1 / 300
	rel_imp_res_comp = 1
	biggest_spends_method = 'aggressive'
	df = section6_3_1_helper_comp(ftp_cal_equiv, rel_imp_res_comp, biggest_spends_method, rowname, df)

	rowname = 'Computation, central, bracketed weigh. avg.'
	left = section6_2_3(virtual_successes=1, disp=False)
	right = section6_2_3(virtual_successes=.5, disp=False)
	left = round(float(left) * 100, 2, type=str) + "%"
	right = round(float(right) * 100, 2, type=str) + "%"
	df = df.append(pd.Series(name=rowname, data={'1 VS': left, '0.5 VS': right}))

	rowname = 'Computation, high, bracketed weigh. avg.'
	left = 0.5*(functions.fourParamFrameworkComp(rel_imp_res_comp=1,
												 ftp_cal_equiv=1/300,
												 biggest_spends_method='aggressive',
												 virtual_successes=1) +
				functions.logUniform('aggressive'))
	right = 0.5*(functions.fourParamFrameworkComp(rel_imp_res_comp=1,
												  ftp_cal_equiv=1/300,
												  biggest_spends_method='aggressive',
												  virtual_successes=.5) +
				functions.logUniform('aggressive'))
	left = round(float(left) * 100, 2, type=str) + "%"
	right = round(float(right) * 100, 2, type=str) + "%"
	df = df.append(pd.Series(name=rowname, data={'1 VS': left, '0.5 VS': right}))


	print(df)

def section6_3_1_regimestart():
	print("\nSection 6.3.1, Researcher-year trial definition with regime start-time of 2000")
	central = functions.fourParamFrameworkResearcher(g_act=11/100,ftp_cal_equiv=1/300,regime_start=2000,g_exp=4.3/100)
	high = functions.fourParamFrameworkResearcher(g_act=16/100,ftp_cal_equiv=1/100,regime_start=2000,g_exp=4.3/100)

	central,high = round(float(central) * 100, 2, type=str) + "%",round(float(high) * 100, 2, type=str) + "%"

	print("Central:",central,"High:",high)

def section7_2_1():
	print("\nSection 7.2.1 Effect of hyper prior updates: first-trial probability")

	df = pd.DataFrame(columns=['pr2036static','pr2036hyper'])  # forces the columns to appear in this order

	row_inputs = [
		[{'ftp': 1/100}, {'ftp':1/1000}],
		[{'ftp': 1/10}, {'ftp': 1/100}],
		[{'ftp': 1/10}, {'ftp': 1/100}, {'ftp': 1/1000}],
		[{'ftp': 1/1000}, {'ftp': 1/10000}]
	]

	for input_list in row_inputs:
		rowname = str([ToFractionStrings(i) for i in input_list])

		for rule in input_list: # the same for every row in the table
			rule['name'] = 'calendar'
			rule['regime_start'] = 1956
			rule['forecast_from'] = 2020

		initial_weights = [1] * len(input_list)

		datadict = functions.hyperPrior(rules=input_list,initial_weights=initial_weights)

		datadict = ToPercentageStrings(datadict)

		df = df.append(pd.Series(name=rowname, data=datadict))

	print(df)

def appendix8():
	print("\nAppendix 8: using a hyper-prior on different trial definitions")

	df = pd.DataFrame(columns=['pr2036static','pr2036hyper'])  # forces the columns to appear in this order

	# First rule is the same in all rows
	rule1 = {'name':'calendar',
			 'regime_start':1956,
			 'ftp': 1/300
			 }

	row_inputs = [
		[rule1,
			{'name':'res-year',
			 'ftp_cal_equiv':1/300,
			 'g_exp':4.3/100,
			 'g_act':11/100,
			 'regime_start':1956}
		 ],

		[rule1,
		 	{'name': 'res-year',
			 'ftp_cal_equiv':1/300,
			 'g_exp':4.3/100,
			 'g_act': 21 / 100,
			 'regime_start': 1956},
		],

		[rule1,
		 	{'name': 'res-year',
			 'ftp_cal_equiv':1/300,
			 'g_exp':4.3/100,
			 'g_act': 21 / 100,
			 'regime_start': 2000}
		],

		[rule1,
		 	{'name': 'computation',
			 'ftp_cal_equiv': 1 / 300,
			 'g_exp': 4.3 / 100,
			 'rel_imp_res_comp': 5,
			 'regime_start': 1956,
			 'biggest_spends_method':'aggressive'},
		],

		[rule1,
		 	{'name': 'computation',
			 'ftp_cal_equiv': 1/300,
			 'g_exp':4.3/100,
			 'rel_imp_res_comp':1,
		 	 'regime_start': 1956,
			 'biggest_spends_method':'aggressive'}
		],

		[rule1,
		 	{'name': 'computation',
			 'biohypothesis':'lifetime',
			 'regime_start': 1956,
			 'biggest_spends_method':'aggressive'},
		],

		[rule1,
		 	{'name': 'computation',
			 'biohypothesis': 'evolution'}
		]
	]


	for input_list in row_inputs:
		rowname = str(ToFractionStrings(input_list[1]))

		initial_weights = [1] * len(input_list)

		datadict = functions.hyperPrior(input_list,initial_weights)

		# We only display the second weight
		datadict['wt2020'] = datadict['wts2020'][1]
		del datadict['wts2020']

		datadict = ToPercentageStrings(datadict)

		df = df.append(pd.Series(name=rowname, data=datadict))

	print(df)

def section7_3():
	print("\n7.3 Allow some probability that AGI is impossible")
	df = pd.DataFrame(columns=['pr2036 calendar-year','pr2036 20% impossible'])  # forces this order

	input_ftps = [
		1 / 1000,
		1 / 300,
		1 / 200,
		1 / 100,
		1 / 50,
		1 / 20,
		1 / 10
	]

	rows = []
	for input_ftp in input_ftps:
		rows.append(
			[
				{'name':'calendar', # Rule 1
				 'regime_start':1956,
				  'ftp': input_ftp},

				 {'name':'impossible'}  # Rule 2
			]
		)

	for row in rows:
		rowname = ToFractionStrings(row[0]['ftp'])

		datadict = functions.hyperPrior(row, initial_weights=[.8, .2])
		datadict['pr2036 20% impossible'] = datadict.pop('pr2036hyper')
		datadict['wt2020'] = datadict.pop('wts2020')[1]
		del datadict['pr2036static']

		datadict['pr2036 calendar-year'] = functions.fourParamFrameworkCalendar(row[0]['ftp'])

		datadict = ToPercentageStrings(datadict)

		df = df.append(pd.Series(name=rowname, data=datadict))

	print(df)

def appendix9():
	print("\nAppendix 9: AGI Impossible")
	df = pd.DataFrame(columns=['pr2036 No Hyper','pr2036 20% impossible'])  # forces order

	rule2 = {'name':'impossible'}  # the same in every row

	row_inputs = [
		[
			{'name': 'calendar',
			 'ftp': 1/300,
			 'regime_start':1956},

		rule2],

		[
			{'name': 'res-year',
			 'ftp_cal_equiv':1/300,
			 'g_act': 11 / 100,
			 'g_exp': 4.3/100,
			 'regime_start': 1956},

		rule2],

		[
			{'name': 'res-year',
			 'ftp_cal_equiv':1/300,
			 'g_act': 21 / 100,
			 'g_exp': 4.3/100,
			 'regime_start': 1956},

		rule2],

		[
			{'name': 'res-year',
			 'ftp_cal_equiv':1/300,
			 'g_act': 21 / 100,
			 'g_exp': 4.3/100,
			 'regime_start': 2000},

		rule2],

		[
			{'name': 'computation',
			 'biggest_spends_method':'aggressive',
			 'ftp_cal_equiv':1/300,
			 'g_exp': 4.3 / 100,
			 'rel_imp_res_comp': 5,
			 'regime_start': 1956},

		rule2],

		[
			{'name': 'computation',
			 'biggest_spends_method':'aggressive',
			 'ftp_cal_equiv':1/300,
			 'g_exp': 4.3 / 100,
			 'rel_imp_res_comp': 1,
			 'regime_start': 1956},

		rule2],

		[
			{'name': 'computation',
			 'biggest_spends_method':'aggressive',
			 'biohypothesis': 'lifetime',
			 'regime_start': 1956},

		rule2],

		[
			{'name': 'computation',
			 'biggest_spends_method':'aggressive',
			 'biohypothesis': 'evolution'},

		rule2],

		[
			{'name': 'computation-loguniform',
			 'biggest_spends_method':'aggressive'},

		rule2],
	]

	for row in row_inputs:
		rowname = str(row)

		datadict = functions.hyperPrior(row, initial_weights=[0.8,0.2])
		del datadict['pr2036static']

		datadict['pr2036 20% impossible'] = datadict.pop('pr2036hyper')
		datadict['wt2020'] = datadict.pop('wts2020')[1]

		datadict['pr2036 No Hyper'] = functions.hyperPrior(row, initial_weights=[1,0])['pr2036hyper']

		datadict = ToPercentageStrings(datadict)

		df = df.append(pd.Series(name=rowname, data=datadict))

	print(df)

def section_8():
	print("\nSection 8: All-things considered judgement")

	row_inputs = [
		{ # Low
			'rules':
				[
					{# Rule 1
						'name':'calendar',
						'virtual_successes': 0.5,
						'regime_start':1956,
						'ftp':1/1000,

					},

					{# Rule 2
						'name': 'res-year',
						'virtual_successes': 0.5,
						'regime_start': 1956,
						'ftp_cal_equiv': 1 / 1000,
						'g_exp':4.3/100,
						'g_act': 7 / 100,
					},

					{# Rule 3
						'name':'impossible',
					}
				],
			'weights': [50,30,20]
		},

		{  # Central
			'rules':
				[
					{  # Rule 1
						'name': 'calendar',
						'regime_start': 1956,
						'ftp': 1 / 300,

					},

					{  # Rule 2
						'name': 'res-year',
						'regime_start': 1956,
						'ftp_cal_equiv': 1 / 300,
						'g_exp': 4.3 / 100,
						'g_act': 11 / 100,
					},

					{  # Rule 3
						'name': 'computation',
						'biggest_spends_method': 'central',
						'rel_imp_res_comp': 5,
						'g_exp': 4.3 / 100,
						'ftp_cal_equiv': 1 / 300,
						'regime_start': 1956,
					},

					{  # Rule 4
						'name': 'computation',
						'biggest_spends_method': 'central',
						'biohypothesis': 'lifetime',
						'regime_start': 1956
					},

					{  # Rule 5
						'name': 'computation',
						'biggest_spends_method': 'central',
						'biohypothesis': 'evolution'
					},

					{  # Rule 6
						'name':'impossible',
					},
				],
			'weights': [
				.30,  # Rule 1
				.30,  # Rule 2
				.05,  # Rule 3
				.10,  # Rule 4
				.15,  # Rule 5
				.10   # Rule 6
				]
		},

		{  # High
			'rules':
				[
					{  # Rule 1
						'name': 'calendar',
						'regime_start': 1956,
						'ftp': 1 / 100,
					},

					{  # Rule 2
						'name': 'calendar',
						'regime_start': 2000,
						'ftp': 1 / 100,
					},

					{  # Rule 3
						'name': 'res-year',
						'regime_start': 1956,
						'ftp_cal_equiv': 1 / 100,
						'g_exp': 4.3 / 100,
						'g_act': 11 / 100,
					},

					{  # Rule 4
						'name': 'res-year',
						'regime_start': 2000,
						'ftp_cal_equiv': 1 / 100,
						'g_exp': 4.3 / 100,
						'g_act': 11 / 100,
					},

					{  # Rule 5
						'name': 'computation',
						'biggest_spends_method': 'aggressive',
						'rel_imp_res_comp': 5,
						'regime_start': 1956,
						'g_exp': 4.3 / 100,
						'ftp_cal_equiv': 1 / 100,
					},

					{  # Rule 6
						'name': 'computation',
						'biggest_spends_method': 'aggressive',
						'biohypothesis': 'lifetime',
						'regime_start': 1956,
					},

					{  # Rule 7
						'name': 'computation',
						'biggest_spends_method': 'aggressive',
						'biohypothesis': 'evolution',
					},

					{  # Rule 8
						'name': 'impossible',
					},
				],
			'weights': [
				.10*.15,  # Rule 1
				.10*.85,  # Rule 2
				.40*.15,  # Rule 3
				.40*.85,  # Rule 4
				.10,      # Rule 5
				.10,      # Rule 6
				.20,      # Rule 7
				.10       # Rule 8
				]
		}
	]

	for row in row_inputs:
		datadict = functions.hyperPrior(row['rules'], initial_weights=row['weights'])
		print(ToPercentageStrings(datadict['pr2036hyper']))


def ToPercentageStrings(input):
	if isinstance(input, dict):
		input = deepcopy(input)
		for k, v in input.items():
			if isinstance(v, (list, np.ndarray)):
				input[k] = [round(float(i) * 100, 2, type=str) + "%" for i in v]
			if isinstance(v, np.float):
				input[k] = round(float(v) * 100, 2, type=str) + "%"
		return input
	if isinstance(input, (float, int)):
		input = float(input)
		return round(input * 100, 2, type=str) + "%"

def ToFractionStrings(input):
	if isinstance(input, dict):
		input = deepcopy(input)
		for k, v in input.items():
			if isinstance(v, (list, np.ndarray)):
				input[k] = ['1/' + str(int((1 / float(i)))) for i in v]
			if isinstance(v, np.float):
				input[k] = '1/' + str(int((1 / float(v))))
		return input
	if isinstance(input, (float, int)):
		input = float(input)
		return '1/'+str(int((1 /input)))