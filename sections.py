from columnar import columnar
import functions
import numpy as np
from sigfig import round as round_lib
import pandas as pd

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

## From here on, I only create the print output (not the return output), since Tom at OpenPhil has asked me to directly edit the document.
def section6_2_from_research_helper(biggest_spends_method):
	print("Section 6.2 from research, table with biggest_spends_method:",biggest_spends_method)
	columns = [1,5,10]
	df = pd.DataFrame(columns=columns)

	for ftp_cal_equiv in [1 / 50, 1 / 100, 1 / 300, 1 / 1000, 1/3000]:
		dict_comprehension = {relative_impact_research_compute:
								  functions.fourParamFrameworkComp(relative_impact_research_compute=relative_impact_research_compute,
																   regime_start_year=1956,
																   forecast_to_year=2036,
																   biggest_spends_method=biggest_spends_method,
																   ftp_cal_equiv=ftp_cal_equiv)
							  for relative_impact_research_compute in columns}
		dict_comprehension = {k:round(v*100,2,type=str)+"%" for k,v in dict_comprehension.items()}
		row_name = 'ftp_cal = 1/' + str(int(1 / ftp_cal_equiv))
		row = pd.Series(data=dict_comprehension, name=row_name)
		df = df.append(row)

	df.columns = ["X="+str(x) for x in columns]  # rename the columns
	print(df)

def section6_2_from_research():
	section6_2_from_research_helper('conservative')
	section6_2_from_research_helper('aggressive')

def section6_2_from_biology():
	print("Section 6.2 from biology")
	columns = ['lifetime','evolutionary']
	df = pd.DataFrame(columns=columns)

	for method in ['conservative','aggressive']:
		datadict = {'lifetime':functions.lifetimeAnchor(method), 'evolutionary':functions.evolutionaryAnchor(method)}
		datadict = {k:round(v*100,2,type=str)+"%" for k,v in datadict.items()}
		row = pd.Series(data=datadict,name=method)
		df = df.append(row)
	print(df)