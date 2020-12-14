from columnar import columnar
import functions
import numpy as np
from sigfig import round


def section42():
	print("Section 4.2")
	print_output = []
	return_output = []

	for ftp in [1/50,1/100,1/200,1/300,1/500,1/1000,1/2000,1/3000]:
		pragi = functions.fourParamFramework(ftp=ftp, regime_start=1956, forecast_from=2020, forecast_to=2036, virtual_successes=1)
		pragi = round(pragi*100, sigfigs=2)
		return_output.append(pragi)
		print_output.append(['1/'+str(int(1/ftp))] + [str(pragi)+'%'])

	print(columnar(print_output,no_borders=True))
	return return_output


def section522():
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



def section523():
	print("Section 5.2.3")
	print_output = []
	return_output = []
	header = ["Number of virtual successes","ftp 1/100","ftp 1/300","ftp 1/1000"]
	print_output.append(header)
	for vs in [.1,.25,.5,1,2,4,10]:
		pragi = [
			functions.fourParamFramework(ftp=1/100,virtual_successes=vs),
			functions.fourParamFramework(ftp=1/300, virtual_successes=vs),
			functions.fourParamFramework(ftp=1/1000, virtual_successes=vs)]
		pragi = [round(x*100, sigfigs = 2) for x in pragi]
		return_output.append(pragi)

		row = [vs] + [str(x)+'%' for x in pragi]
		print_output.append(row)
	print(columnar(print_output,no_borders=True))
	return return_output