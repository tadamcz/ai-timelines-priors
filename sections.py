from columnar import columnar
import functions
import numpy as np
print("Section 3.4")
output = []
for ftp in [1/50,1/100,1/200,1/300,1/500,1/1000,1/2000,1/3000]:
	pragi = functions.fourParamFramework(ftp=ftp, regime_start=1956, forecast_from=2020, forecast_to=2036, virtual_successes=1)
	output.append(['1/'+str(int(1/ftp))] + [str(np.around(pragi*100,2))+'%'])
print(columnar(output,no_borders=True))


print("Section 5.2.2")
output = []
vs = [.1,.5,1,2,10]
output.append(['Number of virtual successes']+vs)
pr_after_50 = [functions.generalized_laplace(trials=50, failures=50, virtual_successes=i, ftp=1/100) for i in vs]
output.append(['After 50 failures']+['1/'+str(int(1/x)) for x in pr_after_50])
pr_after_100 = [functions.generalized_laplace(trials=100, failures=100, virtual_successes=i, ftp=1/100) for i in vs]
output.append(['After 100 failures']+['1/'+str(int(1/x)) for x in pr_after_100])
print(columnar(output,no_borders=True))


print("Section 5.2.3")
output = []
header = ["Number of virtual successes","ftp 1/100","ftp 1/300","ftp 1/1000"]
output.append(header)
for vs in [.1,.25,.5,1,2,4,10]:
	pragi = [
		functions.fourParamFramework(ftp=1/100,virtual_successes=vs),
		functions.fourParamFramework(ftp=1/300, virtual_successes=vs),
		functions.fourParamFramework(ftp=1/1000, virtual_successes=vs)]
	row = [vs] + [str(np.around(x*100,2))+'%' for x in pragi]

	output.append(row)
print(columnar(output,no_borders=True))