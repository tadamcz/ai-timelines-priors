import sections
import warnings
from time import sleep
from termcolor import colored

'''
NOTE: I used this file for the first part of the report, but then abandoned it after it became clear that it wasn't desired/needed.

Refer to print_output.py, which prints the results for the entire report.
'''


def setup_warning_catcher():
    """ Wrap warnings.showwarning with code that records warnings. """

    caught_warnings = []
    original_showwarning = warnings.showwarning

    def custom_showwarning(*args,  **kwargs):
        caught_warnings.append(args[0])
        return original_showwarning(*args, **kwargs)

    warnings.showwarning = custom_showwarning
    return caught_warnings
caught_warnings_list = setup_warning_catcher()

def warn_if_unequal(vetting, report, location=None):
	if not vetting == report:
		warnstring = '''
		In section ''' + str(location) + ''', vetting value
		''' + str(vetting) +''' is not equal to
		report value
		'''+ str(report)

		sleep(.01)
		# I tried setting sys.stdout.flush() instead of sleep(), but it didn't work for getting the
		# warnings in the right order. sleep() is very hacky but OK for this purpose.

		warnings.warn(warnstring)
		print("\n\n")

section4_2 = [12,
			  8.9,
			  5.7,
			  4.2,
			  2.8,
			  1.5,
			  0.77,
			  0.52]
warn_if_unequal(sections.section4_2(), section4_2, "4.2")


section5_2_2 = [
	[1/600, 1/200, 1/150, 1/125, 1/105],
	[1/1100,1/300,1/200,1/150,1/110]
]
warn_if_unequal(sections.section5_2_2(), section5_2_2, "5.2.2")

section5_2_3_t1 = [
[2,1.6,0.93], # I changed 0.9 to 0.93, I believe you accidentally didn't include 2 significant figures
[4.1,2.7,1.2],
[6.4,3.6,1.4],
[8.9,4.2,1.5],
[11,4.7,1.5],
[13,4.9,1.6],
[14,5.1,1.6]
]
warn_if_unequal(sections.section5_2_3_t1(), section5_2_3_t1, "5.2.3 Table 1")

section5_2_3_t2_left = [
[39,30,13,4.7],
[52,43,23,8.7],
[7.6,6.4,3.6,1.4]
]
print(colored("The differences below are all in the first column, ftp=1/50",'red'))
warn_if_unequal(sections.section5_2_3_t2_left(), section5_2_3_t2_left, "5.2.3 Table 2, Left part")

section5_2_3_t2_right = [
[34,14,4.8],
[50,25,9.1],
[8.9,4.2,1.5]
]
warn_if_unequal(sections.section5_2_3_t2_right(), section5_2_3_t2_right, "5.2.3 Table 2, Right part")

section5_3_t1 =[ # I am adding significant figures to the last column, e.g. 0.2 becomes 0.23.
[19,12,11,3.7,0.23],
[12,8.9,8.4,3.2,0.22],
[4.7,4.2,4.1,2.3,0.22],
[1.5,1.5,1.5,1.2,0.20]
]
print(colored("The differences below are very minor, likely rounding errors",'red'))
warn_if_unequal(sections.section5_3_t1(), section5_3_t1, "5.3 Table 1")

section5_3_t2 = [
[6.4,5.3,7.3,20],
[5.3,4,5,8.9],
[2.9,2.7,3.1,4.2],
[1.3,1.2,1.3,1.4]
]
print(colored("The differences below are for the cells (1/100,168) and (1/1000,0)",'red'))
warn_if_unequal(sections.section5_3_t2(),section5_3_t2,"5.3 Table 2")

section6_1_2 = [
[11,14,16,17,18],
[7.2,11,13,15,16],
[3.2,6.1,8.2,9.7,11], # changed 6 to 6.1 and 8 to 8.2 for consistent use of 2 significant digits
[1.1,2.3,3.3,4.4,5.3]
]
print(colored("There are several differences below, all on the second significant digit (so they are relatively minior)",'red'))
warn_if_unequal(sections.section6_1_2(),section6_1_2,"6.1.2")

assert len(caught_warnings_list) == 4