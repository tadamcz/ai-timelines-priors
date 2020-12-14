import sections
import warnings

def warn_if_unequal(vetting, report, location=None):
	if not vetting == report:
		warnstring = '''
		In section ''' + str(location) + '''
		Vetting value ''' + str(vetting) +''' is not equal to
		Report value ''' + str(report)
		warnings.warn(warnstring)

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
warn_if_unequal(sections.section5_2_3_t2_left(), section5_2_3_t2_left, "5.2.3 Table 2, Left part")


section5_2_3_t2_right = [
[34,14,4.8],
[50,25,9.1],
[8.9,4.2,1.5]
]
warn_if_unequal(sections.section5_2_3_t2_right(), section5_2_3_t2_right, "5.2.3 Table 2, Right part")

