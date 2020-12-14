import sections

section42 = [12,
8.9,
5.7,
4.2,
2.8,
1.5,
0.77,
0.52]
assert sections.section42() == section42


section522 = [
	[1/600, 1/200, 1/150, 1/125, 1/105],
	[1/1100,1/300,1/200,1/150,1/110]
]
assert sections.section522() == section522

section523 = [
[2,1.6,0.93], # I changed 0.9 to 0.93, I believe you accidentally didn't include 2 significant figures
[4.1,2.7,1.2],
[6.4,3.6,1.4],
[8.9,4.2,1.5],
[11,4.7,1.5],
[13,4.9,1.6],
[14,5.1,1.6]
]

assert sections.section523() == section523