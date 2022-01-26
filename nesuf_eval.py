import numpy as np 

files = ['RI.txt', 'DPI.txt', 'HPI.txt', 'SI.txt', 'BI.txt', 'JI.txt', 'BEST.txt']

for file in files:
	f = open('nesuf_scores/'+file, 'r')
	y = f.readlines()
	y = [[float(item) for item in line.split(' ')] for line in y]
	y2 = [[item[0]*item[1], item[2]*item[3], item[4]*item[5]] for item in y]
	print(file)
	print("mean = ", (np.mean(y, axis = 0)))
	print("std = ", (np.std(y, axis = 0)))
	print("y2 mean = ", (np.mean(y2, axis = 0)))