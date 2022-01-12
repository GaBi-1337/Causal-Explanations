import numpy as np 

files = ['RI.txt', 'DPI.txt', 'HPI.txt', 'SI.txt', 'BI.txt', 'JI.txt']

for file in files:
	f = open('nesuf_scores/'+file, 'r')
	y = f.readlines()
	y = [[float(item) for item in line.split(' ')] for line in y]
	print(file)
	print("mean = ", (np.mean(y, axis = 0)))
	print("std = ", (np.std(y, axis = 0)))
	# print("min = %f" % (np.min(y)))