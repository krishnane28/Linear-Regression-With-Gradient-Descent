import csv
from random import *

with open('input.csv', 'w', newline = '') as input_file:
	file_writer = csv.writer(input_file, delimiter = ',')
	for i in range(100):
		file_writer.writerow([randint(1, 100), randint(1, 100)])