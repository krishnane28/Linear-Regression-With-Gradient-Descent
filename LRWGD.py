import numpy as np

# Calculates the total error produced by the model in each iteration
def compute_error(m, b, points):
	total_error = 0
	# Goes through each point in our dataset
	# Calculate error and sums it up with the total points in the dataset
	# Takes the average of the total error the model has produced during each iteration
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		total_error += (y - ((m * x) + b)) ** 2
	return (total_error / float(len(points)))
	
# Calculates gradient in each iteration
def batch_gradient_descent(points, current_m, current_b, learning_rate):
	# This is batch gradient descent i.e. calculate the gradient of the
	# whole data set and performs a single update
	# Since we are doing a single update for the weights and biases
	# approaching to the local minima can be slow and cost effective
	# since we have to calculate the gradient for all the points
	# in our data set
	# But it guarantees near to local minima for convex error function
	# and local maxima for concave error function
	m_gradient = 0
	b_gradient = 0
	total_points = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		m_gradient += (-(2/total_points)) * (y - ((current_m * x) + current_b)) * x
		b_gradient += (-(2/total_points)) * (y - ((current_m * x) + current_b))
	m_new = current_m - (learning_rate * m_gradient)
	b_new = current_b - (learning_rate * b_gradient)
	return [m_new, b_new]
	
	
def stochastic_gradient_descent(points, current_m, current_b, learning_rate):
	# This is stochastic gradient descent i.e. calculate the gradient of
	# each data set and performs the update
	# Here we update the weights and biases n times for each iteration where
	# n is the total number of points in the data set
	# This approaches the local minima or maxima faster but give a lot of 
	# fluctuations during the process
	m_gradient = 0
	b_gradient = 0
	total_points = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		m_gradient = (-2) * (y - ((current_m * x) + current_b)) * x
		b_gradient = (-2) * (y - ((current_m * x) + current_b))
		current_m = current_m - (learning_rate * m_gradient)
		current_b = current_b - (learning_rate * b_gradient)
	return [current_m, current_b]
	
# Calculates the ideal gradient for the model 
def gradient_descent(points, initial_m, initial_b, learning_rate, num_iterations):
	m = initial_m
	b = initial_b
	
	for i in range(num_iterations):
		[m, b] = stochastic_gradient_descent(points, m, b, learning_rate)
		
	return [m, b]


def run():
	points = np.genfromtxt('input.csv', delimiter = ',')
	# learning_rate tells how fast our model learns
	# less learning_rate means the model is too slow to converge
	# high learning_rate means the model never converge
	# converge here is to meet at a point which is the local minima
	learning_rate = 0.00001
	
	# y = mx + b 
	# m is the slope of our line
	# b is the y intercept
	# both are zero because we do not know the correct values for our model
	initial_b = 0
	initial_m = 0
	
	# number of times we want to run the model to learn m and b for the dataset
	num_iterations = 1000
	print('Initial m: %f, Initial b: %f, Initial error: %f' %(initial_m, \
	      initial_b, compute_error(initial_m, initial_b, points)))
	[m, b] = gradient_descent(points, initial_m, initial_b, learning_rate, num_iterations)
	print('Final m: %f, Final b: %f, Final error: %f' %(m, \
	      b, compute_error(m, b, points)))

if __name__ == '__main__':
	run()
	