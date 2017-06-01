# Algorithms from Chapter 4 of 
# Introduction to Algorithms, 3rd Ed
# Written by Everett Sussman
#################################

# Import all libraries 
import sys, time, tqdm, random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns



############################################################
# The Maximum Sub-array Problem                            #
#                                                          #
# Description of algorithm:                                #
# Divide array of values into 3 parts: left half,          #
# right half, and crossover.  Store the maximal subarray   #
# in each part, adding the overall sums in only the        #
# crossover section, then return the initial and final     #
# indices of maximal sub-array, and total value.           #
############################################################

# Find-Max-Crossing-Subarray
def Find_Max_Crossing_Subarray(A, low, mid, high):
	'''A represents the list to analyze,
	low, mid, and high are lowest index, middle index, 
	and highest index respectively.'''
	minint = -sys.maxint - 1
	left_sum, right_sum = minint, minint
	sum_l, sum_r = 0, 0
	for i in xrange(0, mid + 1 - low):
		j = mid - i
		sum_l += A[j]
		if sum_l > left_sum:
			left_sum = sum_l
			max_left = j
	for i in xrange(mid+1, high + 1):
		sum_r += A[i]
		if sum_r > right_sum:
			right_sum = sum_r
			max_right = i
	return max_left, max_right, left_sum + right_sum

# Find Maximal Subarray O(nlog(n))
def Find_Max_Sub(A, low, high):
	'''A represents sub-array to search through,
	low is beginning index of sub-array, 
	high is terminating index of sub-array.'''
	if high == low:
		return low, high, A[low]
	else:
		mid = int(np.floor((low+high)/2))
		(l_l, l_h, l_s) = Find_Max_Sub(A, low, mid)
		(r_l, r_h, r_s) = Find_Max_Sub(A, mid+1, high)
		(c_l, c_h, c_s) = Find_Max_Crossing_Subarray(A, low, mid, high)
		if l_s >= r_s and l_s >= c_s:
			return l_l, l_h, l_s
		elif r_s >= l_s and r_s >= c_s:
			return r_l, r_h, r_s
		else:
			return c_l, c_h, c_s

# Brute Force Algorithm O(n^2)
def Brute_Max_Sub(A, low, high):
	'''A is array to search, low and high are 
	initial and terminating indices of array to search. '''
	minint = -sys.maxint-1
	max_sum = minint
	b_sum = 0
	for j in xrange(low+1, high+2):
		for i in xrange(low, j):
			b_sum = sum(A[i:j])
			if max_sum < b_sum:
				max_sum = b_sum
				max_i = i
				max_j = j-1
	return max_i, max_j, max_sum

def Linear_Max_Sub(A, low, high):
	'''Linear Time Algorithm for finding maximum subarray.'''
	max_i, max_j, max_sum = low, low, A[low]
	run_sum = max_sum
	prop_i = max_i
	j = low+1
	while j <= high:
		if A[j] > 0:
			if run_sum <= 0 and A[j] <= max_sum:
				prop_i = j
				run_sum = A[j]
				j += 1
			elif run_sum <= 0 and A[j] > max_sum:
				prop_i, max_i, max_j, run_sum, max_sum = j, j, j, A[j], A[j]
				j += 1
			elif run_sum + A[j] > max_sum:
				max_i = prop_i
				max_j = j
				run_sum += A[j]
				max_sum = run_sum
				j += 1
			else:
				run_sum += A[j]
				j += 1
		else:
			run_sum += A[j]
			j += 1
	return max_i, max_j, max_sum

############################################################
# Matrix Multiplication                                    #
#                                                          #
# Description of Problem:                                  #
# There are two matrices, A and B.  Each matrix is n x n.  #
# Calculate matrix C = AB where                            #
# c_{ij} = \sum_{k=1}^n a_{ik}b_{kj}.                      #
############################################################

# Square-Matrix-Multiply O(n^3)
def sq_mat_mult(A, B):
	'''A and B are both n x n matrices to be multiplied to find
	C = AB. Both A and B must be numpy.arrays in 2 dimensions.'''
	n = A.shape[0]
	C = np.zeros(shape=(n,n))
	for i in xrange(0, n):
		for j in xrange(0, n):
			for k in xrange(0, n):
				C[i][j] = C.item((i, j)) + A.item((i, k))*B.item((k, j))
	return C

# Recursive Square Matrix Multiply O(n^3)
def rec_sq_mult(A, B):
	'''A and B are both n x n matrices where log2(n) is an integer.'''
	n = A.shape[0]
	C = np.zeros(shape=(n,n))
	if n == 1:
		C = A*B
		return C
	else:
		A_11 = A[0:n/2, 0:n/2]
		A_12 = A[0:n/2, n/2:n]
		A_21 = A[n/2:n, 0:n/2]
		A_22 = A[n/2:n, n/2:n]
		B_11 = B[0:n/2,0:n/2]
		B_12 = B[0:n/2, n/2:n]
		B_21 = B[n/2:n, 0:n/2]
		B_22 = B[n/2:n, n/2:n]
		C[0:n/2, 0:n/2] = rec_sq_mult(A_11, B_11) + rec_sq_mult(A_12, B_21)
		C[0:n/2, n/2:n] = rec_sq_mult(A_11, B_12) + rec_sq_mult(A_12, B_22)
		C[n/2:n, 0:n/2] = rec_sq_mult(A_21, B_11) + rec_sq_mult(A_22, B_21)
		C[n/2:n, n/2:n] = rec_sq_mult(A_21, B_12) + rec_sq_mult(A_22, B_22)
		return C

# Strassen's Method O(n^log(7))
def strassen(A, B):
	'''This implementation only works on square matrices where 
	log2(n) is an integer.'''
	n = A.shape[0]
	C = np.zeros(shape=(n,n))
	if n == 1:
		C = A*B
		return C
	else:
		# Get submatrices
		A_11 = A[0:n/2, 0:n/2]
		A_12 = A[0:n/2, n/2:n]
		A_21 = A[n/2:n, 0:n/2]
		A_22 = A[n/2:n, n/2:n]
		B_11 = B[0:n/2,0:n/2]
		B_12 = B[0:n/2, n/2:n]
		B_21 = B[n/2:n, 0:n/2]
		B_22 = B[n/2:n, n/2:n]

		# Generate 10 S matrices
		S_1 = B_12 - B_22
		S_2 = A_11 + A_12
		S_3 = A_21 + A_22
		S_4 = B_21 - B_11
		S_5 = A_11 + A_22
		S_6 = B_11 + B_22
		S_7 = A_12 - A_22
		S_8 = B_21 + B_22
		S_9 = A_11 - A_21
		S_10 = B_11 + B_12

		# Generate P matrices recursively
		P_1 = strassen(A_11, S_1)
		P_2 = strassen(S_2, B_22)
		P_3 = strassen(S_3, B_11)
		P_4 = strassen(A_22, S_4)
		P_5 = strassen(S_5, S_6)
		P_6 = strassen(S_7, S_8)
		P_7 = strassen(S_9, S_10)

		# Construct C
		C[0:n/2, 0:n/2] = P_5 + P_4 - P_2 + P_6
		C[0:n/2, n/2:n] = P_1 + P_2
		C[n/2:n, 0:n/2] = P_3 + P_4
		C[n/2:n, n/2:n] = P_5 + P_1 - P_3 - P_7
		return C


# Strassen's Method With Leaf Size O(n^log(7))
def helper(A, B, leaf):
	'''This implementation works on any square matrix multiplication 
	of size n.'''
	n = A.shape[0]

	# Trivial Case
	if n == 1:
		return A * B

	# Pad matrix if necessary
	size, pad_width = 1, 0
	while size < n:
		if size * 2 == n:
			break
		elif size * 2 < n:
			size = size * 2
		else:
			pad_width = size*2 - n
			break

	A = np.lib.pad(A, (0,pad_width), 'constant')
	B = np.lib.pad(B, (0,pad_width), 'constant')
	n = A.shape[0]
	C = np.zeros(shape=(n,n))

	if n <= leaf:
		C = sq_mat_mult(A, B)
		return C
	
	else:
		# Get submatrices
		A_11 = A[0:n/2, 0:n/2]
		A_12 = A[0:n/2, n/2:n]
		A_21 = A[n/2:n, 0:n/2]
		A_22 = A[n/2:n, n/2:n]
		B_11 = B[0:n/2,0:n/2]
		B_12 = B[0:n/2, n/2:n]
		B_21 = B[n/2:n, 0:n/2]
		B_22 = B[n/2:n, n/2:n]

		# Generate 10 S matrices
		S_1 = B_12 - B_22
		S_2 = A_11 + A_12
		S_3 = A_21 + A_22
		S_4 = B_21 - B_11
		S_5 = A_11 + A_22
		S_6 = B_11 + B_22
		S_7 = A_12 - A_22
		S_8 = B_21 + B_22
		S_9 = A_11 - A_21
		S_10 = B_11 + B_12

		# Generate P matrices recursively
		P_1 = strassen2(A_11, S_1, leaf)
		P_2 = strassen2(S_2, B_22, leaf)
		P_3 = strassen2(S_3, B_11, leaf)
		P_4 = strassen2(A_22, S_4, leaf)
		P_5 = strassen2(S_5, S_6, leaf)
		P_6 = strassen2(S_7, S_8, leaf)
		P_7 = strassen2(S_9, S_10, leaf)

		# Construct C
		C[0:n/2, 0:n/2] = P_5 + P_4 - P_2 + P_6
		C[0:n/2, n/2:n] = P_1 + P_2
		C[n/2:n, 0:n/2] = P_3 + P_4
		C[n/2:n, n/2:n] = P_5 + P_1 - P_3 - P_7
		return C

def strassen2(A, B, leaf):
	real_n = A.shape[0]
	C = helper(A, B, leaf)
	return C[0:real_n, 0:real_n]

############################################################
# Testing                                                  #
############################################################

# Test Find_Max_Crossing_Subarray
def cross_tests():
	A1 = [-1, 3, -2, 3, -2, 2, -1]
	assert(Find_Max_Crossing_Subarray(A1, 0, 3, 6) == (1, 5, 4))
	A2 = [1, 2, 3, 2, 1]
	assert(Find_Max_Crossing_Subarray(A2, 0, 2, 4) == (0, 4, 9))
	A3 = [-1, -2, 1, -2, -1]
	assert(Find_Max_Crossing_Subarray(A3, 0, 2, 4) == (2, 3, -1))

# Test Find_Max_Sub
def max_tests():
	A1 = [1]
	assert(Find_Max_Sub(A1, 0, len(A1)-1) == (0, 0, 1))
	A2 = [2, -3, 1, 4, -2]
	assert(Find_Max_Sub(A2, 0, len(A2)-1) == (2, 3, 5))
	A3 = [13, -3, -25, 20, -3, -16, -23, 18,
	20, -7, 12, -5, -22, 15, -4, 7]
	assert(Find_Max_Sub(A3, 0, len(A3)-1) == (7, 10, 43))
	A4 = [1, -4, 3, -4]
	assert(Find_Max_Sub(A4, 0, len(A4)-1) == (2, 2, 3))
	A5 = [1, -2, 3, 4]
	assert(Find_Max_Sub(A5, 0, len(A5)-1) == (2, 3, 7))

# Test Brute Force Method for maximal sub-array
def brute_max_tests():
	A1 = [1]
	assert(Brute_Max_Sub(A1, 0, len(A1)-1) == (0, 0, 1))
	A2 = [2, -3, 1, 4, -2]
	assert(Brute_Max_Sub(A2, 0, len(A2)-1) == (2, 3, 5))
	A3 = [13, -3, -25, 20, -3, -16, -23, 18,
	20, -7, 12, -5, -22, 15, -4, 7]
	assert(Brute_Max_Sub(A3, 0, len(A3)-1) == (7, 10, 43))
	A4 = [1, -4, 3, -4]
	assert(Brute_Max_Sub(A4, 0, len(A4)-1) == (2, 2, 3))
	A5 = [1, -2, 3, 4]
	assert(Brute_Max_Sub(A5, 0, len(A5)-1) == (2, 3, 7))

# Test linear algorithm for 
def linear_max_tests():
	A1 = [1]
	assert(Linear_Max_Sub(A1, 0, len(A1)-1) == (0, 0, 1))
	A2 = [2, -3, 1, 4, -2]
	assert(Linear_Max_Sub(A2, 0, len(A2)-1) == (2, 3, 5))
	A3 = [13, -3, -25, 20, -3, -16, -23, 18,
	20, -7, 12, -5, -22, 15, -4, 7]
	assert(Linear_Max_Sub(A3, 0, len(A3)-1) == (7, 10, 43))
	A4 = [1, -4, 3, -4]
	assert(Linear_Max_Sub(A4, 0, len(A4)-1) == (2, 2, 3))
	A5 = [1, -2, 3, 4]
	assert(Linear_Max_Sub(A5, 0, len(A5)-1) == (2, 3, 7))

# Test normal matrix multiplication algorithm
def sq_mat_tests():
	for i in range(0, 10):
		A = np.random.randint(20, size=(3,3))
		B = np.random.randint(20, size=(3,3))
		if sq_mat_mult(A,B).all() != np.dot(A,B).all():
			print "Error!  Failed to multiply {} and {}.".format(A,B)
			break

def rec_sq_mult_tests():
	for i in range(0, 4):
		A = np.random.randint(20, size=(2**i, 2**i))
		B = np.random.randint(20, size=(2**i, 2**i))
		if rec_sq_mult(A,B).all() != np.dot(A,B).all():
			print "Error!  Failed to multiply {} and {}.".format(A,B)
			break

def strassen_tests():
	for i in range(0, 4):
		A = np.random.randint(20, size=(2**i, 2**i))
		B = np.random.randint(20, size=(2**i, 2**i))
		if strassen(A,B).all() != np.dot(A,B).all():
			print "Error!  Failed to multiply {} and {}.".format(A,B)
			break

# Run tests
def run_all_tests():
	cross_tests()
	max_tests()
	brute_max_tests()
	linear_max_tests()
	sq_mat_tests()
	rec_sq_mult_tests()
	strassen_tests()
	print "Import of Chapter 4 algorithms Complete"

run_all_tests()


