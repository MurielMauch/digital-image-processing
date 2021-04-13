import matplotlib
import numpy as np

####################################################################################################
# vectorized kernel example
##################################################################################################

entry = np.asarray([[1,2,3], [4,5,6], [1, 2, 3]], dtype=np.int32)
filter_h1 = np.asarray([-1, -2, -1, 0, 0, 0, 1, 2, 1], dtype=np.int32)

vectorized_matrix = np.matrix.flatten(entry)
print(vectorized_matrix)

filter_result = np.dot(vectorized_matrix, filter_h1)

center = int(len(vectorized_matrix) / 2)

vectorized_matrix[center] = filter_result

print(vectorized_matrix)
