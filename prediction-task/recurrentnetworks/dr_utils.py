import itertools
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
import numpy as np

x1 = list(map(list, itertools.product([0, 1], repeat=3)))
x1 = x1 * 2
# print(x1)

k = list(itertools.combinations(x1, 2))
equal_vec, unequal_vec = [], []

# print(len(equal_vec))
# print(len(unequal_vec))

for i in k:
    if (i[0] == i[1]):
        equal_vec.append(i)
    else:
        unequal_vec.append(i)

train_data, test_data = [], []
train_data.extend(equal_vec[:6])
train_data.extend(unequal_vec[:6])

test_data.extend(equal_vec[6:8])

test_data.extend(unequal_vec[6:8])

flat_list1 = [list(i) for i in train_data]
flat_list2 = [list(i) for i in test_data]

#convert into numpy arrays
# target_tensors_numpy = [i.numpy() for i in flat_list2]
#
# input_tensors_numpy = [i.numpy() for i in flat_list1]
#
# print(target_tensors_numpy)

two_comb = []

for i in flat_list1:
    two_comb.extend(list(itertools.combinations(i,2)))

two_comb_list = [list(i) for i in two_comb]
diff1, diff2 = [], []
for i,j in flat_list1:
    diff1.append([abs(m - n) for m, n in zip(i, j)])

for i,j in flat_list2:
    diff2.append([abs(m - n) for m, n in zip(i, j)])

# print(diff1)

diff_binary = []

for i in diff1:
    if i==j:
        diff_binary.append(0)
    elif i!=j:
        diff_binary.append(1)

two_comb_list = [i+j for i,j in two_comb_list]

model = MLPRegressor()
model.fit(two_comb_list,diff_binary)

count_zero, count_one = 0,0

for i in diff1:
    if i==0:
        count_zero+=1
    elif i==1:
        count_one+=1

#try to predict on test data
test_inputs = []
#repplicate target tensor to make it compatible with zip!!!
test1 = [[i]*3 for i in flat_list2]

#Flat list such that each list of list merges into a single list..
flat_list = list(list(itertools.chain.from_iterable(i)) for i in test1)

for i,j in zip(flat_list1,flat_list):
    test_inputs.append(zip(i,j))

#modify these test inputs-- tuples into lists
#flatten them again
cc = [item for sublist in test_inputs for item in sublist]
test_inputs_list = [list(c) for c in cc]
#NEW LINE ADDED
diff = []
for i,j in cc:
    diff.append([abs(m - n) for m, n in zip(i, j)])

np.corrcoef(diff1,diff) #for bach
flat_list2 = [i+j for i,j in flat_list2]

k = model.predict(flat_list2)
#print(type(k))
#coefficients = [coef.shape for coef in model.coefs_]
#print(coefficients)
bin_diff_chunks = zip(*[iter(diff1)]*6)
bin_out_chunks = zip(*[iter(k)]*2)

bin_out_chunks = [list(i) for i in bin_out_chunks]
print(normalize(bin_out_chunks))
#final_chunks_dict = dict(zip(bin_diff_chunks,bin_out_chunks))

#
np.save('result.txt', bin_out_chunks)