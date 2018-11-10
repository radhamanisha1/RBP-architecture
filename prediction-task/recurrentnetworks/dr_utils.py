import itertools
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
import numpy as np

data = [line.strip() for line in open('/home/radhamanisha/RBP-codebase/data-kern/pitch-plots/data_bach.txt', 'r')]
print(len(data))
vocab_size = list(set(data))

def char_to_tensor(str):
    one_hot_encoded_vector = torch.zeros(len(str)).long()
    for char in range(len(str)):
        one_hot_encoded_vector[char] = vocab_size.index(str[char])
    return (one_hot_encoded_vector)

#split into subsequences
data_modified = zip(*[iter(data)]*6)

#convert into list
data_modified_list = [list(i) for i in data_modified]

inputs, targets = [],[]

for i in data_modified_list:
    inputs.append(i[:-1])
    targets.append(i[-1:])

#convert into tensors
input_tensors, target_tensors = [],[]

input_tensors = [char_to_tensor(i) for i in inputs]
target_tensors = [char_to_tensor(i) for i in targets]

#convert into numpy arrays
target_tensors_numpy = [i.numpy() for i in target_tensors]

input_tensors_numpy = [i.numpy() for i in input_tensors]

two_comb = []

for i in input_tensors_numpy:
    two_comb.extend(list(itertools.combinations(i,2)))

two_comb_list = [list(i) for i in two_comb]
diff = [abs(x-y) for x,y in two_comb]

#get binary differences
diff_binary = []

for i in diff:
    if i==0:
        diff_binary.append(i)
    elif i!=0:
        i=1
        diff_binary.append(i)

model = MLPRegressor()
model.fit(two_comb_list,diff_binary)

count_zero, count_one = 0,0

for i in diff_binary:
    if i==0:
        count_zero+=1
    elif i==1:
        count_one+=1

#try to predict on test data
test_inputs = []
#repplicate target tensor to make it compatible with zip!!!
test1 = [[i]*5 for i in target_tensors_numpy]

#Flat list such that each list of list merges into a single list..
flat_list = list(list(itertools.chain.from_iterable(i)) for i in test1)

for i,j in zip(input_tensors_numpy,flat_list):
    test_inputs.append(zip(i,j))

#modify these test inputs-- tuples into lists
#flatten them again
cc = [item for sublist in test_inputs for item in sublist]
test_inputs_list = [list(c) for c in cc]
#NEW LINE ADDED
diff1 = [abs(x-y) for x,y in cc]
print('len of diff1', len(diff1))

np.corrcoef(diff[:7715],diff1) #for bach

k = model.predict(test_inputs_list)
#print(type(k))
#coefficients = [coef.shape for coef in model.coefs_]
#print(coefficients)
bin_diff_chunks = zip(*[iter(diff_binary)]*10)
bin_out_chunks = zip(*[iter(k)]*5)

print(type(bin_out_chunks))
bin_out_chunks = [list(i) for i in bin_out_chunks]
print(normalize(bin_out_chunks))
final_chunks_dict = dict(zip(bin_diff_chunks,bin_out_chunks))

#
np.save('/home/radhamanisha/RBP-codebase/dr-results/bach_5.txt', bin_out_chunks)

f = open('/home/radhamanisha/RBP-codebase/dr-results/bach_5.txt','w')
for k,v in final_chunks_dict.items():
    print>>f,k,v
