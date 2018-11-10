import itertools
import torch
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# data = [line.strip() for line in open('/Users/ksnmurthy/research-work/data-kern/pitch-plots/data_schweiz.txt', 'r')]
data = [line.strip() for line in
        open('/home/radhamanisha/RBP-codebase/data-kern/pitch-plots/data_bach.txt', 'r')]


def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


def cond_prob(x, y):
    ic = len(set.intersection(*[set(x), set(y)]))
    uc = len(set(y))
    return ic / float(uc)


vocab_size = list(set(data))

char_to_int = dict((c, i) for i, c in enumerate(vocab_size))

ba = [char_to_int[char] for char in data]
# split into subsequences
data_modified = zip(*[iter(ba)] * 6)

# convert into list
data_modified_list = [list(i) for i in data_modified]

inputs, targets = [], []

for i in data_modified_list:
    inputs.append(i[:-1])
    targets.append(i[-1:])

two_comb = []

# ba = [char_to_int[char] for char in data]

for i in inputs:
    two_comb.append(list(itertools.combinations(i, 2)))

# diff = [abs(x-y) for x,y in two_comb]
two_comb_list = [list(i) for i in two_comb]
diff = []

for i in two_comb_list:
    diff.append([abs(x - y) for x, y in i])
# diff = [abs(x-y) for x,y in two_comb]

# get binary differences
diff_binary = []

for i in diff:
    if i == 0:
        diff_binary.append(i)
    elif i != 0:
        i = 1
        diff_binary.append(i)

# model = MLPRegressor()
# model.fit(two_comb_list,diff_binary)
print(diff_binary[:1])
count_zero, count_one = 0, 0

for i in diff_binary:
    if i == 0:
        count_zero += 1
    elif i == 1:
        count_one += 1

# try to predict on test data
test_inputs = []
# repplicate target tensor to make it compatible with zip!!!
test1 = [[i] * 5 for i in targets]

# Flat list such that each list of list merges into a single list..
flat_list = list(list(itertools.chain.from_iterable(i)) for i in test1)

for i, j in zip(inputs, flat_list):
    test_inputs.append(zip(i, j))

# modify these test inputs-- tuples into lists
# flatten them again
# cc = [item for sublist in test_inputs for item in sublist]
test_inputs_list = [list(c) for c in cc]

diff1 = []

for i in test_inputs:
    diff1.append([abs(x - y) for x, y in i])

jack_sim = []
jc_c = []

for i, j in zip(diff, diff1):
    jack_sim.append(jaccard_similarity(i, j))

for i, j in zip(diff, diff1):
    jc_c.append(cond_prob(i, j))

diff_binary1 = []
for i in test_inputs_list:
    if i == 0:
        diff_binary1.append(i)
    elif i != 0:
        i = 1
        diff_binary1.append(i)

j = jack_sim[:50]

j = np.array(j).reshape(5, 10)

jc = jc_c[:50]

jc = np.array(jc).reshape(5, 10)

plt.imshow(j, cmap=plt.cm.RdBu);
plt.title('Jaccard Similarity distribution')
plt.xlabel('DR inputs')
plt.ylabel('DR outputs')
plt.colorbar()
plt.show()

plt.imshow(jc, cmap=plt.cm.RdBu);
plt.title('Conditional probability distribution')
plt.xlabel('DR inputs')
plt.ylabel('DR outputs')
plt.colorbar()
plt.show()

for i in diff1:
    if (i[4] == 0):
        d15 += 1
d = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15]
for i in d:
    i = i * 0.01
h1 = ['DR-I1', 'DR-I2', 'DR-I3', 'DR-I4', 'DR-I5', 'DR-I6', 'DR-I7', 'DR-I8', 'DR-I9', 'DR-I10']
h2 = ['DR-O1', 'DR-O2', 'DR-O3', 'DR-O4', 'DR-O5']
y_pos = np.arange(len(h2))
print(diff_binary1[:1])
plt.title('Occurrence of DR outputs over 8 datasets')
plt.ylabel('Percentage of occurrence')
plt.xlabel('DR outputs')
plt.xticks(y_pos, h2)
plt.bar(y_pos, de[10:15], align='center', alpha=0.5)

# k = model.predict(test_inputs_list)

# coefficients = [coef.shape for coef in model.coefs_]
# print(coefficients)
# bin_diff_chunks = zip(*[iter(diff_binary)]*10)
# bin_out_chunks = zip(*[iter(k)]*5)

# final_chunks_dict = dict(zip(bin_diff_chunks,bin_out_chunks))

# keys = final_chunks_dict.keys()
# values=final_chunks_dict.values()
# get combinations of key,value

print(len(diff_binary))
print(len(diff_binary1))

drd = list(itertools.product(diff_binary[:5], diff_binary1[:5]))
# print(len(drd))
drd_arr = np.array(drd).reshape(5, 10)
# print(drd_arr)
plt.imshow(drd_arr, cmap=plt.cm.RdBu)
h1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
# plt.xticks(np.arange(0,9),h1)
# plt.yticks(np.arange(0,4))
plt.colorbar()
plt.show()

# flatten list of lists into one


# flat_list = [item for sublist in drd for item in sublist]
# split flatlist into two lists using zip and map
# w1, w2 = map(list,zip(*flat_list))
# create third column w3 which is for dataset
# w3 = ['Chinese']*len(w1)
# convert into panda frame
# pdd = pd.DataFrame({'DR-input':w1, 'DR-output':w2, 'dataset':w3})
# sns.factorplot(x="DR-input",y="DR-output",col="dataset", data=pdd,kind="strip", jitter=True)
