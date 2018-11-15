#mid fusion

import itertools
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import time
import argparse
import random
import torch.utils.data as utils_data

batch_size = 1
#data generation
#Case1:ABA vs other
#Case2:ABB vs other
#Case3:ABA vs ABB
#Case4:ABC vs other
#Case5: ABA-BAB vs other
#Case6:Mixed patterns
alp1 = ['c','d','e','f', 'i', 'j']
alp2 = ['g', 'a', 'b','h', 'k', 'l']
p1 = list(itertools.product(alp1,repeat=3))
p2 = list(itertools.product(alp2,repeat=3))

data1 = [list(i) for i in p1]
data2 = [list(i) for i in p2]

train_aba_patterns, train_abb_patterns, train_abc_patterns, train_aaa_patterns,train_aab_patterns = [], [], [], [],[]
test_aba_patterns, test_abb_patterns, test_abc_patterns, test_aaa_patterns, test_aab_patterns = [], [], [], [], []

for i in data1:
    if(i[0]==i[2] and i[0]!= i[1] and i[1]!=i[2]):
        train_aba_patterns.append(i)
    elif(i[0]!= i[1] and i[0]!=i[2] and i[1]==i[2]):
        train_abb_patterns.append(i)
    elif(i[0]!=i[1] and i[1]!=i[2] and i[0] != i[2]):
        train_abc_patterns.append(i)
    elif(i[0] == i[1] and i[1] == i[2] and i[0] == i[2]):
        train_aaa_patterns.append(i)
    elif(i[0] == i[1] and i[1]!=i[2] and i[0]!=i[2]):
        train_aab_patterns.append(i)

random.shuffle(train_aba_patterns)
random.shuffle(train_abb_patterns)
random.shuffle(train_abc_patterns)
random.shuffle(train_aaa_patterns)
random.shuffle(train_aab_patterns)

train_aba_patterns = train_aba_patterns[:30]
train_abb_patterns = train_abb_patterns[:8]
train_aaa_patterns = train_aaa_patterns[:6]
train_abc_patterns = train_abc_patterns[:8]
train_aab_patterns = train_aab_patterns[:8]

training_data = train_aaa_patterns + train_aba_patterns + train_abc_patterns + train_abb_patterns + train_aab_patterns

for i in data2:
    if (i[0]==i[2] and i[0]!= i[1] and i[1]!=i[2]):
        test_aba_patterns.append(i)
    elif (i[0]!= i[1] and i[0]!=i[2] and i[1]==i[2]):
        test_abb_patterns.append(i)
    elif (i[0]!=i[1] and i[1]!=i[2] and i[0] != i[2]):
        test_abc_patterns.append(i)
    elif (i[0] == i[1] and i[1] == i[2] and i[0] == i[2]):
        test_aaa_patterns.append(i)
    elif (i[0] == i[1] and i[1] != i[2] and i[0] != i[2]):
        test_aab_patterns.append(i)

random.shuffle(test_aba_patterns)
random.shuffle(test_abb_patterns)
random.shuffle(test_abc_patterns)
random.shuffle(test_aaa_patterns)
random.shuffle(test_aab_patterns)

test_aba_patterns = test_aba_patterns[:30]
test_abb_patterns = test_abb_patterns[:8]
test_aaa_patterns = test_aaa_patterns[:6]
test_abc_patterns = test_abc_patterns[:8]
test_aab_patterns = test_aab_patterns[:8]


testing_data = test_aaa_patterns[:3]+ test_aba_patterns[:15] + test_abc_patterns[:4] + test_abb_patterns[:4] + test_aab_patterns[:4]
validation_data = test_aaa_patterns[3:]+ test_aba_patterns[15:] + test_abc_patterns[4:] + test_abb_patterns[4:] + test_aab_patterns[4:]

target1, target2, target3 = [],[],[]

for i in training_data:
    if (i[0] == i[2] and i[0] != i[1] and i[1] != i[2]):
        target1.append([1])
    else:
        target1.append([0])

for i in validation_data:
    if (i[0] == i[2] and i[0] != i[1] and i[1]!=i[2]):
        target2.append([1])
    else:
        target2.append([0])

for i in testing_data:
    if (i[0] == i[2] and i[0] != i[1] and i[1]!=i[2]):
        target3.append([1])
    else:
        target3.append([0])

final_data1 = [''.join(i) for i in training_data]
final_data2 = [''.join(i) for i in validation_data]
final_data3 = [''.join(i) for i in testing_data]

alphabet_size = 12
sample_space = 'cdefgabhijkl'
sample_space_len = len(sample_space)
# decoding char from int
char_from_int = dict((i, c) for i, c in enumerate(sample_space))

char_to_int = dict((c, i) for i, c in enumerate(sample_space))

#create one hot encodings of data
def letterToIndex(letter):
    return sample_space.find(letter)


#< 1 * alphabet_size > one _hot_encoded
def letterToTensor(letter):
    tensor = torch.zeros(1, sample_space_len)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def wordToTensor(word):
    tensor = torch.zeros(len(word), 1, sample_space_len)
    for li, letter in enumerate(word):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

train_data, valid_data, test_data = [],[], []
diff1, diff2, diff3 = [], [], []

for i in final_data1:
    train_data.append(wordToTensor(i))

for i in final_data2:
    valid_data.append(wordToTensor(i))

for i in final_data3:
    test_data.append(wordToTensor(i))

for i in train_data:
    d1 = torch.FloatTensor.sub(i[0], i[1]).abs()
    d2 = torch.FloatTensor.sub(i[0], i[2]).abs()
    d3 = torch.FloatTensor.sub(i[1], i[2]).abs()
    s1 = torch.sum(d1)
    s2 = torch.sum(d1)
    s3 = torch.sum(d1)
    # print(k+p+q)
    diff1.append(s1)
    diff1.append(s2)
    diff1.append(s3)

for i in valid_data:
    d1 = torch.FloatTensor.sub(i[0], i[1]).abs()
    d2 = torch.FloatTensor.sub(i[0], i[2]).abs()
    d3 = torch.FloatTensor.sub(i[1], i[2]).abs()
    s1 = torch.sum(d1)
    s2 = torch.sum(d1)
    s3 = torch.sum(d1)
    # print(k+p+q)
    diff2.append(s1)
    diff2.append(s2)
    diff2.append(s3)

for i in test_data:
    d1 = torch.FloatTensor.sub(i[0], i[1]).abs()
    d2 = torch.FloatTensor.sub(i[0], i[2]).abs()
    d3 = torch.FloatTensor.sub(i[1], i[2]).abs()
    s1 = torch.sum(d1)
    s2 = torch.sum(d1)
    s3 = torch.sum(d1)
    # print(k+p+q)
    diff3.append(s1)
    diff3.append(s2)
    diff3.append(s3)

diff1 = zip(*[iter(diff1)]*3)
diff2 = zip(*[iter(diff2)]*3)
diff3 = zip(*[iter(diff3)]*3)

diff1 = [torch.FloatTensor(i) for i in diff1]
diff2 = [torch.FloatTensor(i) for i in diff2]
diff3 = [torch.FloatTensor(i) for i in diff3]

f1 = [torch.LongTensor(np.array(i)) for i in target1]
f2 = [torch.LongTensor(np.array(i)) for i in target2]
f3 = [torch.LongTensor(np.array(i)) for i in target3]

training_samples = utils_data.TensorDataset(torch.stack(train_data),torch.stack(f1))
validation_samples = utils_data.TensorDataset(torch.stack(valid_data),torch.stack(f2))
testing_samples = utils_data.TensorDataset(torch.stack(test_data),torch.stack(f3))
print(len(training_samples))
print(len(validation_samples))
print(len(testing_samples))
print(testing_samples)

dataloader1 = utils_data.DataLoader(training_samples,1)
dataloader2 = utils_data.DataLoader(validation_samples,1)
dataloader3 = utils_data.DataLoader(testing_samples,1)


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden,dr_data):
        combined = torch.cat((input.view(1,-1), hidden.view(1,-1), dr_data.view(1,-1)), 1)
        self.dropout = torch.nn.Dropout(0.2)
        hidden = self.i2h(combined.view(batch_size, -1))
        output = self.i2o(combined.view(batch_size, -1))
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


model = RNN(15, 50, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate():
    total_loss = 0
    correct = 0
    total = 5
    hidden = model.init_hidden()
    for i,j in dataloader3:
        #print('hidden', hidden)
        model.zero_grad()
        for k in diff3:
            for c in range(i.size()[0]):
                hidden = model.init_hidden()
                output, hidden = model(Variable(i[:,c].float()), hidden.float(), Variable(k))
                total_loss = criterion(output, Variable(j).view(-1))
            values, target = torch.max(output, 1)

                #total += j.size(0)
            #correct += (target == j).sum()
            # print('predicted value', target)
            # print('actual value', j)
            correct += (target.view(-1, 1) == Variable(j)).sum()
        # print('no of corrects', correct)

        return total_loss.data[0]/len(dataloader3), correct

def train():
    model.train()
    total_loss = 0
    correct = 0
    start_time = time.time()
    hidden = model.init_hidden()
    #for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    for i, j in dataloader1:
        model.zero_grad()
        for k in diff1:
            for c in range(i.size()[0]):
                #print('c',i[:,c])
                output, hidden = model(Variable(i[:,c].float()), hidden.float(), Variable(k))
                total_loss += criterion(output, Variable(j).view(-1))
                total_loss.backward(retain_graph=True)
                optimizer.step()

            values, target = torch.max(output, 1)

            correct += (target.view(-1, 1) == Variable(j)).sum()

    return total_loss.data[0]/ len(dataloader1)


def validate():
    model.eval()
    total_loss = 0
    correct = 0
    start_time = time.time()
    hidden = model.init_hidden()
    #for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    for i, j in dataloader2:
        model.zero_grad()
        for k in diff3:
            for c in range(i.size()[0]):
                output, hidden = model(Variable(i[:,c].float()), hidden.float(), Variable(k))
                total_loss += criterion(output, Variable(j).view(-1))

            values, target = torch.max(output, 1)
            correct += (target.view(-1, 1) == Variable(j)).sum()
    return total_loss.data[0]/ len(dataloader2)
# Loop over epochs.
lr = 0.01
best_val_loss = None
acc=0
nsim = 10
for sim in range(nsim):
    model = RNN(15, 50, 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    try:
        for epoch in range(1, 10):
            epoch_start_time = time.time()
            loss = train()
            val_loss = validate()

            if not best_val_loss or val_loss < best_val_loss:
                with open('res.pt', 'wb') as f:
                    torch.save(model, f)
                    best_val_loss = val_loss
                    # print('best val loss after ', best_val_loss)
                    # print('training loss', loss)

            else:
                lr /= 0.001
    except KeyboardInterrupt:
        print('Exiting from training early')

    # Load the best saved model.
    with open('res.pt', 'rb') as f:
        model = torch.load(f)
        #print('model ready', model)

    # Run on test data.
        #for i,j in dataloader3:
        test_loss, correct = evaluate()
        print('Simulation: ', sim, 'test loss', test_loss)
        print('Accuracy of the network {} %'.format((correct.data.numpy() * [100]) / len(dataloader3)))
        acc += (correct.data.numpy() * [100]) / len(dataloader3)

print('Avg Accuracy: ', acc / nsim)
