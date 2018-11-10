#early fusion

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

train_data, valid_data, test_data = [],[],[]

diff1, diff2, diff3 = [], [], []

for i in final_data1:
    train_data.append(wordToTensor(i))

for i in final_data2:
    valid_data.append(wordToTensor(i))

for i in final_data3:
    test_data.append(wordToTensor(i))


for i in train_data:
   diff1.append(torch.FloatTensor.sub(i[0],i[1]).abs())

for i in train_data:
   diff1.append(torch.FloatTensor.sub(i[0],i[2]).abs())

for i in train_data:
   diff1.append(torch.FloatTensor.sub(i[1],i[2]).abs())

for i in valid_data:
   diff2.append(torch.FloatTensor.sub(i[0],i[1]).abs())

for i in valid_data:
   diff2.append(torch.FloatTensor.sub(i[0],i[2]).abs())

for i in valid_data:
   diff2.append(torch.FloatTensor.sub(i[1],i[2]).abs())


for i in test_data:
   diff3.append(torch.FloatTensor.sub(i[0],i[1]).abs())

for i in test_data:
   diff3.append(torch.FloatTensor.sub(i[0],i[2]).abs())

for i in test_data:
   diff3.append(torch.FloatTensor.sub(i[1],i[2]).abs())

DR_train_list, DR_valid_list, DR_test_list = [],[],[]


train_data = [i.numpy() for i in train_data]
valid_data = [i.numpy() for i in valid_data]
test_data = [i.numpy() for i in test_data]


diff1 = [d.numpy() for d in diff1]
diff2 = [d.numpy() for d in diff2]
diff3 = [d.numpy() for d in diff3]


new_train_data = zip(train_data, diff1)
new_valid_data = zip(valid_data, diff2)
new_test_data = zip(test_data, diff3)

## new_train = np.concatenate(new_train_data).ravel()
new_train, new_valid, new_test = [], [], []

t1 = [list(i) for i in new_train_data]
t2 = [list(i) for i in new_valid_data]
t3 = [list(i) for i in new_test_data]

q1 = list(itertools.chain(*t1))
q2 = list(itertools.chain(*t2))
q3 = list(itertools.chain(*t3))


for i in q1:
    new_train.extend(np.row_stack(i))
for i in q2:
    new_valid.extend(np.row_stack(i))
for i in q3:
    new_test.extend(np.row_stack(i))


train_data_modified = zip(*[iter(new_train)]*6)
valid_data_modified = zip(*[iter(new_valid)]*6)
test_data_modified = zip(*[iter(new_test)]*6)


t1 = [torch.LongTensor(np.array(i)) for i in train_data_modified]
f1 = [torch.LongTensor(np.array(i)) for i in target1[:40]]

t2 = [torch.LongTensor(np.array(i)) for i in valid_data_modified]
f2 = [torch.LongTensor(np.array(i)) for i in target2[:20]]

t3 = [torch.LongTensor(np.array(i)) for i in test_data_modified]
f3 = [torch.LongTensor(np.array(i)) for i in target3[:20]]


training_samples = utils_data.TensorDataset(torch.stack(t1),torch.stack(f1))
validation_samples = utils_data.TensorDataset(torch.stack(t2),torch.stack(f2))
testing_samples = utils_data.TensorDataset(torch.stack(t3),torch.stack(f3))


dataloader1 = utils_data.DataLoader(training_samples,1)
dataloader2 = utils_data.DataLoader(validation_samples,1)
dataloader3 = utils_data.DataLoader(testing_samples,1)

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x.view(1,-1))
        out = self.relu(out.view(1,-1))
        out = self.fc2(out.view(1,-1))

        return out

model = NeuralNet(72, 50, 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    correct = 0
    predictions, actuals = [], []
    for i, j in dataloader3:
        model.zero_grad()
        output = model(Variable(i.float()))
        total_loss += criterion(output, Variable(j).view(-1))
        values, target = torch.max(output, 1)
        correct += (target.view(-1, 1) == Variable(j)).sum()
            # predictions.append(target.data.numpy())
            # actuals.append(j.view(-1).numpy())

    # print('no of corrects', correct)
    # print(total_loss)
    return total_loss.data[0] / len(dataloader3), correct


def validate():
    total_loss = 0
    correct = 0
    predictions, actuals = [], []
    for i, j in dataloader2:
        model.zero_grad()
        # for k in diff2:
        output = model(Variable(i.float()))
        total_loss += criterion(output, Variable(j).view(-1))
        values, target = torch.max(output, 1)
        correct += (target.view(-1, 1) == Variable(j)).sum()
            # predictions.append(target.data.numpy())
            # actuals.append(j.view(-1).numpy())
    return total_loss.data[0] / len(dataloader2), correct

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    correct = 0
    for i, j in dataloader1:
        model.zero_grad()
        output = model(Variable(i.float()))
        total_loss = criterion(output, Variable(j).view(-1))
        total_loss.backward(retain_graph=True)
        optimizer.step()

    values, target = torch.max(output, 1)
    correct += (target.view(-1, 1) == Variable(j)).sum()

    return total_loss.data[0] / len(dataloader1)


# Loop over epochs.
lr = 0.01
nsim = 10
acc=0
best_val_loss = None

for sim in range(nsim):
    model = NeuralNet(72, 50, 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
                    print('best val loss after ', best_val_loss)
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 0.1
    except KeyboardInterrupt:
        print('Exiting from training early')

    # Load the best saved model.
    with open('res.pt', 'rb') as f:
        model = torch.load(f)
        # print('model ready', model)

        # Run on test data.
        # for i,j in dataloader3:
        test_loss, correct = evaluate()
        print('Simulation: ', sim, 'test loss', test_loss)
        print('Accuracy of the network {} %'.format((correct.data.numpy() * [100]) / len(dataloader3)))
        acc += (correct.data.numpy() * [100]) / len(dataloader3)

print('Avg Accuracy: ', acc / nsim)
