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
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
import random
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score, accuracy_score


batch_size = 1

alp1 = ['c','d','e','f', 'i', 'j']
alp2 = ['g', 'a', 'b','h', 'k', 'l']
p1 = list(itertools.product(alp1,repeat=3))
p2 = list(itertools.product(alp2,repeat=3))

data1 = [list(i) for i in p1]
data2 = [list(i) for i in p2]

train_aba_patterns, train_abb_patterns, train_abc_patterns, train_aaa_patterns = [], [], [], []
test_aba_patterns, test_abb_patterns, test_abc_patterns, test_aaa_patterns = [], [], [], []
train_aab_patterns, test_aab_patterns = [],[]

for i in data1:
    if (i[0]==i[2] and i[0]!= i[1] and i[1]!=i[2]):
        train_aba_patterns.append(i)
    elif (i[0]!= i[1] and i[0]!=i[2] and i[1]==i[2]):
        train_abb_patterns.append(i)
    elif (i[0]!=i[1] and i[1]!=i[2] and i[0] != i[2]):
        train_abc_patterns.append(i)
    elif (i[0] == i[1] and i[1] == i[2] and i[0] == i[2]):
        train_aaa_patterns.append(i)
    elif (i[0] == i[1] and i[1] == i[2] and i[0] != i[2]):
        train_aab_patterns.append(i)

test_abb_patterns = test_abb_patterns[:8]
test_aaa_patterns = test_aaa_patterns[:6]
test_abc_patterns = test_abc_patterns[:8]
test_aab_patterns = test_aab_patterns[:8]

testing_data = test_aaa_patterns[:3]+ test_aba_patterns[:15] + test_abc_patterns[:4] + test_abb_patterns[:4] + test_aab_patterns[:4]

validation_data = test_aaa_patterns[3:]+ test_aba_patterns[15:] + test_abc_patterns[4:] + test_abb_patterns[4:] + test_aab_patterns[4:]


target1, target2, target3 = [],[],[]
for i in training_data:
    if (i[0] == i[2] and i[0] != i[1]):
        target1.append([1])
    else:
        target1.append([0])

for i in validation_data:
    if (i[0] == i[2] and i[0] != i[1]):
        target2.append([1])
    else:
        target2.append([0])

for i in testing_data:
    if (i[0] == i[2] and i[0] != i[1]):
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

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        for i in range(hidden_size):
            for j in range(input_size):
                with torch.no_grad():
                    if (j % 2 == 0):
                        self.fc1.weight[i][j] = 1
                    else:
                        self.fc1.weight[i][j] = -1

    # def dr_trainable(self, x):
    #     out1 = self.relu(self.fc1(x))
    #     out2 = self.relu(self.fc2(out1))
    #     out3 = self.sigm(self.fc3(out2))

    def forward(self, x):
        out = self.fc1(x.view(1, -1))
        out = self.relu(out.view(1, -1))
        out = self.fc2(out.view(1, -1))
        out = F.softmax(out)
        return out


# def weights_init(m):
#     if isinstance(m, torch.nn.Linear):
#         self.fc1(m.weight.data)
#         self.fc1(m.bias.data)


model = NeuralNet(9, 12, 2)
# model.apply(weights_init)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.1)
criterion = torch.nn.CrossEntropyLoss()


def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    correct = 0
    predictions, actuals = [], []
    for i, j in dataloader2:
        model.zero_grad()
        output = model(Variable(i.float()))
        total_loss += criterion(output, Variable(j).view(-1))
        values, target = torch.max(output, 1)
        correct += (target.view(-1, 1) == Variable(j)).sum()
            # predictions.append(target.data.numpy())
            # actuals.append(j.view(-1).numpy())

    print('no of corrects', correct)
    print(total_loss)
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
for sim in range(nsim):
    model = NeuralNet(12, 12, 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    try:
        for epoch in range(1, 10):
            epoch_start_time = time.time()
            loss = train()
            # print(loss)
            with open('1.pt', 'wb') as f:
                torch.save(model, f)

    except KeyboardInterrupt:
        print('Exiting from training early')

    # Load the best saved model.
    with open('1.pt', 'rb') as f:
        model = torch.load(f)
        # print('model ready', model)

        # Run on test data.
        # for i,j in dataloader3:
        test_loss, correct = evaluate()
        print('test loss', test_loss)
        print('Accuracy of the network {} %'.format((correct.data.numpy() * [100]) / len(dataloader2)))
