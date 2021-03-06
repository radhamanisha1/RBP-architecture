
import random
import itertools
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import time
import argparse
import torch.utils.data as utils_data
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


parser = argparse.ArgumentParser(description='PyTorch prediction')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

batch_size = 1

#training
# le - a
# li - b
# we - c
# wi - d
# je - e
# ji - f
# de - g
# di - h

#testing
# ba - i
# po -j
# ko - k
# ga - l


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


random.shuffle(train_aba_patterns)

for i in data2:
    if (i[0]==i[2] and i[0]!= i[1] and i[1]!=i[2]):
        test_aba_patterns.append(i)

random.shuffle(test_aba_patterns)

aba_train_patterns = train_aba_patterns[:30]
aba_valid_patterns = test_aba_patterns[:15]
aba_test_patterns = test_aba_patterns[16:30]


alphabet_size = 12
sample_space = 'cdefgabhijkl'
sample_space_len = len(sample_space)
# decoding char from int
char_from_int = dict((i, c) for i, c in enumerate(sample_space))

char_to_int = dict((c, i) for i, c in enumerate(sample_space))

training_data, validation_data, testing_data = [], [], []
tr_data, ts_data, vs_data, vs_tar_data, tr_tar_data, ts_tar_data = [], [], [], [], [], []


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

diff1, diff2, diff3 = [], [], []

for i in aba_train_patterns:
    training_data.append([wordToTensor(char) for char in i])
for i in aba_valid_patterns:
    validation_data.append([wordToTensor(char) for char in i])
for i in aba_test_patterns:
    testing_data.append([wordToTensor(char) for char in i])

# print(training_data)

for i in training_data:
    tr_data.append(i[:-1])

for i in training_data:
    tr_tar_data.append(i[-1:])

for i in validation_data:
    vs_data.append(i[:-1])

for i in validation_data:
    vs_tar_data.append(i[-1:])

for i in testing_data:
    ts_data.append(i[:-1])

for i in testing_data:
    ts_tar_data.append(i[-1:])

# print(tr_data[0][0])

# print(tr_data[:1])

flat_list1 = [item for sublist in tr_data for item in sublist]
flat_list2 = [item for sublist in vs_data for item in sublist]
flat_list3 = [item for sublist in ts_data for item in sublist]

tr_tar_data = [item for sublist in tr_tar_data for item in sublist]
vs_tar_data = [item for sublist in vs_tar_data for item in sublist]
ts_tar_data = [item for sublist in ts_tar_data for item in sublist]


# print(flat_list)
train_data = [i.numpy() for i in flat_list1]
valid_data = [i.numpy() for i in flat_list2]
test_data = [i.numpy() for i in flat_list3]

train_targets = [i.numpy() for i in tr_tar_data]
valid_targets = [i.numpy() for i in vs_tar_data]
test_targets = [i.numpy() for i in ts_tar_data]


train_data = zip(*[iter(train_data)]*2)
valid_data = zip(*[iter(valid_data)]*2)
test_data = zip(*[iter(test_data)]*2)

# print(tr_tar_data)
t1 = [torch.LongTensor((i)) for i in train_data]
#t1 = [torch.LongTensor(np.array(i)) for i in DR_train_list]
f1 = [torch.LongTensor((i)) for i in train_targets]

#t2 = [torch.LongTensor(np.array(i)) for i in DR_valid_list]
t2 = [torch.LongTensor(np.array(i)) for i in valid_data]
f2 = [torch.LongTensor(np.array(i)) for i in valid_targets]

#t3 = [torch.LongTensor(np.array(i)) for i in DR_test_list]
t3 = [torch.LongTensor(np.array(i)) for i in test_data]
f3 = [torch.LongTensor(np.array(i)) for i in test_targets]

training_samples = utils_data.TensorDataset(torch.stack(t1),torch.stack(f1))
validation_samples = utils_data.TensorDataset(torch.stack(t2),torch.stack(f2))
testing_samples = utils_data.TensorDataset(torch.stack(t3),torch.stack(f3))

dataloader1 = utils_data.DataLoader(training_samples,1)
dataloader2 = utils_data.DataLoader(validation_samples,1)
dataloader3 = utils_data.DataLoader(testing_samples,1)

class CharRNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, model, n_layers):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = torch.nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = torch.nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(hidden_size, hidden_size, n_layers)
        elif self.model == "rnn":
            self.rnn = torch.nn.RNN(hidden_size, hidden_size, n_layers)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = input.view(1,-1)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        out = self.h2o(output.view(1, -1))
        out1 = F.softmax(out)
        return out1, hidden

    def init_hidden(self, batch_size):

        if self.model == "lstm":
            return (torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

        return torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


model = CharRNN(alphabet_size, 50, alphabet_size, 'rnn', batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

criterion = torch.nn.CrossEntropyLoss()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def accu(y_true, y_pred):
    y_pred = np.concatenate(tuple(y_pred))
    y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(y_pred.shape)
    return (y_true == y_pred).sum() / float(len(y_true))

print('len of dataloader3', len(dataloader3))


def evaluate():
    # Turn on evaluation mode which disables dropout.
    #model.eval()
    total_loss = 0
    correct = 0
    total = 0
    hidden = model.init_hidden(batch_size)
    for i,j in dataloader3:
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            for c in range(i.size()[0]):
                output, hidden = model(Variable(i)[:,c].float(), hidden)
                output = output.view(1,-1)
                total_loss += criterion(output, Variable(j).float())
                values, target = torch.max(output, 1)

            total += j.size(0)

            correct += (target.view(-1, 1) == Variable(j)).sum()


    return total_loss.data[0]/len(dataloader3), correct


def train():
    model.train()
    total_loss = 0
    correct = 0
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    #for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    for i, j in dataloader1:
        #print('i', i)
        #print('j', j)
        model.zero_grad()

        for c in range(i.size()[0]):
            output, hidden = model(Variable(i)[:,c].float(), hidden)
            total_loss += criterion(output, Variable(j).float())
        total_loss.backward()
        optimizer.step()

        values, target = torch.max(output, 1)

        correct += (target.view(-1, 1) == Variable(j)).sum()
        return total_loss.data[0]/ len(dataloader1), correct


def validate():
    # Turn on training mode which enables dropout.
    model.eval()
    total_loss = 0
    correct = 0
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    #for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    for i, j in dataloader2:
        hidden = repackage_hidden(hidden)
        model.zero_grad()

        for c in range(i.size()[0]):
            output, hidden = model(Variable(i)[:,c].float(), hidden)
            total_loss += criterion(output, Variable(j).float())

        values, target = torch.max(output, 1)

        correct += (target.view(-1, 1) == Variable(j)).sum()

        return total_loss.data[0]/ len(dataloader2), correct
# Loop over epochs.
lr = 0.1
best_val_loss = None
acc = 0
nsim = 10
for sim in range(nsim):
    model = CharRNN(alphabet_size, 12, alphabet_size, 'rnn', batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    try:
        for epoch in range(1, 10):
            epoch_start_time = time.time()
            loss, cr = train()

            val_loss, corr = validate()
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                    best_val_loss = val_loss
                    print('best val loss after ', best_val_loss)
                    print('training loss', loss)

            else:
                lr /= 0.001
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

        test_loss, correct = evaluate()
        print('Simulation: ', sim, 'test loss', test_loss)
        print('Accuracy of the network {} %'.format((correct.data.numpy() * [100]) / len(dataloader3)))
        acc += (correct.data.numpy() * [100]) / len(dataloader3)

print('Avg Accuracy: ', acc / nsim)
