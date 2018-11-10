#early fusion

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
parser.add_argument('--model', type=str, default='lstm',
                     help='type of recurrent net (rnn, gru, lstm)')
# parser.add_argument('--model', type=str, default='LSTM',
#                     help='type of recurrent net (LSTM, GRU,RNN_TANH or RNN_RELU)')
parser.add_argument('--emsize', type=int, default=20,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=20,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=2,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

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


#abb_train_patterns = np.array(['dhh', 'dee', 'dbb', 'dcc', 'ahh', 'aee', 'abb', 'acc', 'fhh', 'fee', 'fbb', 'fcc', 'ghh', 'gee', 'gbb', 'gcc'])
#abb_test_patterns = np.array(['ijj', 'kll'])
#
# kf = KFold(n_splits=10)
#
# train_data, test_data, train_d, test_d, training_data, teste_data = [], [], [], [], [], []
#
# for train, test in kf.split(aba_patterns):
#     train_data.append(train)
#     test_data.append(test)
#
# train_d = list(itertools.chain.from_iterable(train_data))
# test_d = list(itertools.chain.from_iterable(test_data))
#
# for i in train_d:
#     training_data.append(aba_patterns[i])
#
# for i in test_d:
#     teste_data.append(aba_patterns[i])
#
#
# tr_data, ts_data, vs_data, vs_tar_data, tr_tar_data, ts_tar_data = [], [], [], [], [], []
#
# len_train_data = len(training_data)
# m_index = int(len_train_data / 10)
# k_index = len_train_data - m_index
# train_data = training_data[:k_index]
# valid_data = training_data[k_index:]

aba_train_patterns = np.array([['a','h','a'], ['a','e','a'], ['a','b','a'], ['a', 'c', 'a'], ['d', 'h', 'd'], ['d', 'e', 'd'], ['d', 'b' , 'd'], ['d', 'c', 'd'], ['f', 'h', 'f'], ['f', 'e', 'f'], ['f', 'b', 'f'], ['f', 'c', 'f'], ['g', 'h', 'g'], ['g', 'e', 'g']])

#abb_train_patterns = np.array([['a','h','h'], ['a','e','e'], ['a','b','b'], ['a', 'c', 'c'], ['d', 'h', 'h'], ['d', 'e', 'e'], ['d', 'b' , 'b'], ['d', 'c', 'c'], ['f', 'h', 'h'], ['f', 'e', 'e'], ['f', 'b', 'b'], ['f', 'c', 'c'], ['g', 'h', 'h'], ['g', 'e', 'e']])

aba_valid_patterns = np.array([['g', 'b', 'g'], ['g', 'c', 'g']])

aba_test_patterns = np.array([['i','j','i'], ['k', 'l', 'k']])

#abb_train_patterns = np.array([['a','h','h'], ['a','e','e'], ['a','b','b'], ['a', 'c', 'c'], ['d', 'h', 'h'], ['d', 'e', 'e'], ['d', 'b' , 'b'], ['d', 'c', 'c'], ['f', 'h', 'h'], ['f', 'e', 'e'], ['f', 'b', 'b'], ['f', 'c', 'c'], ['g', 'h', 'h'], ['g', 'e', 'e']])

#abb_valid_patterns = np.array([['g', 'b', 'b'], ['g', 'c', 'c']])

#abb_test_patterns = np.array([['i','j','j'], ['k', 'l', 'l']])


alphabet_size = 12
sample_space = ['a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
sample_space_len = len(sample_space)
# decoding char from int
char_from_int = dict((i, c) for i, c in enumerate(sample_space))

char_to_int = dict((c, i) for i, c in enumerate(sample_space))

training_data, validation_data, testing_data = [], [], []
tr_data, ts_data, vs_data, vs_tar_data, tr_tar_data, ts_tar_data = [], [], [], [], [], []

for i in aba_train_patterns:
    training_data.append([char_to_int[char] for char in i])
for i in aba_valid_patterns:
    validation_data.append([char_to_int[char] for char in i])
for i in aba_test_patterns:
    testing_data.append([char_to_int[char] for char in i])

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


# #create one hot encodings of data
# def letterToIndex(letter):
#     return sample_space.find(letter)
#
#
# #< 1 * alphabet_size > one _hot_encoded
# def letterToTensor(letter):
#     tensor = torch.zeros(1, sample_space_len)
#     tensor[0][letterToIndex(letter)] = 1
#     return tensor

#print(letterToTensor('a'))


# for i in aba_train_patterns:
#     train_data.append(i[:-1])
#
# for i in aba_train_patterns:
#     target_data.append(i[-1:])


out, inp1, inp2 = [], [], []


# for i in train_data:
#     inp1.append(letterToTensor(i[0][0]))
#     inp2.append(letterToTensor(i[1][0]))
#
# for i in target_data:
#     out.append(letterToTensor(i))


# for i in tr_data:
#     inp1.append(i[0][0])
#     inp2.append(i[1][0])
#
#
# for i in tr_tar_data:
#     out.append(i)
#
# out_t, inp3, inp4 = [], [], []
#
# for i in ts_data:
#     inp3.append(i[0][0])
#     inp4.append(i[1][0])
#
# for i in ts_tar_data:
#     out_t.append(i)
#
# out_t_t, inp5, inp6 = [], [], []
#
# for i in vs_data:
#     inp5.append(i[0][0])
#     inp6.append(i[1][0])
#
# for i in vs_tar_data:
#     out_t_t.append(i)

diff1, diff2, diff3 = [], [], []

for i in tr_data:
   diff1.append(abs(i[0]-i[1]))

for i in vs_data:
   diff2.append(abs(i[0]-i[1]))

for i in ts_data:
   diff3.append(abs(i[0]-i[1]))

# x1 = Variable(torch.cat(inp1), requires_grad=True)
# x2 = Variable(torch.cat(inp2), requires_grad=True)
# y = Variable(torch.cat(out), requires_grad=False)
#
#
# x_1 = Variable(torch.cat(inp3), requires_grad=True)
# x_2 = Variable(torch.cat(inp4), requires_grad=True)
# y_y = Variable(torch.cat(out_t), requires_grad=False)

DR_train_list, DR_valid_list, DR_test_list = [],[],[]

diff1 = [[i] for i in diff1]
diff2 = [[i] for i in diff2]
diff3 = [[i] for i in diff3]


for i,j in zip(tr_data, diff1):
    DR_train_list.append(i+j)

for i,j in zip(vs_data, diff2):
    DR_valid_list.append(i+j)

for i,j in zip(ts_data, diff3):
    DR_test_list.append(i+j)

#t1 = [torch.LongTensor(np.array(i)) for i in tr_data]
t1 = [torch.LongTensor(np.array(i)) for i in DR_train_list]
f1 = [torch.LongTensor(np.array(i)) for i in tr_tar_data]

t2 = [torch.LongTensor(np.array(i)) for i in DR_valid_list]
#t2 = [torch.LongTensor(np.array(i)) for i in vs_data]
f2 = [torch.LongTensor(np.array(i)) for i in vs_tar_data]

t3 = [torch.LongTensor(np.array(i)) for i in DR_test_list]
#t3 = [torch.LongTensor(np.array(i)) for i in ts_data]
f3 = [torch.LongTensor(np.array(i)) for i in ts_tar_data]

dr_units = [[0,0,0,0,0,0,0,0,0.8,0,0,0], [0,0,0,0,0,0,0,0,0,0,0.8,0]]
dr_data1 = [torch.FloatTensor(i) for i in dr_units]

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
        encoded = self.embed(input.view(1,-1))
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        out = self.h2o(output.view(batch_size, -1))
        out1 = F.softmax(out)
        return out1, hidden

    def init_hidden(self, batch_size):

        if self.model == "lstm":
            return (torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

        return torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


class CharRNN1(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, model, n_layers):
        super(CharRNN1, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = torch.nn.Dropout(0.2)

        self.embed = torch.nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = torch.nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(hidden_size, hidden_size, n_layers)
        elif self.model == "rnn":
            self.rnn = torch.nn.RNN(hidden_size, hidden_size, n_layers)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, dr_data):
        batch_size = input.size(0)
        encoded = self.embed(input.view(1,-1))
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        out = self.h2o(output.view(batch_size, -1))
        out1 = F.softmax(out)
        #output = torch.add(out1,dr_data)
        output = torch.cat((out1,dr_data),0)
        return output, hidden

    # def forward(self, input, hidden):
    #     encoded = self.embed(input.view(1, -1))
    #     output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
    #     output = self.h2o(output.view(1, -1))
    #     return output, hidden

    def init_hidden(self, batch_size):

        if self.model == "lstm":
            return (torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

        return torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


model = CharRNN(alphabet_size, 20, alphabet_size, args.model, batch_size)
# model1 = CharRNN1(alphabet_size, 20, alphabet_size, args.model, batch_size)

#model = model.RNNModel(args.model, alphabet_size, 20, alphabet_size,  args.batch_size, dropout=0.5)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
    hidden = model.init_hidden(args.batch_size)
    for i,j in dataloader3:
        # for l in dr_data1:
            hidden = repackage_hidden(hidden)
            #print('hidden', hidden)
            model.zero_grad()
            for c in range(i.size()[0]):
                output, hidden = model(Variable(i)[:,c], hidden)
                # o = [x+y for x,y in zip(output, Variable(i).view(-1,12))]
                # output= torch.stack(o)
            #print('o', torch.max(output,1))
                #print('output afr', output.view(1,-1))
                output = output.view(1,-1)
                total_loss = criterion(output.view(1,-1), Variable(j).view(-1))
            values, target = torch.max(output, 1)

            total += j.size(0)
                # print('total', total)
                # print('predicted value', target)
                # print('actual value', j)
        # print('tar', target.view(-1, 1).data.numpy())
        #correct += (target == j).sum()
            correct += (target.view(-1, 1) == Variable(j)).sum()
            #print('total_loss', total_loss[0] / args.bptt)
            #print('correct',correct)

    return total_loss[0]/args.bptt, correct


def train():
    model.train()
    total_loss = 0
    correct = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    #for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    for i, j in dataloader1:
        #print('i', i)
        #print('j', j)
        model.zero_grad()

        for c in range(i.size()[0]):
            output, hidden = model(Variable(i)[:,c], hidden)
            total_loss += criterion(output, Variable(j).view(-1))
        total_loss.backward()
        optimizer.step()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)
        #
        # total_loss += loss.data

        # if 1 % args.log_interval == 0 and 1 > 0:
        #     cur_loss = total_loss[0] / args.log_interval
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
        #             'loss {:5.2f} '.format(
        #         epoch, 1, len(train_data) // args.bptt, lr,
        #         cur_loss))
        #     total_loss = 0
        #     start_time = time.time()
        values, target = torch.max(output, 1)
        #print('j', j)
        #print('tar', target.view(-1, 1).data.numpy())
        correct += (target.view(-1, 1) == Variable(j)).sum()
        return total_loss.data[0]/ args.bptt, correct


def validate():
    # Turn on training mode which enables dropout.
    model.eval()
    total_loss = 0
    correct = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    #for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    for i, j in dataloader2:
        hidden = repackage_hidden(hidden)
        model.zero_grad()

        for c in range(i.size()[0]):
            output, hidden = model(Variable(i)[:,c], hidden)
            total_loss += criterion(output, Variable(j).view(-1))

        values, target = torch.max(output, 1)

        correct += (target.view(-1, 1) == Variable(j)).sum()

        return total_loss.data[0]/ args.bptt, correct
# Loop over epochs.
lr = args.lr
best_val_loss = None
acc = 0
nsim = 10
for sim in range(nsim):
    model = CharRNN(alphabet_size, 20, alphabet_size, args.model, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    try:
        for epoch in range(1, 10):
            epoch_start_time = time.time()
            loss, cr = train()
            #print('Corrected accuracy', cr)
            #for i,j in dataloader2:
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
        #print('model ready', model)

    # # Run on test data.
    #     #for i,j in dataloader3:
    #     test_loss, correct = evaluate()
    #     print('-' * 89)
    #     print('-' * 89)
    #     print('test loss', test_loss)
    #     print('Accuracy of the network {} %'.format((correct.data.numpy()* [100]) / args.bptt))

        test_loss, correct = evaluate()
        print('Simulation: ', sim, 'test loss', test_loss)
        print('Accuracy of the network {} %'.format((correct.data.numpy() * [100]) / len(dataloader3)))
        acc += (correct.data.numpy() * [100]) / len(dataloader3)

print('Avg Accuracy: ', acc / nsim)
