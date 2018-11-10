import numpy as np
from rnn import model
seed = 7
np.random.seed(seed)
from sklearn import GridSearchCV

# optimizer = ['Adagrad', 'Adadelta', 'Adam']
hidden_layers = [1,2,3]
hidden_neurons = [10,20,30,40,50]
dropout = [0.1,0.2,0.4]
lr = [0.01,0.1,0.2,0.4]
param_grid = dict(hidden_layers = hidden_layers, hidden_neurons = hidden_neurons, dropout = dropout, lr= lr)
grid = GridSearchCV(estimator=model, param_grid=param_grid,n_jobs = 1)
grid_result = grid.fit(grid)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))