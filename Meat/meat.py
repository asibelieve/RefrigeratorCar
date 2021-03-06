# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 00:32:33 2021

@author: Administrator
"""
import os
import warnings
from abc import ABCMeta

import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer, AdamOptimizer
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.utils import gen_batches, check_random_state
from sklearn.utils import shuffle
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

basePath = os.getcwd() + '/Meat/'

_STOCHASTIC_SOLVERS = ['sgd', 'adam']


def _pack(coefs_, intercepts_):
    return np.hstack([l.ravel() for l in coefs_ + intercepts_])


class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def __init__(self, hidden_layer_sizes, activation, solver,
                 alpha, batch_size, learning_rate, learning_rate_init, power_t,
                 max_iter, loss, shuffle, random_state, tol, verbose,
                 warm_start, momentum, nesterovs_momentum, early_stopping,
                 validation_fraction, beta_1, beta_2, epsilon,
                 n_iter_no_change):
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.loss = loss
        self.hidden_layer_sizes = hidden_layer_sizes
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change

    def _unpack(self, packed_parameters):
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            self.coefs_[i] = np.reshape(packed_parameters[start:end], shape)

            start, end = self._intercept_indptr[i]
            self.intercepts_[i] = packed_parameters[start:end]

    def _forward_pass(self, activations):
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i],
                                                 self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                activations[i + 1] = hidden_activation(activations[i + 1])

        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        activations[i + 1] = output_activation(activations[i + 1])

        return activations

    def _compute_loss_grad(self, layer, n_samples, activations, deltas,
                           coef_grads, intercept_grads):
        coef_grads[layer] = safe_sparse_dot(activations[layer].T,
                                            deltas[layer])
        coef_grads[layer] += (self.alpha * self.coefs_[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)

        return coef_grads, intercept_grads

    def _loss_grad_lbfgs(self, packed_coef_inter, X, y, activations, deltas,
                         coef_grads, intercept_grads):
        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self._backprop(
            X, y, activations, deltas, coef_grads, intercept_grads)
        self.n_iter_ += 1
        grad = _pack(coef_grads, intercept_grads)
        return loss, grad

    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        values = np.sum(
            np.array([np.dot(s.ravel(), s.ravel()) for s in self.coefs_]))
        loss += (0.5 * self.alpha) * values / n_samples

        last = self.n_layers_ - 2

        deltas[last] = activations[-1] - y

        coef_grads, intercept_grads = self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)

        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self.activation]
            inplace_derivative(activations[i], deltas[i - 1])

            coef_grads, intercept_grads = self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

        return loss, coef_grads, intercept_grads

    def _initialize(self, y, layer_units):
        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]

        self.n_layers_ = len(layer_units)

        if not is_classifier(self):
            self.out_activation_ = 'identity'
        elif self._label_binarizer.y_type_ == 'multiclass':
            self.out_activation_ = 'softmax'
        else:
            self.out_activation_ = 'logistic'

        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(layer_units[i],
                                                        layer_units[i + 1])
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
            else:
                self.best_loss_ = np.inf

    def _init_coef(self, fan_in, fan_out):
        # Use the initialization method recommended by
        # Glorot et al.
        factor = 6.
        if self.activation == 'logistic':
            factor = 2.
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.uniform(-init_bound, init_bound,
                                               (fan_in, fan_out))
        intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                    fan_out)
        return coef_init, intercept_init

    def _fit(self, X, y, incremental=False):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        # Validate input parameters.
        self._validate_hyperparameters()
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." %
                             hidden_layer_sizes)

        X, y = self._validate_input(X, y, incremental)
        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = ([n_features] + hidden_layer_sizes +
                       [self.n_outputs_])

        # check random state
        self._random_state = check_random_state(self.random_state)

        if not hasattr(self, 'coefs_') or (not self.warm_start and not
        incremental):
            # First time training the model
            self._initialize(y, layer_units)

        # lbfgs does not support mini-batches
        if self.solver == 'lbfgs':
            batch_size = n_samples
        elif self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > n_samples:
                warnings.warn("Got `batch_size` less than 1 or larger than "
                              "sample size. It is going to be clipped")
            batch_size = np.clip(self.batch_size, 1, n_samples)

        # Initialize lists
        activations = [X]
        activations.extend(np.empty((batch_size, n_fan_out))
                           for n_fan_out in layer_units[1:])
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_,
                                                            n_fan_out_ in zip(layer_units[:-1],
                                                                              layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                           layer_units[1:]]

        # Run the Stochastic optimization solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(X, y, activations, deltas, coef_grads,
                                 intercept_grads, layer_units, incremental)

        # Run the LBFGS solver
        elif self.solver == 'lbfgs':
            self._fit_lbfgs(X, y, activations, deltas, coef_grads,
                            intercept_grads, layer_units)
        return self

    def _validate_hyperparameters(self):
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be either True or False, got %s." %
                             self.shuffle)
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0, got %s." % self.max_iter)
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0, got %s." % self.alpha)
        if (self.learning_rate in ["constant", "invscaling", "adaptive"] and
                self.learning_rate_init <= 0.0):
            raise ValueError("learning_rate_init must be > 0, got %s." %
                             self.learning_rate)
        if self.momentum > 1 or self.momentum < 0:
            raise ValueError("momentum must be >= 0 and <= 1, got %s" %
                             self.momentum)
        if not isinstance(self.nesterovs_momentum, bool):
            raise ValueError("nesterovs_momentum must be either True or False,"
                             " got %s." % self.nesterovs_momentum)
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be either True or False,"
                             " got %s." % self.early_stopping)
        if self.validation_fraction < 0 or self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be >= 0 and < 1, "
                             "got %s" % self.validation_fraction)
        if self.beta_1 < 0 or self.beta_1 >= 1:
            raise ValueError("beta_1 must be >= 0 and < 1, got %s" %
                             self.beta_1)
        if self.beta_2 < 0 or self.beta_2 >= 1:
            raise ValueError("beta_2 must be >= 0 and < 1, got %s" %
                             self.beta_2)
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0, got %s." % self.epsilon)
        if self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be > 0, got %s."
                             % self.n_iter_no_change)

        # raise ValueError if not registered
        supported_activations = ('identity', 'logistic', 'tanh', 'relu')
        if self.activation not in supported_activations:
            raise ValueError("The activation '%s' is not supported. Supported "
                             "activations are %s." % (self.activation,
                                                      supported_activations))
        if self.learning_rate not in ["constant", "invscaling", "adaptive"]:
            raise ValueError("learning rate %s is not supported. " %
                             self.learning_rate)
        supported_solvers = _STOCHASTIC_SOLVERS + ["lbfgs"]
        if self.solver not in supported_solvers:
            raise ValueError("The solver %s is not supported. "
                             " Expected one of: %s" %
                             (self.solver, ", ".join(supported_solvers)))

    def _fit_lbfgs(self, X, y, activations, deltas, coef_grads,
                   intercept_grads, layer_units):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run LBFGS
        packed_coef_inter = _pack(self.coefs_,
                                  self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        optimal_parameters, self.loss_, d = fmin_l_bfgs_b(
            x0=packed_coef_inter,
            func=self._loss_grad_lbfgs,
            maxfun=self.max_iter,
            iprint=iprint,
            pgtol=self.tol,
            args=(X, y, activations, deltas, coef_grads, intercept_grads))

        self._unpack(optimal_parameters)

    def _fit_stochastic(self, X, y, activations, deltas, coef_grads,
                        intercept_grads, layer_units, incremental):

        if not incremental or not hasattr(self, '_optimizer'):
            params = self.coefs_ + self.intercepts_

            if self.solver == 'sgd':
                self._optimizer = SGDOptimizer(
                    params, self.learning_rate_init, self.learning_rate,
                    self.momentum, self.nesterovs_momentum, self.power_t)
            elif self.solver == 'adam':
                self._optimizer = AdamOptimizer(
                    params, self.learning_rate_init, self.beta_1, self.beta_2,
                    self.epsilon)

        # early_stopping in partial_fit doesn't make sense
        early_stopping = self.early_stopping and not incremental
        if early_stopping:
            # don't stratify in multilabel classification
            should_stratify = is_classifier(self) and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            X, X_val, y, y_val = train_test_split(
                X, y, random_state=self._random_state,
                test_size=self.validation_fraction,
                stratify=stratify)
            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None

        n_samples = X.shape[0]

        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            for it in range(self.max_iter):
                if self.shuffle:
                    X, y = shuffle(X, y, random_state=self._random_state)
                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    activations[0] = X[batch_slice]
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X[batch_slice], y[batch_slice], activations, deltas,
                        coef_grads, intercept_grads)
                    accumulated_loss += batch_loss * (batch_slice.stop -
                                                      batch_slice.start)

                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_,
                                                         self.loss_))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(early_stopping, X_val, y_val)

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = ("Validation score did not improve more than "
                               "tol=%f for %d consecutive epochs." % (
                                   self.tol, self.n_iter_no_change))
                    else:
                        msg = ("Training loss did not improve more than tol=%f"
                               " for %d consecutive epochs." % (
                                   self.tol, self.n_iter_no_change))

                    is_stopping = self._optimizer.trigger_stopping(
                        msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % self.max_iter, ConvergenceWarning)
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    def _update_no_improvement_count(self, early_stopping, X_val, y_val):
        if early_stopping:
            # compute validation score, use that for stopping
            self.validation_scores_.append(self.score(X_val, y_val))

            if self.verbose:
                print("Validation score: %f" % self.validation_scores_[-1])
            # update best parameters
            # use validation_scores_, not loss_curve_
            # let's hope no-one overloads .score with mse
            last_valid_score = self.validation_scores_[-1]

            if last_valid_score < (self.best_validation_score_ +
                                   self.tol):
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

            if last_valid_score > self.best_validation_score_:
                self.best_validation_score_ = last_valid_score
                self._best_coefs = [c.copy() for c in self.coefs_]
                self._best_intercepts = [i.copy()
                                         for i in self.intercepts_]
        else:
            if self.loss_curve_[-1] > self.best_loss_ - self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]

    def fit(self, X, y):
        return self._fit(X, y, incremental=False)

    def partial_fit(self):
        if self.solver not in _STOCHASTIC_SOLVERS:
            raise AttributeError("partial_fit is only available for stochastic"
                                 " optimizers. %s is not stochastic."
                                 % self.solver)
        return self._partial_fit

    def _partial_fit(self, X, y):
        return self._fit(X, y, incremental=True)

    def _predict(self, X):
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + \
                      [self.n_outputs_]

        # Initialize layers
        activations = [X]

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],
                                         layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations)
        y_pred = activations[-1]

        return y_pred


class BP(BaseMultilayerPerceptron, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=2000, shuffle=True,
                 random_state=1, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=1000):

        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, loss='squared_loss', shuffle=shuffle,
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change)

    def predict(self, X):
        check_is_fitted(self, "coefs_")
        y_pred = self._predict(X)
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def _validate_input(self, X, y, incremental):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         multi_output=True, y_numeric=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        return X, y


data = pd.DataFrame(pd.read_excel(io=(os.path.join(basePath, '????????????.xlsx'))))  # ??????excel????????????
x = data.iloc[:, :3]
y = data.iloc[:, 3:]

model = BP(hidden_layer_sizes=(30,), random_state=10, learning_rate_init=0.08, max_iter=4000)  # BP????????????????????????
model.fit(x, y)  # ????????????
zz = list(data.iloc[0:-1, -1])
pre1 = model.predict(x.iloc[0:-1, :])  # ????????????
print('????????????????????????', np.abs(zz - pre1).mean())  # ????????????


# i=2
# while(i):
#     i=i-1
#     print('??????????????????:\n')
#     Tem=input('??????????????????:')
#     O2=input('??????????????????:')
#     NH3=input('??????????????????:')
#     x_test=np.array((float(Tem),float(O2),float(NH3))).reshape(1,3)
#
#     pre = model.predict(x_test)  # ????????????
#     if pre<15:
#         print("TVBN?????????{:.2f} < 15???????????????????????????".format(float(pre)))
#     elif 15<pre<25:
#         print("TVBN?????????15 < {:.2f} < 25???????????????????????????".format(float(pre)))
#     else :
#         print("TVBN?????????{:.2f} > 25?????????????????????".format(float(pre)))

def fresh_level(Tem, O2, NH3):
    x_test = np.array((float(Tem), float(O2), float(NH3))).reshape(1, 3)
    pre = model.predict(x_test)  # ????????????
    if pre < 15:
        return "TVBN?????????{:.2f} < 15???????????????????????????".format(float(pre))
    elif 15 < pre < 25:
        return "TVBN?????????15 < {:.2f} < 25???????????????????????????".format(float(pre))
    else:
        return "TVBN?????????{:.2f} > 25?????????????????????".format(float(pre))
