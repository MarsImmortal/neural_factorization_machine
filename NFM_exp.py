import os
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from time import time
import argparse
import LoadData as DATA
from tensorflow.keras.layers import BatchNormalization

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Factorization Machine")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--hidden_factor', type=int, required=True, help='Number of hidden factors')
    parser.add_argument('--layers', type=str, required=True, help='Comma-separated list of layer sizes')
    parser.add_argument('--keep_prob', type=str, required=True, help='Comma-separated list of dropout probabilities')
    parser.add_argument('--loss_type', type=str, required=True, help='Loss type: square_loss or log_loss')
    parser.add_argument('--activation', type=str, required=True, help='Activation function: relu, sigmoid, tanh, identity')
    parser.add_argument('--pretrain', type=int, required=True, help='Pretrain flag: 1 or 0')
    parser.add_argument('--optimizer', type=str, required=True, help='Optimizer: e.g., AdagradOptimizer')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--batch_norm', type=int, required=True, help='Batch normalization flag: 1 or 0')
    parser.add_argument('--verbose', type=int, required=True, help='Verbosity level')
    parser.add_argument('--early_stop', type=int, required=True, help='Early stopping flag: 1 or 0')
    parser.add_argument('--epoch', type=int, required=True, help='Number of epochs')
    return parser.parse_args()

class NeuralFM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, hidden_factor, layers, loss_type, pretrain_flag, epoch, batch_size, learning_rate, lamda_bilinear,
                 keep_prob, optimizer_type, batch_norm, activation_function, verbose, early_stop, random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.layers = layers
        self.loss_type = loss_type
        self.pretrain_flag = pretrain_flag
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])  # replaced xrange with range
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.activation_function = activation_function
        self.early_stop = early_stop
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], [] 
        
        # init all variables in a TensorFlow graph
        self._init_graph()  # You'll need to update this method for TensorFlow 2.x
    def _init_graph(self):
            '''
            Initialize the TensorFlow model containing: input data, variables, model, loss, optimizer
            '''
            # Set random seed
            tf.random.set_seed(self.random_seed)

            # Define inputs
            self.train_features = tf.keras.Input(shape=(None,), dtype=tf.int32)
            self.train_labels = tf.keras.Input(shape=(1,), dtype=tf.float32)

            # Initialize weights
            self.weights = self._initialize_weights()

            # Define the model architecture
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
            summed_features_emb = tf.reduce_sum(nonzero_embeddings, axis=1)
            summed_features_emb_square = tf.square(summed_features_emb)
            squared_features_emb = tf.square(nonzero_embeddings)
            squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1)
            FM = 0.5 * (summed_features_emb_square - squared_sum_features_emb)

            if self.batch_norm:
                batch_norm_layer = tf.keras.layers.BatchNormalization()
                FM = batch_norm_layer(FM, training=True)
            
            FM = tf.keras.layers.Dropout(rate=self.keep_prob[-1])(FM)

            for i in range(len(self.layers)):
                FM = tf.keras.layers.Dense(units=self.layers[i], use_bias=True)(FM)
                if self.batch_norm:
                    batch_norm_layer = tf.keras.layers.BatchNormalization()
                    FM = batch_norm_layer(FM, training=True)
                FM = self.activation_function(FM)
                FM = tf.keras.layers.Dropout(rate=self.keep_prob[i])(FM)

            FM = tf.keras.layers.Dense(units=1, use_bias=False)(FM)
            Bilinear = tf.reduce_sum(FM, axis=1, keepdims=True)
            Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features), axis=1)
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)
            self.out = Bilinear + Feature_bias + Bias

            # Loss calculation
            if self.loss_type == 'square_loss':
                self.loss = tf.keras.losses.MeanSquaredError()(self.train_labels, self.out)
                if self.lamda_bilinear > 0:
                    self.loss += tf.keras.regularizers.l2(self.lamda_bilinear)(self.weights['feature_embeddings'])
            elif self.loss_type == 'log_loss':
                self.out = tf.keras.activations.sigmoid(self.out)
                self.loss = tf.keras.losses.BinaryCrossentropy()(self.train_labels, self.out)
                if self.lamda_bilinear > 0:
                    self.loss += tf.keras.regularizers.l2(self.lamda_bilinear)(self.weights['feature_embeddings'])

            # Optimizer
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate, initial_accumulator_value=1e-8)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.95)

            # Initialize Keras Model
            self.model = tf.keras.Model(inputs=[self.train_features], outputs=self.out)
            self.model.compile(optimizer=self.optimizer, loss=self.loss)

            # Calculate the number of parameters
            total_parameters = sum(np.prod(var.shape) for var in self.model.trainable_variables)
            if self.verbose > 0:
                print("#params: %d" % total_parameters)
    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:  # with pretrain
            pretrain_file = '../pretrain/%s_%d/%s_%d' % (self.dataset, self.hidden_factor, self.dataset, self.hidden_factor)
            # Load pretrained weights
            weight_saver = tf.compat.v1.train.import_meta_graph(pretrain_file + '.meta')
            pretrain_graph = tf.compat.v1.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.compat.v1.Session() as sess:
                weight_saver.restore(sess, pretrain_file)
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32)
        else:  # without pretrain
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random.normal([self.features_M, self.hidden_factor], mean=0.0, stddev=0.01), name='feature_embeddings')
            all_weights['feature_bias'] = tf.Variable(
                tf.random.uniform([self.features_M, 1], minval=0.0, maxval=0.0), name='feature_bias')
            all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')
        
        # Deep layers
        num_layer = len(self.layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            all_weights['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.layers[0])), dtype=tf.float32)
            all_weights['bias_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])), dtype=tf.float32)
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i-1] + self.layers[i]))
                all_weights['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i-1], self.layers[i])), dtype=tf.float32)
                all_weights['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=tf.float32)
            # Prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)), dtype=tf.float32)
        else:
            all_weights['prediction'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=tf.float32))
        
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Define batch normalization layer
        bn = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-5,
            center=True,
            scale=True,
            name=scope_bn
        )
        
        # Apply batch normalization
        def apply_batch_norm():
            return bn(x, training=True)

        def apply_batch_norm_inference():
            return bn(x, training=False)
        
        # Conditionally apply batch normalization
        z = tf.cond(train_phase, apply_batch_norm, apply_batch_norm_inference)
        return z

    def partial_fit(self, data):
        # Convert the data into TensorFlow tensors
        x = tf.convert_to_tensor(data['X'], dtype=tf.int32)
        y = tf.convert_to_tensor(data['Y'], dtype=tf.float32)
        dropout_keep = tf.convert_to_tensor(self.keep_prob, dtype=tf.float32)
        train_phase = tf.convert_to_tensor(True, dtype=tf.bool)
        
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self._forward_pass(x, dropout_keep, train_phase)
            loss_value = self._compute_loss(y, logits)
        
        # Compute gradients
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss_value.numpy()

    def _forward_pass(self, x, dropout_keep, train_phase):
        # Define the forward pass
        nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], x)
        summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1)
        summed_features_emb_square = tf.square(summed_features_emb)
        squared_features_emb = tf.square(nonzero_embeddings)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)
        fm = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        
        if self.batch_norm:
            fm = self.batch_norm_layer(fm, train_phase, 'bn_fm')
        
        fm = tf.nn.dropout(fm, dropout_keep[-1])
        
        for i in range(len(self.layers)):
            fm = tf.matmul(fm, self.weights['layer_%d' % i])
            fm = tf.nn.bias_add(fm, self.weights['bias_%d' % i])
            if self.batch_norm:
                fm = self.batch_norm_layer(fm, train_phase, 'bn_%d' % i)
            fm = self.activation_function(fm)
            fm = tf.nn.dropout(fm, dropout_keep[i])
        
        logits = tf.matmul(fm, self.weights['prediction'])
        return logits

    def _compute_loss(self, y_true, y_pred):
        if self.loss_type == 'square_loss':
            loss_value = tf.reduce_mean(tf.square(y_true - y_pred))
            if self.lamda_bilinear > 0:
                regularizer = tf.reduce_sum(tf.square(self.weights['feature_embeddings']))
                loss_value += self.lamda_bilinear * regularizer
        elif self.loss_type == 'log_loss':
            y_pred = tf.sigmoid(y_pred)
            loss_value = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
            if self.lamda_bilinear > 0:
                regularizer = tf.reduce_sum(tf.square(self.weights['feature_embeddings']))
                loss_value += self.lamda_bilinear * regularizer
        return loss_value

    def get_random_block_from_data(self, data, batch_size):
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y = [], []
        
        # Collect samples forward from start_index
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i += 1
            else:
                break
        
        # Collect samples backward from start_index
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i -= 1
            else:
                break
        
        return {'X': np.array(X), 'Y': np.array(Y)}
    def shuffle_in_unison_scary(self, a, b):
        rng_state = np.random.get_state()  # Save the current random state
        np.random.shuffle(a)  # Shuffle the first array
        np.random.set_state(rng_state)  # Restore the saved random state
        np.random.shuffle(b)  # Shuffle the second array
        
    import time

    def train(self, Train_data, Validation_data, Test_data):
        # Check Init performance
        if self.verbose > 0:
            t2 = time.time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            init_test = self.evaluate(Test_data)
            print(f"Init: \t train={init_train:.4f}, validation={init_valid:.4f}, test={init_test:.4f} [{time.time()-t2:.1f} s]")
        
        for epoch in range(self.epoch):  # Use range instead of xrange
            t1 = time.time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'])
            total_batch = len(Train_data['Y']) // self.batch_size  # Use integer division
            for i in range(total_batch):  # Use range instead of xrange
                # Generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                loss = self.partial_fit(batch_xs)
            
            t2 = time.time()
            
            # Output validation
            train_result = self.evaluate(Train_data)
            valid_result = self.evaluate(Validation_data)
            test_result = self.evaluate(Test_data)
            
            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            self.test_rmse.append(test_result)
            
            if self.verbose > 0 and epoch % self.verbose == 0:
                print(f"Epoch {epoch+1} [{t2-t1:.1f} s]\ttrain={train_result:.4f}, validation={valid_result:.4f}, test={test_result:.4f} [{time.time()-t2:.1f} s]")
            
            if self.early_stop > 0 and self.eva_termination(self.valid_rmse):
                # print(f"Early stop at {epoch+1} based on validation result.")
                break
    def eva_termination(self, valid):
        # Ensure that there are enough values to check
        if len(valid) > 5:
            if self.loss_type == 'square_loss':
                # Check if the last 5 validation losses are increasing
                if all(valid[i] > valid[i + 1] for i in range(-1, -5, -1)):
                    return True
            else:
                # Check if the last 5 validation losses are decreasing
                if all(valid[i] < valid[i + 1] for i in range(-1, -5, -1)):
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        
        # Convert data to appropriate format for TensorFlow 2.x
        feed_dict = {
            self.train_features: data['X'],
            self.train_labels: np.expand_dims(data['Y'], axis=-1),  # Ensure correct shape for labels
            self.dropout_keep: self.no_dropout,
            self.train_phase: False
        }

        # Run the session to get predictions
        predictions = self.sess.run(self.out, feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))

        # Evaluate based on loss type
        if self.loss_type == 'square_loss':
            predictions_bounded = np.clip(y_pred, a_min=np.min(y_true), a_max=np.max(y_true))  # Bound the predictions
            RMSE = np.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred)  # Calculate log loss
            return logloss

        # Uncomment for classification accuracy
        # predictions_binary = (y_pred > 0.5).astype(float)
        # Accuracy = accuracy_score(y_true, predictions_binary)
        # return Accuracy

if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)
    
    if args.verbose > 0:
        print("Neural FM: dataset=%s, hidden_factor=%d, dropout_keep=%s, layers=%s, loss_type=%s, pretrain=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d" 
              % (args.dataset, args.hidden_factor, args.keep_prob, args.layers, args.loss_type, args.pretrain, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer, args.batch_norm, args.activation, args.early_stop))
    
    # Set activation function
    if args.activation == 'sigmoid':
        activation_function = tf.nn.sigmoid
    elif args.activation == 'tanh':
        activation_function = tf.nn.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity
    else:
        raise ValueError(f"Unsupported activation function: {args.activation}")

    # Training
    t1 = time()
    model = NeuralFM(
        features_M=data.features_M,
        hidden_factor=args.hidden_factor,
        layers=eval(args.layers),
        loss_type=args.loss_type,
        pretrain_flag=args.pretrain,
        epoch=args.epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lamda_bilinear=args.lamda,
        keep_prob=eval(args.keep_prob),
        optimizer_type=args.optimizer,
        batch_norm=args.batch_norm,
        activation_function=activation_function,
        verbose=args.verbose,
        early_stop=args.early_stop
    )
    model.train(data.Train_data, data.Validation_data, data.Test_data)
    
    # Find the best validation result across iterations
    best_valid_score = float('inf') if args.loss_type == 'square_loss' else float('-inf')
    for i, score in enumerate(model.valid_rmse):
        if args.loss_type == 'square_loss' and score < best_valid_score:
            best_valid_score = score
            best_epoch = i
        elif args.loss_type == 'log_loss' and score > best_valid_score:
            best_valid_score = score
            best_epoch = i

    print("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]" 
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], model.test_rmse[best_epoch], time() - t1))
