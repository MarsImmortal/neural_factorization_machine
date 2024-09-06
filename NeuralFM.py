import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
import LoadData as DATA

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[0.8,0.5]', 
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='Adam',
                        help='Specify an optimizer type (Adam, Adagrad, SGD, Momentum).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                    help='Whether to perform batch normalization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--early_stop', type=int, default=1,
                    help='Whether to perform early stop (0 or 1)')
    return parser.parse_args()

class NeuralFM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, hidden_factor, layers, loss_type, pretrain_flag, epoch, batch_size, learning_rate, lamda_bilinear,
                 keep_prob, optimizer_type, batch_norm, activation_function, verbose, early_stop, random_seed=2016):
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
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.activation_function = activation_function
        self.early_stop = early_stop
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        # Initialize model
        self._init_model()

    def _init_model(self):
        '''
        Initialize the TensorFlow 2.x model.
        '''
        # Set random seed
        tf.random.set_seed(self.random_seed)

        # Define inputs
        self.inputs = Input(shape=(None,), dtype=tf.int32, name='features')
        self.labels = Input(shape=(1,), dtype=tf.float32, name='labels')

        # Embedding layer
        embeddings = Embedding(input_dim=self.features_M, output_dim=self.hidden_factor)(self.inputs)
        summed_features_emb = tf.reduce_sum(embeddings, axis=1)
        summed_features_emb_square = tf.square(summed_features_emb)

        squared_features_emb = tf.square(embeddings)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1)

        # FM part
        FM = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        if self.batch_norm:
            FM = self.batch_norm_layer(FM)
        FM = Dropout(self.keep_prob[-1])(FM)

        # Deep layers
        for i in range(len(self.layers)):
            FM = Dense(self.layers[i], activation=self.activation_function, use_bias=True)(FM)
            if self.batch_norm:
                FM = self.batch_norm_layer(FM)
            FM = Dropout(self.keep_prob[i])(FM)
        
        output = Dense(1, activation=None)(FM)

        # Add bias
        bias = tf.Variable(0.0, dtype=tf.float32, name='bias')
        feature_bias = tf.reduce_sum(Embedding(input_dim=self.features_M, output_dim=1)(self.inputs), axis=1)
        prediction = tf.reduce_sum(output, axis=1, keepdims=True) + feature_bias + bias

        # Model definition
        self.model = Model(inputs=[self.inputs, self.labels], outputs=prediction)
        
        # Loss function
        if self.loss_type == 'square_loss':
            self.loss_fn = MeanSquaredError()
        elif self.loss_type == 'log_loss':
            self.loss_fn = BinaryCrossentropy()

        # Optimizer
        if self.optimizer_type == 'Adam':
            self.optimizer = Adam(learning_rate=self.learning_rate)
        elif self.optimizer_type == 'Adagrad':
            self.optimizer = Adagrad(learning_rate=self.learning_rate)
        elif self.optimizer_type == 'SGD':
            self.optimizer = SGD(learning_rate=self.learning_rate)
        elif self.optimizer_type == 'Momentum':
            self.optimizer = SGD(learning_rate=self.learning_rate, momentum=0.95)

        
        # Compile model
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
    
    def batch_norm_layer(self, x):
        return BatchNormalization()(x)

    def partial_fit(self, data):
        '''
        Fit a batch
        '''
        with tf.GradientTape() as tape:
            predictions = self.model([data['X'], data['Y']], training=True)
            loss = self.loss_fn(data['Y'], predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss.numpy()

    def get_random_block_from_data(self, data, batch_size):
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y = [], []
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i += 1
            else:
                break
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
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def train(self, Train_data, Validation_data, Test_data):
        '''
        Fit a dataset
        '''
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            init_test = self.evaluate(Test_data)
            print(f"Init: \t train={init_train:.4f}, validation={init_valid:.4f}, test={init_test:.4f} [{time()-t2:.1f} s]")
        
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'])
            total_batch = len(Train_data['Y']) // self.batch_size
            for i in range(total_batch):
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                loss = self.partial_fit(batch_xs)
                
            # Evaluation
            if (epoch+1) % 1 == 0:
                train_rmse = self.evaluate(Train_data)
                valid_rmse = self.evaluate(Validation_data)
                test_rmse = self.evaluate(Test_data)
                self.train_rmse.append(train_rmse)
                self.valid_rmse.append(valid_rmse)
                self.test_rmse.append(test_rmse)
                
                if self.verbose > 0:
                    print(f"Epoch {epoch+1}: \t train={train_rmse:.4f}, validation={valid_rmse:.4f}, test={test_rmse:.4f} [{time()-t1:.1f} s]")

                if self.early_stop and len(self.valid_rmse) > 1 and self.valid_rmse[-1] >= min(self.valid_rmse[:-1]):
                    print("Early stopping...")
                    break

    def evaluate(self, data):
        '''
        Evaluate the model
        '''
        predictions = self.model([data['X'], data['Y']], training=False)
        if self.loss_type == 'square_loss':
            loss = self.loss_fn(data['Y'], predictions)
            return np.sqrt(loss.numpy())
        elif self.loss_type == 'log_loss':
            loss = self.loss_fn(data['Y'], predictions)
            return loss.numpy()

    def predict(self, X):
        '''
        Predict
        '''
        return self.model.predict(X)

    def eva_termination(self):
        '''
        Get the last evaluation result
        '''
        if len(self.valid_rmse) > 0:
            return self.valid_rmse[-1]
        else:
            return None

if __name__ == "__main__":
    args = parse_args()

    data_path = args.path
    dataset = args.dataset
    hidden_factor = args.hidden_factor
    layers = eval(args.layers)
    keep_prob = eval(args.keep_prob)
    lamda_bilinear = args.lamda
    learning_rate = args.lr
    loss_type = args.loss_type
    optimizer_type = args.optimizer
    batch_norm = args.batch_norm
    activation_function = args.activation
    epoch = args.epoch
    batch_size = args.batch_size
    pretrain_flag = args.pretrain
    verbose = args.verbose
    early_stop = args.early_stop

    # Load data
    Train_data, Validation_data, Test_data = DATA.load_data(data_path, dataset)

    model = NeuralFM(features_M=len(Train_data['X'][0]), 
                     hidden_factor=hidden_factor, 
                     layers=layers, 
                     loss_type=loss_type, 
                     pretrain_flag=pretrain_flag, 
                     epoch=epoch, 
                     batch_size=batch_size, 
                     learning_rate=learning_rate, 
                     lamda_bilinear=lamda_bilinear, 
                     keep_prob=keep_prob, 
                     optimizer_type=optimizer_type, 
                     batch_norm=batch_norm, 
                     activation_function=activation_function, 
                     verbose=verbose, 
                     early_stop=early_stop)

    model.train(Train_data, Validation_data, Test_data)
