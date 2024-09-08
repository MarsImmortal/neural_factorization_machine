import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
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
    parser.add_argument('--optimizer', nargs='?', default='Adagrad',
                        help='Specify an optimizer type (Adam, Adagrad, GradientDescent, Momentum).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                    help='Whether to perform batch normalization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--early_stop', type=int, default=1,
                    help='Whether to perform early stop (0 or 1)')
    return parser.parse_args()

class NeuralFM(tf.keras.Model):
    def __init__(self, features_M, hidden_factor, layers, loss_type, pretrain_flag, epoch, batch_size, learning_rate, lamda_bilinear,
                 keep_prob, optimizer_type, batch_norm, activation_function, verbose, early_stop, random_seed=2016):
        super(NeuralFM, self).__init__()
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.layers_sizes = layers
        self.loss_type = loss_type
        self.pretrain_flag = pretrain_flag
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for _ in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.activation_function = activation_function
        self.early_stop = early_stop
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []
        
        # Initialize model components
        self._init_model()

    def _init_model(self):
        self.feature_embeddings = tf.keras.layers.Embedding(input_dim=self.features_M, output_dim=self.hidden_factor)
        self.feature_bias = tf.keras.layers.Embedding(input_dim=self.features_M, output_dim=1)
        self.bias = tf.keras.layers.Dense(1, use_bias=False)

        # Deep layers
        self.deep_layers = []
        previous_size = self.hidden_factor
        for size in self.layers_sizes:
            self.deep_layers.append(tf.keras.layers.Dense(size, activation=self.activation_function))
            previous_size = size
        self.prediction = tf.keras.layers.Dense(1)

        # Ensure optimizer_type is set to 'Adagrad' in the arguments or code
        self.optimizer_type = 'Adagrad'  # or set this dynamically based on input

        # Initialize the optimizer
        self.optimizer = getattr(tf.keras.optimizers, self.optimizer_type)(learning_rate=self.learning_rate)


    def call(self, inputs, training=False):
        features, labels = inputs

        nonzero_embeddings = tf.gather(self.feature_embeddings.weights[0], features)
        summed_features_emb = tf.reduce_sum(nonzero_embeddings, axis=1)
        summed_features_emb_square = tf.square(summed_features_emb)
        
        squared_features_emb = tf.square(nonzero_embeddings)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1)
        
        FM = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        if self.batch_norm:
            FM = tf.keras.layers.BatchNormalization()(FM, training=training)
        FM = tf.nn.dropout(FM, self.keep_prob[-1])
        
        for layer in self.deep_layers:
            FM = layer(FM)
            if self.batch_norm:
                FM = tf.keras.layers.BatchNormalization()(FM, training=training)
            FM = tf.nn.dropout(FM, self.keep_prob[len(self.deep_layers)])
        
        FM = self.prediction(FM)
        Bilinear = tf.reduce_sum(FM, axis=1, keepdims=True)
        Feature_bias = tf.reduce_sum(tf.gather(self.feature_bias.weights[0], features), axis=1, keepdims=True)
        Bias = self.bias(tf.ones_like(labels))
        
        out = Bilinear + Feature_bias + Bias
        return out

    def train_step(self, data):
        features, labels = data
        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = self.compiled_loss(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        features, labels = data
        predictions = self(features, training=False)
        loss = self.compiled_loss(labels, predictions)
        return {"loss": loss}

    def compile(self, **kwargs):
        super(NeuralFM, self).compile(**kwargs)

def main():
    # Data loading
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)

    if args.verbose > 0:
        print("Neural FM: dataset=%s, hidden_factor=%d, dropout_keep=%s, layers=%s, loss_type=%s, pretrain=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d" 
              %(args.dataset, args.hidden_factor, args.keep_prob, args.layers, args.loss_type, args.pretrain, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer, args.batch_norm, args.activation, args.early_stop))
    
    activation_function = tf.keras.activations.relu
    if args.activation == 'sigmoid':
        activation_function = tf.keras.activations.sigmoid
    elif args.activation == 'tanh':
        activation_function = tf.keras.activations.tanh
    elif args.activation == 'identity':
        activation_function = tf.keras.activations.linear

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

    model.compile(optimizer=model.optimizer, loss='mean_squared_error' if args.loss_type == 'square_loss' else 'binary_crossentropy')

    history = model.fit(
        x={'features': data.Train_data['X'], 'labels': data.Train_data['Y']},
        epochs=args.epoch,
        batch_size=args.batch_size,
        validation_data=({'features': data.Validation_data['X'], 'labels': data.Validation_data['Y']}),
        verbose=args.verbose
    )

    # Find the best validation result
    min_loss = np.min(history.history['val_loss'])
    print("Minimum validation loss: ", min_loss)
    
    # Evaluate on test set
    test_loss = model.evaluate({'features': data.Test_data['X'], 'labels': data.Test_data['Y']})
    print("Test Loss: ", test_loss)

if __name__ == "__main__":
    main()
