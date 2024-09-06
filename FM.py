import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, log_loss

class FM(tf.keras.Model):
    def __init__(self, features_M, hidden_factor, loss_type='square_loss', 
                 epoch=100, batch_size=128, learning_rate=0.05, lamda_bilinear=0,
                 keep_prob=0.5, optimizer_type='adam', batch_norm=False, verbose=1):
        super(FM, self).__init__()
        self.features_M = features_M
        self.hidden_factor = hidden_factor
        self.loss_type = loss_type
        self.lamda_bilinear = lamda_bilinear
        self.keep_prob = keep_prob
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        
        # Model Layers
        self.feature_embeddings = tf.keras.layers.Embedding(input_dim=features_M, output_dim=hidden_factor)
        self.feature_bias = tf.keras.layers.Embedding(input_dim=features_M, output_dim=1)
        self.bias = tf.Variable(initial_value=tf.constant(0.0))

        # Batch normalization
        if batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        else:
            self.batch_norm_layer = None

        # Optimizer
        if optimizer_type == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_type == 'momentum':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95)
        
    def call(self, inputs, training=False):
        features, labels = inputs
        nonzero_embeddings = tf.nn.embedding_lookup(self.feature_embeddings, features)
        summed_features_emb = tf.reduce_sum(nonzero_embeddings, axis=1)
        summed_features_emb_square = tf.square(summed_features_emb)
        
        squared_features_emb = tf.square(nonzero_embeddings)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1)
        
        FM = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        
        if self.batch_norm_layer:
            FM = self.batch_norm_layer(FM, training=training)
        
        FM = tf.nn.dropout(FM, rate=1-self.keep_prob)
        Bilinear = tf.reduce_sum(FM, axis=1, keepdims=True)
        Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.feature_bias, features), axis=1, keepdims=True)
        Bias = self.bias * tf.ones_like(labels)
        out = Bilinear + Feature_bias + Bias
        
        if self.loss_type == 'log_loss':
            out = tf.sigmoid(out)
        
        return out

    def train_step(self, data):
        features, labels = data
        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            if self.loss_type == 'square_loss':
                loss = tf.reduce_mean(tf.square(labels - predictions))
            elif self.loss_type == 'log_loss':
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, predictions))
            if self.lamda_bilinear > 0:
                loss += self.lamda_bilinear * tf.reduce_sum(tf.square(self.feature_embeddings))
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {'loss': loss}

    def test_step(self, data):
        features, labels = data
        predictions = self(features, training=False)
        if self.loss_type == 'square_loss':
            loss = tf.reduce_mean(tf.square(labels - predictions))
        elif self.loss_type == 'log_loss':
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, predictions))
        
        return {'loss': loss}

    def fit(self, train_dataset, val_dataset, epochs=1):
        for epoch in range(epochs):
            for train_data in train_dataset:
                self.train_step(train_data)
            for val_data in val_dataset:
                self.test_step(val_data)
            
            # Evaluation metrics
            val_loss = np.mean([self.test_step(data)['loss'].numpy() for data in val_dataset])
            if self.verbose > 0:
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")