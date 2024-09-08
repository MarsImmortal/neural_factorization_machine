import numpy as np
import tensorflow as tf

class LoadData(object):
    '''Given the path of data, return the data format for DeepFM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    def __init__(self, path, dataset, loss_type):
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset + ".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.features_M = self.map_features()
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data(loss_type)

    def map_features(self):
        '''Map feature indices and get total number of features.'''
        self.features = {}
        for file in [self.trainfile, self.testfile, self.validationfile]:
            self.read_features(file)
        return len(self.features)

    def read_features(self, file):
        '''Read features from a file and map them to indices.'''
        try:
            with open(file) as f:
                i = len(self.features)
                for line in f:
                    items = line.strip().split(' ')
                    for item in items[1:]:
                        if item not in self.features:
                            self.features[item] = i
                            i += 1
        except FileNotFoundError:
            print(f"File not found: {file}")
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    def construct_data(self, loss_type):
        '''Construct training, validation, and test datasets.'''
        Train_data = self.construct_dataset(*self.read_data(self.trainfile), loss_type)
        print("# of training:", len(Train_data['Y']))

        Validation_data = self.construct_dataset(*self.read_data(self.validationfile), loss_type)
        print("# of validation:", len(Validation_data['Y']))

        Test_data = self.construct_dataset(*self.read_data(self.testfile), loss_type)
        print("# of test:", len(Test_data['Y']))

        return Train_data, Validation_data, Test_data

    def read_data(self, file):
        '''Read data from a file.'''
        X_ = []
        Y_ = []
        Y_for_logloss = []
        try:
            with open(file) as f:
                for line in f:
                    items = line.strip().split(' ')
                    Y_.append(float(items[0]))

                    v = 1.0 if float(items[0]) > 0 else 0.0
                    Y_for_logloss.append(v)

                    X_.append([self.features.get(item, -1) for item in items[1:]])
        except FileNotFoundError:
            print(f"File not found: {file}")
        except Exception as e:
            print(f"Error reading file {file}: {e}")
        return X_, Y_, Y_for_logloss

    def construct_dataset(self, X_, Y_, loss_type):
        '''Construct dataset dictionary.'''
        Data_Dic = {}
        X_lens = [len(line) for line in X_]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = [Y_[i] for i in indexs]
        Data_Dic['X'] = [X_[i] for i in indexs]
        return Data_Dic

    def truncate_features(self):
        '''Truncate feature vectors to the minimum length.'''
        num_variable = len(self.Train_data['X'][0])
        for data in [self.Train_data, self.Validation_data, self.Test_data]:
            for i in range(len(data['X'])):
                num_variable = min(num_variable, len(data['X'][i]))
        # Truncate data
        for data in [self.Train_data, self.Validation_data, self.Test_data]:
            for i in range(len(data['X'])):
                data['X'][i] = data['X'][i][:num_variable]
        return num_variable

    def to_tf_dataset(self, data_dict):
        '''Convert data dictionary to TensorFlow dataset.'''
        features = np.array(data_dict['X'])
        labels = np.array(data_dict['Y'])
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return dataset

    def get_tf_datasets(self, batch_size=32):
        '''Convert train, validation, and test data to TensorFlow datasets with batching.'''
        train_dataset = self.to_tf_dataset(self.Train_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset = self.to_tf_dataset(self.Validation_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = self.to_tf_dataset(self.Test_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train_dataset, validation_dataset, test_dataset
