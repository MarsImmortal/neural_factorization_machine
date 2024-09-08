import numpy as np
import os

class LoadData:
    '''
    Given the path of data, return the data format for DeepFM.
    :param path: Path to the data files
    :param dataset: Name of the dataset
    :param loss_type: Type of loss function ('log_loss' or other)
    :return: Train_data, Validation_data, Test_data: Dictionaries with keys 'Y' and 'X'
    '''

    def __init__(self, path, dataset, loss_type):
        self.path = os.path.join(path, dataset)
        self.trainfile = f"{self.path}/{dataset}.train.libfm"
        self.testfile = f"{self.path}/{dataset}.test.libfm"
        self.validationfile = f"{self.path}/{dataset}.validation.libfm"
        self.features_M = self.map_features()
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data(loss_type)

    def map_features(self):
        ''' Map feature entries in all files and return the total number of features '''
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        return len(self.features)

    def read_features(self, file):
        ''' Read a feature file and map features to indices '''
        with open(file, 'r') as f:
            i = len(self.features)
            for line in f:
                items = line.strip().split(' ')
                for item in items[1:]:
                    if item not in self.features:
                        self.features[item] = i
                        i += 1

    def construct_data(self, loss_type):
        ''' Construct training, validation, and test datasets '''
        Train_data = self.construct_dataset_from_file(self.trainfile, loss_type)
        Validation_data = self.construct_dataset_from_file(self.validationfile, loss_type)
        Test_data = self.construct_dataset_from_file(self.testfile, loss_type)
        print("# of training:", len(Train_data['Y']))
        print("# of validation:", len(Validation_data['Y']))
        print("# of test:", len(Test_data['Y']))
        return Train_data, Validation_data, Test_data

    def construct_dataset_from_file(self, file, loss_type):
        ''' Read data from a file and construct dataset '''
        X_, Y_, Y_for_logloss = self.read_data(file)
        if loss_type == 'log_loss':
            return self.construct_dataset(X_, Y_for_logloss)
        else:
            return self.construct_dataset(X_, Y_)

    def read_data(self, file):
        ''' Read data file and return feature and label lists '''
        X_, Y_, Y_for_logloss = [], [], []
        with open(file, 'r') as f:
            for line in f:
                items = line.strip().split(' ')
                Y_.append(float(items[0]))
                Y_for_logloss.append(1.0 if float(items[0]) > 0 else 0.0)
                X_.append([self.features[item] for item in items[1:]])
        return X_, Y_, Y_for_logloss

    def construct_dataset(self, X_, Y_):
        ''' Construct dataset from features and labels '''
        Data_Dic = {}
        X_lens = [len(line) for line in X_]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = [Y_[i] for i in indexs]
        Data_Dic['X'] = [X_[i] for i in indexs]
        return Data_Dic

    def truncate_features(self):
        ''' Ensure all feature vectors are of the same length '''
        num_variable = len(self.Train_data['X'][0])
        num_variable = min(len(line) for line in self.Train_data['X'])
        self.Train_data['X'] = [x[:num_variable] for x in self.Train_data['X']]
        self.Validation_data['X'] = [x[:num_variable] for x in self.Validation_data['X']]
        self.Test_data['X'] = [x[:num_variable] for x in self.Test_data['X']]
        return num_variable
