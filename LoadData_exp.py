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
        '''
        Map feature indices across all data files and store in self.features dictionary.
        :return: Total number of unique features
        '''
        self.features = {}
        
        # Read features from each data file
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        
        # Return the number of unique features
        return len(self.features)
    
    def read_features(self, file):
        '''
        Read feature indices from a data file and update the features dictionary.
        :param file: Path to the feature file
        '''
        with open(file, 'r') as f:
            i = len(self.features)  # Initialize feature index based on current length of self.features
            for line in f:
                items = line.strip().split(' ')
                for item in items[1:]:  # Skip the label (first item)
                    if item not in self.features:
                        self.features[item] = i
                        i += 1
    def construct_data(self, loss_type):
        '''
        Construct training, validation, and test datasets from the specified files.
        :param loss_type: Type of loss function ('log_loss' or other)
        :return: Train_data, Validation_data, Test_data: Dictionaries with keys 'Y' and 'X'
        '''
        # Read and construct training data
        X_train, Y_train, Y_for_logloss_train = self.read_data(self.trainfile)
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_train, Y_for_logloss_train)
        else:
            Train_data = self.construct_dataset(X_train, Y_train)
        print("# of training samples:", len(Y_train))

        # Read and construct validation data
        X_valid, Y_valid, Y_for_logloss_valid = self.read_data(self.validationfile)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_valid, Y_for_logloss_valid)
        else:
            Validation_data = self.construct_dataset(X_valid, Y_valid)
        print("# of validation samples:", len(Y_valid))

        # Read and construct test data
        X_test, Y_test, Y_for_logloss_test = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_test, Y_for_logloss_test)
        else:
            Test_data = self.construct_dataset(X_test, Y_test)
        print("# of test samples:", len(Y_test))

        return Train_data, Validation_data, Test_data

    def read_data(self, file):
        '''
        Read data from a file and return feature and label arrays.
        :param file: Path to the data file
        :return: X_ (list of feature vectors), Y_ (list of labels), Y_for_logloss (list of labels for log loss)
        '''
        X_ = []
        Y_ = []
        Y_for_logloss = []

        with open(file, 'r') as f:
            for line in f:
                items = line.strip().split(' ')
                label = float(items[0])
                Y_.append(label)
                
                # For log loss, convert positive labels to 1 and others to 0
                Y_for_logloss.append(1.0 if label > 0 else 0.0)
                
                # Map feature entries to indices
                features = [self.features.get(item, 0) for item in items[1:]]
                X_.append(features)
        
        return X_, Y_, Y_for_logloss

    def construct_dataset(self, X_, Y_):
        Data_Dic = {}
        X_lens = [ len(line) for line in X_]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = [ Y_[i] for i in indexs]
        Data_Dic['X'] = [ X_[i] for i in indexs]
        return Data_Dic
    
    def truncate_features(self):
        """
        Make sure each feature vector is of the same length
        """
        num_variable = len(self.Train_data['X'][0])
        for i in xrange(len(self.Train_data['X'])):
            num_variable = min([num_variable, len(self.Train_data['X'][i])])
        # truncate train, validation and test
        for i in xrange(len(self.Train_data['X'])):
            self.Train_data['X'][i] = self.Train_data['X'][i][0:num_variable]
        for i in xrange(len(self.Validation_data['X'])):
            self.Validation_data['X'][i] = self.Validation_data['X'][i][0:num_variable]
        for i in xrange(len(self.Test_data['X'])):
            self.Test_data['X'][i] = self.Test_data['X'][i][0:num_variable]
        return num_variable
