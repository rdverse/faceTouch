import pandas as pd
import numpy as np

'''
@ Load a csv file, get filenames, features and labels
'''
class  GetData():
    def __init__(self, PATH = '../AllResults/Files/compare_AUC_integer_float/NIU/Files60/integer.csv'):
        self.data = pd.read_csv(PATH)
        self.features, self.labels, self.fileNames = self.__transform()

    def __transform(self):
        feat_columns = [str(i) for i in np.arange(7)]
        features = self.data[[feat_columns]]
        labels = self.data['label'].values
        fileNames = self.data['file_path'].values
        fileNames = np.array([f.split('/')[-1]
                             .split('.txt')[0] for
                              f in fileNames])

        return features, labels, fileNames

