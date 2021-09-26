import pandas as pd
import numpy as np


class get_data():
    def __init__(self, PATH=''):

        assert PATH != '', "Please specify path variable as PATH= 'file_path.csv'"

        print("#" * 60)
        print(PATH)
        df = pd.read_csv(PATH)
        # Shuffle the dataframe
        self.data = df.sample(frac=1).reset_index(drop=True)
        self.features, self.pids, self.descriptions, self.labels = self.__transform()
        print("Data loaded from {}".format(PATH))

    def __transform(self):

        #feat_columns = [str(i) for i in np.arange(1,self.data[1]-3)]

        # for peak files make sure there are three additional columns added. just in case.
        colsToBeDropped = [
            "labels", "descriptions", "pids", "Unnamed: 0"
        ]

        featuresDf = self.data.copy()

        for col in colsToBeDropped:
            if col in featuresDf.columns:
                featuresDf.drop(columns=[col], inplace=True)

        features = featuresDf.values

        pids = self.data["pids"].values
        descriptions = self.data["descriptions"].values
        labels = self.data["labels"].values
        
        return features, pids, descriptions, labels