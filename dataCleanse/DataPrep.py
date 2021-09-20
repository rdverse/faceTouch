import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from magic import magicData
import FeatureEngineering
import os
import tqdm


class makeData():
    def __init__(self,
                 featureType='feature_extraction_new',
                 root_name='covidData',
                 kind="auc",
                 normalizeInt=True,
                 callibrant=False,
                 baseline=False,
                 correctionType='none'):

        #print statement
        magicData.start()

        self.kind = kind
        self.normalizeInt = normalizeInt
        self.callibrant = callibrant
        self.correctionType = correctionType
        self.baseline = baseline
        self.featureType = featureType

        featureBuilder = FeatureEngineering.__dict__

        # root path
        self.root_name = root_name

        # Test file for testing the data
        # self.filePath = os.path.join(
        #     self.root_name, 'PCR_Positive/No symptom information/AT_Day6.txt')

        self.featureLength = featureBuilder[self.featureType + '_length']
        self.featureFunction = featureBuilder[self.featureType]

        cols = self._get_cols()

        if not self.kind == "raw":
            self.final_data = pd.DataFrame(data=[], columns=cols)
        else:
            self.final_data = list()

        self.features = list()
        self.labels = list()
        self.filePaths = list()

        #Instantiate the process
        self.file_parser()

        if not self.kind == "raw":
            self.final_data.columns = [str(c) for c in self.final_data.columns]

    def _get_cols(self):
        cols = list(range(self.featureLength()))
        cols.append("label")
        cols.append("sub_label")
        cols.append("file_path")
        return (cols)

    def _get_data(self, filePath):

        with open(filePath, 'r') as f:

            content = f.readlines()
            content = content[8:]
            content = np.array([[float(item) for item in c.split()]
                                for c in content if c[0] != '#'])

            if self.featureType == 'feature_extraction_new':
                feature = self.featureFunction(
                    content,
                    filePath,
                    kind=self.kind,
                    normalizeInt=self.normalizeInt,
                    baseline=self.baseline,
                    callibrant=self.callibrant,
                    correctionType=self.correctionType)
            else:
                feature = self.featureFunction(content,
                                               filePath,
                                               callibrant=self.callibrant
                                               )

            return (feature)

    def file_parser(self):
        for root, dirs, files in tqdm.tqdm(
                os.walk(self.root_name, topdown=False)):

            for f in files:

                if len(files) > 2:

                    # first x columns in df are the features
                    filePath = os.path.join(root, f)

                    print(filePath)

                    if self.kind != "raw":
                        df_feature = self._get_data(filePath)
                        # add label
                        df_feature.append(root.split('/')[-2])
                        # add sub-label
                        df_feature.append(root.split('/')[-1])
                        # add filepath
                        df_feature.append(filePath)

                        #append the df_feature to the dataframe
                        df_len = len(self.final_data)
                        self.final_data.loc[df_len] = df_feature

                    else:
                        raw_data = self._get_data(filePath)
                        self.final_data.append(raw_data)
                        # pcr positive, negative
                        self.labels.append(root.split('/')[-2])
                        # Add file Paths
                        self.filePaths.append(filePath)
