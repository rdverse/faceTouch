import os
import numpy as np
import pandas as pd
from IPython.display import clear_output
import tqdm
from makeResults import result_bars, result_monitor, get_results
from csvLoader import get_data

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning

import tensorflow as tf
from keras.utils import np_utils
# import kerastuner as kt
from tensorflow.keras import layers, Model, optimizers, losses


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        clear_output(wait=True)


class allModels():
    # Initialize and run all models
    def __init__(self, csvPATH, run=True):
        # Data Zone
        data = get_data(PATH=csvPATH)
        self.features, self.pids, self.descriptions, self.labels = data.features, data.pids, data.descriptions, data.labels
        
        self.X_Train, self.X_Test, self.y_Train, self.y_Test = train_test_split(self.features,
                                            self.labels,
                                            test_size=0.2,
                                            shuffle=True,
                                            random_state=11)
        self.dataset = [self.X_Train, self.X_Test, self.y_Train, self.y_Test]
        self.savePath = 'plots/results/stat/'

        self.fileName = "g"#csvPATH.split('/')[-1].strip('.csv')
        self.run = run
        self.algs = dict()

        if self.run:
            # Algorithms zone

            self.random_forest()
            #self.decision_tree()
            #self.XgBoost()
            #self.logistic_regression()
            #self.svm()
            #self.nn()
            #self.KNN()
            #self.LDA()
            #self.GNB()

            # Plotting zone

            result_bars(self.savePath + self.fileName, self.algs)

    #################Utils for models###########################
    def print_stats(self, name, results):

        Train = results["train"]
        Test = results["test"]

        print("\nAlgorithm : {}".format(name))
        print(
            "Test Metrics \nAccuracy : {} \nPrecision : {} \nRecall : {} \nF1 : {}"
            .format(Test.accuracy * 100, Test.precision, Test.recall,
                    Test.f1))
        print(
            "Train Metrics \nAccuracy : {} \nPrecision : {} \nRecall : {} \nF1 : {}"
            .format(Train.accuracy * 100, Train.precision, Train.recall,
                    Train.f1))
        abbrev = [n[0] for n in name.split()]
        self.algs[''.join(abbrev)] = Test
        print()

    @ignore_warnings(category=FitFailedWarning)
    def random_forest(self):
        # Random Forest with PCA
        hyper = dict()
        hyper['pca__n_components'] = [2, 5]
        pca = PCA()
        rf = RandomForestClassifier(n_jobs=-1, verbose=0)
        poly  = PolynomialFeatures(2)
        param_grid = {
            'rf__max_depth': [3, 5,10,20],
            'rf__min_samples_leaf': [5, 10,30,50,100, 150],
            'rf__min_samples_split': [10,20,30,50,100],
            'rf__n_estimators': [10,50,100,150,200,300],
            #[int(x) for x in np.linspace(start=2, stop=20, num=1)],
            'rf__max_features': ['auto', 'sqrt'],
            'rf__bootstrap': [True, False]
        }

        # param_grid['pca__n_components'] = hyper['pca__n_components']
        pipe = Pipeline([('poly',poly), ('rf', rf)])

        grid_rf = RandomizedSearchCV(pipe, param_grid, verbose=1, n_jobs=-1, cv=3,n_iter=30)
        grid_rf.fit(self.X_Train, self.y_Train)
        print(grid_rf.best_params_)

        results_rf = get_results(self.dataset,
                                        grid_rf.best_estimator_,
                                        )

        self.print_stats("Random Forest", results_rf)

    @ignore_warnings(category=FitFailedWarning)
    def logistic_regression(self):

        logreg_grid = {
     
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10,100]
        }

        print(logreg_grid)

        clf = LogisticRegression(verbose=0, max_iter=5000)
        lr_grid = RandomizedSearchCV(clf,
                               logreg_grid,
                               cv=3,
                               verbose=1,
                               n_jobs=-1,n_iter=30)
        #with ignore_warnings(category=[ConvergenceWarning, FitFailedWarning]):

        lr_grid.fit(self.features, self.labels)

        print(lr_grid.best_params_)
        results_lr = get_results(self.dataset,
                                        lr_grid.best_estimator_)

        self.print_stats("Logistic Regression", results_lr)
        #return (lr_grid.best_estimator_)

    @ignore_warnings(category=FitFailedWarning)
    def svm(self):
        #     ######## For Classifiers

        svm_params = {
            'C': [ 0.0001, 0.001, 0.1, 1, 5, 10,100],
            'gamma': [0.0001, 0.001, 0.1, 0.001, 1, 5, 10,100],
            'kernel': ['linear', 'rbf', 'sigmoid']
        }

        clf = SVC(max_iter=3000, probability=True)
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        svm_grid = RandomizedSearchCV(clf,
                                svm_params,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,n_iter=30)

        svm_grid.fit(self.features, self.labels)

        print(svm_grid.best_params_)

        results_svm = get_results(self.dataset,
                                        svm_grid.best_estimator_)


        self.print_stats("Support Vector Machines", results_svm)

    @ignore_warnings(category=FitFailedWarning)
    def XgBoost(self):
        xgb_params = {
            "xgb__learning_rate": [
        0.001,  0.10,
             0.25, 1, 10
            ],
            "xgb__max_depth": [3,5,10,20,50,100,150,200,300],
            "xgb__min_child_weight": [5,10,30,100],
            "xgb__gamma": [0.1, 1, 10, 100],
            "xgb__colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
        }
        xgb = XGBClassifier(use_label_encoder=False)

        pipe = Pipeline([('xgb', xgb)])

        xgb_grid = RandomizedSearchCV(pipe,
                                xgb_params,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,n_iter=30)

        xgb_grid.fit(self.features, self.labels)
        print(xgb_grid.best_params_)

        results_xgb = get_results(self.dataset,
                                        xgb_grid.best_estimator_)

        self.print_stats("X G Boost", results_xgb)
        return (xgb_grid.best_estimator_)

    @ignore_warnings(category=FitFailedWarning)
    def KNN(self):
        #     ######## For Classifiers

        knn_params = {
            'n_neighbors': [2, 3, 4, 5, 6],
            'leaf_size': [3, 5, 10, 25],
            'weights': ['distance', 'uniform'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

        knn_clf = KNeighborsClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        knn_grid = RandomizedSearchCV(knn_clf,
                                knn_params,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,n_iter=30)
        knn_grid.fit(self.features, self.labels)

        print(knn_grid.best_params_)

        results_knn = get_results(self.dataset,
                                        knn_grid.best_estimator_)


        self.print_stats("K Nearest Neighbors", results_knn )

    @ignore_warnings(category=FitFailedWarning)
    def GNB(self):
        gnb = GaussianNB()
        gnb.fit(self.features, self.labels)

        results_gnb = get_results(self.dataset,
                                        gnb)

        self.print_stats("Gaussian Naive Bayes", results_gnb)

    @ignore_warnings(category=FitFailedWarning)
    def LDA(self):
        #     ######## For Classifiers
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        lda_params = {'solver': ['svd', 'lsqr', 'eigen']}

        lda_clf = LinearDiscriminantAnalysis()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        lda_grid = RandomizedSearchCV(lda_clf,
                                lda_params,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,n_iter=30)
        lda_grid.fit(self.features, self.labels)

        print(lda_clf.get_params())

        results_lda = get_results(self.dataset,
                                        lda_grid.best_estimator_)

        self.print_stats("Latent Dirichlet Allocation", results_lda)

    @ignore_warnings(category=FitFailedWarning)
    def decision_tree(self):
        # Random Forest with PCA
        hyper = dict()
        hyper['pca__n_components'] = [2, 5, 10, 25, 40, 50, 80, 100]
        pca = PCA()

        dt = DecisionTreeClassifier()

        dt_params = {
            'dt__max_depth': [3,5,10,20,30,50],
            'dt__min_samples_leaf': [5,10,100,50,20],
            'dt__min_samples_split': [5,10,20,50,100],
            'dt__max_features': ['auto', 'sqrt', 'log2'],
            'dt__criterion': ["gini", "entropy"],
            'dt__splitter': ["best", "random"],
            'dt__max_leaf_nodes': [5,10, 20,50,100]
        }

        # param_grid['pca__n_components'] = hyper['pca__n_components']
        pipe_dt = Pipeline([('dt', dt)])

        dt_grid = RandomizedSearchCV(pipe_dt, dt_params, verbose=1, n_jobs=-1, cv=3,n_iter=30)
        dt_grid.fit(self.features, self.labels)

        print(dt_grid.best_params_)
        results_dt = get_results(self.dataset,
                                        dt_grid.best_estimator_)


        self.print_stats("Decision Tree", results_dt)

    # @ignore_warnings(category=FitFailedWarning)
    # def mlpc(self):
    #     # Random Forest with PCA
    #     hyper = dict()
    #     hyper['pca__n_components'] = [2, 5, 10, 25, 40, 50, 80, 100]
    #     pca = PCA()

    #     dt = MLPClassifier()

    #     mlp_params = {
    #         'dt__max_depth': [int(x) for x in np.linspace(2, 12, num=10)],
    #         'dt__min_samples_leaf': [1, 2, 4, 6],
    #         'dt__min_samples_split': [2, 4, 6, 8, 10],
    #         'dt__max_features': ['auto', 'sqrt', 'log2'],
    #         'dt__criterion': ["gini", "entropy"],
    #         'dt__splitter': ["best", "random"],
    #         'dt__max_leaf_nodes': [2, 4, 5, 8, 10, 15]
    #     }

    #     # param_grid['pca__n_components'] = hyper['pca__n_components']
    #     pipe_mlp = Pipeline([('mlp', mlp)])

    #     mlp_grid = RandomSearchCV(pipe_dt,
    #                               dt_params,
    #                               verbose=0,
    #                               n_jobs=-1,
    #                               cv=3)
    #     mlp_grid.fit(self.features, self.labels)

    #     print(mlp_grid.best_params_)

    #     results_dt = multipleIterations(self.features,
    #                                     self.labels,
    #                                     mlp_grid.best_estimator_,
    #                                     calibrate=False,
    #                                     calibrateMethod="isotonic")

    #     self.print_stats("Multi Layer Perceptron", results_dt)

    # def build_model2(self, hp):
    #     input = layers.Input(shape=(self.features.shape[-1]))

    #     model = input
    #     for i in range(hp.Int(name='num_layers', min_value=1, max_value=4)):
    #         model = layers.Dense(hp.Int(name='hidden_size_{}'.format(i),
    #                                     min_value=2,
    #                                     max_value=100,
    #                                     step=5,
    #                                     default=6),
    #                              activation='relu')(model)
    #         model = layers.Dropout(
    #             hp.Choice(name='Dropout_{}'.format(i),
    #                       values=[0.1, 0.2, 0.3, 0.4]))(model)

    #     output = layers.Dense(2, activation='softmax')(model)

    #     optimizer = optimizers.Adam(
    #         hp.Choice(name='learningRate_{}'.format(i),
    #                   values=[
    #                       0.000001, 0.00000001, 0.0001, 0.00001, 0.001, 0.01,
    #                       0.1, 1.0
    #                   ]))

    #     loss = losses.CategoricalCrossentropy()
    #     model = Model(input, output)
    #     model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    #     return model

    # def nn(self):
    #     try:
    #         physical_devices = tf.config.list_physical_devices('CPU')
    #         tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #     except:
    #         # Invalid device or cannot modify virtual devices once initialized.
    #         print('Could not initialize the tensorflow gpu')
    #         pass
    #     encLabels = np_utils.to_categorical(self.labels)
    #     # Split the dataset
    #     X_Train, X_Test, y_Train, y_Test = train_test_split(
    #         self.features.reshape(self.features.shape[0],
    #                               self.features.shape[1], 1),
    #         np.array(encLabels),
    #         test_size=0.3,
    #         shuffle=True,
    #         random_state=11)
    #     X_Test, X_Val, y_Test, y_Val = train_test_split(X_Test,
    #                                                     y_Test,
    #                                                     test_size=0.5,
    #                                                     shuffle=True,
    #                                                     random_state=11)

    #     earlystopping = tf.keras.callbacks.EarlyStopping(
    #         monitor='val_accuracy', patience=5)

    #     reduceLRplateau = tf.keras.callbacks.ReduceLROnPlateau(
    #         monitor='val_accuracy',
    #         min_lr=1e-6,
    #         patience=4,
    #         factor=0.9,
    #         verbose=0)

    #     tuner2 = kt.Hyperband(
    #         self.build_model2,
    #         objective='val_accuracy',
    #         max_epochs=20,
    #         factor=20,
    #         hyperband_iterations=1,
    #         seed=20,
    #         tune_new_entries=True,
    #         allow_new_entries=True,
    #         directory='Basic{}'.format(np.random.randint(100)),
    #         project_name=self.fileName + str(np.random.randint(100)))

    #     print(tuner2.search_space_summary())

    #     tuner2.search(
    #         X_Train,
    #         y_Train,
    #         validation_data=(X_Val, y_Val),
    #         verbose=0,
    #         callbacks=[earlystopping, reduceLRplateau,
    #                    ClearTrainingOutput()])

    #     best_hps = tuner2.get_best_hyperparameters(num_trials=1)[0]

    #     model2 = tuner2.hypermodel.build(best_hps)

    #     # fit model
    #     hist = model2.fit(X_Train,
    #                       y_Train,
    #                       epochs=20,
    #                       verbose=0,
    #                       validation_data=(X_Val, y_Val),
    #                       callbacks=[earlystopping, reduceLRplateau])

    #     # Get predictions
    #     pTest2 = model2.predict(X_Test)
    #     pTrain2 = model2.predict(X_Train)

    #     pTest2 = [0 if p[0] > p[1] else 1 for p in pTest2]
    #     pTrain2 = [0 if p[0] > p[1] else 1 for p in pTrain2]
    #     # get model scores
    #     Train_2 = {}
    #     Test_2 = {}

    #     yTr = [0 if p[0] > p[1] else 1 for p in y_Train]
    #     yTe = [0 if p[0] > p[1] else 1 for p in y_Test]

    #     Test_2["precision"], Test_2["recall"], Test_2[
    #         "f1"], _ = precision_recall_fscore_support(yTe,
    #                                                    pTest2,
    #                                                    average='macro')
    #     Test_2["accuracy"] = accuracy_score(pTest2, yTe)

    #     Train_2["precision"], Train_2["recall"], Train_2[
    #         "f1"], _ = precision_recall_fscore_support(yTr,
    #                                                    pTrain2,
    #                                                    average='macro')
    #     Train_2["accuracy"] = accuracy_score(pTrain2, yTr)

    #     print('\nNeural Networks')
    #     print(
    #         "Test Metrics \nAccuracy : {} \nPrecision : {} \nRecall : {} \nF1 : {}"
    #         .format(Test_2["accuracy"], Test_2["precision"], Test_2["recall"],
    #                 Test_2["f1"]))
    #     print(
    #         "\nTrain Metrics \nAccuracy : {} \nPrecision : {} \nRecall : {} \nF1 : {}"
    #         .format(Train_2["accuracy"], Train_2["precision"],
    #                 Train_2["recall"], Train_2["f1"]))
    #     print()
    #     history = pd.DataFrame(hist.history)

    #     results_m2 = multipleIterations_nn2(self.features, encLabels, model2)
    #     Test_m2 = results_m2["test"]
    #     Train_m2 = results_m2["train"]

    #     monitorPath = self.savePath + 'NN/' + self.fileName + '.png'
    #     result_monitor(monitorPath, history)
    #     self.algs["NN"] = Test_m2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

   
    fileNames = [
    "../data/dataset/statFeatures/stat50_8.csv"
    ]

    for fileName in tqdm.tqdm(fileNames):
        runInstance = allModels(fileName)
