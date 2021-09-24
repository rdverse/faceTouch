from scipy.stats import kurtosis, skew, iqr
import pandas as pd
import numpy as np

def get_labels():
    labels = [
        'mean',
        'variance',
        'standard_deviation',
        'kurtosis',
        'skewness',
        'minimum_value',
        'maximum_value',
        '25_percentile',
        '75_percentile',
        'inter_quartile_range',
        'auto_correlation_sequence',
    ]

    finalLabels = list()

    axes = ['x', 'y', 'z', 'r']
    sensors = ['accel', 'gyro']

    finalLabels = [[l + '_' + a for l in labels] for a in axes]
    finalLabels = list(np.array(finalLabels).flatten())
    finalLabels = np.array([[fl + '_' + s for fl in finalLabels]
                            for s in sensors])

    return finalLabels.flatten()


def HeuristicBuilder(feature):
    # Time Domain Features
    heuristicFeature = np.empty(0)
    feature = feature.flatten()
    #index x,y,z at a time

    heuristics = {
        'mean': np.mean(feature),
        'variance': np.var(feature),
        'std_dev': np.std(feature),
        'kurtosis': kurtosis(feature.flatten()),
        'skewness': skew(a=feature.flatten()),
        'min_val': feature.min(),
        'max_val': feature.max(),
        'perc25': np.percentile(feature, 25),
        'perc75': np.percentile(feature, 75),
        'inter_quart_range': iqr(feature),
        'auto_corr_seq': _autocorrelation(feature),
    }

    heuristicFeature = np.hstack(list(heuristics.values()))

    return (heuristicFeature)


def _autocorrelation(feature):

    result = np.correlate(feature, feature)
    #result = result[int(len(result) / 2):][:3]
    return (result)