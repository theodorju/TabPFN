import numpy as np
import pandas as pd

fontsizes = {'title': 24, 'label': 16, 'legend_title': 18, 'subplot_title': 18, 'ticklabel': 16}

open_ml_name_dict = {
    '11': 'balance-scale',
    '18': 'mfeat-morphological',
    '16': 'mfeat-karhunen',
    '22': 'mfeat-zernike',
    '31': 'credit-g',
    '37': 'diabetes',
    '23': 'cmc',
    '14': 'mfeat-fourier'
}

# Dataset feature mapping
dataset_features_num_dict = {
    'iris': 4,
    '11': 4,
    '18': 6,
    '37': 8,
    'breast_cancer': 10,
    '23': 9,
    'titanic': 12,
    '31': 20,
    '22': 47,
    'digits': 64,
    '16': 64,
    '14': 76,
}


# SKLearn datasets
datasets_sk = ["breast_cancer", "iris"]

# OpenML datasets (tabpfn crashed on 14, 29)
datasets_open = ["11", '16', '22', '37', '14']
datasets_open_full_results = ['11', '16', '22', '31']

spearman_corr_df = pd.DataFrame(np.array([['iris', 4, 'features', 0.986],
                                          ['iris', 4, 'numerical', 0.986],
                                          ['iris', 0, 'symbolic', 0.986],
                                          ['iris', 3, 'classes', 0.986],
                                          ['breast_cancer', 30, 'features', 0.898],
                                          ['breast_cancer', 30, 'numerical', 0.898],
                                          ['breast_cancer', 0, 'symbolic', 0.898],
                                          ['breast_cancer', 2, 'classes', 0.898],
                                          ['digits', 64, 'features', 0.971],
                                          ['digits', 64, 'numerical', 0.971],
                                          ['digits', 0, 'symbolic', 0.971],
                                          ['digits', 10, 'classes', 0.971],
                                          ['titanic', 11, 'features', 0.989],
                                          ['titanic', 1, 'numerical', 0.989],
                                          ['titanic', 9, 'symbolic', 0.989],
                                          ['titanic', 2, 'classes', 0.989],
                                          ['11', 4, 'features', 0.992],
                                          ['11', 4, 'numerical', 0.992],
                                          ['11', 0, 'symbolic', 0.992],
                                          ['11', 3, 'classes', 0.992],
                                          ['18', 6, 'features', 0.981],
                                          ['18', 6, 'numerical', 0.981],
                                          ['18', 0, 'symbolic', 0.981],
                                          ['18', 10, 'classes', 0.981],
                                          ['37', 8, 'features', 0.986],
                                          ['37', 8, 'numerical', 0.986],
                                          ['37', 0, 'symbolic', 0.986],
                                          ['37', 2, 'classes', 0.986],
                                          ['23', 9, 'features', 0.990],
                                          ['23', 2, 'numerical', 0.990],
                                          ['23', 8, 'symbolic', 0.990],
                                          ['23', 3, 'classes', 0.990],
                                          ['31', 20, 'features', 0.991],
                                          ['31', 7, 'numerical', 0.991],
                                          ['31', 14, 'symbolic', 0.991],
                                          ['31', 2, 'classes', 0.991],
                                          ['22', 47, 'features', 0.983],
                                          ['22', 47, 'numerical', 0.983],
                                          ['22', 0, 'symbolic', 0.983],
                                          ['22', 10, 'classes', 0.983],
                                          ['16', 64, 'features', 0.985],
                                          ['16', 64, 'numerical', 0.985],
                                          ['16', 0, 'symbolic', 0.985],
                                          ['16', 10, 'classes', 0.985],
                                          ['14', 76, 'features', 0.864],
                                          ['14', 76, 'numerical', 0.864],
                                          ['14', 0, 'symbolic', 0.864],
                                          ['14', 10, 'classes', 0.864]]),
                                columns=['name', 'val', 'type', 'corr'])
