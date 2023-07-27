import os
import pickle
import numpy as np
import nibabel as nib
from scipy.stats import skew, entropy
from tqdm import tqdm

def get_stat_comp_name_pairs(args):
    stat_func_map = {'mean':np.mean, 'std':np.std, 'skewness': skew}
    stat_comp_name_pairs = []
    for x in args.stats.split(','):
        stat_comp_name_pairs.append((stat_func_map[x], x))
    return stat_comp_name_pairs


def getthresholds(DTIDKI_Threshold=0.4, WMTI_Threshold=0.4):
    assert DTIDKI_Threshold >=0 and DTIDKI_Threshold <= 1.0, "invalid range of DTIDKI_Threshold"
    assert WMTI_Threshold >=0 and WMTI_Threshold <= 1.0, "invalid range of WMTI_Threshold"

    res = {}
    for metric in ['fa', 'md', 'ad', 'rd', 'mk', 'ak', 'rk']:
        res[metric] = DTIDKI_Threshold
    for metric in ['awf', 'ias_Da', 'eas_De_par', 'eas_De_perp']:
        res[metric] = WMTI_Threshold
    return res

def feature_gen(subjs, metrics, rois, root_dir, DTIDKI_Threshold=0.4, WMTI_Threshold=0.4, stat_comp_name_pairs=[(np.mean, 'mean'), (np.std, 'std'), (skew, 'skewness')], suffix='.nii', registered=False, save_name=None):
    roi_metric_pairs = []
    for roi in rois:
        for metric in metrics:
            roi_metric_pairs.append((roi,metric))
    print('roi_metric_pairs:')
    for i, roi_metric_pair in enumerate(roi_metric_pairs):
    print(i, ':', roi_metric_pair)

    thresholds = getthresholds(DTIDKI_Threshold=DTIDKI_Threshold, WMTI_Threshold=WMTI_Threshold)

    features = np.zeros((len(subjs),len(roi_metric_pairs),len(stat_comp_name_pairs)),dtype=np.float32)

    for (i, subj) in tqdm(enumerate(subjs)):

        fapath = os.path.join(root_dir, 'fa', subj+suffix)
        assert os.path.isfile(fapath), f'directory does not exist {fapath}'
        fa =  ((nib.load(fapath)).get_fdata()).astype(np.float32)

        labelpath = os.path.join(root_dir, 'roi', f"region{roi}"+suffix if registered else f"{subj}_region{roi}"+suffix)
        assert os.path.isfile(labelpath), f'directory does not exist {labelpath}'
        label = ((nib.load(labelpath)).get_fdata()).astype(np.float32)

        for (j, roi_metric_pair) in enumerate(roi_metric_pairs):
            roi, metric = roi_metric_pair
            threshold = thresholds[metric]
            metricpath = os.path.join(root_dir, metric, subj+suffix)
            assert os.path.isfile(metricpath), f'directory does not exist {fapath}'
            metric = ((nib.load(metricpath)).get_fdata()).astype(np.float32)
            metric = (fa > threshold) * metric
            metric = metric[label==roi]

            for k, stat_comp_name_pair in enumerate(stat_comp_name_pairs):
                stat_comp, stat_name =stat_comp_name_pair
                features[i][j][k] = stat_comp(metric[metric != 0], axis=None)
    
    N, M, S = features.shape
    features = features.reshape(N, M*S)
    
    res = {'features':features, 
           'subjs':subjs, 
           'roi_metric_pairs':roi_metric_pairs, 
           'stat_comp_name_pairs':stat_comp_name_pairs}
    
    if save_name is not None:
        save_dir = os.path.join(root_dir, 'features')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True) 
        with open(os.path.join(save_dir, f'{save_name}.pickle'), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return res

def z_score(train_X, test_X):
    trainXmean = np.mean(train_X, axis=0)
    trainXstd = np.std(train_X, axis=0)
    train_X = (train_X - trainXmean) / trainXstd
    test_X= (test_X - trainXmean) / trainXstd
    return train_X, test_X









