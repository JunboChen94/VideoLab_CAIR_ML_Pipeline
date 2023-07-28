import os
import pickle
import numpy as np
from utils.wrapper import best_first_search_mg
from utils.feature_generation import feature_gen, get_stat_comp_name_pairs, z_score
from utils.evaluation import evaluator
from models import get_model
import argparse

# pipeline configuration with the arguments
parser = argparse.ArgumentParser(description='feature_election_and_prediction')
parser.add_argument(
    "--root_dir",
    default='',
    help="root directory where data is saved",
    type=str,
)
parser.add_argument(
    '--config_file',  
    type=str, default='', 
    help="file saving info like subjects")
parser.add_argument(
    '--preload_data',  
    type=int,
    default=0,
    help='if load data from files')
parser.add_argument(
    '--DTIDKI_Threshold',  
    type=float,
    help='fa threshold for DTI, DKI')
parser.add_argument(
    '--WMTI_Threshold',  
    type=float,
    help='fa threshold for DTI, DKI')
parser.add_argument(
    '--stats',  
    type=str, default='mean,std,skewness', 
    help="statistics for feature generation")
parser.add_argument(
    '--metrics',  
    type=str, default='fa,md,mk', 
    help="metrics for feature generation")
parser.add_argument(
    '--rois',  
    type=str, default='0,1,2', 
    help="rois for feature generation")
parser.add_argument(
    '--suffix',  
    type=str, default='.nii', 
    help="suffix for sample file")
parser.add_argument(
    '--registered',  
    type=int, default=0, 
    help="if data is registered")
parser.add_argument(
    '--save_name',  
    type=str, default='features', 
    help="name of feature files")
parser.add_argument(
    '--random_state',  
    type=int, default=89, 
    help="random seed")
parser.add_argument(
    '--patience',  
    type=int,
    default=100,
    help='wrapper params')
parser.add_argument(
    '--eps',  
    type=float,
    default=0.001,
    help='wrapper params')
parser.add_argument(
    '--kernel',  
    type=str,  
    default='linear',
    help="kernel for SVC, e.g. linear ")
parser.add_argument(
    '--C',  
    type=float, 
    default=1.0, 
    help="params for model")
parser.add_argument(
    '--probability',  
    type=int,
    default=0,  
    help="param for model")
parser.add_argument(
    '--cache_size',  
    type=int,
    default=200,  
    help="param for model")
parser.add_argument(
    '--penalty',  
    type=str,
    default='l2',  
    help="param for model")
parser.add_argument(
    '--max_iter',  
    type=int,
    default=100,  
    help="param for model")
parser.add_argument(
    '--learning_rate',  
    type=float,
    default=0.1,  
    help="param for model")
parser.add_argument(
    '--max_depth',  
    type=int,
    default=3,  
    help="param for model")
parser.add_argument(
    '--n_estimators',  
    type=int,
    default=100,  
    help="param for model")
parser.add_argument(
    '--hidden_layer_sizes',  
    type=int,
    default=100,  
    help="param for model")
parser.add_argument(
    '--activation',  
    type=str,
    default='relu',  
    help="param for model")
parser.add_argument(
    '--solver',  
    type=str,
    default='adam',  
    help="param for model") 
parser.add_argument(
    '--alpha',  
    type=float,
    default=0.0001,  
    help="param for model") 
parser.add_argument(
    '--learning_rate_init',  
    type=float,
    default=0.001,  
    help="param for model") 
parser.add_argument(
    '--momentum',  
    type=float,
    default=0.9,  
    help="param for model") 
parser.add_argument(
    '--beta_1',  
    type=float,
    default=0.9,  
    help="param for model") 
parser.add_argument(
    '--beta_2',  
    type=float,
    default=0.999,  
    help="param for model") 
parser.add_argument(
    '--auto_batch_size',  
    type=int,
    default=0,  
    help="param for model") 
parser.add_argument(
    '--hidden_layer_sizes',  
    type=int,
    default=100,  
    help="param for model")
parser.add_argument(
    '--scoring',  
    type=str,
    default='roc_auc',  
    help="metric of performance") 
parser.add_argument(
    '--n_splits',  
    type=int,
    default=5,  
    help="param for evalutaor")
parser.add_argument(
    '--n_repeats',  
    type=int,
    default=10,  
    help="param for evalutaor")
args = parser.parse_args()


# load configuration files saving train/test samples/subjects defined
assert os.path.isfile(args.config_file), "config file was not found!"
with open(args.config_file, 'rb') as handle:
    configs = pickle.load(handle)
train_samples, train_sample_labels = configs['train_samples'], configs['train_sample_labels']
test_samples, test_sample_labels = configs['test_samples'], configs['test_sample_labels']
assert len(list(set(train_samples) & set(test_samples))) == 0
subject_split = False
if 'train_subjects' in configs:
    train_subjects = configs['train_subjects']
    train_subject_labels = configs['train_subjects_labels']
    test_subjects = configs['test_subjects']
    test_subject_labels = configs['test_subjects_labels']
    assert len(list(set(train_subjects) & set(test_subjects))) == 0
    if len(train_subjects) != len(train_samples):
        subject_split = True
# feature generations
if args.preload_data:
    # load feature generated and saved previously
    train_features_file = os.path.join(args.root_dir, 'features', f"{args.save_name}_train.pickle")
    with open(train_features_file, 'rb') as handle:
        train_features = pickle.load(handle)
        train_features = train_features['features']
    test_features_file = os.path.join(args.root_dir, 'features', f"{args.save_name}_test.pickle")
    with open(test_features_file, 'rb') as handle:
        test_features = pickle.load(handle)
        test_features = test_features['features']
else:
    # generate train and test set's features
    metrics = args.metrics.split(',')
    rois = args.rois.split(',')
    stat_comp_name_pairs = get_stat_comp_name_pairs(args)
    # training set feature generation
    train_features = feature_gen(train_samples, 
                                 metrics, 
                                 rois, 
                                 root_dir=args.root_dir, 
                                 DTIDKI_Threshold=args.DTIDKI_Threshold, 
                                 WMTI_Threshold=args.WMTI_Threshold, 
                                 stat_comp_name_pairs=stat_comp_name_pairs, 
                                 suffix=args.suffix, 
                                 registered=args.registered, 
                                 save_name=args.save_name+'_train')
    # test set feature generation
    test_features = feature_gen(test_samples, 
                                 metrics, 
                                 rois, 
                                 root_dir=args.root_dir, 
                                 DTIDKI_Threshold=args.DTIDKI_Threshold, 
                                 WMTI_Threshold=args.WMTI_Threshold, 
                                 stat_comp_name_pairs=stat_comp_name_pairs, 
                                 suffix=args.suffix, 
                                 registered=args.registered, 
                                 save_name=args.save_name+'_test')
    train_features = train_features['features']
    test_features = test_features['features']

# normalize features with z-score based on the training set
train_features, test_features = z_score(train_features, test_features)

# initialize model
model = get_model(args)

# run feature selection based on cross-validation done on the training set
best_feature_set_accu, _, best_feature_set, record = best_first_search_mg(train_features, 
                                                                          train_sample_labels, 
                                                                          model,
                                                                          evaluator,
                                                                          args,
                                                                          verbose=False,
                                                                          mega_step=1, 
                                                                          patience=args.patience, 
                                                                          eps=args.eps,
                                                                          train_samples=train_samples if subject_split else None,
                                                                          train_subjs=train_subjects if subject_split else None, 
                                                                          train_subjs_label=train_subject_labels if subject_split else None)
# show performance of cross-validation with best features found 
print("best feature set performance for cross-validation")
print(best_feature_set_accu)
print("selected one hot")
print(best_feature_set)
best_feature_set = best_feature_set.nonzero()[0]
print("selected feature")
print(best_feature_set)

# with the best feature subset, retrain the model with all samples in the training set
# and show its performance on the test set
train_features_selected = train_features[:, best_feature_set]
test_features_selected = test_features[:, best_feature_set]
clf = estimator.fit(train_features_selected, train_sample_labels)
score = clf.score(test_features_selected, test_sample_labels)
print('performance', score)


















