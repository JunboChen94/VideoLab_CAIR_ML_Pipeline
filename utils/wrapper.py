import os 
import time
import timeit
import pandas as pd 
import numpy as np

from copy import deepcopy

def bool_reverse(int_value):
    
    return (int_value == 0).astype(np.int16)

def in_list(target, search_list, where=False):

    if not where:

        for item in search_list:
            if (target == item).all():
                return True

        return False

    else:

        for item_idx, item in enumerate(search_list):
            if (target == item).all():
                return item_idx

        return -1

def best_first_search_mg(X, y, estimator, evaluator, args, verbose=False, mega_step=False, patience=100, eps=1e-6, train_samples=None, train_subjs=None, train_subjs_label=None):
    _ , num_features = X.shape
    # 1. put init state on open list, close list = empty, best = initial
    init_feature, init_score = np.zeros([num_features]).astype(np.int16), 0
    open_list = {}
    open_list[tuple(init_feature)] = init_score 
    close_list = set()
    best = init_feature
    best_score = init_score
    # cross score
    current_compound = init_feature
    current_compound_ts = init_score
    current_score = init_score
    # start searching 
    k = 0
    while 1: 
        
        start_time = time.time()
        
        # print(k, mega_step)
        if (current_compound_ts > current_score + eps) and mega_step and (tuple(current_compound) in open_list and tuple(current_compound) not in close_list):
            # if the compound is better than current
            current = current_compound
            current_score = current_compound_ts
        else:
            # let v = argmax(w in open)(open), get state from open with maximal f(w)
            current = max(open_list, key=open_list.get)
            current_score = open_list[current]
            current = np.array(current)
        
        del open_list[tuple(current)]
        close_list.add(tuple(current))

        # check for best update
        if current_score > best_score + eps:
            # reset the patience counter
            k = 0 
            # best is updated by current
            best = current
            best_score = current_score
            print('local best = {0:.4f}, features = {1}'.format(best_score, best.nonzero()[0])) 
        else:
            # increment counter
            k += 1 
        
        # check stop criterion
        if k >= patience:
            history = {'open_list' : open_list, 'close_list' : close_list}
            return best_score, 0, best, history

        # expand the child of current
        
        local_feature_list, local_score_list = [], []
        
        for f, feature_state in enumerate(current):

            current_temp = deepcopy(current)
            current_temp[f] = bool_reverse(feature_state) # get the child of current

            # for child not in open or close, evaluate and add to open

            if (tuple(current_temp) not in open_list) and (tuple(current_temp) not in close_list):

                current_temp_idx = current_temp.nonzero()[0]

                if current_temp_idx.shape[0] == 0:
                    current_temp_ts, current_temp_tr = (-1e8, -1e8)
                else:
                    current_temp_ts, current_temp_tr = evaluator(X[:, current_temp_idx], y, estimator, args, train_samples, train_subjs, train_subjs_label)
                    print(current_temp_idx)
                    print("score: {}".format(current_temp_ts))                
                
                local_feature_list.append(current_temp)
                local_score_list.append(current_temp_ts)

                open_list[tuple(current_temp)] = current_temp_ts

        local_sort = np.argsort(local_score_list)
        first, second = local_feature_list[local_sort[-1]], local_feature_list[local_sort[-2]]
        # calculate the current compound list
        current_compound = first + second - current
        current_compound_idx = current_compound.nonzero()[0]
        current_compound_ts, _ = evaluator(X[:, current_compound_idx], y, estimator, args, train_samples, train_subjs, train_subjs_label)
        print(current_compound_idx)
        print("score: {}".format(current_compound_ts))

        if (tuple(current_compound) not in open_list) and (tuple(current_compound) not in close_list):        
            open_list[tuple(current_compound)] = current_compound_ts           
        
        # 6. if best set/acc change in last k expansion, goto 2
        print("K: {}".format(k))
        end_time = time.time()
        print("iteration time=", end_time - start_time)