from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

def evaluator(X, Y, estimator, args, train_samples=None, train_subjs=None, train_subjs_label=None):
    # define stratified k fold cross validation
    rskf = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.random_state)
    
    if train_subjs is not None:
        # make the fold splits based on subjects
        iterator = []
        for trainindex, valindex in rskf.split(train_subjs_label, train_subjs_label):
            iterator = iterator + [(findSampleIndex(trainindex, train_subjs, train_samples), 
                                    findSampleIndex(valindex, train_subjs, train_samples))]
        cv = iterator
    else:
        # make the fold splits based on samples
        cv=rskf.split(X, Y)
    
    # conduct cross-validation, get validation performance
    scores = cross_validate(estimator, X, Y, scoring=args.scoring, n_jobs=-1, cv=cv, verbose=0, return_estimator=True, return_train_score=True)
    
    return np.mean(scores['test_score']), np.mean(scores['train_score'])












