from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def get_model_params(args):
    params = {'class_weight': 'balanced', 'random_state': args.random_state}
    if args.model == 'SVC':
        model_params = {'kernel':args.kernel, 'C':args.C, 'probability':args.probability, 'cache_size': args.cache_size}
    elif args.model == 'LogisticRegression':
        model_params = {'penalty': args.penalty, 'C': args.C, 'max_iter': args.max_iter}
    elif model == 'GradientBoostingClassifier':
        model_params = {'learning_rate': args.learning_rate, 'max_depth': args.max_depth, 'n_estimators': args.n_estimators}
    elif model == 'MLPClassifier':
        if args.auto_batch_size:
            args.batch_size = 'auto'
        model_params = {'hidden_layer_sizes': args.hidden_layer_sizes, 
                        'activation': args.activation, 
                        'solver': args.solver, 
                        'alpha': args.alpha, 
                        'learning_rate_init': args.learning_rate_init, 
                        'momentum': args.momentum,
                        'beta_1': args.beta_1, 
                        'beta_2': args.beta_2, 
                        'max_iter': args.max_iter, 
                        'batch_size': args.batch_size}
        del params['class_weight']
    else:
        model_params = {}
    params = {**params, **model_params}

    return params

def get_model(args):
    
    params = get_model_params(args)

    if args.model == 'SVC':
        model = SVC
    elif args.model == 'LogisticRegression':
        model = LogisticRegression
    elif model == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier
    elif model == 'MLPClassifier':
        model = MLPClassifier

    estimator = model(**params)

    return estimator



