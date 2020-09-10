def compute_metrics(metrics, y_true, y_pred):
    scores = {}
    for metric in metrics:
        scorer = _get_scorer(metric)
        scores[metric] = scorer(y_true, y_pred)
    return scores

def _get_scorer(metric):
    if metric == 'r2':
        return _r2
    
    elif metric == 'explained_variance':
        return _explained_variance
    
    elif metric == 'max_error':
        return _max_error
    
    elif metric == 'mean_absolute_error':
        return _mean_absolute_error
    
    elif metric == 'mean_squared_error':
        return _mean_squared_error
    
    elif metric == 'root_mean_squared_error':
        return _root_mean_squared_error
    
    elif metric == 'mean_squared_log_error':
        return _mean_squared_log_error
    
    elif metric == 'median_absolute_error':
        return _median_absolute_error
    
    elif metric == 'mean_tweedie_deviance':
        return _mean_tweedie_deviance
    
    elif metric == 'mean_poisson_deviance':
        return _mean_poisson_deviance
    
    elif metric == 'mean_gamma_deviance':
        return _mean_gamma_deviance
    
    else:
        raise ValueError(metric)

def _r2(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

def _explained_variance(y_true, y_pred):   
    from sklearn.metrics import explained_variance_score
    return explained_variance_score(y_true, y_pred)

def _max_error(y_true, y_pred):
    from sklearn.metrics import max_error
    return max_error(y_true, y_pred)

def _mean_absolute_error(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)
    
def _mean_squared_error(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)

def _root_mean_squared_error(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred, squared = False)

def _mean_squared_log_error(y_true, y_pred):
    from sklearn.metrics import mean_squared_log_error
    return mean_squared_log_error(y_true, y_pred)

def _median_absolute_error(y_true, y_pred):
    from sklearn.metrics import median_absolute_error
    return median_absolute_error(y_true, y_pred)

def _mean_tweedie_deviance(y_true, y_pred):
    from sklearn.metrics import mean_tweedie_deviance
    return mean_tweedie_deviance(y_true, y_pred)

def _mean_poisson_deviance(y_true, y_pred):
    from sklearn.metrics import mean_poisson_deviance
    return mean_poisson_deviance(y_true, y_pred)

def _mean_gamma_deviance(y_true, y_pred):
    from sklearn.metrics import mean_gamma_deviance
    return mean_gamma_deviance(y_true, y_pred)