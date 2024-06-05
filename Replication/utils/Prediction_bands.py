import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression



def split_conformal_bands(predictor, 
                    X_test,  
                    X_cal, 
                    y_cal,
                    alpha = 0.95
                    ):
    cal_mean_predictions = predictor.predict(X_cal)
    S_score = np.abs(y_cal - cal_mean_predictions)
    quantile = np.nanquantile(S_score, 1 - alpha, method = "higher")
    
    mean_predictions = predictor.predict(X_test)
    prediction_bands = np.stack([
        mean_predictions - quantile,
        mean_predictions + quantile
    ], axis = 1)
    
    return prediction_bands, quantile, mean_predictions


def weighted_conformal_prediction(predictor,
                                        X_cal,
                                        y_cal,
                                        X_test,
                                        cal_weights,
                                        alpha=0.95
                                        ):
    """ 
    Weighted Split Conformal Prediction (taken from github.io/code/nonexchangeable_conformal.zip) 
    """

    # normalize weights (we add +1 in the denominator for the test point at n+1)
    

    weights_normalized = cal_weights / (np.sum(cal_weights) + 1)

    if(np.sum(weights_normalized) >= 1-alpha):
        # calibration scores: |y_i - x_i @ betahat|
        R = np.abs(y_cal - predictor.predict(X_cal))
        ord_R = np.argsort(R)
        # from when are the cumulative quantiles at least 1-\alpha
        ind_thresh = np.min(np.where(np.cumsum(weights_normalized[ord_R])>=1-alpha)) if not None else len(weights_normalized)
        # get the corresponding residual
        quantile = np.sort(R)[ind_thresh]
        
    else:
        quantile = 1
    
    # Standard prediction intervals using the absolute residual score quantile
    mean_prediction = predictor.predict(X_test)
    prediction_bands = np.stack([
        mean_prediction - quantile,
        mean_prediction + quantile
    ], axis=1)
    
    return mean_prediction, prediction_bands, quantile
