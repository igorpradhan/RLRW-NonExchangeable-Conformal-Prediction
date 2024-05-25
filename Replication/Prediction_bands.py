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


def weighted_split_conformal_prediction(predictor,
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
    weights_normalized = cal_weights / (np.sum(cal_weights)+1)

    if(np.sum(weights_normalized) >= 1-alpha):
        # calibration scores: |y_i - x_i @ betahat|
        R = np.abs(y_cal - predictor.predict(X_cal))
        ord_R = np.argsort(R)
        # from when are the cumulative quantiles at least 1-\alpha
        ind_thresh = np.min(np.where(np.cumsum(weights_normalized[ord_R])>=1-alpha))
        # get the corresponding residual
        quantile = np.sort(R)[ind_thresh]
    else:
        quantile = np.inf
    
    # Standard prediction intervals using the absolute residual score quantile
    mean_prediction = predictor.predict(X_test)
    prediction_bands = np.stack([
        mean_prediction - quantile,
        mean_prediction + quantile
    ], axis=1)

    return mean_prediction, prediction_bands, quantile



def run_trial(itrial,
              PI_split_CP,
              train_lag,
              methods,
              rho, rho_LS,
              alpha,
              X, Y, N):
    for pred_idx in np.arange(train_lag, N):
        # we predict the point at pred_idx (n+1) starting at pred_idx=train_lag (101-th datapoint)

        for method_idx, method in enumerate(methods):
            # calibration weights for non-exchangeable conformal prediction (nexCP)
            if method in ['nexCP+LS', 'nexCP+WLS']:
                # weights at 1, ..., n (notice: in Python arrays this becomes 0, ..., n-1)
                weights = rho**(np.arange(pred_idx,0,-1))
                # weight n+1 should always be 1
                weights = np.r_[weights,1]
            else:
                weights = np.ones(pred_idx+1)
            
            # weights for weighted linear regression (WLS)
            if method == 'nexCP+WLS':
                # tags 1, ..., n+1
                tags = rho_LS**(np.arange(pred_idx,-1,-1))
            else:
                tags = np.ones(pred_idx+1)
                
            random_ind = int(np.where(np.random.multinomial(1,weights,1))[1])
            tags[np.c_[random_ind,n]] = tags[np.c_[n,random_ind]]

            # odd data points for training, even ones for calibration
            inds_odd = np.arange(1,int(np.ceil(pred_idx/2)*2-1),2) # excludes pred_idx
            inds_even = np.arange(2,int(np.floor(pred_idx/2)*2),2) # excludes pred_idx

            # train a weighted least squares regression (tags are the weights)
            predictor = LinearRegression()
            predictor.fit(X[itrial, inds_odd], Y[itrial, inds_odd], tags[inds_odd])

            mean_prediction, prediction_bands, quantile = weighted_split_conformal_prediction(
                predictor, # the trained weighted linear regression model
                X[itrial, inds_even], # calibration inputs
                Y[itrial, inds_even], # calibration targets
                X[itrial, pred_idx][np.newaxis, :], # test point to predict
                weights[inds_even], # calibration score weights
                alpha # target miscoverage rate
            )
            PI_split_CP[method_idx,itrial,pred_idx,:] = prediction_bands

            return PI_split_CP