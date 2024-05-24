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

def reproduction_pred(X, Y,
                      alpha,
                      rho,
                      train_lag,
                      rho_ls,
                      ntrial,
                      N
                      ):
    methods = ["CP+LS", "NexCP+LS", "NexCP+WLS"]

    # Initialize prediction intervals
    PI_split_CP = np.zeros((len(methods), ntrial, N, 2))
    PI_split_CP[:, :, :train_lag, 0] = -np.inf
    PI_split_CP[:, :, :train_lag, 1] = np.inf

    # Main loop
    for trial in tqdm(np.arange(ntrial)):
        for pred_idx in range(train_lag, N):
            for method_idx, method in enumerate(methods):
                if method == "CP+LS":
                    weights = np.ones(pred_idx+1)
                    tags = np.ones(pred_idx)
                else:
                    weights = rho ** np.arange(pred_idx, 0, -1)
                    weights = np.r_[weights, 1]  # Fix append

                if method == "NexCP+WLS":
                    tags = rho_ls ** np.arange(pred_idx, -1, -1)
                else:
                    tags = np.ones(pred_idx + 1)  # Default tags for other methods

                idx_odd = np.arange(1, int(np.ceil(pred_idx / 2) * 2), 2)
                idx_even = np.arange(0, int(np.floor(pred_idx / 2) * 2), 2)

                predictor = LinearRegression()
                predictor.fit(X[idx_odd], Y[idx_odd], tags[idx_odd])  # Correct indexing

                mean_predictions, prediction_bands, quantile = weighted_split_conformal_prediction(
                    predictor,
                    X[idx_even],
                    Y[idx_even],
                    X[pred_idx][np.newaxis, :],  # Correct indexing
                    weights[idx_even],
                    alpha
                )

                PI_split_CP[method_idx, trial, pred_idx, :] = prediction_bands
    
    print("Finished processing.")
    return PI_split_CP