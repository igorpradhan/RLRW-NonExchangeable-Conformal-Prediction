import numpy as np
from scipy.stats import multivariate_normal

class METHODS():

    """
        GENERATING \Tilde{X} FOR BOX KERNEL
    """

    @staticmethod
    def runif_ball(n, center, radius):
        d = len(center)
        U = np.random.rand(n)
        Z = np.random.normal(size=(n, d))
        norms = np.linalg.norm(Z, axis=1)
        data = center + radius * (U ** (1 / d))[:, np.newaxis] * Z / norms[:, np.newaxis]
        return data
        
    
    """
        LOCALISATION KERNEL FOR BOX KERNEL
    """
    @staticmethod
    def euclid_distance(x,y):
        return np.linalg.norm(x-y)
    
    
    @staticmethod
    def weighted_quantile(x, q, w):
        weights = w/np.sum(w)
        sorted_indices = np.argsort(x)
        cum_sum = np.cumsum(weights[sorted_indices])
        quantile = np.interp(q, cum_sum, x[sorted_indices])
        return 
    

    """
        Computing Smoothed Weighted Quantile
    """
    @staticmethod
    def smoothed_weighted_quantile(v, alpha, w, indices):
        """
        input:
            v: Unique scores
            alpha: smoothing parameter
            w: weights
            indices: indices of the scores
        output: 
            smoothed_weighted_quantile: smoothed weighted quantile
        """
        weight = w / np.sum(w)
        U = np.random.rand()
        v_tilde = v
        w_tilde = np.sum(weight[indices[v_tilde]])
        
        p_values = np.zeros(len(v_tilde))
        for i in range(len(v_tilde) - 1):
            p_values[i] = np.sum(w_tilde[1: len(v_tilde) - i]) + U * w_tilde[-1]
            
        p_values[-1] = U * w_tilde[-1]

        if np.sum(p_values[p_values > alpha]) > 0: 
            id = np.max(np.where(p_values > alpha))
            quantile = v_tilde[id]
            
        if id < len(v_tilde) - 1:
            subarray_sum = np.sum(w_tilde[(id + 1):(len(v_tilde) - 1)])
            closed = (subarray_sum + U * (w_tilde[id] + w_tilde[-1]) > alpha)
        elif id == len(v_tilde):
            closed = False
        elif id == len(v_tilde) - 1:
            closed = (U * (w_tilde[id] + w_tilde[-1]) > alpha)
        else:
            quantile = float('-inf')
            closed = False
        return quantile, closed
        

    """
        OPTIMUM BANDWIDTH FOR RLCP
    """
    
    @staticmethod
    def opt_RLCP_h(X_train, kernel, h_min, eff_size):
        n_train, d = X_train.shape
        
        def effsize(h):
            H = np.zeros((n_train, n_train))
            for i in range(n_train):
                if kernel == "gaussian":
                    x_tilde_train = np.random.normal(loc=X_train[i, :], scale=np.diag(np.ones(d)) * h**2, size=1)
                    H[i, :] = multivariate_normal.pdf(X_train, mean=x_tilde_train, cov=np.diag(d) * h**2)
                    
                elif kernel == "box":
                    x_tilde_trains = METHODS.runif_ball(n_train, X_train[i, :], h)
                    H[i, :] = np.apply_along_axis(lambda x: np.prod(np.abs(x - X_train[i, :]) <= h), 1, x_tilde_trains)
                    
                H[i,:] = H[i,:] / np.sum(H[i,:])
            effective_size = n_train/np.linalg.norm(H, ord = "fro")**2 - 1
            return effective_size

        
        candidate_bandwidths = np.arrrage(h_min, 6.02, 0.02)
        
        i, optimiser = 1, 0
        while optimiser < eff_size or i > len(candidate_bandwidths):
            optimiser  = effsize(candidate_bandwidths[i])
            h_opt = candidate_bandwidths[i]
            i += 1
        
        return h_opt

        
    """
        Locally weighted CP methods
    """
    
    def RLCP(X_calib, scores_calib, X_test, scores_test, kernel, h, alpha):
        n_test, d = Xtest.shape
        coverage = score_threshold = [0] * n_test
        
        Xcalib = Xcalib[np.argsort(scores_calib), :]
        scores_calib = np.sort(scores_calib)        
        
        scores = (scores, np.inf)
        indices = []
        j = i = 1
        scores_unique = np.array()
        while i < len(scores):
                current_score = scores[i]
                scores_unique.append(current_score)
                current_indices = np.where(scores == current_score)[0]
                indices.append(current_indices)
                i += len(current_indices)
            
        for i in range(n_test):
            xtest = X_test[i]
            test_score = scores_test[i]
            cov_data = np.vstack([Xcalib, xtest])
            
            # Finding the weights and the score threshold
            if kernel == "gaussian":
                xtilde_test = multivariate_normal.rvs(mean=xtest, cov=np.diag(np.ones(d)) * h**2)
                weights = multivariate_normal.pdf(cov_data, mean=xtilde_test, cov=np.diag(np.ones(d)) * h**2)
                result = METHODS.smoothed_weighted_quantile(scores_unique, alpha, weights, indices)
            elif kernel == "box":
                xtilde_test = METHODS.runifball(1, xtest, h).flatten()
                weights = np.array([1 if METHODS.euclid_distance(x, xtilde_test) <= h else 0 for x in cov_data])
                result = METHODS.smoothed_weighted_quantile(scores_unique, alpha, weights, indices)
            
            score_threshold[i] = result[0]  # score_threshold
            closed = result[1]   # whether it's a closed interval
            
            # Coverage
            coverage[i] = int(test_score < score_threshold[i])
            if closed:
                coverage[i] = int(test_score <= score_threshold[i])
        
        return np.column_stack((coverage, score_threshold))

    