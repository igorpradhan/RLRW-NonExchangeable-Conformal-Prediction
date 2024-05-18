import numpy as np

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
        
