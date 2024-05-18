import numpy as np

class METHODS():
    """
        GENERATING \TildeP{X} FOR BOX KERNEL
    """
    @staticmethod
    def runif_ball(n ,center, radius):
        d = len(center)
        data = np.zeros((n, d))
        
        U = np.random.rand(n)
        Z = np.random.normal(size = (n, d))
        
        
        
        return data    
        
    
    