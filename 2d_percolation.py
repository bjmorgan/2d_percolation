from scipy.ndimage import label
import numpy as np
from typing import Optional

def is_percolating(f: np.ndarray,
                   axis: int,
                   periodic: Optional[bool]=False) -> bool:
    """Test whether a binary 2D cluster is percolating between two opposite edges.
    
    Requires that all nodes in the cluster are contiguous.
    
    Args:
        f (np.ndarray): A binary 2D numpy array.
        axis (int): The axis to test along: 0 (x) or 1 (y).
        periodic (optional[bool]): Only consider clusters to be percolating if they
            are periodic along the specified axis. Default is False.
            
    Returns:
        bool
    
    """
    if np.all(np.any(f, axis=axis)):
        if periodic:
            match axis:
                case 0:
                    if not np.any(np.bitwise_and(f[:,0], f[:,-1])):
                        return False
                case 1:
                    if not np.any(np.bitwise_and(f[0], f[-1])):
                        return False
                case _:
                    raise ValueError("axis should be 0 (x) or 1 (y)")
        return True
    else:
        return False
            
    
def percolation_threshold(landscape: np.ndarray,
                          axis: int,
                          conv: float,
                          periodic=False) -> [float, np.ndarray]:
    """Find the lowest-maximum-score percolating cluster for a 2D graph.
    
    Uses scipy.ndimage.label to find all clusters of contiguous points with
    "heights" below a threshold value.
    For each cluster, test whether this percolates along the specified axis.
    Uses a bracketing algorithm to find the threshold height for percolation,
    within a window set by the conv parameter.
    
    Args:
        landscape (np.ndarray): 2D numpy array of "heights" for each node.
        axis (int): Axis to compute the threshold percolating path for (x=0, y=1).
        conv (float): Threshold bracketing convergence.
        periodic (optional[bool]): Only consider clusters to be percolating if they
            are periodic along the specified axis. Default is False.
            
    Returns:
        float, np.ndarray: Returns the threshold value, 
            and a map of the corresponding percolating cluster.
    
    """
    threshold_upper = landscape.max()
    threshold_lower = landscape.min()
    percolating_cluster = None
    while threshold_upper - threshold_lower > conv:
        mid_point = (threshold_upper + threshold_lower)/2
        image = (landscape <= mid_point).astype(int)
        features, labels = label(image)
        for l in range(labels):
            f = features == l + 1
            if is_percolating(f, axis=axis, periodic=periodic):
                threshold_upper = mid_point
                percolating_cluster = f.astype(int)
                break
        else:
            threshold_lower = mid_point    
    return threshold_upper, percolating_cluster

