import tensorflow as tf, numpy as np, rasterio as rio

def _normalize_DEM(
    filepath
    selectors = {'DEM'} ):
    """
    Description:
    This function implements the preprocessing pipeline for the 
    Digital-Elevation-Model of a region. The following steps 
    are implemented: 

    -> The original `DEM` band is scaled down to [0, 1] range by dividing 
    -> The `slope` band is scaled down to [0, 1] range by applying a pixel-wise sin(*) function
    -> The `aspect` band is also scaled down t
    -> 
    """

    # Open DEM file
    DEM = rio.open()[:,:,:,:]

    # Separate DEM, slope and aspect bands
    dem = DEM[]
    slope = DEM[]
    aspect = DEM[]

    # Replace NaN pixel values with zeroes

    # Normalize 'DEM' band

    # Normalize 'slope' band

    # Normalize 'aspect' band

    # Add a DEM band in the final image 
    if 'DEM' in selectors: 
        pass

    # Add a slope band in the final image
    if 'slope' in selectors: 
        pass

    # Add an aspect band in the final image
    if 'aspect' in selectors:
        pass

def _normalize_SAR(filepath):
    """
    Description:
    """
    
    pass

def _normalize_spectral_indices(filepath): 
    """
    Description:
    
    """
    
    pass

def _mask_nan_pixels(frame_sequence, window_length = 3):
    """
    """
    
    pass

class GenericClassName(tf.keras.utils.Sequence): 
    def __init__(self):
        pass

    def __iter__(self):
        pass