import numpy as np, rasterio as rio, matplotlib.pyplot as plt
import os, PIL

# imgs = np.random.randint(0, 255, (100, 50, 50, 3), dtype=np.uint8)

# imgs[:,:,:,0] = imgs[:,:,:,0]
# imgs[:,:,:,1] = imgs[:,:,:,0]
# imgs[:,:,:,2] = imgs[:,:,:,0]

# imgs = [PIL.Image.fromarray(img) for img in imgs]

# # duration is the number of milliseconds between frames; this is 40 frames per second
# imgs[0].save("array.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)

def array2gif(
    arr, 
    cloud_mask = None, 
    filename = 'array.gif', 
    fps = 12, 
    label = 'mask'):

    """
    Description:
    Given a 3D numpy array, this function will generate an animated GIF using 2D slices
    from this array as video-frames. Useful for visualization purposes on sequences of satelite images.

    Arguments List:
     -> arr: (3D numpy array) {timestep, height, width} 
     Data to convert into an animated GIF
     
     -> cloud_mask: (None or a numpy array with the same shape as `arr`) {timestep, height, width} 
     optional boolean array with cloud mask indices. The default is `None` (no cloud mask). 

     -> filename: (str) 
     Filename for the output GIF file. The default is 'array.gif'
     
     -> fps: (int)
     Frames per second. This parameter controls the playback speed of the generated GIF. The default is 12 fps
     
     -> label: (str)
      * 'VV':    Indicates that the input array contains frames from the VV band of the Sentinel-1 mission 
      * 'VH':    Indicates that the input array contains frames from the VH band of the Sentinel-1 mission
      * 'NDWI':  Indicates that the input array consists of NDWI maps for a particular region
      * 'NDVI':  Indicates that the input array consists of NDVI maps for a particular region
      * 'mNDWI': Indicates that the input array consists of mNDWI maps for a particular region
      * 'mask':  Indicates that the input array contains segmentation masks (soil vs water) for a particular region

     The default value is 'mask'. 
     
    Returns:
     -> None. 
    """

    lo, hi = None, None 
    
    if label in {'NDWI', 'mNDWI', 'NDVI'}:
        # Spectral indices range: [-1, +1].  
        lo, hi = -1.0 , +1.0
    elif label in {'VV', 'VH'}:
        # VV and VH SAR bands range: [-50, +10] dB.  
        lo, hi = -50.0, +10.0
    elif label == 'mask':
        # Segmentation masks: 0 -> soil, +1 -> water
        lo, hi = 0.0, +1.0
    else:
        raise ValueError("Invalid data format")

    # Normalize to [0, 1] range
    grayscale = (arr - lo) / (hi - lo)

    # Rescale to [0, 255]
    grayscale = (255.0 - 0.0) * grayscale + 0.0
    
    # Thresholding (Make sure that all pixels stay within the [0, 255] range)
    grayscale[grayscale < 0] = 0
    grayscale[grayscale > 255] = 255

    # Convert 3D grayscale array to 4D RGB array. That is:
    # {timestep, height, width} => {timestep, height, width, RGB-channel}
    rgb = np.zeros_like()

    if label in {'NDWI', 'mNDWI', 'NDVI', 'VV', 'VH'}: 
        rgb[:,:,:,0] = grayscale[:,:,:]
        rgb[:,:,:,1] = grayscale[:,:,:]
        rgb[:,:,:,2] = grayscale[:,:,:]
    elif label == 'mask':
        rgb[:,:,:,0] = 0
    
    # Filter out cloud-pixels (if a cloud-mask is provided).
    if cloud_mask is not None:
        if label == 'mask':
            # Black pixels indicate cloud presence in a segmentation mask
            rgb[cloud_mask, 0] = 0
            rgb[cloud_mask, 1] = 0
            rgb[cloud_mask ,2] = 0
        elif label in {'NDVI', 'NDWI', 'mNDWI'}: 
            # Red pixels indicate cloud presence in a NDWI, NDVI, mNDWI map
            rgb[cloud_mask, 0] = 255
            rgb[cloud_mask, 1] = 0
            rgb[cloud_mask, 2] = 0
        elif label in {'VV', 'VH'}:
            # SAR data are not affected by cloud presence for the most part.
            pass

    # Filter out NaN-pixels
    if label == 'mask':
        pass
    elif label in {'NDVI', 'NDWI', 'mNDWI'}:
        pass

    # Cast to unsigned, 8bit integer
    rgb = rgb.real.astype(dtype = np.uint8)
    
    # Convert 4D numpy array to a list of PIL.Image instances (frames)
    gif = [PIL.Image.fromarray(frame) for frame in rgb]
    
    # Save frames as GIF
    gif[0].save(
        fp = filename, 
        save_all = True, 
        append_images = gif[1:], 
        duration = 1000 / fps, 
        loop = 0
    )

def plot_lulc(sequence):
    """
    Description: 
    Given a sequence of frames (sequence of segmentation masks), this function will 
    """
    pass