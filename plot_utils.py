import numpy as np, rasterio as rio, matplotlib.pyplot as plt
import os, PIL

def array2gif(
    arr, 
    cloud_mask = None, 
    filename = 'array.gif', 
    fps = 12, 
    selector = 'mask'):

    """
    Description:
    Given a 3D numpy array, this function will generate an animated GIF using 2D slices
    from this array as video-frames. Useful for visualization purposes on sequences of satelite images.

    Arguments List:
     -> arr: (3D numpy array) {timestep, height, width} 
     Data to convert into an animated GIF
     
     -> cloud_mask: (None or a numpy array with the same shape as `arr`) {timestep, height, width} 
     optional boolean array with cloud mask indices. The default is `None` (no cloud mask applied). 

     -> filename: (str) 
     Filename for the output GIF file. The default is 'array.gif'
     
     -> fps: (int)
     Frames per second. This parameter controls the playback speed of the generated GIF. The default is 12 fps
     
     -> label: (str)
      * 'VV':        Indicates that the input array contains frames from the VV band of the Sentinel-1 mission 
      * 'VH':        Indicates that the input array contains frames from the VH band of the Sentinel-1 mission
      * 'NDWI':      Indicates that the input array consists of NDWI maps for a particular region
      * 'NDVI':      Indicates that the input array consists of NDVI maps for a particular region
      * 'mNDWI':     Indicates that the input array consists of mNDWI maps for a particular region
      * 'selector':  Indicates that the input array contains segmentation masks (soil vs water) for a particular region

     The default value is 'mask', which means that the array is assumed to contain segmentation masks
     
    Returns:
     -> gif: (Python List) Every entry in the list is a 3D numpy array and corresponds to a single rgb frame. {height x width x 3}
    """

    lo, hi = None, None 
    
    if selector in {'NDWI', 'mNDWI', 'NDVI'}:
        # Spectral indices range: [-1, +1].  
        lo, hi = -1.0 , +1.0
    elif selector in {'VV', 'VH'}:
        # VV and VH SAR bands range: [-50, +10] dB.  
        lo, hi = -50.0, +10.0
    elif selector == 'mask':
        # Segmentation masks: 0 -> soil, +1 -> water
        lo, hi = 0.0, 1.0
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
    rgb = np.zeros_like(grayscale)
    rgb = np.expand_dims(rgb, axis=-1)
    rgb = np.repeat(rgb, repeats=3, axis=-1)
    
    if selector in {'NDWI', 'mNDWI', 'NDVI', 'VV', 'VH'}: 
        rgb[:,:,:,0] = grayscale[:,:,:]
        rgb[:,:,:,1] = grayscale[:,:,:]
        rgb[:,:,:,2] = grayscale[:,:,:]
    elif selector == 'mask':
        rgb[:,:,:,0] = 0
        rgb[:,:,:,1] = 255 * (grayscale >  127)
        rgb[:,:,:,2] = 255 * (grayscale <= 127)
    
    # Filter out cloud-pixels (if a cloud-mask is provided).
    if cloud_mask is not None:
        if selector == 'mask':
            # Black pixels indicate cloud presence in a segmentation mask
            rgb[cloud_mask, 0] = 0
            rgb[cloud_mask, 1] = 0
            rgb[cloud_mask ,2] = 0
        elif selector in {'NDVI', 'NDWI', 'mNDWI'}: 
            # Red pixels indicate cloud presence in a NDWI, NDVI, mNDWI map
            rgb[cloud_mask, 0] = 255
            rgb[cloud_mask, 1] = 0
            rgb[cloud_mask, 2] = 0
        elif selector in {'VV', 'VH'}:
            # SAR data are not affected by cloud presence
            pass

    # Remove NaN-pixels and infinities
    pixel_is_not_finite = np.isfinite(grayscale) == False
    
    if selector == 'mask':
        # Black pixels indicate NaN values and infinities in a segmentation mask 
        rgb[pixel_is_not_finite, 0] = 0
        rgb[pixel_is_not_finite, 1] = 0
        rgb[pixel_is_not_finite, 2] = 0
    elif selector in {'NDVI', 'NDWI', 'mNDWI', 'VV', 'VH'}:
        # Red Pixels indicate NaN values and infinities in NDWI, NDVI, mNDWI, VV and VH maps
        rgb[pixel_is_not_finite, 0] = 255
        rgb[pixel_is_not_finite, 1] = 0
        rgb[pixel_is_not_finite, 2] = 0
    else:
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

    return gif

def plot_landcover(
    ground_truth_sequence,
    predicted_sequence = None,
    figure_index = 1, 
    decision_threshold = 0.5, 
    epsilon = 1e-5):
    
    """
    Description: 
    Given a sequence of frames (sequence of segmentation masks), this function will estimate 
    the water coverage of every frame in the sequence and generate a plot of the water-coverage 
    as a function of time. Useful for visualizing the extent of a flood event over time. 

    Arguments List:
    -> ground_truth_sequence: (3D numpy array) Sequence of ground-truth segmentation masks { timestep x height x width }

    -> predicted_sequence: (3D numpy array) Sequence of model-prediction, segmentation masks { timestep x height x width } 

    -> figure_index: (int) Figure index for pop-up, matplotlib window. The default is 1. See matplotlib.pyplot.figure() 
       for more info.
    
    -> decision_threshold: (float) Decision threshold for discriminating between water and soil pixels. 
       All pixels above the decision threshold are classified as water pixels and all pixels below the 
       decision threshold are classified as soil pixels. The default is 0.5
    
    -> epsilon: (float) A small positive constant for numerical stability in division operations. The 
       default is 1e-5.

    Returns:

    if `predicted_sequence` is not None:
        A two-element tuple with the following entries:
        -> ground_truth_water_coverage: (1D numpy array) A list of water-coverage estimations for every frame in the original sequence.
        -> predicted_water_coverage: (1D numpy array) A list of water-coverage estimations for every frame in the output of the prediction model.

    if `predicted_sequence` is None:
        A single Python List:
        -> ground_truth_water_coverage: (1D numpy array) A list of water-coverage estimations for every frame in the original sequence.

    Warnings: 
        -> This function implicitely assumes that the segmentation-mask sequences are normalized in [0, 1) 
        -> This function implicitely assumes that ground-truth- and model-prediction-sequences are properly aligned along the time-axis
    """

    num_of_frames = ground_truth_sequence.shape()[0]
    
    ground_truth_water_coverage = [0] * num_of_frames
    predicted_water_coverage = [0] * num_of_frames
    
    for i in range(num_of_frames):
        # Extract the current frame from the ground-truth sequence
        ground_truth_frame = ground_truth_sequence[i,:,:] 

        # Count total number of water pixels and soil pixels
        ground_truth_water_pixels = ground_truth_frame[ground_truth_frame > decision_threshold].sum() 
        ground_truth_soil_pixels = ground_truth_frame[ground_truth_frame <= decision_threshold].sum()

        # Estimate ground-truth water-coverage for the current frame
        ground_truth_water_coverage[i] = (ground_truth_water_pixels + epsilon) / (ground_truth_soil_pixels + ground_truth_water_pixels + epsilon)

        # Do the same for model predictions (if a sequence is provided)
        if predicted_sequence is not None:
            # Extract the current frame from the model-predictions sequence
            predicted_frame = predicted_sequence[i,:,:]

            # Count total number of water pixels and soil pixels
            predicted_water_pixels = predicted_frame[predicted_frame > prediction_threshold].sum()
            predicted_soil_pixels = predicted_frame[predicted_frame <= prediction_threshold].sum()

            predicted_water_coverage[i] = (predicted_water_pixels + epsilon) / (predicted_soil_pixels + predicted_water_pixels + epsilon)

    # Cast Python Lists to Numpy arrays
    ground_truth_water_coverage = np.array(ground_truth_water_coverage)

    if predicted_sequence is not None:
        predicted_water_coverage = np.array(predicted_water_coverage)
    
    # New pop-up window for matplotlib plot
    matplotlib.pyplot.figure(figure_index)
    matplotlib.pyplot.clf()
    
    # Plot the original ground-truth time-series of water-coverage
    matplotlib.pyplot.plot(
        np.array(ground_truth_water_coverage), 
        linestyle = '-', 
        linewidth = 0.5, 
        marker = '.', 
        markersize = 4, 
        color = 'g'
    )

    # Plot the model prediction time series (if one is provided)
    if predicted_sequence  is not None:
        matplotlib.pyplot.plot(
            np.array(predicted_water_coverage), 
            linestyle = '-', 
            linewidth = 0.5, 
            marker = '.', 
            markersize = 4, 
            color = 'b'
        )

    # y-axis limits: 0% to 100% water coverage
    # matplotlib.pyplot.ylim(0, 100 + 5)

    if predicted_sequence is not None: 
        matplotlib.pyplot.legend(['Ground Truth', 'Model prediction'], loc='upper left')
    else: 
        matplotlib.pyplot.legend(['Ground Truth'], loc='upper left')

    matplotlib.pyplot.grid(axis='both', linestyle=':')
    matplotlib.pyplot.xlabel('Time/Frame index')
    matplotlib.pyplot.ylabel('Water coverage percentage')
    matplotlib.pyplot.title('Water coverage vs time')
    matplotlib.pyplot.show()

    if predicted_sequence is not None:
        return ground_truth_water_coverage, predicted_water_coverage
    else: 
        return ground_truth_sequence

def plot_dem(
    dem, 
    selector = 'DEM', 
    figure_index = 1, 
    save_png = True, 
    png_name = 'dem.png'):
    
    """
    Description:
    A utility function which displays heatmaps of DEM data (elevation, terrain slope and terrain aspect)

    Arguments List:
    -> dem: (numpy array)
    -> selector: (str) 'DEM' for terrain elevation, 'slope' for terrain slope, 'aspect' for terrain aspect.
    -> figure_index: (int) Figure index for matplotlib pop-up window
    -> save_png: (bool) 

    Returns: 
    -> None

    Warnings:
    -> None
    """

    # Remove NaN values and infinities
    is_not_finite = np.isfinite(arr) == False
    arr[is_not_finite] = 0
    
    if selector == 'DEM': 
        pass
    elif selector == 'aspect':
        pass
    elif selector == 'slope':
        pass
    else: 
        raise ValueError('Invalid DEM selector')

    matplotlib.pyplot.figure(figure_index)
    matplotlib.pyplot.clf()
    matplotlib.pyplot.imshow(dem)
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.show() 

    if save_png == True:
        matplotlib.pyplot.savefig(png_name)

def plot_precipitation(
    arr, 
    figure_index = 1, 
    normalize = False): 
    """
    Description:
    A utility function for generating plots from precipitation timeseries. Useful for discovering potential 
    correlations between flash-flood-events and precipitation

    Arguments List: 
    -> arr: (1D numpy array) Precipitation time-series
    -> figure_index: (int) Figure index for matplotlib pop-up window
    -> normalize: (bool) if set to `True`, the precipitation timeseries is normalized by subtracting its mean and dividing it with the standard deviation.

    Returns: 
    -> None

    Warnings: 
    -> None
    """

    # Replace Infinities and NaN values with zero (if there are any)
    is_not_finite = np.isfinite(arr) == False
    arr[is_not_finite] = 0

    # Open a new matplotlib, pop-up window. Enable grid lines and add axis labels
    matplotlib.pyplot.figure(figure_index)
    matplotlib.pyplot.clf()
    matplotlib.pyplot.grid(axis='both', linestyle=':')
    matplotlib.pyplot.xlabel('Time Index')

    if normalize == True:
        avg = np.mean(arr)
        std = np.std(arr)
        arr = (arr - avg) / std
    
    matplotlib.pyplot.plot(
        arr, 
        linestyle='-', 
        linewidth=1, 
        marker='.', 
        markersize=4,
        color='b'
    )

    if normalize == True:
        matplotlib.pyplot.ylabel('Normalized Precipitation')
        matplotlib.pyplot.title('Normalized Precipitation vs Time')
    else: 
        matplotlib.pyplot.ylabel('Precipitation ()')
        matplotlib.pyplot.title('Precipitation vs Time')
    
    matplotlib.pyplot.show()
    
def plot_model_history(
    history,
    figure_index = 1 ):

    """
    Description: 
    A utility function which generates a plot of the loss function with respect to 
    the training epoch index. Useful for studying the convergence of a potential 
    model during training and adjusting optimization parameters accordingly.

    Arguments List: 
    -> history: (model history). See tensorflow.keras.model.fit() for more info
    -> figure_index: (int) Figure index for matplotlib pop-up window.

    Returns: 
    -> None

    Warnings:
    -> None
    """

    matplotlib.pyplot.figure(figure_index)
    matplotlib.pyplot.clf()

    matplotlib.pyplot.plot(
        history.history['loss'],
        linestyle = '-',
        linewidth = 1, 
        marker = '.', 
        markersize = 8, 
        color = 'g'
    )

    matplotlib.pyplot.plot(
        history.history['val_loss'],
        linestyle = '-',
        linewidth = 1, 
        marker = '.', 
        markersize = 8,
        color = 'b'
    )

    matplotlib.pyplot.legend(['training loss', 'validation loss'], loc='upper right')
    matplotlib.pyplot.grid(axis='both', linestyle=':')
    matplotlib.pyplot.xlabel('Epoch Index')
    matplotlib.pyplot.ylabel('Segmentation Loss')
    matplotlib.pyplot.title('Model-Loss vs Epoch-Index')
    matplotlib.pyplot.show()