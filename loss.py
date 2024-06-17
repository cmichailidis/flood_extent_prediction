import tensorflow as tf

def BinaryJaccardLoss(
    y_true, 
    y_pred, 
    smoothing = 1e-5, 
    axis = -1, 
    num_of_prediction_steps = 1): 
    
    """
    Description: 
    The Jaccard Loss function is defined as: 
      Jac(A,B) = 1 - IoU(A,B)

    Since all keras optimizers work by minimizing a loss function, 
    the Jaccard Loss function should be used when attempting to 
    maximize the IoU metric of a prediction model.

    Arguments List: 
    -> y_true: (5D tensor) Ground truth segmentation mask + cloud mask {batch x timestep x height x width x 2}
    -> y_pred: (5D tensor) Model predictions {batch x timestep x height x width x 1}
    -> smoothing: (float) Small positive constant to avoid division-by-zero issues
    -> axis: (int)
    -> num_of_prediction_steps: (int) Number of prediction steps to take into account when calculating the loss function

    Returns: 
    -> loss: scalar 1D tensor
    """

    # Separate cloud-probabilities and ground-truth masks and drop un-necessary dimensions
    cloud_prob, y_mask = y_true[:,:num_of_prediction_steps,:,:,0], y_true[:,:num_of_prediction_steps,:,:,1]
    y_pred = y_pred[:,:num_of_prediction_steps,:,:,0]
    
    # Flatten tensors down to 1 dimension
    y_mask = tf.keras.backend.flatten(y_mask)
    y_pred = tf.keras.backend.flatten(y_pred)

    # "Fuzzy" Intersection (Overlap between ground-truth and predictions)
    intersection = tf.keras.backend.abs(y_mask * y_pred)

    # "Fuzzy" Union (Required for scaling / normalization step)
    union = tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred) - intersection

    # Cloud mask
    intersection = intersection * cloud_mask
    union = union * cloud_mask

    # Estimate total size of intersection and union sets 
    union = tf.keras.backend.sum(union, axis=axis)
    intersection = tf.keras.backend.sum(intersection, axis=axis)

    # Intersection-over-Union (Normalized overlap between ground-truth and predictions)
    IoU = (intersection + smoothing) / (union + smoothing) 

    return 1 - IoU 

def BinaryDiceLoss(
    y_true, 
    y_pred, 
    smoothing = 1e-5, 
    num_of_prediction_steps = 1):
    
    """
    Description: 
    The Dice Loss is defined as: 
      DiceLoss(A, B) = 1 - DiceCoeff(A,B)

    Since keras optimizers expect a loss function to minimize, 
    the Dice Loss function should be used when attempting to maximize
    the Dice coefficient of a prediction model.

    Arguments List:
    -> y_true: (5D tensor) Ground-truth segmentation masks + cloud mask {batch x timestep x height x width x 2}
    -> y_pred: (5D tensor) Model predictions {batch x timestep x height x width x 1} 
    -> smoothing: (float) small positive constant to avoid division-by-zero errors
    -> num_of_prediction_steps: (int) Number of prediction steps to take into account when calculating the loss function

    Returns:
    -> loss: scalar 1D tensor
    """

    # Separate cloud-probabilities and ground-truth masks and drop un-necessary dimensions
    cloud_prob, y_mask = y_true[:,:num_of_prediction_steps,:,:,0], y_true[:,:num_of_prediction_steps,:,:,1]
    y_pred = y_pred[:,:num_of_prediction_steps,:,:,0]
    
    # Flatten tensors down to 1D vectors
    y_true = tf.keras.backend.flatten(y_mask)
    y_pred = tf.keras.backend.flatten(y_pred)

    # "Fuzzy" Intersection (Overlap between ground-truth and predictions)
    intersection = tf.keras.backend.abs(y_true * y_pred)

    # "Fuzzy" Union (Required for normalization)
    union = tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred) - intersection

    # Cloud mask
    intersection = intersection * cloud_mask
    union = union * cloud_mask

    # Estimate total size of intersection and union sets
    union = tf.keras.backend.sum(union, axis = axis)
    intersection = tf.keras.backend.sum(intersection, axis = axis)

    # Dice coefficient with smoothing factor
    dice = ( 2 * intersection + smoothing ) / ( union + intersection + smoothing ) 

    # Dice Loss
    return 1 - dice 

def binary_iou_func(smoothing = 1e-5, num_of_prediction_steps = 1): 
    """
    Description:

    Arguments List: 
    -> smoothing: (float) 
    -> num_of_prediction_steps: (int)

    Returns:
    A function handle (lambda function) for calculating the binary IoU loss function 
    """
    
    return lambda y_true,  y_pred : BinaryJaccardLoss(y_true, y_pred, smoothing=smoothing, num_of_prediction_steps = num_of_prediction_steps)

def binary_dice_func(smoothing = 1e-5, num_of_prediction_steps = 1):
    """
    Description:

    Arguments List: 
    -> smoothing: (float) 
    -> num_of_prediction_steps: (int)

    Returns: 
    A function handle (lambda function) for calculating the binary Dice loss function
    """
    
    return lambda  y_true, y_pred: BinaryDiceLoss(y_true, y_pred, smoothing=smoothing, num_of_prediction_steps = num_of_prediction_steps)