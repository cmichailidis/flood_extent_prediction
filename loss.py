import tensorflow as tf

def BinaryJaccardLoss(
    y_true, 
    y_pred, 
    smoothing = 1e-5, 
    axis = -1, 
    use_cloud_mask = False, 
    num_of_prediction_steps = 1): 
    
    """
    Description: 
    The Jaccard Loss function is defined as: 
      Jac(A,B) = 1 - IoU(A,B)

    Since all keras optimizers work by minimizing a loss function, 
    the Jaccard Loss function should be used when attempting to 
    maximize the IoU metric of a prediction model.

    Arguments List: 
    -> y_true: (tensor)
    -> y_pred: (tensor) 
    -> smoothing: (float)
    -> axis: (int)
    -> use_cloud_mask: (bool) 
    -> num_of_prediction_steps: int

    Returns: 
    -> loss: scalar 1D tensor
    """

    # Separate cloud-probabilities and ground-truth masks
    cloud_prob, y_mask = y_true[:,:,:,:,0], y_true[:,:,:,:,1]
    
    # Flatten tensors down to 1 dimension
    y_mask = tf.keras.backend.flatten(y_mask)
    y_pred = tf.keras.backend.flatten(y_pred)

    # "Fuzzy" Intersection (Overlap between ground-truth and predictions)
    intersection = tf.keras.backend.abs(y_mask * y_pred)
    intersection = tf.keras.backend.sum(intersection, axis=axis)

    # "Fuzzy" Union (Required for scaling / normalization step)
    union = tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred)
    union = tf.keras.backend.sum(union, axis=axis)
    union = union - intersection

    # Intersection-over-Union (Normalized overlap between ground-truth and predictions)
    IoU = (intersection + smoothing) / (union + smoothing) 

    return 1 - IoU 

def BinaryDiceLoss(
    y_true, 
    y_pred, 
    smoothing = 1e-5, 
    use_cloud_mask = False, 
    num_of_prediction_steps = 1):
    
    """
    Description: 
    The Dice Loss is defined as: 
      DiceLoss(A, B) = 1 - DiceCoeff(A,B)

    Since keras optimizers expect a loss function to minimize, 
    the Dice Loss function should be used when attempting to maximize
    the Dice coefficient of a prediction model.

    Arguments List:
    -> y_true: (tensor) Ground-truth segmentation masks as a binary tensor.
    -> y_pred: (tensor) Model predictions.
    -> smoothing: (float) 
    -> use_cloud_mask: (bool)

    Returns:
    -> loss: scalar 1D tensor
    """

    # Separate cloud-probabilities and ground-truth masks
    cloud_prob, y_mask = y_true[:,:,:,:,0], y_true[:,:,:,:,1]
    
    # Flatten tensors down to 1D vectors
    y_true = tf.keras.backend.flatten(y_mask)
    y_pred = tf.keras.backend.flatten(y_pred)

    # "Fuzzy" Intersection (Overlap between ground-truth and predictions)
    intersection = tf.keras.backend.abs(y_true * y_pred)
    intersection = tf.keras.backend.sum(intersection, axis = axis)

    # "Fuzzy" Union (Required for normalization)
    union = tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred)
    union = tf.keras.backend.sum(union, axis = axis)
    union = union - intersection

    # Dice coefficient with smoothing factor
    dice = ( 2 * intersection + smoothing ) / ( union + intersection + smoothing ) 

    # Dice Loss
    return 1 - dice 
