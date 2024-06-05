import tensorflow a tf

def BinaryJaccardIndex(
    y_true, 
    y_pred, 
    epsilon = 1e-5, 
    axis = -1, 
    use_cloud_mask = True, 
    cloud_threshold = 0.5,
    num_of_prediction_steps = 1):
    
    """
    Description:
    The Jaccard Index (also known as Intersection-over-Union, IoU) is a 
    commonly used metric in image segmentation tasks. 

    Given two segmentation masks A and B, the IoU is defined as: 
      IoU(A,B) = intersection(A,B) / union(A,B)

    IoU values close to 1, indicate good overlap between A and B
    IoU values close to 0, indicate poor overlap between A and B

    Arguments List:
    -> y_true: (5D tensor) Ground-truth segmentation mask + optional cloud-probability masks { batch x timesteps x height x width x 2 }
    -> y_pred: (5D tensor) Predicted segmentation mask { batch x timesteps x height x width x 1 } 
    -> epsilon: (float) small positive constant for numeric stability in tensor division operations
    -> axis: (int)
    -> use_cloud_mask: (bool) if True, all pixels with a cloud-probability greater than 'cloud_threshold' are not taken into account
       The default is True
    -> cloud_threshold: (float) See above
    -> num_of_prediction_steps: (int) Number of future prediction steps to take into account when calculating this metric. The default
       is 1 which means that only the next prediction step is taken into account.

    Returns:
    -> IoU: scalar, 1D tensor

    References: 
    -> https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """

    # Separate cloud-probabilities and ground-truth masks
    cloud_prob, y_mask = y_true[:,:,:,:,0], y_true[:,:,:,:,1]
    
    # Flatten tensors down to 1 dimension
    y_mask = tf.keras.backend.flatten(y_mask)
    y_pred = tf.keras.backend.flatten(y_pred)

    # Intersection (Overlap between ground-truth and predictions)
    intersection = tf.keras.backend.abs(y_mask * y_pred)
    intersection = tf.keras.backend.sum(intersection, axis=axis)

    # Union (Required for scaling / normalization step)
    union = tf.keras.backend.abs(y_mask) + tf.keras.backend.abs(y_pred)
    union = tf.keras.backend.sum(union, axis=axis)
    union = union - intersection

    return  intersection / (union + epsilon)

def BinaryDiceCoefficient(
    y_true, 
    y_pred, 
    epsilon = 1e-5, 
    axis = -1, 
    use_cloud_mask = False, 
    num_of_prediction_steps = 1):
    
    """
    Description:
    Much like the Jaccard Index, the Dice coefficient is also used 
    in image segmentation tasks to measure the similarity between two
    segmentation maps. 

    Given two segmentation masks A and B, the Dice coefficient is defined as: 
      Dice(A,B) = 2 * intersection(A,B) / ( union(A,B) + intersection(A,B) )

    Values close to 1 indicate strong agreement between A and B, whereas 
    values close to 0 indicate poor agreement between A and B.

    Arguments List:
    -> y_true: (tensor)
    -> y_pred: (tensor)
    -> epsilon: (float)
    -> axis: (int)
    -> use_cloud_mask: (bool)

    Returns:
    -> coeff: scalar 1D tensor
    """

    # Separate cloud-probabilities and ground-truth masks
    cloud_prob, y_mask = y_true[:,:,:,:,0], y_true[:,:,:,:,1]
    
    # Flatten tensors down to 1 dimension
    y_mask = tf.keras.backend.flatten(y_mask)
    y_pred = tf.keras.backend.flatten(y_pred)

    # "Fuzzy" Intersection (Overlap between ground-truth and predictions)
    intersection = tf.keras.backend.abs(y_mask * y_pred)
    intersection = tf.keras.backend.sum(intersection, axis = axis)

    # "Fuzzy" Union (Required for scaling / normalization step)
    union = tf.keras.backend.abs(y_mask) + tf.keras.backend.abs(y_pred)
    union = tf.keras.backend.sum(union, axis = axis)
    union = union - intersection

    # Dice Coefficient 
    return ( 2 * intersection ) / ( union + intersection + epsilon )