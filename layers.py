import tensorflow as tf

class ConvBlock2D(tf.keras.layers.Layer):
    """
    Description:
    This class implemenents the basic convolution block of the architecture.
    It consists of an input 'Conv2D[1x1]-BatchNorm-ReLU' block followed by a number
    of hidden, 'Conv2D[nxn]-BatchNorm-ReLU' blocks. Residual connections are added 
    between the input and output of every hidden block. 

    - The (1x1) convolution-layer (sometimes referred to as depthwise convolution) can be used 
      for dimensionality reduction. This idea was popularized by the authors of the InceptionNet 
      architecture
    
    - Residual connections are used to achieve better gradient flow during training 
      This idea was popularized by the authors of the ResNET architecture
    """

    def __init__(
        self,
        num_of_filters: int,
        num_of_blocks: int = 3,
        kernel_size: tuple = (3,3),
        leaky_relu_slope: float = 0.10 ):

        """
        Description:
        Class Constructor.

        Arguments:
         - num_of_filters: (int) number of convolution kernels per convolution layer.
         - num_of_blocks: (int) number of Conv2D-BatchNorm-ReLU blocks. The default value is 3: 1 input block and 2 hidden blocks
         - kernel_size: (tuple of ints) kernel dimensions for hidden convolution blocks in pixels (height x width). The default is (3,3)
         - leaky_relu_slope: (float) ReLU slope for negative inputs. The default is 0.15 (A negative sign is implicitely assumed)
        """

        assert(num_of_filters > 0)
        assert(num_of_blocks > 1)
        assert(len(kernel_size) == 2)
        assert(kernel_size[0] > 0 and kernel_size[1] > 0)
        
        self._layer_name = "[Conv2D[1x1]*{input_depth}-BatchNorm-ReLU]->[Conv2D[{height}x{width}]*{hidden_depth}-BatchNorm-ReLU]*{blocks}".format(
            input_depth = num_of_filters,
            height = kernel_size[0], 
            width = kernel_size[1], 
            hidden_depth = num_of_filters, 
            blocks = num_of_blocks - 1
        )
        
        # super().__init__( name = self._layer_name )
        super().__init__()

        # Total number of Conv2D-BN-ReLU blocks
        self._num_of_blocks = num_of_blocks

        # Lists for storing Conv2D, BatchNorm, ReLU and Residual layer instances
        self._conv_layers = []
        self._batch_norm_layers = [ tf.keras.layers.BatchNormalization() for i in range(self._num_of_blocks) ]
        self._relu_layers = [ tf.keras.layers.LeakyReLU( alpha = abs(leaky_relu_slope)) for i in range(self._num_of_blocks) ]
        self._residual_layers = [ tf.keras.layers.Add() if i > 0 else None for i in range(self._num_of_blocks) ]

        # Optional Dropout layer (it is meant to be used after the last Conv2D-BN-ReLU block)
        self._dropout_rate = 0.05
        self._dropout_layer = tf.keras.layers.Dropout(rate = self._dropout_rate)
        
        for i in range(self._num_of_blocks):
            self._conv_layers.append(
                tf.keras.layers.SeparableConv2D(
                    filters = num_of_filters,
                    kernel_size = kernel_size if i > 0 else (1,1),
                    padding = "same"
                )
            )

    def call(self, inputs, training = False):
        """
        Description:
        The 'call' method defines the computation graph of this custom layer.
        The 'inputs' tensor first goes through the 1x1 Conv-BatchNorm-ReLU block. 
        The output of this computation is then fed to the hidden Conv-BatchNorm-ReLU blocks.
        With the exception of the first 1x1 conv block, residual connections are used to between
        the input and output of every hidden conv block. The final Conv-BatchNorm-ReLU block 
        may apply a Dropout layer with a small dropout probability.

        Arguments List: 
        -> inputs: (4D tensor) {batch, height, width, channels}
        -> training: (bool) Set to True during training. Set to False during inference mode

        Returns: 
        -> output: (4D tensor) {batch, height, width, channels} The output of the final hidden block.
           * 'batch', 'height' and 'width' are equal to the corresponding dimensions of the input-tensor.
           * 'channels' is the equal to 'num_of_filters' argument provided in the constructor.
        """

        # Placeholder variable for the input tensor of the current Conv2D-BN-ReLU block
        previous_tensor = inputs

        # Placeholder variable for the output tensor of the current Conv2D-BN-ReLU block
        current_tensor = None

        # Define the computation graph of this custom-layer
        for i in range(self._num_of_blocks):

            # Conv2D -> BatchNorm -> ReLU
            current_tensor = self._conv_layers[i](previous_tensor)
            current_tensor = self._batch_norm_layers[i](current_tensor, training = training)
            current_tensor = self._relu_layers[i](current_tensor)

            # Apply the Residual connection (the first Conv2D-BN-ReLU block does not use one)
            if i > 0:
                current_tensor = self._residual_layers[i]([current_tensor, previous_tensor])

            # Save the output to use it as an input for the next iteration
            previous_tensor = current_tensor

        # Uncomment the following line to enable the optional Dropout layer
        # return self._dropout_layer(current_tensor, training=training)
        
        # Return the final output tensor
        return current_tensor

class ConvBlockLSTM(tf.keras.layers.Layer):
    
    """
    Description:
    This class implements the basic Conv-LSTM layer of the architecture. It consists
    of multiple, stacked ConvLSTM layers with additional Residual connections and 
    Batch-Normalization layers in-between. 
    """

    def __init__(
        self, 
        num_of_filters: int, 
        kernel_size: tuple = (3,3),
        num_of_layers: int = 1, 
        bidirectional: bool = False):
        
        """
        Class constructor: 

        Arguments List: 
        num_of_filters: (int) Number of filters per ConvLSTM layer.
        kernel_size: (int) pixel dimensions for convolution kernels {height x width}. The default is [3x3] pixels.
        num_of_layers: (int) Total number of stacked ConvLSTM layers. The default is 1 (single ConvLSTM layer)
        bidirectional: (bool) if True, use bidirectional ConvLSTM layers. The default is False (feedforward, unidirectional layers).
        """

        # Invoke the constructor of the base class (tf.keras.layers.Layer)
        super().__init__()

        # Placeholder attributes for the arguments of the constructor
        self._num_of_layers = num_of_layers
        self._conv_lstm_layers = [None] * self._num_of_layers
        self._batch_norm_layers = [None] * self._num_of_layers
        self._residual_layers = [None] * self._num_of_layers

        # Stack multiple ConvLSTM Layers. 
        for i in range(self._num_of_layers):

            # (Time-Distributed) Batch-Normalization layers
            self._batch_norm_layers[i] = tf.keras.layers.BatchNormalization()
            self._batch_norm_layers[i] = tf.keras.layers.TimeDistributed(self._batch_norm_layers[i])

            # ConvLSTM layers
            self._conv_lstm_layers[i] = tf.keras.layers.ConvLSTM2D(
                filters = num_of_filters,
                kernel_size = kernel_size, 
                padding = "same",
                return_sequences = True, 
                return_state = False, 
                dropout = 0.0,
                recurrent_dropout = 0.0,
            )

            # If the bidirectional option is enabled, wrap the ConvLSTM layer with the BiDirectional layer
            if bidirectional == True: 
                self._conv_lstm_layers[i] = tf.keras.layers.BiDirectional(
                    layer = self._conv_lstm_layers[i],
                    merge_mode = 'sum',
                    backward_layer = None           # TODO: Use a distinct ConvLSTM layer for the backward direction (maybe?)
                )

            # Residual connection (channel-wise addition) between the input and 
            # the output of the ConvLSTM layer.
            # The first ConvLSTM layer cannot have a residual connection, since we don't 
            # know in advance how many feature-maps its input tensor has.
            if i > 0: 
                self._residual_layers[i] = tf.keras.layers.Add()

    def call(self, inputs, training = False):
        
        """
        Description: 
        The 'call' method defines the computation graph of this custom layer. 

        Arguments List: 
        -> inputs:  (5D tensor) { batch, timestep, height, width, channels }
        -> training: (bool) True during training mode, False during inference mode.  

        Returns: 
        -> output: (5D tensor) { batch, timestep, height, width, channels }
        """

        # Placeholder variable for the output of the previous iteration
        temp = inputs

        # Placeholder variable for the output of the current iteration
        output = None

        # Define the computation graph of the stacked ConvLSTM layer
        for i in range(self._num_of_levels):
            
            # Batch-Norm -> ConvLSTM 
            output = self._batch_norm_layers[i](temp, training = training)
            output = self._conv_lstm_layers[i](output, training = training)

            # All ConvLSTM layers use a Residual connection between their input and output ( with the exception of the first layer )
            if i > 0: 
                output = self._residual_layers[i]([output, temp])

            # Save the output tensor of this iteration and used it as input for the next one
            temp = output

        # Return the output tensor of the last iteration
        return output