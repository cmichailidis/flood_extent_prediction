import tensorflow as tf, layers

class TemporalUNET(tf.keras.models.Model):

    """
    This class implements a modified version of the UNET architecture. 
    UNET networks fall under the Fully-Convolutional-Neural-Networks (FCNN)
    category, because they do not make use of Dense Layers. Instead, they rely 
    solely on convolution layers and they attempt to learn an image-to-image 
    mapping. 

    A typical UNET network consists of:
    1) an encoder block (contracting path) with alternating Convolution and Downsampling layers (or strided Convolutions in some cases)
    2) a decoder block (expanding path) with alternating Convolution and Upsampling layers (or Deconvolutions in some cases)
    3) skip connections between all encoder-decoder levels with compatible dimensions

    UNET models are widely used for semantic segmentation tasks, because they are usually able to 
    achieve good performance even in small datasets.

    This particular implementation simply adds ConvLSTM layers in parallel to all skip connections between 
    the encoder and the decoder levels thus making it possible to extract temporal features from
    sequences of satelite images at different resolution levels. The Convolution layers in the 
    encoder and decoder are left as is and they are applied independently for every timestep of the 
    input tensors. 
    """

    def __init__(
        self,
        num_of_levels: int, 
        num_of_filters: list, 
        conv_blocks_per_level: int = 3, 
        kernel_size: tuple = (3,3), 
        leaky_relu_slope: float = 0.10):

        """
        Class Constructor: 

        Arguments List: 
        -> num_of_levels: (int) Number of resolution levels in the encoder and decoder.
        -> num_of_filters: (list of ints) Number of convolution kernels at every resolution level
        -> conv_blocks_per_level: 
        -> kernel_size: (tuple of ints) Kernel dimensions in pixels [height x width]
        -> leaky_relu_slope: ReLU slope for negative inputs.
        """

        # Invoke the constructor of the base class.
        super().__init__()
        
        self._num_of_levels = num_of_levels
        
        # Time-Distributed,  2D-Convolution layers on the encoder path 
        self._encoder_conv_blocks_2D = [None] * self._num_of_levels

        # Time-Distributed, 2D-Convolution layers on the decoder path
        self._decoder_conv_blocks_2D = [None] * self._num_of_levels

        # 2D-Conv-LSTM layers between the encoder and decoder path 
        self._lstm_layers_2D = [None] * self._num_of_levels

        # Skip connections (depth-wise concatenation) on the decoder path
        self._concatenate_layers = [None] * self._num_of_levels
        
        # Downsampling layers on the encoder path 
        self._pooling_layers = [None] * self._num_of_levels

        # Upsampling layers on the decoder path 
        self._upsampling_layers = [None] * self._num_of_levels

        # Build the encoder network / contracting path 
        for i in range(self._num_of_levels): 
            self._encoder_conv_blocks_2D[i] = ConvBlock2D(
                num_of_filters = num_of_filters[i],
                num_of_blocks = conv_blocks_per_level, 
                kernel_size = kernel_size, 
                leaky_relu_slope = leaky_relu_slope                
            )

            # Wrap the previous Convolution Layer with the TimeDistributed layer to let Keras know 
            # that this convolution operation is meant to be applied independently for every frame
            # of the input sequence.
            self._encoder_conv_blocks_2D[i] = tf.keras.layers.TimeDistributed(self._encoder_conv_blocks_2D[i])

            # The last resolution level on the encoder path does not require a Pooling layer 
            if i != self._num_of_levels - 1: 
                self._pooling_layers[i] = tf.keras.layers.MaxPooling2D(pool_size = (2,2), padding = "valid")
                self._pooling_layers[i] = tf.keras.layers.TimeDistributed(self._pooling_layers[i])

        # Add Conv2D-LSTM layers and residual connections between the corresponding encoder and decoder levels.
        for i in range(self._num_of_levels):
            self._lstm_layers_2D[i] = ConvBlockLSTM(
                num_of_filters = num_of_filters[i], 
                kernel_size = kernel_size, 
                num_of_layers = 2, 
                bidirectional = False
            )

        # Build the decoder network / expanding path
        for i in range(self._num_of_levels): 
            self._decoder_conv_blocks_2D[i] = ConvBlock2D(
                num_of_filters = num_of_filters[i], 
                num_of_blocks = conv_blocks_per_level, 
                kernel_size = kernel_size, 
                leaky_relu_slope = leaky_relu_slope
            )

            # Wrap the previous Convolution Layer with the TimeDistributed layer, to let Keras know 
            # that this convolution operation is meant to be applied independently for every frame
            # of the input sequence.
            self._decoder_conv_blocks_2D[i] = tf.keras.layers.TimeDistributed(self._decoder_conv_blocks_2D[i])

            # The first resolution level on the decoder path does not require an Upsampling layer
            if i != 0: 
                self._upsampling_layers[i] = tf.keras.layers.UpSampling2D(size = (2,2), interpolation = "bilinear")
                self._upsampling_layers[i] = tf.keras.layers.TimeDistributed(self._upsampling_layers[i])

            # The last resolution level does not require a channel-wise, concatenation layer
            if i != self._num_of_levels - 1: 
                self._concatenate_layers[i] = tf.keras.layers.Concatenate(axis = -1)

        self._output_layer = tf.keras.layers.Conv3D(
            filters = 1,
            kernel_size = (3,3,3),
            padding = "same",
            activation = "sigmoid"
        )
    
    def call(self, inputs, training = False):
        
        """
        Description:
        The 'call' method describes the computation graph of a model.
        First, the input-stream of satellite images is processed by the 
        convolution layers of the encoder. Then, the output of every 
        encoder-stage is fed to the corresponding ConvLSTM layer. The 
        output of these ConvLSTM layers is then fed to the decoder of the
        architecture. Pooling and Upsampling connections are used to 
        connect the different resolution levels of the encoder and decoder
        blocks. Finally, a 3D Convolution layer is used to produce the 
        segmentation masks of the next timesteps.

        Arguments List: 
         -> inputs: (tensor) The input of the model. A 5D tensor with the following dimensions: {batch, timestep, height, width, channels}
         -> training: (bool) True indicates that the model is in 'training' mode whereas False indicates that the model is in 'inference' 
            mode. For most layers this makes no difference, however certain types of layers (such as batch-norm layers) need this information 
            to work as expected.

        Return List: 
         -> output: 5D tensor { batch x timesteps x height x width x 1 } Model predictions (future segmentation masks)
        """

        temp = inputs
        
        encoder_outputs = [None] * self._num_of_levels
        lstm_outputs = [None] * self._num_of_levels
        decoder_outputs = [None] * self._num_of_levels

        # Define the computation graph of the contracting path / encoder network
        for i in range(self._num_of_levels):             
            encoder_outputs[i] = self._encoder_conv_blocks_2D[i](temp, training = training)

            if i != self._num_of_levels - 1:
                temp = self._pooling_layers[i](encoder_outputs[i])
        
        # Define the computation graph of the ConvLSTM layers between the encoder and decoder
        for i in range(self._num_of_levels): 
            lstm_outputs[i] = self._lstm_layers_2D[i](encoder_outputs[i], training = training)
        
        # Define the computation graph of the expanding path / decoder network
        for i in range(self._num_of_levels - 1, -1, -1):
            
            temp = lstm_outputs[i]
            
            # Concatenation
            if i != self._num_of_levels - 1:
                temp = self._concatenate_layers[i]([lstm_outputs[i], decoder_outputs[i+1]])
            
            # Convolution 
            decoder_outputs[i] = self._decoder_conv_blocks_2D[i](temp, training = training)

            # Upsampling 
            if i != 0:
                decoder_outputs[i] = self._upsampling_layers[i](decoder_outputs[i])

        output_tensor = self._output_layer(decoder_outputs[0], training = training)
        
        return output_tensor