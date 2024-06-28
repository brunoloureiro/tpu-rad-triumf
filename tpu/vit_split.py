from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras import activations
import tensorflow.keras as tfa
import tensorflow.keras.layers as nn
from tensorflow.keras import Sequential
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import numpy as np
#0.04553992412
pi=3.141592653589793

#0.5 * x * (1 + tf.tanh(tf.sqrt(2 / pi) * (x + 0.044715 * tf.pow(x,3))))
def other_gelu(x):
    temp = x * x * x
    #return 0.5 * x * (1 + tf.math.erf(x / tf.sqrt(2.0)))
    return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + 0.044715 * temp)))
    #return 0.5 * x * (1.0 + tf.tanh(0.7978845608028653 * (x + 0.04553992412 * tf.pow(x, 3))))

get_custom_objects().update({'other_gelu': Activation(other_gelu)})

#https://gist.github.com/ekreutz/160070126d5e2261a939c4ddf6afb642
class DotProductAttention(keras.layers.Layer):
    def __init__(self, use_scale=True, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.use_scale = use_scale

    def build(self, input_shape):
        query_shape = input_shape[0]
        if self.use_scale:
            dim_k = tf.cast(query_shape[-1], tf.float32)
            self.scale = 1 / tf.sqrt(dim_k)
        else:
            self.scale = None

    def call(self, input):
        query, key, value = input
        score = tf.matmul(query, key, transpose_b=True)
        if self.scale is not None:
            score *= self.scale
        return tf.matmul(tf.nn.softmax(score), value)

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, h=8, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.h = h

    def build(self, input_shape):
        query_shape, key_shape, value_shape = input_shape
        print(query_shape)
        d_model = query_shape[-1]

        # Note: units can be anything, but this is what the paper does
        units = d_model // self.h

        self.layersQ = []
        for _ in range(self.h):
            layer =  layers.Dense(units, activation=None, use_bias=False)
            layer.build(query_shape)
            self.layersQ.append(layer)

        self.layersK = []
        for _ in range(self.h):
            layer =  layers.Dense(units, activation=None, use_bias=False)
            layer.build(key_shape)
            self.layersK.append(layer)

        self.layersV = []
        for _ in range(self.h):
            layer =  layers.Dense(units, activation=None, use_bias=False)
            layer.build(value_shape)
            self.layersV.append(layer)

        self.attention = DotProductAttention(True)

        self.out =  layers.Dense(d_model, activation=None, use_bias=False)
        self.out.build((query_shape[0], query_shape[1], self.h * units))

    def call(self, input):
        query = input[0]
        key = input[1]
        value = input[2]

        q = [layer(query) for layer in self.layersQ]
        k = [layer(key) for layer in self.layersK]
        v = [layer(value) for layer in self.layersV]

        # Head is in multi-head, just like the paper
        head = [self.attention([q[i], k[i], v[i]]) for i in range(self.h)]

        out = self.out(tf.concat(head, -1))
        return out

class CreatePatches( tf.keras.layers.Layer ):

  def __init__( self , patch_size,num_patches,input_image_size ):
    super( CreatePatches , self ).__init__()
    self.patch_size = patch_size
    self.num_patches = num_patches
    self.input_image_size = input_image_size
  def call(self, inputs ):
    patches = []
    # For square images only ( as inputs.shape[ 1 ] = inputs.shape[ 2 ] )
    
    for i in range( 0 , self.input_image_size , self.patch_size ):
        for j in range( 0 , self.input_image_size , self.patch_size ):
            patches.append( inputs[ : , i : i + self.patch_size , j : j + self.patch_size , : ] )
    
    return  tf.concat(patches,axis=-2)

class Patches2(layers.Layer):
    """Create a a set of image patches from input. The patches all have
    a size of patch_size * patch_size.
    """

    def __init__(self, patch_size,num_patches,input_image_size):
        super(Patches2, self).__init__()
        self.patch_size = patch_size
        self.patches_layer = CreatePatches(patch_size = patch_size, num_patches = num_patches,input_image_size=input_image_size)
        self.num_patches = num_patches
    def call(self, images):
        #batch_size = tf.shape(images)[0]
        patches = self.patches_layer(images)
        patches = tf.keras.layers.Reshape([self.patch_size*self.patch_size,self.num_patches*3])(patches)#tf.reshape(patches,[batch_size,self.patch_size,self.patch_size,self.num_patches*3])
        #print(patches.shape)
        #patches = tf.keras.layers.Reshape([ self.patch_size*self.patch_size, self.num_patches*3])(patches)
        #patch_dims = self.num_patches * 3
        #patches = tf.reshape(patches, [batch_size, self.patch_size*self.patch_size, patch_dims])
        return patches

class Mlp( tf.keras.layers.Layer ):
    """Multi-Layer Perceptron

    Args:
        x (tf.Tensor): Input
        hidden_units (List[int])
        dropout_rate (float)

    Returns:
        tf.Tensor: Output
    
    """
    def __init__( self , hidden_units,dropout_rate ):
        super( Mlp , self ).__init__()
        
        self.net=[]
        for units in hidden_units:
            self.net.append(layers.Dense(units, activation=other_gelu))
            self.net.append(layers.Dropout(dropout_rate))
        self.net = Sequential(self.net)
    def call(self, x, training=True):
        return self.net(x, training=training)



class PatchEncoder(layers.Layer):
    """The `PatchEncoder` layer will linearly transform a patch by projecting it into a
    vector of size `projection_dim`. In addition, it adds a learnable position
    embedding to the projected vector.
    """
    
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class ViTModel(keras.Model):
    def __init__(
        self,
        in_shape,
        num_classes: int,
        image_size: int,
        patch_size: int,
        num_patches: int,
        projection_dim: int,
        dropout: float,
        n_transformer_layers: int,
        num_heads: int,
        transformer_units: List[int],
        mlp_head_units: List[int],
        normalization: bool=False,
        partial_model_op: int = 0,
        **kwargs,
    ):
        super().__init__(self, **kwargs)
        
        # Create patches.
        self.patches_2 = Patches2(patch_size,num_patches,input_image_size=image_size)
        
        # Encode patches.
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)

        self.head_layer_norm = []
        self.head_mha = []
        self.head_skip_1 = []
        self.head_layer_norm_2 = []
        self.head_mlp = []
        self.head_skip_2 = []
        
        # Create multiple layers of the Transformer block.
        for _ in range(n_transformer_layers):
            # Layer normalization 1.
            self.head_layer_norm.append(layers.LayerNormalization(epsilon=1e-6))
            # Create a multi-head attention layer.
            self.head_mha.append(MultiHeadAttention(h=num_heads))
            # Skip connection 1.
            self.head_skip_1.append(layers.Add())
            # Layer normalization 2.
            self.head_layer_norm_2.append(layers.LayerNormalization(epsilon=1e-6))
            # Mlp.
            self.head_mlp.append(Mlp( hidden_units=transformer_units, dropout_rate=0.1))
            # Skip connection 2.
            self.head_skip_2.append(layers.Add())

        # Create a [batch_size, projection_dim] tensor.
        self.norm_after_heads = layers.LayerNormalization(epsilon=1e-6)
        self.flatten_after_heads = layers.Flatten()
        self.dropout_after_heads = layers.Dropout(dropout)
        
        # Add Mlp.
        self.mlp_after_heads = Mlp( hidden_units=mlp_head_units, dropout_rate=dropout)
        
        # Classify outputs.
        self.classifier = layers.Dense(num_classes,activation='softmax')

        self.partial_model_op = partial_model_op

        self.n_transformer_layers = n_transformer_layers

    def call(self, inputs):
        augmented = inputs
        
        # Create patches.
        patches = self.patches_2(augmented)
        
        # Encode patches.
        encoded_patches = self.patch_encoder(patches)

        if self.partial_model_op == 1:
            return encoded_patches

        ops_per_head = 6
        ops_before = 1

        # Create multiple layers of the Transformer block.
        for head_iter in range(self.n_transformer_layers):
            i = head_iter
            # Layer normalization 1.
            x1 = self.head_layer_norm[i](encoded_patches)
            if self.partial_model_op == (head_iter*ops_per_head) + 1 + ops_before:
                return x1
            # Create a multi-head attention layer.
            attention_output = self.head_mha[i]([x1,x1,x1])
            if self.partial_model_op == (head_iter*ops_per_head) + 2 + ops_before:
                return attention_output
            # Skip connection 1.
            x2 = self.head_skip_1[i]([attention_output, encoded_patches])
            if self.partial_model_op == (head_iter*ops_per_head) + 3 + ops_before:
                return x2
            # Layer normalization 2.
            x3 = self.head_layer_norm_2[i](x2)
            if self.partial_model_op == (head_iter*ops_per_head) + 4 + ops_before:
                return x3
            # Mlp.
            x3 = self.head_mlp[i](x3)
            if self.partial_model_op == (head_iter*ops_per_head) + 5 + ops_before:
                return x3
            # Skip connection 2.
            encoded_patches = self.head_skip_2[i]([x3, x2])
            if self.partial_model_op == (head_iter*ops_per_head) + 6 + ops_before:
                return encoded_patches

            head_iter += 1

        # Create a [batch_size, projection_dim] tensor.
        representation = self.norm_after_heads(encoded_patches)
        if self.partial_model_op == (head_iter*ops_per_head) + 7 + ops_before:
                return representation
        representation = self.flatten_after_heads(representation)
        if self.partial_model_op == (head_iter*ops_per_head) + 8 + ops_before:
                return representation
        representation = self.dropout_after_heads(representation)
        if self.partial_model_op == (head_iter*ops_per_head) + 9 + ops_before:
                return representation
        
        # Add Mlp.
        features = self.mlp_after_heads(representation)
        if self.partial_model_op == (head_iter*ops_per_head) + 10 + ops_before:
            return features
        
        # Classify outputs.
        logits = self.classifier(features)
        #print(logits.shape)
        # Create the Keras model.
        return logits

    def model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        outputs = self.call(inputs)

        return keras.Model(inputs=inputs, outputs=outputs)