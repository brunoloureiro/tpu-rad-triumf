#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

from transformers.tf_utils import shape_list, stable_softmax

import collections.abc
from typing import Optional, Tuple, Union
#from keras.utils.generic_utils import get_custom_objects

#from transformers.activations_tf import get_tf_activation

from transformers.modeling_tf_utils import (
	get_initializer,
)

# based on timm's implementation of DeiT for TensorFlow

'''

TIMM's DeiT:

TFDLeiTForImageClassification
	dleit = TFDLeiTMainLayer
	classifier = keras.layers.Dense (K classes) or keras.layers.Activation (binary yes/no)


TFDLeiTMainLayer
	embeddings = TFDLeiTEmbeddings
	encoder = TFDLeiTEncoder
	layernorm = keras.layers.LayerNormalization
	pooler = TFDLeiTPooler

TFDLeiTPooler
	dense = keras.layers.Dense(activation="tanh")

Normalization Layer
	keras.layers.LayerNormalization

TFDLeiTEncoder
	layers = array: TFDLeiTLayer

TFDLeiTLayer
	attention = TFDLeiTAttention
	intermediate = TFDLeiTIntermediate
	dleit_output = TFDLeiTOutput

	layernorm_before = keras.layers.LayerNormalization
	layernorm_after = keras.layers.LayerNormalization

TFDLeiTOutput
	dense = keras.layers.Dense
	dropout = keras.layers.Dropout

TFDLeiTIntermediate
	dense = keras.layers.Dense
	intermediate_activation_function = get_tf_activation()

TFDLeiTAttention
	self_attention = TFDLeiTSelfAttention
	dense_output = TFDLeiTSelfOutput

TFDLeiTSelfOutput
	dense = keras.layers.Dense
	dropout = keras.layers.Dropout

TFDLeiTSelfAttention
	query = keras.layers.Dense
	key = keras.layers.Dense
	value = keras.layers.Dense
	dropout = keras.layers.Dropout

	call():
		a lot of things
		including stable_softmax, multiply, matmul, tranpose, ...

TFDLeiTEmbeddings
	patch_embeddings = TFDLeiTPatchEmbeddings
	dropout = keras.layers.Dropout

TFDLeiTPatchEmbeddings
	projection = keras.layers.Conv2D

'''

'''

hidden_size=768,
num_hidden_layers=12,
num_attention_heads=12,
intermediate_size=3072,
hidden_act="gelu",
hidden_dropout_prob=0.0,
attention_probs_dropout_prob=0.0,
initializer_range=0.02,
layer_norm_eps=1e-12,
image_size=224,
patch_size=16,
num_channels=3,
qkv_bias=True,
encoder_stride=16,

'''

DEBUG_SHAPES = False

class DLeiTConfig():
	def __init__(
		self,
		hidden_size=768,
		num_hidden_layers=12,
		num_attention_heads=12,
		intermediate_size=3072,
		hidden_act="gelu",
		hidden_dropout_prob=0.0,
		attention_probs_dropout_prob=0.0,
		initializer_range=0.02,
		layer_norm_eps=1e-12,
		image_size=224,
		patch_size=16,
		num_channels=3,
		qkv_bias=True,
		encoder_stride=16,
		num_labels=1000,
		use_return_dict=False,
		iterative_qkv=False,
		stacked_heads=False,
	):
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.intermediate_size = intermediate_size
		self.hidden_act = hidden_act
		self.hidden_dropout_prob = hidden_dropout_prob
		self.attention_probs_dropout_prob = attention_probs_dropout_prob
		self.initializer_range = initializer_range
		self.layer_norm_eps = layer_norm_eps
		self.image_size = image_size
		self.patch_size = patch_size
		self.num_channels = num_channels
		self.qkv_bias = qkv_bias
		self.encoder_stride = encoder_stride
		self.batch_size = 1
		self.num_labels = num_labels
		self.use_return_dict = use_return_dict
		self.iterative_qkv = iterative_qkv
		self.stacked_heads = stacked_heads

		image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
		patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

		num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

		self.num_patches = num_patches

class DLeiT(keras.Model):
	def __init__(self, config: DLeiTConfig, **kwargs) -> None:
		super().__init__(**kwargs)

		self.config = config

		self.num_channels = config.num_channels
		self.batch_size = config.batch_size

		image_size = config.image_size
		image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)

		# TPU must always use batch=1
		self.image_shape = (*image_size, config.num_channels)

		self.num_labels = config.num_labels
		self.dleit = TFDLeiTMainLayer(config, add_pooling_layer=False, name="dleit")

		# Classifier head
		self.classifier = (
			keras.layers.Dense(config.num_labels, name="classifier")
			if config.num_labels > 0
			else keras.layers.Activation("linear", name="classifier")
		)
		self.config = config

	def call(
		self,
		pixel_values: tf.Tensor,
		const_tokens: tf.Tensor,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		training: Optional[bool] = False,
	):
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.dleit(
			pixel_values,
			const_tokens,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			training=training,
		)

		#sequence_output = outputs[0]
		sequence_output = outputs

		if DEBUG_SHAPES: print(f"Shape of output of DLeiTMainLayer: {sequence_output.shape}")

		logits = self.classifier(sequence_output[:, 0, :])
		# we don't use the distillation token

		if DEBUG_SHAPES: print(f"Logits shape: {logits.shape}")

		return logits

	def model(self, input_shape=None):
		input_shape = input_shape if input_shape is not None else self.image_shape

		input_pixels = keras.layers.Input(shape=input_shape, batch_size=self.batch_size)
		const_tokens = keras.layers.Input(shape=(self.batch_size, self.config.num_patches+4, self.config.hidden_size), batch_size=self.batch_size)

		return keras.Model(
			inputs = [input_pixels, const_tokens],
			outputs = self.call(input_pixels, const_tokens),
		)

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "dleit", None) is not None:
			with tf.name_scope(self.dleit.name):
				self.dleit.build(None)
		if getattr(self, "classifier", None) is not None:
			with tf.name_scope(self.classifier.name):
				self.classifier.build([self.config.batch_size, self.config.num_patches+2, self.config.hidden_size])

class TFDLeiTMainLayer(keras.layers.Layer):
	def __init__(self, config: DLeiTConfig, add_pooling_layer: bool = False, **kwargs) -> None:
		super().__init__(**kwargs)

		self.config = config

		self.num_channels = config.num_channels
		self.batch_size = config.batch_size

		image_size = config.image_size
		image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)

		# TPU must always use batch=1
		self.image_shape = (*image_size, config.num_channels)

		self.embeddings = TFDLeiTEmbeddings(
			config,
			name="embeddings",
		)

		# should be encoder after embeddings

		# but I will try with each part of the encoder first

		# TFDLeiTAttention
		# has TFDLeiTSelfAttention
		# and TFDLeiTSelfOutput

		#self.encoder = TFDLeiTSelfAttention(config, name="attention")
		#self.encoder = TFDLeiTAttention(config, name="attention")

		# the encoder itself is several layers of ``TFDleiTLayer''
		# each layer has TFDLeiTAttention,
		# 	followed by TFDLeiTIntermediate
		# 	and then finally TFDLeiTOutput

		# Intermediate is Dense -> activation
		# Output is Dense -> Dropout

		#self.encoder = TFDLeiTLayer(config, name="attention")

		self.encoder = TFDLeiTEncoder(config, name="encoder")

		self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

		self.pooler = TFDLeiTPooler(config, name="pooler") if add_pooling_layer else None

	def ensure_bhwc(self, pixel_values: tf.Tensor):
		batch_size, height, width, num_channels = shape_list(pixel_values)

		if num_channels != self.num_channels and height == self.num_channels:
			# TF 2.0 image layers can't use NCHW format when running on CPU.
			# (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
			pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))

		return pixel_values

	def call(
		self,
		pixel_values: tf.Tensor,
		const_tokens: tf.Tensor,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		training: Optional[bool] = False,
	):
		if DEBUG_SHAPES: print(f"DLeiT received input of shape {pixel_values.shape}")
		#pixel_values = self.ensure_bhwc(pixel_values)
		#if DEBUG_SHAPES: print(f"After ensuring BHWC shape: {pixel_values.shape}")

		embedding_output = self.embeddings(pixel_values, const_tokens[0])

		if DEBUG_SHAPES: print(f"Return of embeddings: {embedding_output.shape}")

		encoder_outputs = self.encoder(
			embedding_output,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			training=training,
		)

		if DEBUG_SHAPES: print(f"Return of encoder: {encoder_outputs.shape}")

		# REMOVE THIS
		#return encoder_outputs

		#sequence_output = encoder_outputs[0]
		sequence_output = self.layernorm(encoder_outputs, training=training)

		#pooled_output = self.pooler(sequence_output, training=training) if self.pooler is not None else None
		assert self.pooler is None, f"Pooling is currently disabled."

		#if DEBUG_SHAPES and pooled_output is not None: print(f"TFDLeiTMainLayer pooler output shape: {pooled_output.shape}")

		#if not return_dict:
		#	head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
		#	final_output = head_outputs + encoder_outputs[1:]

		#	pooled_shape = pooled_output.shape if pooled_output is not None else None
		#	if DEBUG_SHAPES: print(f"Shape of final output of TFDLeiTMainLayer (not dict): {(sequence_output.shape, pooled_shape)}")

		#	return final_output

		#return TFBaseModelOutputWithPooling(
		#	last_hidden_state=sequence_output,
		#	pooler_output=pooled_output,
		#	hidden_states=encoder_outputs.hidden_states,
		#	attentions=encoder_outputs.attentions,
		#)

		if DEBUG_SHAPES: print(f"Shape of final output of TFDLeiTMainLayer: {sequence_output.shape}")

		return sequence_output

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "embeddings", None) is not None:
			with tf.name_scope(self.embeddings.name):
				self.embeddings.build(None)
		if getattr(self, "encoder", None) is not None:
			with tf.name_scope(self.encoder.name):
				self.encoder.build(None)
		if getattr(self, "layernorm", None) is not None:
			with tf.name_scope(self.layernorm.name):
				self.layernorm.build([self.batch_size, self.config.num_patches+2, self.config.hidden_size])
		if getattr(self, "pooler", None) is not None:
			with tf.name_scope(self.pooler.name):
				self.pooler.build(None)
		if getattr(self, "classifier", None) is not None:
			with tf.name_scope(self.classifier.name):
				self.classifier.build((self.batch_size, self.config.num_patches+2, self.config.hidden_size))

def iterative_dense(hidden_states, layer):
	batch_size, dim_0, dim_1 = shape_list(hidden_states)
	output_states = []

	for i in range(dim_0):
		partial_states = tf.reshape(hidden_states[:,i,:], (batch_size, dim_1))
		output_states.append(layer(partial_states))

	return tf.stack(output_states, axis=1)

# timm's implementation with slight adjustments
# their implementation of deit is based on vit:
# copied from transformers.models.vit.modeling_tf_vit.TFViTSelfAttention with ViT->DeiT
# (and then I renamed DeiT -> DLeiT)
class TFDLeiTSelfAttention(keras.layers.Layer):
	def __init__(self, config: DLeiTConfig, **kwargs):
		super().__init__(**kwargs)

		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError(
				f"The hidden size ({config.hidden_size}) is not a multiple of the number "
				f"of attention heads ({config.num_attention_heads})"
			)

		image_size, patch_size = config.image_size, config.patch_size
		num_channels, hidden_size = config.num_channels, config.hidden_size

		image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
		patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

		num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

		self.image_size = image_size
		self.patch_size = patch_size
		self.num_channels = num_channels
		self.num_patches = num_patches
		self.batch_size = config.batch_size

		self.hidden_size = hidden_size

		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

		self.iterative_qkv = config.iterative_qkv

		assert self.iterative_qkv == False, f"Do not enable iterative QKV - it is not correct. I will eventually remove this code."

		self.stacked_heads = config.stacked_heads

		if self.stacked_heads:
			self.query = keras.layers.Dense(
				units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
			)

			self.key = keras.layers.Dense(
				units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
			)

			self.value = keras.layers.Dense(
				units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
			)

		else:
			self.query = [keras.layers.Dense(
				units=self.attention_head_size, kernel_initializer=get_initializer(config.initializer_range), name=f"query_{i}"
			) for i in range(config.num_attention_heads)]

			self.key = [keras.layers.Dense(
				units=self.attention_head_size, kernel_initializer=get_initializer(config.initializer_range), name=f"key_{i}"
			) for i in range(config.num_attention_heads)]

			self.value = [keras.layers.Dense(
				units=self.attention_head_size, kernel_initializer=get_initializer(config.initializer_range), name=f"value_{i}"
			) for i in range(config.num_attention_heads)]

		# -----------------------------------------------------------

		
		self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob, name="attention_dropout")
		self.config = config

	def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
		# Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
		tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

		# Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
		return tf.transpose(tensor, perm=[0, 2, 1, 3])

	def call(
		self,
		hidden_states: tf.Tensor,
		output_attentions: bool,
		training: bool = False,
	) -> Tuple[tf.Tensor]:
		if DEBUG_SHAPES: print(f"SelfAttention received hidden states with shape {hidden_states.shape}")

		#batch_size = shape_list(hidden_states)[0]
		batch_size = self.config.batch_size

		stacked_heads = self.stacked_heads

		if self.iterative_qkv:
			mixed_query_layer = iterative_dense(hidden_states, self.query)
			#mixed_query_layer_split = [iterative_dense(hidden_states, q) for q in self.query]
			#mixed_query_layer = tf.concat(mixed_query_layer_split, axis=-1)

			mixed_key_layer = iterative_dense(hidden_states, self.key)
			mixed_value_layer = iterative_dense(hidden_states, self.value)
		else:
			if stacked_heads:
				mixed_query_layer = self.query(inputs=hidden_states)
				mixed_key_layer = self.key(inputs=hidden_states)
				mixed_value_layer = self.value(inputs=hidden_states)
			else:
				query_layer = [q(inputs=hidden_states) for q in self.query]
				key_layer = [k(inputs=hidden_states) for k in self.key]
				value_layer = [v(inputs=hidden_states) for v in self.value]

		if stacked_heads:
			query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
			key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
			value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

			# Take the dot product between "query" and "key" to get the raw attention scores.
			# (batch size, num_heads, seq_len_q, seq_len_k)
			attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

		else:
			#query_layer = [self.transpose_for_scores(q, batch_size) for q in mixed_query_layer]
			#key_layer = [self.transpose_for_scores(k, batch_size) for k in mixed_key_layer]
			#value_layer = [self.transpose_for_scores(v, batch_size) for v in mixed_value_layer]

			# Take the dot product between "query" and "key" to get the raw attention scores.
			# (batch size, num_heads, seq_len_q, seq_len_k)
			attention_scores = [tf.matmul(query_layer[i], key_layer[i], transpose_b=True) for i in range(self.config.num_attention_heads)]
			

		# REMOVE THIS
		#frankenstein = tf.reshape(attention_scores, (batch_size, self.num_patches+2, (self.num_patches+2)*3))
		#all_ones = tf.ones((batch_size, self.num_patches+2, self.all_head_size))
		#return tf.math.subtract(all_ones, frankenstein[:,:,:self.all_head_size])

		
		if stacked_heads:
			dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
			attention_scores = tf.divide(attention_scores, dk)
		else:
			dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores[0].dtype)
			attention_scores = [tf.divide(attention_scores[i], dk) for i in range(self.config.num_attention_heads)]

		if DEBUG_SHAPES: print(f"SelfAttention scores are shape {attention_scores.shape}")

		# Normalize the attention scores to probabilities.
		if stacked_heads:
			attention_probs = stable_softmax(logits=attention_scores, axis=-1)
		else:
			attention_probs = [stable_softmax(logits=attention_scores[i], axis=-1) for i in range(self.config.num_attention_heads)]

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		if stacked_heads:
			attention_probs = self.dropout(inputs=attention_probs, training=training)
		else:
			attention_probs = [self.dropout(inputs=attention_probs[i], training=training) for i in range(self.config.num_attention_heads)]

		# Mask heads if we want to
		#if head_mask is not None:
		#	attention_probs = tf.multiply(attention_probs, head_mask)

		if stacked_heads:
			attention_output = tf.matmul(attention_probs, value_layer)
			attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
		else:
			attention_output = [tf.matmul(attention_probs[i], value_layer[i]) for i in range(self.config.num_attention_heads)]
			#attention_output = [tf.transpose(attention_output[i], perm=[0, 2, 1, 3]) for i in range(self.config.num_attention_heads)]

		# (batch_size, seq_len_q, all_head_size)
		if stacked_heads:
			attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
		else:
			#print(f'before reshape {attention_output[0].shape}')
			attention_output = [tf.reshape(tensor=attention_output[i], shape=(batch_size, -1, self.attention_head_size)) for i in range(self.config.num_attention_heads)]
			attention_output = tf.concat(attention_output, axis=-1)

		if DEBUG_SHAPES: print(f"SelfAttention output shape is {attention_output.shape}")
		#if DEBUG_SHAPES and not stacked_heads: print(f"SelfAttention output shape is {len(attention_output)}x{attention_output[0].shape}")
		#outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

		return attention_output
		#return outputs

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if self.stacked_heads:
			if getattr(self, "query", None) is not None:
				#for q in self.query:
				with tf.name_scope(self.query.name):
				#with tf.name_scope(q.name):
					if self.iterative_qkv:
						self.query.build([self.batch_size, self.config.hidden_size])
						#q.build([self.batch_size, self.config.hidden_size])
					else:
						self.query.build([self.batch_size, self.num_patches+2, self.config.hidden_size])
			if getattr(self, "key", None) is not None:
				with tf.name_scope(self.key.name):
					if self.iterative_qkv:
						self.key.build([self.batch_size, self.config.hidden_size])
					else:
						self.key.build([self.batch_size, self.num_patches+2, self.config.hidden_size])
			if getattr(self, "value", None) is not None:
				with tf.name_scope(self.value.name):
					if self.iterative_qkv:
						self.value.build([self.batch_size, self.config.hidden_size])
					else:
						self.value.build([self.batch_size, self.num_patches+2, self.config.hidden_size])
		else:
			if getattr(self, "query", None) is not None:
				for q in self.query:
					with tf.name_scope(q.name):
						q.build([self.batch_size, self.num_patches+2, self.config.hidden_size])
			if getattr(self, "key", None) is not None:
				for k in self.key:
					with tf.name_scope(k.name):
						k.build([self.batch_size, self.num_patches+2, self.config.hidden_size])
			if getattr(self, "value", None) is not None:
				for v in self.value:
					with tf.name_scope(v.name):
						v.build([self.batch_size, self.num_patches+2, self.config.hidden_size])

		# build dropout?


# timm's implementation with slight adjustments
# their implementation of deit is based on vit:
# copied from transformers.models.vit.modeling_tf_vit.TFViTSelfOutput with ViT->DeiT
# (and then I renamed DeiT -> DLeiT)
class TFDLeiTSelfOutput(keras.layers.Layer):
	"""
	The residual connection is defined in TFDLeiTLayer instead of here (as is the case with other models), due to the
	layernorm applied before each block.
	"""

	def __init__(self, config: DLeiTConfig, **kwargs):
		super().__init__(**kwargs)

		self.dense = keras.layers.Dense(
			units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
		)
		self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
		self.config = config
		self.batch_size = config.batch_size
		self.num_patches = config.num_patches

	def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
		hidden_states = self.dense(inputs=hidden_states)
		hidden_states = self.dropout(inputs=hidden_states, training=training)

		return hidden_states

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "dense", None) is not None:
			with tf.name_scope(self.dense.name):
				self.dense.build([self.batch_size, self.num_patches+2, self.config.hidden_size])


# timm's implementation with slight adjustments
# their implementation of deit is based on vit:
# copied from transformers.models.vit.modeling_tf_vit.TFViTAttention with ViT->DeiT
# (and then I renamed DeiT -> DLeiT)
class TFDLeiTAttention(keras.layers.Layer):
	def __init__(self, config: DLeiTConfig, **kwargs):
		super().__init__(**kwargs)

		self.self_attention = TFDLeiTSelfAttention(config, name="attention")
		self.dense_output = TFDLeiTSelfOutput(config, name="output")

	def prune_heads(self, heads):
		raise NotImplementedError

	def call(
		self,
		input_tensor: tf.Tensor,
		output_attentions: bool,
		training: bool = False,
	) -> Tuple[tf.Tensor]:
		self_outputs = self.self_attention(
			hidden_states=input_tensor, output_attentions=output_attentions, training=training
		)

		attention_output = self.dense_output(
			#hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
			hidden_states=self_outputs, input_tensor=input_tensor, training=training
		)

		#outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

		#return outputs
		return attention_output

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "self_attention", None) is not None:
			with tf.name_scope(self.self_attention.name):
				self.self_attention.build(None)
		if getattr(self, "dense_output", None) is not None:
			with tf.name_scope(self.dense_output.name):
				self.dense_output.build(None)

# timm's implementation with slight adjustments
# their implementation of deit is based on vit:
# copied from transformers.models.vit.modeling_tf_vit.TFViTIntermediate with ViT->DeiT
# (and then I renamed DeiT -> DLeiT)
class TFDLeiTIntermediate(keras.layers.Layer):
	def __init__(self, config: DLeiTConfig, **kwargs):
		super().__init__(**kwargs)

		self.dense = keras.layers.Dense(
			units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
		)

		if isinstance(config.hidden_act, str):
			#self.intermediate_act_fn = get_tf_activation(config.hidden_act)
			if config.hidden_act.lower() == 'gelu':
				self.intermediate_act_fn = other_gelu
			else:
				raise NotImplementedError("Please use GeLU activation")
		else:
			#self.intermediate_act_fn = config.hidden_act
			raise NotImplementedError(("Please provide the custom GeLU activation function to the DLeiT config object"))
		self.config = config

	def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
		hidden_states = self.dense(inputs=hidden_states)
		'''
		output_states = []

		for i in range(self.config.num_patches+2):
			partial_states = tf.reshape(hidden_states[:,i,:], (self.config.batch_size, self.config.hidden_size))
			output_states.append(self.dense(partial_states))

		hidden_states = tf.stack(output_states, axis=1)
		'''

		hidden_states = self.intermediate_act_fn(hidden_states)

		return hidden_states

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "dense", None) is not None:
			with tf.name_scope(self.dense.name):
				self.dense.build([self.config.batch_size, self.config.num_patches+2, self.config.hidden_size])
				#self.dense.build([self.config.batch_size, self.config.hidden_size])


# timm's implementation with slight adjustments
# their implementation of deit is based on vit:
# copied from transformers.models.vit.modeling_tf_vit.TFViTOutput with ViT->DeiT
# (and then I renamed DeiT -> DLeiT)
class TFDLeiTOutput(keras.layers.Layer):
	def __init__(self, config: DLeiTConfig, **kwargs):
		super().__init__(**kwargs)

		self.dense = keras.layers.Dense(
			units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
		)
		self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
		self.config = config

	def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
		hidden_states = self.dense(inputs=hidden_states)
		hidden_states = self.dropout(inputs=hidden_states, training=training)
		hidden_states = hidden_states + input_tensor

		return hidden_states

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "dense", None) is not None:
			with tf.name_scope(self.dense.name):
				self.dense.build([self.config.batch_size, self.config.num_patches+2, self.config.intermediate_size])

# timm's implementation with slight adjustments
class TFDLeiTLayer(keras.layers.Layer):
	"""This corresponds to the Block class in the timm implementation."""

	def __init__(self, config: DLeiTConfig, **kwargs):
		super().__init__(**kwargs)

		self.attention = TFDLeiTAttention(config, name="attention")
		self.intermediate = TFDLeiTIntermediate(config, name="intermediate")
		self.dleit_output = TFDLeiTOutput(config, name="output")

		self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
		self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")
		self.config = config

	def call(
		self,
		hidden_states: tf.Tensor,
		output_attentions: bool,
		training: bool = False,
	) -> Tuple[tf.Tensor]:
		if DEBUG_SHAPES: print(f"Input of DLeitLayer {hidden_states.shape}")

		# REMOVE THIS
		#return hidden_states

		#attention_outputs = self.attention(
		attention_output = self.attention(
			# in DLeiT, layernorm is applied before self-attention
			input_tensor=self.layernorm_before(inputs=hidden_states, training=training),
			output_attentions=output_attentions,
			training=training,
		)

		#attention_output = attention_outputs[0]
		if DEBUG_SHAPES: print(f"Output of attention of DLeitLayer {attention_output.shape}")

		# REMOVE THIS
		#return attention_output

		# first residual connection
		hidden_states = attention_output + hidden_states

		# in DLeiT, layernorm is also applied after self-attention
		layer_output = self.layernorm_after(inputs=hidden_states, training=training)

		# REMOVE THIS
		#return layer_output

		if DEBUG_SHAPES: print(f"After LayerNormalization {layer_output.shape}")

		# something in intermediate is not supported by the TPU
		# time for the fun part

		# looks like it was the GeLU after all

		intermediate_output = self.intermediate(hidden_states=layer_output, training=training)

		if DEBUG_SHAPES: print(f"Intermediate output {intermediate_output.shape}")

		# REMOVE THIS
		#return intermediate_output

		# second residual connection is done here
		layer_output = self.dleit_output(
			hidden_states=intermediate_output, input_tensor=hidden_states, training=training
		)

		if DEBUG_SHAPES: print(f"DLeiT output {layer_output.shape}")

		#outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

		if DEBUG_SHAPES: print(f"Final output of DLeiTLayer {layer_output.shape}")

		return layer_output
		#return outputs

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "attention", None) is not None:
			with tf.name_scope(self.attention.name):
				self.attention.build(None)
		if getattr(self, "intermediate", None) is not None:
			with tf.name_scope(self.intermediate.name):
				self.intermediate.build(None)
		if getattr(self, "dleit_output", None) is not None:
			with tf.name_scope(self.dleit_output.name):
				self.dleit_output.build(None)
		if getattr(self, "layernorm_before", None) is not None:
			with tf.name_scope(self.layernorm_before.name):
				self.layernorm_before.build([self.config.batch_size, self.config.num_patches+2, self.config.hidden_size])
		if getattr(self, "layernorm_after", None) is not None:
			with tf.name_scope(self.layernorm_after.name):
				self.layernorm_after.build([self.config.batch_size, self.config.num_patches+2, self.config.hidden_size])

# timm's implementation with slight adjustments
# their implementation of deit is based on vit:
# copied from transformers.models.vit.modeling_tf_vit.TFViTEncoder with ViT->DeiT
# (and then I renamed DeiT -> DLeiT)
class TFDLeiTEncoder(keras.layers.Layer):
	def __init__(self, config: DLeiTConfig, **kwargs):
		super().__init__(**kwargs)

		self.layer = [TFDLeiTLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

	def call(
		self,
		hidden_states: tf.Tensor,
		output_attentions: bool,
		output_hidden_states: bool,
		return_dict: bool,
		training: bool = False,
	) -> Tuple[tf.Tensor]:
		#all_hidden_states = () if output_hidden_states else None
		#all_attentions = () if output_attentions else None

		if DEBUG_SHAPES: print(f"TFDLeiTEncoder received input of shape {hidden_states.shape}")

		for i, layer_module in enumerate(self.layer):
			#if output_hidden_states:
			#	all_hidden_states = all_hidden_states + (hidden_states,)

			#layer_outputs = layer_module(
			hidden_states = layer_module(
				hidden_states=hidden_states,
				output_attentions=output_attentions,
				training=training,
			)
			#hidden_states = layer_outputs[0]

			#if output_attentions:
				#all_attentions = all_attentions + (layer_outputs[1],)

		# Add last layer
		#if output_hidden_states:
		#	all_hidden_states = all_hidden_states + (hidden_states,)

		if DEBUG_SHAPES: print(f"Final output of TFDLeiTEncoder has shape {hidden_states.shape}")

		#if not return_dict:
		#	return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

		return hidden_states

		#return TFBaseModelOutput(
		#	last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
		#)

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "layer", None) is not None:
			for layer in self.layer:
				with tf.name_scope(layer.name):
					layer.build(None)

def get_cls_tokens_ones(shape):
	return tf.ones(shape)

# timm's implementation with slight adjustments
class TFDLeiTEmbeddings(keras.layers.Layer):
	"""
	Construct the CLS token, distillation token, position and patch embeddings. Optionally, also the mask token.
	"""

	def __init__(self, config: DLeiTConfig, use_mask_token: bool = False, **kwargs) -> None:
		super().__init__(**kwargs)
		self.config = config
		self.use_mask_token = use_mask_token

		image_size, patch_size = config.image_size, config.patch_size
		num_channels, hidden_size = config.num_channels, config.hidden_size

		image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
		patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

		num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

		self.image_size = image_size
		self.patch_size = patch_size
		self.num_channels = num_channels
		self.num_patches = num_patches
		self.batch_size = config.batch_size

		self.hidden_size = hidden_size

		self.patch_embeddings = TFDLeiTPatchEmbeddings(config=config, name="patch_embeddings")
		self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")

	def build(self, input_shape=None):
		'''
		self.cls_token = self.add_weight(
			shape=(1, 1, self.config.hidden_size),
			initializer=keras.initializers.zeros(),
			trainable=True,
			name="cls_token",
		)
		'''

		#self.cls_token = tf.constant(get_cls_tokens_ones((1, 1, self.config.hidden_size)))
		#self.cls_token = tf.convert_to_tensor(np.load('/home/loureiro/repos/tpu-rad/coral/models/int1_96.nparray.npy'))

		'''
		self.distillation_token = self.add_weight(
			shape=(1, 1, self.config.hidden_size),
			initializer=keras.initializers.zeros(),
			trainable=True,
			name="distillation_token",
		)
		'''

		#self.distillation_token = tf.constant(get_cls_tokens_ones((1, 1, self.config.hidden_size)))
		#self.distillation_token = tf.convert_to_tensor(np.load('/home/loureiro/repos/tpu-rad/coral/models/int1_96.nparray.npy'))

		self.mask_token = None
		if self.use_mask_token:
			self.mask_token = self.add_weight(
				shape=(1, 1, self.config.hidden_size),
				initializer=keras.initializers.zeros(),
				trainable=True,
				name="mask_token",
			)
		num_patches = self.patch_embeddings.num_patches
		
		self.position_embeddings = self.add_weight(
			shape=(1, num_patches + 2, self.config.hidden_size),
			initializer=keras.initializers.zeros(),
			trainable=True,
			name="position_embeddings",
		)
		
		#self.position_embeddings = tf.constant(get_cls_tokens_ones((1, num_patches+2, self.config.hidden_size)))
		#self.position_embeddings = tf.convert_to_tensor(np.load('/home/loureiro/repos/tpu-rad/coral/models/int66_96.nparray.npy'))

		if self.built:
			return
		self.built = True
		if getattr(self, "patch_embeddings", None) is not None:
			with tf.name_scope(self.patch_embeddings.name):
				self.patch_embeddings.build((self.batch_size, *self.image_size, self.num_channels))
		if getattr(self, "dropout", None) is not None:
			with tf.name_scope(self.dropout.name):
				self.dropout.build((self.batch_size, self.num_patches+2, self.hidden_size))

	def call(
		self, pixel_values: tf.Tensor, const_tokens: tf.Tensor, bool_masked_pos: tf.Tensor | None = None, training: bool = False
	) -> tf.Tensor:
		if DEBUG_SHAPES: print(f"Embeddings received pixels of shape {pixel_values.shape}")
		embeddings = self.patch_embeddings(pixel_values)
		if DEBUG_SHAPES: print(f"patch embeddings returned shape {embeddings.shape}")

		#REMOVE THIS
		#return embeddings
		batch_size, seq_length, _ = shape_list(embeddings)

		if bool_masked_pos is not None:
			raise NotImplementedError("Embedding with masked position is not supported in this version.")
			print(f"Using bool_masked_pos")
			'''
			mask_tokens = tf.tile(self.mask_token, [batch_size, seq_length, 1])
			mask_tokens = None
			# replace the masked visual tokens by mask_tokens
			mask = tf.expand_dims(bool_masked_pos, axis=-1)
			mask = tf.cast(mask, dtype=mask_tokens.dtype)
			embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
			'''

		#cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
		#cls_tokens = self.cls_token
		cls_tokens = const_tokens[:, 0:1, :]
		#distillation_tokens = tf.repeat(self.distillation_token, repeats=batch_size, axis=0)
		#distillation_tokens = self.distillation_token
		distillation_tokens = const_tokens[:, 1:2, :]

		if DEBUG_SHAPES: print(f"Shapes of emb, cls, dist: {embeddings.shape}, {cls_tokens.shape}, {distillation_tokens.shape}")

		#REMOVE THIS
		#return embeddings

		# this is not the concat that is failing (yet)

		embeddings = tf.concat((cls_tokens, distillation_tokens, embeddings), axis=1)
		#embeddings = tf.concat((tf.zeros((1,2,self.config.hidden_size)), embeddings), axis=1)

		if DEBUG_SHAPES: print(f"Shape of embeddings: {embeddings.shape}")
		#embeddings = embeddings + self.position_embeddings
		embeddings = tf.math.add(embeddings, const_tokens[:, 2:, :])
		#embeddings = tf.math.add(embeddings, self.position_embeddings)
		#print(f"Shape of embeddings after adding positional: {embeddings.shape}")
		embeddings = self.dropout(embeddings, training=training)
		return embeddings

# This is timm's implementation with slight adjustments
class TFDLeiTPatchEmbeddingsWithConv(keras.layers.Layer):
	"""
	This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
	`hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
	Transformer.
	"""

	def __init__(self, config: DLeiTConfig, **kwargs) -> None:
		super().__init__(**kwargs)

		image_size, patch_size = config.image_size, config.patch_size
		num_channels, hidden_size = config.num_channels, config.hidden_size

		image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
		patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

		num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

		self.image_size = image_size
		self.patch_size = patch_size
		self.num_channels = num_channels
		self.num_patches = num_patches
		self.batch_size = config.batch_size

		self.hidden_size = hidden_size

		self.projection = keras.layers.Conv2D(
			hidden_size,
			kernel_size=patch_size,
			strides=patch_size,
			#padding="same",
			name="projection",
		)

	def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
		batch_size, height, width, num_channels = shape_list(pixel_values)
		if tf.executing_eagerly() and num_channels != self.num_channels:
			raise ValueError(
				"Make sure that the channel dimension of the pixel values match with the one set in the configuration."
			)
		if tf.executing_eagerly() and (height != self.image_size[0] or width != self.image_size[1]):
			raise ValueError(
				f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
			)

		if DEBUG_SHAPES: print(f"TFDLeiTPatchEmbeddingsWithConv received input with shape: {pixel_values.shape}")

		#pixel_values = tf.reshape(pixel_values, (batch_size, num_channels, height, width))
		x = self.projection(pixel_values)
		if DEBUG_SHAPES: print(f"Shape after projection: {x.shape}")
		batch_size, height, width, num_channels = shape_list(x)
		x = tf.reshape(x, (batch_size, height * width, num_channels))
		#x = tf.reshape(x, (1, self.patch_size[0]*self.patch_size[0], self.num_patches*3))
		if DEBUG_SHAPES: print(f"Shape after reshaping: {x.shape}")
		return x

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "projection", None) is not None:
			with tf.name_scope(self.projection.name):
				self.projection.build([self.batch_size, *self.image_size, self.num_channels])

class PatchesLayerIter(keras.layers.Layer):
	def __init__(self, config: DLeiTConfig, **kwargs) -> None:
		super().__init__(**kwargs)

		image_size, patch_size = config.image_size, config.patch_size
		num_channels, hidden_size = config.num_channels, config.hidden_size

		image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
		patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

		num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

		self.image_size = image_size
		self.patch_size = patch_size
		self.num_channels = num_channels
		self.num_patches = num_patches

		self.hidden_size = hidden_size

	def call(self, pixels):
		patches = []
		# For square images only ( as inputs.shape[ 1 ] = inputs.shape[ 2 ] )
		  
		for i in range(0, self.image_size[0], self.patch_size[0]):
			for j in range(0, self.image_size[1], self.patch_size[1]):
				patches.append(
					pixels[
						:,
						i:i + self.patch_size[0],
						j:j + self.patch_size[1],
						:
					]
				)

		if DEBUG_SHAPES: print(f"{len(patches)} entries of shape {patches[0].shape}")
		  
		#patches = tf.concat(patches,axis=-2)

		patches = tf.stack(patches, axis=1)

		#print(f"shape of patches before is {patches.shape}")
		
		#patches_tf = tf.reshape(pixels, (1, self.patch_size[0], int(self.image_size[0] * self.image_size[1] / self.patch_size[1]), 3))

		#print(f"Same result: {np.isclose(patches.eval(), patches_tf.eval())}")

		#patches = self.reshape_patch(patches)

		#print(f"shape of patches after is {patches.shape}")

		if DEBUG_SHAPES: print(f"Created patches with shape {patches.shape}")

		return patches

# This is an adaptation of Pablo's patch embedding
class TFDLeiTPatchEmbeddingsWithDense(keras.layers.Layer):
	def __init__(self, config: DLeiTConfig, **kwargs) -> None:
		super().__init__(**kwargs)

		image_size, patch_size = config.image_size, config.patch_size
		num_channels, hidden_size = config.num_channels, config.hidden_size

		image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
		patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

		num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

		self.image_size = image_size
		self.patch_size = patch_size
		self.num_channels = num_channels
		self.num_patches = num_patches

		self.hidden_size = hidden_size

		projection_dim = hidden_size

		
		self.patches_layer = PatchesLayerIter(config, name="create_patches")
		
		self.reshape_patch = tf.keras.layers.Reshape(
			[
				self.num_patches,
				self.patch_size[0] * self.patch_size[1] * 3,
			],
			name="reshape_patch"
		)

		self.projection = keras.layers.Dense(
			units=projection_dim,
			name="projection",
		)

	def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
		batch_size, height, width, num_channels = shape_list(pixel_values)
		if tf.executing_eagerly() and num_channels != self.num_channels:
			raise ValueError(
				"Make sure that the channel dimension of the pixel values match with the one set in the configuration."
			)
		if tf.executing_eagerly() and (height != self.image_size[0] or width != self.image_size[1]):
			raise ValueError(
				f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
			)

		#print(f"pixel shape: {pixel_values.shape}, return of shape_list: {batch_size, height, width, num_channels}")

		if DEBUG_SHAPES: print(f"TFDLeiTPatchEmbeddingsWithDense received pixels with shape {pixel_values.shape}")

		#pixel_values = tf.reshape(pixel_values, (batch_size, num_channels, height, width))

		x = self.patches_layer(pixel_values)

		#REMOVE THIS

		#return x

		#print(f"After making patches is {x.shape}")

		x = self.reshape_patch(x)

		if DEBUG_SHAPES: print(f"Reshaped patches to {x.shape}")

		#print(f"After reshaping patches is {x.shape}")

		x = self.projection(x)
		#batch_size, height, width, num_channels = shape_list(x)

		if DEBUG_SHAPES: print(f"Projected patches to {x.shape}. Returning this.")

		#x = tf.reshape(x, (batch_size, height * width, num_channels))

		#x = tf.reshape(x, (1, self.patch_size[0]*self.patch_size[0], self.num_patches*3))
		return x

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True

		if getattr(self, "patches_layer", None) is not None:
			with tf.name_scope(self.patches_layer.name):
				image_size = (1, *self.image_size, self.num_channels)
				self.patches_layer.build(image_size)

		if getattr(self, "reshape_patch", None) is not None:
			with tf.name_scope(self.reshape_patch.name):
				#reshape_size = (1, self.patch_size[0], int(self.image_size[0] * self.image_size[1] / self.patch_size[1]), self.num_channels)
				reshape_size = (1, self.num_patches, self.patch_size[0], self.patch_size[1], 3)
				self.reshape_patch.build(reshape_size)

		if getattr(self, "projection", None) is not None:
			with tf.name_scope(self.projection.name):
				projection_size = [
					1,
					self.num_patches,
					self.patch_size[0] * self.patch_size[1] * 3,
					#self.hidden_size,
				]
				self.projection.build(input_shape=projection_size)


use_dense_embeddings = False

if use_dense_embeddings:
	TFDLeiTPatchEmbeddings = TFDLeiTPatchEmbeddingsWithDense
else:
	TFDLeiTPatchEmbeddings = TFDLeiTPatchEmbeddingsWithConv

# Pablo's implementation of GeLU
# The TPU does not support the base tensorflow implementation

#0.5 * x * (1 + tf.tanh(tf.sqrt(2 / pi) * (x + 0.044715 * tf.pow(x,3))))
def other_gelu(x):
	temp = x * x * x
	#return 0.5 * x * (1 + tf.math.erf(x / tf.sqrt(2.0)))
	return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / math.pi) * (x + 0.044715 * temp)))
	#return 0.5 * x * (1.0 + tf.tanh(0.7978845608028653 * (x + 0.04553992412 * tf.pow(x, 3))))

#get_custom_objects().update({'other_gelu': Activation(other_gelu)})