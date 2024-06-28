from __future__ import annotations

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

#from ...activations_tf import get_tf_activation
from transformers.activations_tf import get_tf_activation

#from ...modeling_tf_outputs import (
from transformers.modeling_tf_outputs import (
	TFBaseModelOutput,
	TFBaseModelOutputWithPooling,
	TFImageClassifierOutput,
	TFMaskedImageModelingOutput,
)


#from ...modeling_tf_utils import (
from transformers.modeling_tf_utils import (
	TFPreTrainedModel,
	TFSequenceClassificationLoss,
	get_initializer,
	#keras,
	keras_serializable,
	unpack_inputs,
)

from tensorflow import (
	keras,
	#keras_serializable,
)

#from ...tf_utils import shape_list, stable_softmax
from transformers.tf_utils import shape_list, stable_softmax

#from ...utils import (
from transformers.utils import (
	ModelOutput,
	add_code_sample_docstrings,
	add_start_docstrings,
	add_start_docstrings_to_model_forward,
	logging,
	replace_return_docstrings,
)


#from .configuration_deit import DeiTConfig
from transformers.models.deit.configuration_deit import DeiTConfig

from dleit import *

# General docstring
_CONFIG_FOR_DOC = "DeiTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/deit-base-distilled-patch16-224"
_EXPECTED_OUTPUT_SHAPE = [1, 198, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/deit-base-distilled-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


# Adding one layer at a time to see what works with quantization and what does not

class DLeiTIncremental(TFDLeiTPreTrainedModel, TFSequenceClassificationLoss):
	def __init__(self, config: DeiTConfig):
		super().__init__(config)

		self.num_labels = config.num_labels
		self.dleit = TFDLeiTLayersIncremental(config, add_pooling_layer=False, name="dleit")

		# Classifier head
		self.classifier = (
			keras.layers.Dense(config.num_labels, name="classifier")
			if config.num_labels > 0
			else keras.layers.Activation("linear", name="classifier")
		)
		self.config = config

	@unpack_inputs
	@add_start_docstrings_to_model_forward(DLEIT_INPUTS_DOCSTRING)
	@replace_return_docstrings(output_type=TFImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
	def call(
		self,
		pixel_values: tf.Tensor | None = None,
		head_mask: tf.Tensor | None = None,
		labels: tf.Tensor | None = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		training: bool = False,
	) -> Union[tf.Tensor, TFImageClassifierOutput]:
		r"""
		labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
			config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
			`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

		Returns:

		Examples:

		```python
		>>> from transformers import AutoImageProcessor, TFDLeiTForImageClassification
		>>> import tensorflow as tf
		>>> from PIL import Image
		>>> import requests

		>>> keras.utils.set_random_seed(3)  # doctest: +IGNORE_RESULT
		>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		>>> image = Image.open(requests.get(url, stream=True).raw)

		>>> # note: we are loading a TFDLeiTForImageClassificationWithTeacher from the hub here,
		>>> # so the head will be randomly initialized, hence the predictions will be random
		>>> image_processor = AutoImageProcessor.from_pretrained("facebook/dleit-base-distilled-patch16-224")
		>>> model = TFDLeiTForImageClassification.from_pretrained("facebook/dleit-base-distilled-patch16-224")

		>>> inputs = image_processor(images=image, return_tensors="tf")
		>>> outputs = model(**inputs)
		>>> logits = outputs.logits
		>>> # model predicts one of the 1000 ImageNet classes
		>>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
		>>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
		Predicted class: little blue heron, Egretta caerulea
		```"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.dleit(
			pixel_values,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			training=training,
		)

		return outputs

		'''

		sequence_output = outputs[0]

		logits = self.classifier(sequence_output[:, 0, :])
		# we don't use the distillation token

		loss = None if labels is None else self.hf_compute_loss(labels, logits)

		if not return_dict:
			output = (logits,) + outputs[1:]
			return ((loss,) + output) if loss is not None else output

		return TFImageClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

		'''

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "dleit", None) is not None:
			with tf.name_scope(self.dleit.name):
				self.dleit.build(None)
		if getattr(self, "classifier", None) is not None:
			with tf.name_scope(self.classifier.name):
				self.classifier.build([None, None, self.config.hidden_size])


@keras_serializable
class TFDLeiTLayersIncremental(keras.layers.Layer):
	config_class = DeiTConfig

	def __init__(
		self, config: DeiTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, **kwargs
	) -> None:
		super().__init__(**kwargs)
		self.config = config

		self.embeddings = TFDLeiTEmbeddings(config, use_mask_token=use_mask_token, name="embeddings")
		#self.encoder = TFDLeiTEncoder(config, name="encoder")

		#self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
		#self.pooler = TFDLeiTPooler(config, name="pooler") if add_pooling_layer else None

	def get_input_embeddings(self) -> TFDLeiTPatchEmbeddings:
		return self.embeddings.patch_embeddings

	def _prune_heads(self, heads_to_prune):
		"""
		Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
		class PreTrainedModel
		"""
		raise NotImplementedError

	def get_head_mask(self, head_mask):
		if head_mask is not None:
			raise NotImplementedError
		else:
			head_mask = [None] * self.config.num_hidden_layers

		return head_mask

	@unpack_inputs
	def call(
		self,
		pixel_values: tf.Tensor | None = None,
		bool_masked_pos: tf.Tensor | None = None,
		head_mask: tf.Tensor | None = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		training: bool = False,
	) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor, ...]]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if pixel_values is None:
			raise ValueError("You have to specify pixel_values")

		# TF 2.0 image layers can't use NCHW format when running on CPU.
		# (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
		pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))

		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		head_mask = self.get_head_mask(head_mask)

		embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, training=training)

		return embedding_output

		'''

		encoder_outputs = self.encoder(
			embedding_output,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			training=training,
		)
		sequence_output = encoder_outputs[0]
		sequence_output = self.layernorm(sequence_output, training=training)
		pooled_output = self.pooler(sequence_output, training=training) if self.pooler is not None else None

		if not return_dict:
			head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
			return head_outputs + encoder_outputs[1:]

		return TFBaseModelOutputWithPooling(
			last_hidden_state=sequence_output,
			pooler_output=pooled_output,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
		)

		'''

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
				self.layernorm.build([None, None, self.config.hidden_size])
		if getattr(self, "pooler", None) is not None:
			with tf.name_scope(self.pooler.name):
				self.pooler.build(None)