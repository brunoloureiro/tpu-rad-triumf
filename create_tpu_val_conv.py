#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np

from transformers import (
	TFDeiTModel,
	TFDeiTForImageClassification,
	TFDeiTForImageClassificationWithTeacher,
	AutoImageProcessor,
	#DeiTImageProcessor,
)

import argparse

# TO-DO
# proper import
from common_tpu import *

class ConvOnly(keras.Model):
	def __init__(self,
		image_size,
		batch_size=1,
		channels=3,
		kernel_size=3,
		kernel_initializer="glorot_uniform",
		filters=8,
		data_type=tf.dtypes.float32,
		**kwargs,
	):
		super().__init__(self, **kwargs)

		self.image_size = image_size
		self.channels = channels
		self.batch_size = batch_size
		self.filters = filters
		self.kernel_size = kernel_size

		self.data_type = data_type

		self.conv_layer = keras.layers.Conv2D(
			filters = filters,
			kernel_size = kernel_size,
			kernel_initializer = kernel_initializer,
			dtype = data_type,
		)

	def call(self, x):
		#dot_axes = 1 if transpose_b else (2, 1)
		return self.conv_layer(x)

	def model(self):
		input_shape = (*self.image_size, self.channels)
		x = keras.layers.Input(shape=input_shape, batch_size=self.batch_size)
		return keras.Model(
			inputs = x,
			outputs = self.call(x),
		)

class DatasetGenerator():
	def __init__(
		self,
		image_size,
		train_start=0,
		train_end=10,
		eval_start=10,
		eval_end=13,
		data_type=tf.dtypes.float32,
		image_processor="facebook/deit-base-distilled-patch16-224",
		to_converter=True,
		max_samples=100,
	):

		self.train_dataset, self.val_dataset = tfds.load(
			"cifar100",
			split = [
				f"train[{train_start}%:{train_end}%]",
				f"train[{eval_start}%:{eval_end}%]"
			],
			as_supervised = True,
		)

		self.image_size = image_size

		self.train_start = train_start
		self.train_end = train_end
		self.eval_start = eval_start
		self.eval_end = eval_end

		self.data_type = data_type

		if image_processor is not None:
			self.image_processor = AutoImageProcessor.from_pretrained(image_processor)
		else:
			self.image_processor = None

		self.to_converter = to_converter

		self.max_samples = max_samples
		self._iter = 0


	def gen(self):
		for x in self.train_dataset:
			if self.max_samples is not None:
				if self._iter >= self.max_samples:
					break

			image = tf.image.resize(x[0], self.image_size)
			if self.image_processor is not None:
				image = self.image_processor(
					#[tf.expand_dims(image,axis=0)],
					tf.image.convert_image_dtype(image, dtype=self.data_type, saturate=False),
					return_tensors="tf",
				)
			else:
				image = tf.image.convert_image_dtype(image, dtype=self.data_type, saturate=False),

			if self.to_converter:
				image = [image['pixel_values']] # only what matters to converter

			#yield image
			if self.max_samples is not None:
				self._iter += 1
			yield [tf.convert_to_tensor(image)]
			
def main():
	#model_file = '/home/carol/validate_tpu/conv_edgeai_resnet.tflite'
	model_file = '/home/loureiro/repos/tpu-rad/coral/conv_edgeai_conv2k.tflite'

	use_float_input = True

	if use_float_input:
		input_type = tf.dtypes.float32 # changed from tf.uint8
		output_type = tf.dtypes.float32 # changed from tf.uint8
	else:
		input_type = tf.dtypes.uint8
		output_type = tf.dtypes.uint8

	use_resnet = False

	if use_resnet:
		cnn = tf.keras.applications.ResNet50V2()
		input_shape = cnn.input_shape
		batch_size = 1
		image_size = input_shape[1:3] # (Batch, X, Y, D)
		input_shape = (batch_size, *image_size, 3)
	else:
		batch_size = 1
		image_size = (2048, 2048)
		channels = 3
		input_shape = (*image_size, channels)
		cnn = ConvOnly(
			image_size,
			batch_size = 1,
			kernel_size = 3,
			kernel_initializer = "glorot_uniform",
			filters = 8,
			data_type = input_type,
		)

		cnn = cnn.model()

	print(f"Loaded CNN")

	sample_generator = DatasetGenerator(
		image_size,
		image_processor = None,
		to_converter = False,
		data_type = input_type,
		train_start=0,
		train_end=3,
		eval_start=3,
		eval_end=4,
		max_samples=200,
	)

	print(f"Loaded dataset")

	print(f"Starting quantization...")

	converter_quant = tf.lite.TFLiteConverter.from_keras_model(cnn)
	converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
	converter_quant.representative_dataset = sample_generator.gen
	converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
	converter_quant.target_spec.supported_types = [tf.dtypes.int8]
	converter_quant.inference_input_type = input_type
	converter_quant.inference_output_type = output_type
	converter_quant.experimental_new_converter = True
	converter_quant.allow_custom_ops = True
	converter_quant.input_shape = input_shape

	converted_model = converter_quant.convert()

	print(f"Finished quantization. Saving model to {model_file}")

	with open(model_file, "wb") as f:
		f.write(converted_model)


if __name__ == '__main__':
	main()