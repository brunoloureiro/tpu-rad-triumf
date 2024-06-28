#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np


#from transformers.models.deit.modeling_tf_deit import (
from transformers import (
	TFDeiTModel,
	TFDeiTForImageClassification,
	TFDeiTForImageClassificationWithTeacher,
	AutoImageProcessor,
	#DeiTImageProcessor,
)
from transformers.models.deit.configuration_deit import DeiTConfig
import tensorflow_datasets as tfds

import os
import sys

import dleit
import incremental_dleit

with_teacher = False

if with_teacher:
	DLeiT = dleit.TFDLeiTForImageClassificationWithTeacher
	DeiT = TFDeiTForImageClassificationWithTeacher
else:
	#DLeiT = dleit.TFDLeiTForImageClassification
	DLeiT = incremental_dleit.DLeiTIncremental
	DeiT = TFDeiTForImageClassification
	

DLeiTConfig = DeiTConfig

class SampleGenerator():
	def __init__(
		self,
		shape_a,
		batch_size,
		max_samples=1000,
	):
		self.shape_a = shape_a
		self.batch_size = batch_size
		self.max_samples = max_samples

	def gen(self):
		for i in range(self.max_samples):
			a = np.random.rand(self.batch_size, *self.shape_a).astype(np.float32)
			yield a

class DatasetGenerator():
	def __init__(
		self,
		image_size,
		train_start=0,
		train_end=10,
		eval_start=10,
		eval_end=13,
		image_processor="facebook/deit-base-distilled-patch16-224",
		to_converter=True,
	):

		self.train_dataset, self.val_dataset = tfds.load(
			"tf_flowers",
			split = [
				f"train[{train_start}%:{train_end}%]",
				f"train[{eval_start}%:{eval_end}%]"
			],
			as_supervised = True,
		)

		self.sample_dataset, _ = tfds.load(
			"tf_flowers",
			split = [
				f"train[{train_start}%:{train_end//10}%]",
				f"train[{eval_start}%:{eval_end}%]"
			],
			as_supervised = True,
		)

		self.image_size = image_size

		self.train_start = train_start
		self.train_end = train_end
		self.eval_start = eval_start
		self.eval_end = eval_end

		self.image_processor = AutoImageProcessor.from_pretrained(image_processor)

		self.to_converter = to_converter

		self._sample = None


	def gen(self):
		for x in self.train_dataset:
			image = tf.image.resize(x[0], (self.image_size, self.image_size))
			image = self.image_processor(
				#[tf.expand_dims(image,axis=0)],
				tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=False),
				return_tensors="tf",
			)

			if self.to_converter:
				image = [image['pixel_values']] # only what matters to converter

			yield image

	@property
	def sample(self):
		if self._sample is None:
			for x in self.gen_sample():
				self._sample = x
				break

		return self._sample

	def gen_sample(self):
		for x in self.sample_dataset:
			image = tf.image.resize(x[0], (self.image_size, self.image_size))
			image = self.image_processor(
				#[tf.expand_dims(image,axis=0)],
				tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=False),
				return_tensors="tf",
			)

			if self.to_converter:
				image = [image['pixel_values']] # only what matters to converter

			yield image

def main():
	np.random.seed(1234)

	compare_to_deit = False

	image_size = 224
	shape_a = (image_size, image_size, 3)
	batch_size = 1

	if compare_to_deit:

		deit_cfg = DeiTConfig()
		deit = DeiT(deit_cfg)
		deit.build()

	dleit_cfg = DLeiTConfig()

	dleit = DLeiT(dleit_cfg)
	#dleit = DeiT(DeiTConfig())
	dleit.build(input_shape=(batch_size, *shape_a))
	dleit.compile()

	print(f"Built dleit")

	#sample_generator = SampleGenerator(shape_a, batch_size)
	sample_generator = DatasetGenerator(image_size)

	sample = sample_generator.sample

	if compare_to_deit:
		deit(sample)
	dleit(sample)

	print(f"Loaded dataset generator")

	test_dleit_inout = False

	if test_dleit_inout:
		for sample in sample_generator.gen():
			#print(f"Image: {sample}")
			#print(f"Config: {dleit_cfg.image_size}")

			dleit_res = dleit(sample)
			dleit_predicted_class_idx = tf.math.argmax(dleit_res.logits, axis=-1)[0]

			if compare_to_deit:
				deit_res = deit(sample)
				deit_predicted_class_idx = tf.math.argmax(deit_res.logits, axis=-1)[0]
			
			#print(f"Result: {res}")
			
			if compare_to_deit:
				if deit_predicted_class_idx == dleit_predicted_class_idx:
					print("Same predicted class: ", dleit.config.id2label[int(deit_predicted_class_idx)])
				else:
					print("Different predictions!")
					print(f"DeiT: {deit.config.id2label[int(deit_predicted_class_idx)]}")
					print(f"DLeiT: {dleit.config.id2label[int(dleit_predicted_class_idx)]}")
			else:
				print(f"Prediction: {dleit.config.id2label[int(dleit_predicted_class_idx)]}")

			break

		return

	converter_quant = tf.lite.TFLiteConverter.from_keras_model(dleit)
	converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
	converter_quant.representative_dataset = sample_generator.gen
	converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
	converter_quant.target_spec.supported_types = [tf.dtypes.int8]
	converter_quant.inference_input_type = tf.dtypes.float32 # changed from tf.uint8
	converter_quant.inference_output_type = tf.dtypes.float32 # changed from tf.uint8
	converter_quant.experimental_new_converter = True
	converter_quant.allow_custom_ops=True
	converter_quant.input_shape=(batch_size, *shape_a)

	converted_model = converter_quant.convert()

	output_file = 'dleit_model.tflite'
	with open(output_file, "wb") as f:
		f.write(converted_model)

	print(f"Saved model to {output_file}")

if __name__ == '__main__':
	main()