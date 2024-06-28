#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import itertools

from transformers.tf_utils import shape_list

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

from pathlib import Path

import dleit_from_scratch
import incremental_dleit

with_teacher = False

if with_teacher:
	raise NotImplementedError
	DLeiT = dleit.TFDLeiTForImageClassificationWithTeacher
	DeiT = TFDeiTForImageClassificationWithTeacher
else:
	#DLeiT = dleit.TFDLeiTForImageClassification
	#DLeiT = incremental_dleit.DLeiTIncremental
	DLeiT = dleit_from_scratch.DLeiT
	DeiT = TFDeiTForImageClassification
	

DLeiTConfig = dleit_from_scratch.DLeiTConfig

def create_configs():
	cfgs = {}

	# constant params
	hidden_act="gelu"
	hidden_dropout_prob=0.0
	attention_probs_dropout_prob=0.0
	initializer_range=0.02
	layer_norm_eps=1e-12	
	num_channels=3
	qkv_bias=True
	num_labels=1000
	use_return_dict=False

	# may become variable params
	list_image_size = [64, 112, 224]
	list_patch_size = [8, 16]
	iterative_qkv = False
	stacked_heads = False

	# must be a multiple of number of attention heads
	list_intermediate_size = [0]#[192, 384, 768]#, 3072, 1536] #384 #192

	# int_size 96; hidden_size 48; 1 attn head works without SPLIT

	# variable params
	list_hidden_size = [96, 192, 384]#, 768]
	list_num_hidden_layers = [3, 4, 12]
	list_num_attention_heads = [3, 6, 12]

	all_cfgs = [list_image_size, list_patch_size, list_intermediate_size, list_hidden_size, list_num_hidden_layers, list_num_attention_heads]
	#all_possible_cfgs =

	expected_cfgs = 1
	for cfg_type in all_cfgs:
		expected_cfgs *= len(cfg_type) 

	# simple logic for now, might make more complex combinations later
	num_configs = 1

	#for i in range(num_configs):
	for possible_cfg in itertools.product(*all_cfgs):
		image_size = possible_cfg[0] #list_image_size[i]
		patch_size = possible_cfg[1] #list_patch_size[i]
		intermediate_size = possible_cfg[3] * 2 #possible_cfg[2] #list_intermediate_size[i]
		hidden_size = possible_cfg[3] #list_hidden_size[i]
		num_hidden_layers = possible_cfg[4] #list_num_hidden_layers[i]
		num_attention_heads = possible_cfg[5] #list_num_attention_heads[i]
		

		# not sure if this is correct
		# tried fixing some shape mismatches but it turns out it was the image pre-processing
		#intermediate_size = patch_size * patch_size * num_hidden_layers

		assert image_size % patch_size == 0, f"Please ensure the image size ({image_size}) is a multiple of the patch size ({patch_size})."

		assert intermediate_size % num_attention_heads == 0, f"Please ensure the intermediate size ({intermediate_size}) is a multiple of the number of heads ({num_attention_heads})."

		encoder_stride = patch_size

		cfg_name = f"dleit_fs_hs{hidden_size}_hl{num_hidden_layers}_ah{num_attention_heads}_is{intermediate_size}_im{image_size}_ps{patch_size}"

		if iterative_qkv:
			cfg_name += f"_iterqkv"

		if not stacked_heads:
			cfg_name += f"_layeredheads"

		#print(cfg_name)

		model_exists = os.path.isfile(Path('./models') / (cfg_name+'.tflite'))

		num_configs += 1

		#not interesting
		if image_size == 256 and patch_size == 8:
			continue

		if model_exists:
			continue
		else:
			print(f"({num_configs}/{expected_cfgs}) Creating config for model {cfg_name}.tflite")
			cfgs[cfg_name] = DLeiTConfig(
				hidden_size=hidden_size,
				num_hidden_layers=num_hidden_layers,
				num_attention_heads=num_attention_heads,
				intermediate_size=intermediate_size,
				hidden_act=hidden_act,
				hidden_dropout_prob=hidden_dropout_prob,
				attention_probs_dropout_prob=attention_probs_dropout_prob,
				initializer_range=initializer_range,
				layer_norm_eps=layer_norm_eps,
				image_size=image_size,
				patch_size=patch_size,
				num_channels=num_channels,
				qkv_bias=qkv_bias,
				encoder_stride=encoder_stride,
				num_labels=num_labels,
				use_return_dict=use_return_dict,
				iterative_qkv=iterative_qkv,
				stacked_heads=stacked_heads,
			)

			break

	return cfgs

def get_simple_resize(image_size):
	(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
	data_resize = keras.Sequential(
		[
			keras.layers.Normalization(),
			keras.layers.Resizing(image_size, image_size),
		],
		name="data_resize",
	)
	data_resize.layers[0].adapt(x_test)

	return data_resize

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
		batch_size=1,
		num_channels=3,
		*,
		const_tokens_shape,
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

		self.const_tokens_shape = const_tokens_shape

		self.const_token = tf.random.uniform(self.const_tokens_shape)

		#self.image_processor = AutoImageProcessor.from_pretrained(image_processor)
		self.image_processor = image_processor #get_simple_resize(image_size)

		self.to_converter = to_converter

		self._sample = None

		#image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
		
		#self.image_size = image_size
		self.num_channels = num_channels
		self.batch_size = batch_size

	def ensure_bhwc(self, pixel_values):
		batch_size, height, width, num_channels = shape_list(pixel_values)

		#print(f"Picture was {pixel_values.shape}")

		if num_channels != self.num_channels and height == self.num_channels:
			# TF 2.0 image layers can't use NCHW format when running on CPU.
			# (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
			pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))

		batch_size, height, width, num_channels = shape_list(pixel_values)
		assert batch_size == self.batch_size and height == width and height == self.image_size and num_channels == self.num_channels

		return pixel_values

	def gen(self):
		for x in self.train_dataset:
			image = tf.image.resize(x[0], (self.image_size, self.image_size))
			image = self.image_processor(
				#[tf.expand_dims(image,axis=0)],
				tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=False),
				#return_tensors="tf",
			)

			if self.to_converter:
				image = image['pixel_values'] # only what matters to converter

			yield [self.ensure_bhwc(image), tf.random.uniform(self.const_tokens_shape)]

	@property
	def sample(self):
		if self._sample is None:
			for x in self.gen_sample():
				self._sample = x
				break

		return [self._sample, tf.random.uniform(self.const_tokens_shape)]

	def gen_sample(self):
		for x in self.sample_dataset:
			image = tf.image.resize(x[0], (self.image_size, self.image_size))
			image = self.image_processor(
				#[tf.expand_dims(image,axis=0)],
				tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=False),
				#return_tensors="tf",
			)

			if self.to_converter:
				image = image['pixel_values'] # only what matters to converter

			yield self.ensure_bhwc(image)

def convert_and_export_model(
	model,
	sample_generator,
	model_file,
	file_path,
):
	print(f"Starting conversion of model {model_file}")

	dleit = model.model()

	batch_size = model.config.batch_size
	shape_a = model.image_shape
	shape_b = (1, model.config.num_patches+4, model.config.hidden_size)

	#print(f"inputs are shape {[(batch_size, *shape_a), (batch_size, *shape_b)]}")

	converter_quant = tf.lite.TFLiteConverter.from_keras_model(dleit)
	converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
	converter_quant.representative_dataset = sample_generator.gen
	converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
	converter_quant.target_spec.supported_types = [tf.dtypes.int8]
	converter_quant.inference_input_type = tf.dtypes.float32 # changed from tf.uint8
	converter_quant.inference_output_type = tf.dtypes.float32 # changed from tf.uint8
	converter_quant.experimental_new_converter = True
	converter_quant.allow_custom_ops = True
	converter_quant.input_shape=[(batch_size, *shape_a), (batch_size, *shape_b)]

	converted_model = converter_quant.convert()

	output_file = f'{model_file}.tflite'
	fd = Path(file_path)
	fd.mkdir(exist_ok=True, parents=True)
	with open(fd / output_file, "wb") as f:
		f.write(converted_model)

	print(f"Saved model to {output_file} (full path = {fd/output_file})")

def generate_inputs():
	np.random.seed(1234)

	image_size = 64
	ps = 8
	num_patches = (image_size//ps)**2
	hidden_size = 96

	embed_shape = (1, 1, num_patches+4, hidden_size)

	simple_resize = get_simple_resize(image_size)

	ds = DatasetGenerator(
		image_size,
		image_processor=simple_resize,
		to_converter=False,
		const_tokens_shape=embed_shape,
	)

	max_samples = 8
	curr_sample = 0

	imgs = []
	tokens = None

	for img, new_token in ds.gen():
		imgs.append(img)
		if tokens is None:
			tokens = new_token
		curr_sample += 1

		if curr_sample >= max_samples:
			break

	print(f"Generated {curr_sample} samples")

	save_path = Path('./inputs/')
	save_path.mkdir(exist_ok=True, parents=True)

	images_path = save_path / f"images_hs{hidden_size}_im{image_size}_ps{ps}"

	print(f"Saving images to {images_path}")

	np.save(images_path, imgs)

	tokens_path = save_path / f"tokens_hs{hidden_size}_im{image_size}_ps{ps}"

	print(f"Saving embed tokens to {tokens_path}")

	np.save(tokens_path, tokens)


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

	multiple_configs = True

	if multiple_configs:
		cfgs = create_configs()
		#return
	else:
		cfgs = {'dleit_fs_model_default':DLeiTConfig()}

	infer_sample = False

	dleit_from_scratch.DEBUG_SHAPES = False

	simple_resize = None

	for cfg_name, dleit_cfg in cfgs.items():
		dleit = DLeiT(dleit_cfg)
		#dleit = DeiT(DeiTConfig())
		#dleit.build(input_shape=(batch_size, *shape_a))
		#dleit.compile()

		if simple_resize is None:
			simple_resize = get_simple_resize(dleit_cfg.image_size)

		dleit.build()
		print(f"Built dleit")

		image_size = dleit_cfg.image_size
		shape_a = (image_size, image_size, 3)

		emb_shape = (dleit.config.batch_size, dleit.config.batch_size, dleit.config.num_patches+4, dleit.config.hidden_size)

		sample_generator = DatasetGenerator(image_size, to_converter=False, const_tokens_shape=emb_shape, image_processor=simple_resize)

		if infer_sample:
			#sample_generator = SampleGenerator(shape_a, batch_size)
			sample, tokens = sample_generator.sample

			if compare_to_deit:
				deit(sample)
			res = dleit(sample, tokens)
			print(f"Output of calling dleit has shape {res.shape}")
			return

		print(f"Loaded dataset generator")

		#dleit = dleit.model()

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

		export_path = './models/'

		convert_and_export_model(dleit, sample_generator, cfg_name, export_path)

if __name__ == '__main__':
	#main()
	generate_inputs()