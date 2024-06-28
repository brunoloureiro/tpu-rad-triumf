#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import math
#import torch.utils.data

from pathlib import Path

import os
import sys

#import torchvision as tv
from dleit_for_training import (
	DLeiTConfig,
	DLeiTForTraining,
	DEBUG_SHAPES,
)

def to_tf(tensor):
	return tf.convert_to_tensor(tensor.numpy)

# Based on Pablo's code
class ImagePreProcessor():
	def __init__(
		self,
		upscale_size,
		image_size,
		num_classes,
		is_training=True,
	):
		self.upscale_size = upscale_size
		self.image_size = image_size
		self.num_classes = num_classes
		self.is_training = is_training

	def preprocess(self):
		def _preprocess(image, label):
			#image = to_tf(image)
			# create "rgb" images from bw
			if image.shape[-1] != 3:
				image = tf.concat([image, image, image], axis=-1)

			if self.is_training:
				# Resize to a bigger spatial resolution and take the random
				# crops.
				image = tf.image.resize(image, (self.upscale_size, self.upscale_size))
				image = tf.image.random_crop(image, (self.image_size, self.image_size, 3))
				image = tf.image.random_flip_left_right(image)
			else:
				image = tf.image.resize(image, (self.image_size, self.image_size))
			label = tf.one_hot(label, depth=self.num_classes)
			return image, label

		return _preprocess

def prepare_dataset(
	dataset,
	preprocess_dataset,
	batch_size,
	num_batches=8,
	is_training=True,
):
	if is_training:
		dataset = dataset.shuffle(batch_size * num_batches)
	dataset = dataset.map(preprocess_dataset(), num_parallel_calls=tf.data.AUTOTUNE)
	return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
	batch_size = 64

	hidden_size = 192#96
	n_hidden_layers = 4#3
	n_attention_heads = 12#6

	# some configs do not follow this pattern
	intermediate_size = 2 * hidden_size

	image_size = 64
	patch_size = 8

	dleit_cfg = DLeiTConfig(
		# constants, do not change
		hidden_act="gelu",
		num_channels=3,
		qkv_bias=True,
		encoder_stride=patch_size,
		use_return_dict=False,
		iterative_qkv=False,
		stacked_heads=False,
		# meta params, up to you
		hidden_dropout_prob=0.0,
		attention_probs_dropout_prob=0.0,
		initializer_range=0.02,
		layer_norm_eps=1e-12,
		num_labels=100,
		# params, you can change these based on the model (ideally change the variables above, though)
		hidden_size=hidden_size,
		num_hidden_layers=n_hidden_layers,
		num_attention_heads=n_attention_heads,
		intermediate_size=intermediate_size,
		image_size=image_size,
		patch_size=patch_size,
		batch_size=batch_size,
	)

	DEBUG_SHAPES = False

	if dleit_cfg.num_labels != 1000:
		print(f"REMINDER THIS IS USING ONLY {dleit_cfg.num_labels} LABELS (ImageNet should be 1000)")

	dleit = DLeiTForTraining(dleit_cfg)

	run_inference = False

	if run_inference:
		res = dleit(tf.random.uniform((dleit_cfg.batch_size, dleit_cfg.image_size, dleit_cfg.image_size, dleit_cfg.num_channels)))
		print(f"This is the result shape: {res.shape}")

		return

	# training params

	epochs = 30
	#learning_rate = 0.002
	label_smoothing_factor = 0.1

	learning_rate = 0.001
	weight_decay = 0.0001

	#optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

	optimizer = tf.keras.optimizers.AdamW(
		learning_rate=learning_rate,
		weight_decay=weight_decay,
	)

	loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_factor)

	metrics = [
		tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
		tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
	]

	dleit.compile(optimizer, loss_fn, metrics=metrics)

	dleit.build()

	checkpoint_dir = Path('/tmp/dleit_checkpoints')
	checkpoint_dir.mkdir(exist_ok=True, parents=True)
	cfg_name = f"dleit_ft_hs{dleit_cfg.hidden_size}"
	cfg_name += f"_hl{dleit_cfg.num_hidden_layers}_ah{dleit_cfg.num_attention_heads}"
	cfg_name += f"_is{dleit_cfg.intermediate_size}_im{dleit_cfg.image_size}_ps{dleit_cfg.patch_size}"

	cfg_name += f"_layeredheads"

	checkpoint_dir = checkpoint_dir / f"{cfg_name}.checkpoint"

	checkpoint_callback = keras.callbacks.ModelCheckpoint(
		checkpoint_dir,
		monitor="val_accuracy",
		save_best_only=True,
		save_weights_only=True,
	)

	upscale_factor = 2
	upscale_size = dleit_cfg.image_size * upscale_factor

	train_preprocessor = ImagePreProcessor(
		upscale_size,
		dleit_cfg.image_size,
		dleit_cfg.num_labels,
		is_training=True,
	)

	val_preprocessor = ImagePreProcessor(
		upscale_size,
		dleit_cfg.image_size,
		dleit_cfg.num_labels,
		is_training=False,
	)

	'''
	train_dataset, val_dataset = tfds.load(
		#"tf_flowers",
		"mnist",
		#'huggingface:imagenet-1k',
		split=["train[:100%]", "test[:100%]"],
		#split=["train[:90%]", "train[90%:]"],
		#split=["train[:40%]", "train[40%:50%]"],
		#split=["train[:9%]", "train[9%:10%]"],
		as_supervised=True,
	)
	'''

	 # Pablo's code to generate cifar train/test splits

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

	y_train = tf.keras.utils.to_categorical(y_train)
	y_test = tf.keras.utils.to_categorical(y_test)

	'''
	train_dataset = prepare_dataset(
		train_dataset,
		train_preprocessor.preprocess,
		batch_size,
		num_batches=16,
		is_training=True,
	)

	val_dataset = prepare_dataset(
		val_dataset,
		val_preprocessor.preprocess,
		batch_size,
		num_batches=16,
		is_training=False,
	)
	'''

	data_resize_aug = tf.keras.Sequential(
		[
			keras.layers.Normalization(),
			keras.layers.Resizing(image_size, image_size),
			keras.layers.RandomFlip("horizontal"),
			keras.layers.RandomRotation(factor=0.02),
			keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
		],
		name="data_resize_aug",
	)

	data_resize_aug.layers[0].adapt(x_train)

	data_resize = tf.keras.Sequential(
			[
				keras.layers.Normalization(),
				keras.layers.Resizing(image_size, image_size),               
			],
			name="data_resize",
		)
	data_resize.layers[0].adapt(x_test)

	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	train_dataset = train_dataset.batch(batch_size).map(lambda x, y: (data_resize_aug(x), y))
	test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	val_dataset = test_dataset.batch(batch_size).map(lambda x, y: (data_resize(x), y))

	print(f"Loaded everything. Starting training.")

	dleit.fit(
		x=train_dataset,
		validation_data=val_dataset,
		epochs=epochs,
		#callbacks=[checkpoint_callback],
		verbose=1,
	)

	if dleit_cfg.num_labels != 1000:
		print(f"REMINDER THIS IS USING ONLY {dleit_cfg.num_labels} LABELS (ImageNet should be 1000)")

	#dleit.load_weights(checkpoint_dir)
	#_, accuracy = dleit.evaluate(val_dataset)
	#print(f"Validation accuracy: {round(accuracy * 100, 2)}%")

if __name__ == '__main__':
	main()