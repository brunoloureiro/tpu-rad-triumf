#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import vit
from vit_split import ViTModel
from tensorflow.keras.utils import to_categorical
import os
from pathlib import Path

def quantize_model(model, image_size, prev_layermodel=None):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	data_resize_aug = tf.keras.Sequential(
				[               
					tf.keras.layers.Normalization(),
					tf.keras.layers.Resizing(image_size, image_size),
					tf.keras.layers.RandomFlip("horizontal"),
					tf.keras.layers.RandomRotation(factor=0.02),
					tf.keras.layers.RandomZoom(
						height_factor=0.2, width_factor=0.2
					),
				],
				name="data_resize_aug",
			)

	data_resize_aug.layers[0].adapt(x_train)

	data_resize = tf.keras.Sequential(
				[               
					tf.keras.layers.Normalization(),
					tf.keras.layers.Resizing(image_size, image_size),               
				],
				name="data_resize",
			)
	data_resize.layers[0].adapt(x_test)


	batch_size=1
	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	train_dataset = train_dataset.batch(1).map(lambda x, y: (data_resize_aug(x), y))
	test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	test_dataset = test_dataset.batch(1).map(lambda x, y: (data_resize(x), y))


	if prev_layermodel is None:
		new_in_shape = (1,image_size,image_size,3)
	else:
		new_in_shape = tuple(model.layers[0].input_shape[0]) #get_batched_input_shape(model.layers[0].input_shape, batch_size=1)

	newInput = tf.keras.layers.Input(batch_shape=new_in_shape)

	newOutputs = model(newInput)
	newModel = tf.keras.Model(newInput,newOutputs)
	newModel.set_weights(model.get_weights())
	model = newModel
	X = tf.random.uniform(new_in_shape)
	y_pred = model.predict(X)

	#print([tf.expand_dims(tf.dtypes.cast(x_train[0], tf.float32),0)])
	if prev_layermodel is None:
		def representative_data_gen():
			for input_value in train_dataset.take(1000):
				yield [tf.dtypes.cast(input_value[0],tf.float32)]
	else:
		def representative_data_gen():
			for input_value in train_dataset.take(1000):
				prev_output = prev_layermodel(tf.dtypes.cast(input_value[0],tf.float32))
				'''
				print(f"Previous layermodel:")
				for layer in prev_layermodel.layers:
					print(f"\t{layer.name}")
				print(f"Shape of prev output is {prev_output.shape}")
				'''
				yield [prev_output]
	
	converter_quant = tf.lite.TFLiteConverter.from_keras_model(model) 
	converter_quant.input_shape=(1,image_size,image_size,3)
	converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
	converter_quant.representative_dataset = representative_data_gen
	converter_quant.target_spec.supported_ops = [
	  tf.lite.OpsSet.TFLITE_BUILTINS_INT8, # enable TensorFlow Lite ops.
	  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
	]
	converter_quant.target_spec.supported_types = [tf.int8]
	converter_quant.experimental_new_converter = True
	converter_quant.allow_custom_ops=True
	converter_quant._experimental_new_quantizer = True

	return converter_quant.convert()

def copy_tf_tensor(tensor):
	try:
		np_array = tensor.numpy()
	except AttributeError as e:
		np_array = tensor

	return np_array

def get_batched_input_shape(input_shape, batch_size):
	new_input_shape = []
	try:
		for dim in input_shape:
			# MHA and some other layers may have multiple inputs
			new_subdim_shape = []
			for subdim in dim:
				if subdim is None:
					new_subdim_shape.append(batch_size)
				else:
					new_subdim_shape.append(int(subdim))
			#input_layers.append(tf.keras.layers.Input(tuple(new_subdim_shape)))
			new_input_shape.append(tuple(new_subdim_shape))
			assert tuple(new_subdim_shape) == new_input_shape[0], f"Shapes do not match {new_subdim_shape} and {new_input_shape[0]}"
		new_input_shape = [len(new_input_shape)] + list(new_input_shape[0])
	except TypeError as e:
		for dim in input_shape:
			# this is the normal scenario where an input dim is 1-D array
			if dim is None:
				new_input_shape.append(batch_size)
			else:
				new_input_shape.append(int(dim))
		#input_layers = tf.keras.layers.Input(tuple(input_layers))

	return new_input_shape#, multi_input 

def test_layer_only_model(model, batch_size=1):
	new_shape = get_batched_input_shape(model.layers[0].input_shape, batch_size)
	random_input = tf.random.uniform(new_shape)
	out = model(tuple(random_input))
	print(f"Layer-only model inference worked correctly.")

def get_layer_only_model(model, layer_id, batch_size=1, mha=False, final_block=False):
	i = layer_id
	layer = model.layers[i]

	new_input_shape = get_batched_input_shape(layer.input_shape, batch_size)
	print(f"New input shape {new_input_shape}")
	input_layers = tf.keras.layers.Input(batch_shape=tuple(new_input_shape))
	x = layer(input_layers)

	if mha:
		# CHANGE THIS LATER
		# THIS IS HARD-CODED FOR MHA + NORM
		x = model.layers[i+1](x)
		x = model.layers[i+2](x)
	elif final_block:
		x = model.layers[-5](x)
		x = model.layers[-4](x)
		x = model.layers[-3](x)
		x = model.layers[-2](x)
		x = model.layers[-1](x)

	return tf.keras.Model(inputs=input_layers, outputs=x)

def generate_output(model_file, input_file, batch_inputs=None):
	input_images = np.load(input_file)

	output_data = []
	for input_data in input_images:
		if batch_inputs:
			input_data = [input_data, input_data, input_data]
		interpreter = tf.lite.Interpreter(model_path=str(model_file))
		interpreter.allocate_tensors()

		# Get input and output tensors.
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		# Test model on random input data.
		input_shape = input_details[0]['shape']
		#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
		interpreter.set_tensor(input_details[0]['index'], input_data)

		interpreter.invoke()

		# The function `get_tensor()` returns a copy of the tensor data.
		# Use `tensor()` in order to get a pointer to the tensor.
		img_output = interpreter.get_tensor(output_details[0]['index'])
		output_data.append(copy_tf_tensor(img_output))
	return output_data

def get_model_by_cfg(model_cfg, max_ops=0, model_weights=None):
	model = None
	if model_cfg == 'vit_im64_ps8_proj128_nlayers3_nheads8_mlphead256.pb':
		num_classes=100
		image_size = 64  # We'll resize input images to this size
		patch_size = 8
		projection_dim = 128		
		num_heads = 8
		n_layers = 3
		mlp_head_size = 256
		input_shape = (image_size, image_size, 3)

		model_name="vit8"
		model = ViTModel(
			in_shape=input_shape,
			num_classes=num_classes,
			image_size=image_size,
			patch_size=patch_size,
			num_patches=(image_size // patch_size) ** 2,
			projection_dim=projection_dim,
			dropout=0.2,
			n_transformer_layers=n_layers,
			num_heads=num_heads,
			transformer_units=[
				projection_dim*2,
				projection_dim,
			],
			mlp_head_units=[mlp_head_size],
			partial_model_op=max_ops,
		)
		model = model.model(input_shape)

	elif model_cfg == 'vit16_im64_ps8_proj256_nlayers3_nheads16_mlphead256.pb':
		num_classes=100
		image_size = 64  # We'll resize input images to this size
		patch_size = 8
		projection_dim = 256		
		num_heads = 16
		n_layers = 3
		mlp_head_size = 256
		input_shape = (image_size, image_size, 3)

		model_name="vit16"
		model = ViTModel(
			in_shape=input_shape,
			num_classes=num_classes,
			image_size=image_size,
			patch_size=patch_size,
			num_patches=(image_size // patch_size) ** 2,
			projection_dim=projection_dim,
			dropout=0.2,
			n_transformer_layers=n_layers,
			num_heads=num_heads,
			transformer_units=[
				projection_dim*2,
				projection_dim,
			],
			mlp_head_units=[mlp_head_size],
			partial_model_op=max_ops,
		)
		model = model.model(input_shape)

	return model

def main():
	num_layers = 3
	max_ops = 12 + (6*num_layers)

	weights_path = '/home/carol/pablo-tpu/transformer'

	vit16 = True

	if vit16:
		cfg_name = 'vit16_im64_ps8_proj256_nlayers3_nheads16_mlphead256.pb'
		dir_name = 'vit16'
	else:
		cfg_name = 'vit_im64_ps8_proj128_nlayers3_nheads8_mlphead256.pb'
		dir_name = 'vit_files'

	model_path = Path('/home/carol/tpu-rad/models/partial16')
	model_path.mkdir(exist_ok=True, parents=True)
	weights_file = Path(weights_path) / dir_name

	#vit_model = get_model_by_cfg(cfg_name)

	#vit_model.load_weights(weights_file, custom_objects = vit.MultiHeadAttention)
	vit_model = tf.keras.models.load_model(weights_file, custom_objects = {'MultiHeadAttention': vit.MultiHeadAttention})
	#weights = vit_model.trainable_variables

	# beware the confusion between layers and "op"
	# model.layers is one thing
	# the "return at specific OP number" splitting is a different thing
	# sometimes I interchange both in the variable names (oops)
	# will definitely refactor this in the future (ETA: March 2063)
	random_sample = tf.random.uniform((1, 64, 64, 3))
	vit_weights = [l.get_weights() for l in vit_model.layers]
	specific_op_only = False
	specific_op_id = 24
	enable_model_quantization = True

	previous_layermodel = None

	mha = False
	final_block = False

	# 0 and max_ops are the same, but starting from 0 you do not have
	# the previous model, so I will just do this the lazy way
	for op_id in range(max_ops+1):
		i = op_id
		if final_block:
			previous_op = 19
		else:
			previous_op = specific_op_id - 1 
		if specific_op_only and i != specific_op_id and i != previous_op:
			continue

		# TO-DO
		# CHANGE THIS LATER
		# HARD-CODED FOR MHA
		if mha and i == specific_op_id:
			i = i + 2

		model = get_model_by_cfg(cfg_name, i, vit_weights)
		#model.build((1, 64, 64, 3))
		print(f"Created partial with {i} ops")
		model_output = model(random_sample)
		next_weight = 0
		for l, w in enumerate(model.layers):
			#print(f"Weights for layer {l}: {w.shape} vs full model {vit_weights[l].shape}")
			#print(model.layers[l])
			print(f"\t{model.layers[l].name}")
			try:
				model.layers[l].set_weights(vit_weights[next_weight])
				next_weight += 1
			except ValueError as e:
				print(f"Failed to set weights for layer {l}")
		#return
		#model.set_weights(vit_model.get_weights()) 
		print(f"Successfully loaded weights from trained model into partial with {i} ops")

		vit_output = vit_model(random_sample)
		model_output = model(random_sample)

		# TO-DO
		# CHANGE THIS LATER
		if mha and op_id == specific_op_id:
			i = op_id

		if vit_output.shape != model_output.shape:
			print(f"Models have different outputs! (That is a good thing)")
		else:
			if np.all(vit_output == model_output):
				if i == 0:
					print(f"Full model had same outputs; that is a good thing.")
				else:
					print(f"ERROR: Outputs are the same for vit and partial={i}")	
			else:
				print(f"WARNING: Outputs have same shape but different values for vit and partial={i}")

		if specific_op_only:
			if i == specific_op_id:
				if mha:
					layer_id = -3
				elif final_block:
					layer_id = -5
				else:
					layer_id = -1 # last layer
				layermodel = get_layer_only_model(model, layer_id, mha=mha, final_block=final_block)
				test_layer_only_model(layermodel)
			elif i == previous_op:
				prev_layermodel = model
				continue

		if not enable_model_quantization:
			continue

		if not specific_op_only:
			if vit16:
				model_name = f'vit16_im64_ps8_proj256_nlayers3_nheads16_mlphead256_ops{i}.tflite'
			else:
				model_name = f'vit_im64_ps8_proj128_nlayers3_nheads8_mlphead256_ops{i}.tflite'
			file_exists = os.path.isfile(model_path / model_name)
			if not file_exists:
				quantized_model = quantize_model(model, image_size=64)
				print(f"Finished quantizing model with {i} ops")
			else:
				print(f'MODEL ALREADY EXISTS -- SKIPPING QUANTIZATION OF MODEL {model_name}')

		else:
			# CHANGE THIS
			if vit16:
				if mha:
					model_name = f'vit16_im64_ps8_proj256_nlayers3_nheads16_mlphead256_MHA_ONLY.tflite'
				elif final_block:
					model_name = f'vit16_im64_ps8_proj256_nlayers3_nheads16_mlphead256_FINAL_BLOCK.tflite'
				
			else:
				if mha:
					model_name = f'vit_im64_ps8_proj128_nlayers3_nheads8_mlphead256_MHA_ONLY.tflite'
				elif final_block:
					model_name = f'vit_im64_ps8_proj128_nlayers3_nheads8_mlphead256_FINAL_BLOCK.tflite'
			file_exists = os.path.isfile(model_path / model_name)
			#file_exists = False
			if not file_exists:
				# CHANGE THIS CODE TO CONSIDER WHAT OP WE WANT
				# FOR NOW IT IS HARD CODED FOR MHA BECAUSE I NEED IT
				quantized_model = quantize_model(layermodel, image_size=64, prev_layermodel=prev_layermodel)
				print(f"Finished quantizing last-layer model with {i} ops.")
			else:
				print(f'MODEL ALREADY EXISTS -- SKIPPING QUANTIZATION OF LAYER-ONLY MODEL {model_name}')
			
		if not file_exists:
			with open(model_path / model_name, "wb") as f:
				f.write(quantized_model)
			print(f"Saved model to: {model_path / model_name}")

		if not specific_op_only:
			if vit16:
				input_file = Path('/home/carol/tpu-rad/inputs/vit_base_16_images.npy')
			else:		
				input_file = Path('/home/carol/tpu-rad/inputs/vit_base_8_images.npy')
			output = generate_output(model_path/model_name, input_file)
			print(f"Generated output for model with {i} ops")
			if vit16:
				output_path = Path('/home/carol/tpu-rad/golden/partial16')
			else:
				output_path = Path('/home/carol/tpu-rad/golden/partial')
			output_path.mkdir(exist_ok=True, parents=True)
			if vit16:
				output_name = f'vit_base_16_golden_ops{i}.npy'
			else:
				output_name = f'vit_base_8_golden_ops{i}.npy'
		else:
			if vit16:
				if mha:
					# is it really supposed to be specific ops id? I do not remember anymore
					input_file = Path(f'/home/carol/tpu-rad/golden/partial16/vit_base_16_golden_ops{previous_op}.npy')
				elif final_block:
					input_file = Path(f'/home/carol/tpu-rad/golden/partial16/vit_base_16_golden_ops{previous_op}.npy')
			else:
				if mha:
					# is it really supposed to be specific ops id? I do not remember anymore
					# CHANGED FROM specific_op_id
					input_file = Path(f'/home/carol/tpu-rad/golden/partial/vit_base_8_golden_ops{previous_op}.npy')
				elif final_block:
					input_file = Path(f'/home/carol/tpu-rad/golden/partial/vit_base_8_golden_ops{previous_op}.npy')
			batch_inputs = 3 if mha else None
			output = generate_output(model_path/model_name, input_file, batch_inputs=batch_inputs)
			print(f"Generated output for layer-only model with {i} ops")
			if vit16:
				output_path = Path('/home/carol/tpu-rad/golden/partial16')
			else:
				output_path = Path('/home/carol/tpu-rad/golden/partial')
			output_path.mkdir(exist_ok=True, parents=True)
			if vit16:
				if mha:
					output_name = f'vit_base_16_golden_MHA_ONLY.npy'
				elif final_block:
					output_name = f'vit_base_16_golden_FINAL_BLOCK.npy'
			else:
				if mha:
					output_name = f'vit_base_8_golden_MHA_ONLY.npy'
				elif final_block:
					output_name = f'vit_base_8_golden_FINAL_BLOCK.npy'

		np.save(output_path / output_name, output)
		print(f"Done saving {output_path/output_name}")
		
if __name__ == '__main__':
	main()