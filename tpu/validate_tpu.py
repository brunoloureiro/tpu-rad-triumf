#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np

# TO-DO: better import
from common_tpu import *

def create_input():
	image_size = (2048, 2048)
	channels = 3
	input_shape = (*image_size, channels)

	carol_path = True

	if carol_path:
		input_path = '/home/carol/data_validate_tpu/input_val'
	else:
		input_path = '/home/loureiro/repos/tpu-rad/coral/input_val'

	tf.random.set_seed(5312)

	data = tf.random.uniform(input_shape, seed=None)

	print(data)

	save_data(data, input_path)

def check_golden():
	gold_path = '/home/carol/data_validate_tpu/gold_val.npy'

	golden = load_data(gold_path)
	print(golden.shape)
	print(golden)
	print(golden.shape)

def main():
	compute_golden = True

	carol_path = True

	if carol_path:
		model_file = '/home/carol/data_validate_tpu/conv2k_edgeai.tflite'
		input_path = '/home/carol/data_validate_tpu/input_val.npy'
		gold_path = '/home/carol/data_validate_tpu/gold_val.npy'
	else:
		model_file = '/home/loureiro/repos/tpu-rad/coral/dleit_model.tflite'
		input_path = '/home/loureiro/repos/tpu-rad/coral/input_val.npy'
		gold_path = '/home/loureiro/repos/tpu-rad/coral/gold_val.npy'


	interpreter = load_model(model_file)
	print(f"Loaded model")

	input_data = load_data(input_path)
	print(f"Loaded input data")

	input_data = [input_data]

	if compute_golden:
		print(f"Computing golden...")
		output = run_inference(interpreter, input_data)
		print(f"Finished computing golden. Saving output.")
		save_data(output, gold_path)
		print(f"Finished saving golden to {gold_path}.")
		del output

	print(f"Validating TPU...")

	golden = load_data(gold_path)
	print(f"Loaded golden from {gold_path}.")

	print(f"Running inference...")
	output = run_inference(interpreter, input_data)

	print(f"Finished running inference. Comparing to golden...")

	eq = compare_output(output, golden)

	if eq:
		print(f"Compared to golden. All OK!")
	else:
		print(f"Golden and output do not match!")
		raise RuntimeError(f"Golden does not match output.")

	del output
	del golden

	return


if __name__ == '__main__':
	try:
		main()
	finally:
		#create_input()
		check_golden()