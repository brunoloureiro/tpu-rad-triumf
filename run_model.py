#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np

import argparse
import logging
import os
import time
from typing import Tuple, List, Union

#from rad_setup_modules import configs
#from rad_setup_modules import console_logger
#from rad_setup_modules import dnn_log_helper

import console_logger

# TO-DO: better import
from common_tpu import *

import log_helper as lh

def copy_tf_tensor(tensor):
	try:
		np_array = tensor.numpy()
	except AttributeError as e:
		np_array = tensor

	return np.copy(np_array)

def start_setup_log_file(
	framework_name: str,
	framework_version: str,
	device: str,
	model_name: str,
	args_conf: list,
) -> None:
	log_header = f"framework:{framework_name} framework_version:{framework_version} Device:{device}"
	log_header += " ".join(args_conf)
	bench_name = f"{framework_name}-{model_name}"
	lh.start_log_file(bench_name, log_header)
	lh.set_max_errors_iter(1024*9)
	lh.set_max_infos_iter(512)
	interval_print = 10
	lh.set_iter_interval_print(interval_print)


class LHLogger():
	def __init__(self):
		pass

	def start_log_file(self, *args, **kwargs):
		start_setup_log_file(*args, **kwargs)

	def end_log_file(self, *args, **kwargs):
		lh.end_log_file(*args, **kwargs)

	def start_iteration(self, *args, **kwargs):
		lh.start_iteration(*args, **kwargs)

	def end_iteration(self, *args, **kwargs):
		lh.end_iteration(*args, **kwargs)

	def perf(self, text, *args, **kwargs):
		lh.log_info_detail(f"PERF {text}", *args, **kwargs)

	def error(self, *args, **kwargs):
		lh.log_error_detail(*args, **kwargs)

	def info(self, *args, **kwargs):
		lh.log_info_detail(*args, **kwargs)

	def debug(self, text, *args, **kwargs):
		lh.log_info_detail(f"DEBUG {text}", *args, **kwargs)

	def warning(self, text, *args, **kwargs):
		lh.log_info_detail(f"WARNING {text}", *args, **kwargs)

class Timer:
	time_measure = 0

	def tic(self): self.time_measure = time.perf_counter()

	def toc(self): self.time_measure = time.perf_counter() - self.time_measure

	@property
	def diff_time(self): return self.time_measure

	@property
	def diff_time_str(self): return str(self)

	def __str__(self): return f"{self.time_measure:.4f}s"

	def __repr__(self): return str(self)

	def to_str(self, some_time): return f"{some_time:.4f}s"

def are_equal(lhs: tf.Tensor, rhs: tf.Tensor, threshold: Union[None, float]) -> bool:
	""" Compare based or not in a threshold, if the threshold is none then it is equal comparison    """
	if threshold is not None:
		np.all(
			np.le(
				np.abs(
					np.subtract(lhs.numpy(), rhs.numpy())
				), threshold
			)
		)
	else:
		return np.all(tf.equal(lhs, rhs))


def describe_error(flattened_tensor: np.ndarray) -> Tuple[int, int, float, float]:
	is_nan_tensor, is_inf_tensor = np.isnan(flattened_tensor), np.isinf(flattened_tensor)
	has_nan, has_inf = int(np.any(is_nan_tensor)), int(np.any(is_inf_tensor))
	filtered_tensor = flattened_tensor[~is_nan_tensor & ~is_inf_tensor]
	min_val = float(np.min(filtered_tensor)) if filtered_tensor.size > 0 else 0
	max_val = float(np.max(filtered_tensor)) if filtered_tensor.size > 0 else 0
	return has_nan, has_inf, min_val, max_val

def check_and_setup_gpu() -> None:
	# Disable all torch grads
	torch.set_grad_enabled(mode=False)
	if torch.cuda.is_available() is False:
		dnn_log_helper.log_and_crash(fatal_string=f"Device {configs.DEVICE} not available.")
	dev_capability = torch.cuda.get_device_capability()
	if dev_capability[0] < configs.MINIMUM_DEVICE_CAPABILITY:
		dnn_log_helper.log_and_crash(fatal_string=f"Device cap:{dev_capability} is too old.")

def compare_output(
	output: tf.Tensor,
	golden: tf.Tensor,
	image_index: int,
	logger: LHLogger,
	terminal_logger: logging.Logger = None,
	threshold: float = None,
	max_errors_terminal: int = 10,
):
	equal = are_equal(output, golden, threshold) #np.all(tf.equal(output, golden))

	# equal means no error
	if equal:
		return False
	else:
		logger.error(f"SDC image_index:{image_index}")
		if terminal_logger is not None:
			terminal_logger.error(f"SDC detected for image index {image_index}")
		if output.shape != golden.shape:
			logger.error(f"shape_error e:{golden.shape} r:{output.shape}")

			if terminal_logger is not None:
				terminal_logger.warning(f"Output shape does not match golden shape! Output: {output.shape} Golden: {golden.shape}")

		try:
			output = output.numpy()
			golden = golden.numpy()
		except AttributeError as e:
			pass #print(e)

		output = copy_tf_tensor(output).flatten()
		golden = copy_tf_tensor(golden).flatten()

		if output.size == golden.size:
			diff = golden - output
		else:
			if output.size > golden.size:
				expanded_golden = np.zeros((output.size))
				expanded_golden[:golden.size] = golden
				diff = output - expanded_golden
			else:
				expanded_output = np.zeros((golden.size))
				expanded_output[:output.size] = output
				diff = expanded_output - golden

		has_nan, has_inf, min_val, max_val = describe_error(output)

		logger.error(f"output_t has_nan:{has_nan} has_inf:{has_inf} min_val:{min_val} max_val:{max_val}")

		if terminal_logger is not None:
			terminal_logger.warning(f"Description of output has_nan:{has_nan} has_inf:{has_inf} min_val:{min_val} max_val:{max_val}")

		has_nan, has_inf, min_val, max_val = describe_error(diff)

		logger.error(f"diff_t has_nan:{has_nan} has_inf:{has_inf} min_val:{min_val} max_val:{max_val}")

		if terminal_logger is not None:
			terminal_logger.warning(f"Description of difference has_nan:{has_nan} has_inf:{has_inf} min_val:{min_val} max_val:{max_val}")

		num_elem_golden = golden.size
		curr_err_num = 0
		for i in range(output.size):
			expected = golden[i] if i < num_elem_golden else None
			read = output[i]

			if read != expected:
				logger.error(f"index:{i} e:{expected} r:{read}")
				if terminal_logger is not None:
					curr_err_num += 1
					if curr_err_num < max_errors_terminal:
						terminal_logger.error(f"Error on index:{i} expected: {expected} read: {read} (difference = {read - expected})")

		return True

def parse_args() -> Tuple[argparse.Namespace, List[str]]:
	""" Parse the args and return an args namespace and the tostring from the args    """
	parser = argparse.ArgumentParser(description='D(L)eiT TPU radiation setup', add_help=True)
	parser.add_argument('--iterations', '-it', default=int(1e12), help="Maximum iterations to run.", type=int)
	parser.add_argument('--testsamples', '-n', default=128, help="Test samples (images) to be used in the test.", type=int)
	parser.add_argument('--generate', '-gen', default=False, action="store_true", help="Set this flag to generate the golden output.")
	parser.add_argument('--enableconsolelog', '-log', default=False, action="store_true",
						help="Set this flag enable console logging")
	parser.add_argument('--model', '-m', required=True, help="Path to the *_edgetpu_.tflite file with the model.")
	parser.add_argument('--tokens', '-t', required=False, help="Path to the file with the embedding tokens for this model.")
	parser.add_argument('--input', '-i', required=True, help="Path to the file with the input for this model.")
	parser.add_argument('--golden', '-g', required=True, help="Path to the gold file (expected output).")
	parser.add_argument('--reload', '-r', default=False, action="store_true",
		help="(DEPRECATED: Is now default) Set this flag to reload all the data after a radiation-induced error.")
	parser.add_argument('--vit', '-v', '--notokens', '-nt', default=False, action="store_true", help="Set this flag to use ViT/models without token inputs.")
	parser.add_argument('--noreload', default=False, action="store_true",
		help="Set this flag to DISABLE reloading all the data after a radiation-induced error (not recommended).")
	

	args = parser.parse_args()

	# Check if it is only to generate the gold values
	if args.generate is True:
		args.iterations = 1

	args_text_list = [f"{k}={v}" for k, v in vars(args).items()]
	return args, args_text_list

def main():
	args, formatted_args = parse_args()

	# Possible TO-DOs:
	# Check if a device is ok and disable grad
	#check_and_setup_gpu()
	# (obviously not a GPU in my case)

	# Terminal console
	main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
	terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.enableconsolelog else None

	logger = LHLogger()

	model_name = Path(args.model).stem

	logger.start_log_file(
		framework_name="TensorFlow",
		framework_version=tf.__version__,
		device='EdgeTPU',
		model_name=model_name,
		args_conf=formatted_args,
	)

	if terminal_logger is not None:
		terminal_logger.debug("Started log file for setup. Params:" + "\n".join(formatted_args))

	model_file = args.model
	input_file = args.input
	gold_file = args.golden

	tokens_file = args.tokens

	n_images = args.testsamples

	max_iterations = args.iterations
	compute_golden = args.generate

	reload_on_error = not args.noreload

	if not reload_on_error:
		logger.warning(f"reload_data:False")
		if terminal_logger:
			terminal_logger.warning(f"You have DISABLED reloading the data after a radiation-induced error.")

	use_tokens = not args.vit

	timer = Timer()

	timer.tic()
	interpreter = load_model(model_file, cpu=False)
	timer.toc()

	logger.perf(f"loaded_object:model load_time:{timer.diff_time_str}")

	if terminal_logger is not None:
		terminal_logger.debug(f"Loaded model {model_file} ({timer.diff_time_str})")

	timer.tic()
	images = load_input_data(input_file)
	golden = load_golden(gold_file) if not compute_golden else []
	if use_tokens:
		tokens = load_tokens(tokens_file)
	else:
		tokens = None
	timer.toc()

	logger.perf(f"loaded_object:data load_time:{timer.diff_time_str}")

	if terminal_logger is not None:
		terminal_logger.debug(f"Loaded input and golden ({timer.diff_time_str})")

	if compute_golden:
		max_iterations = 1

	if n_images > len(images):
		logger.info(f'WARNING n_images:{n_images} input_images:{len(images)}')
		if terminal_logger:
			terminal_logger.warning(f"Script is configured to use {n_images} images, but input file only has {len(images)}. Continuing with available images.")
		n_images = len(images)

	if not compute_golden and n_images > len(golden):
		logger.info(f'WARNING n_images:{n_images} golden_outputs:{len(golden)}')
		if terminal_logger:
			terminal_logger.warning(f"Script is configured to use {n_images} images, but saved golden only has {len(golden)} outputs. Continuing with available golden outputs.")
		n_images = min(len(images), len(golden))


	for it in range(max_iterations):
		iter_ker_time = 0
		iter_compare_time = 0
		iter_err_count = 0
		for img_index in range(n_images):
			image = images[img_index]

			err = False

			timer.tic()
			logger.start_iteration()
			output = run_inference(interpreter, image, tokens)
			logger.end_iteration()
			timer.toc()

			kernel_time = timer.diff_time

			iter_ker_time += kernel_time

			compare_time = None

			if not compute_golden:
				sim_error = False
				if sim_error and it == 3:
					output[0][1] += 10

				timer.tic()
				err = compare_output(output, golden[img_index], img_index, logger, terminal_logger)
				timer.toc()

				compare_time = timer.diff_time

				iter_compare_time += compare_time
			else:
				golden.append(copy_tf_tensor(output))

			del output

			if err and reload_on_error:
				iter_err_count += 1
				# this should never happen
				if compute_golden:
					logger.error(f"logic_error: Inference error happened while computing golden")
					raise RuntimeError("Inference error happened while computing golden.")

				timer.tic()
				del interpreter
				timer.toc()

				delete_model_time = timer.diff_time

				logger.perf(f"destroyed_object:model destroy_time:{timer.diff_time_str}")

				if terminal_logger is not None:
					terminal_logger.debug(f"Deleted model from memory ({timer.diff_time_str})")

				timer.tic()
				del images
				del golden
				del tokens
				timer.toc()

				delete_data_time = timer.diff_time

				logger.perf(f"destroyed_object:data destroy_time:{timer.diff_time_str}")

				if terminal_logger is not None:
					terminal_logger.debug(f"Deleted input and golden from memory ({timer.diff_time_str})")
					terminal_logger.info(f"Finished deleting data from memory ({timer.to_str(delete_model_time + delete_data_time)}). Reloading everything.")

				timer.tic()
				interpreter = load_model(model_file)
				timer.toc()

				load_model_time = timer.diff_time

				logger.perf(f"loaded_object:model load_time:{timer.diff_time_str}")

				if terminal_logger is not None:
					terminal_logger.debug(f"Loaded model {model_file} ({timer.diff_time_str})")

				timer.tic()
				images = load_input_data(input_file)
				golden = load_golden(gold_file)
				if use_tokens:
					tokens = load_tokens(tokens_file)
				else:
					tokens = None
				timer.toc()

				load_data_time = timer.diff_time

				logger.perf(f"loaded_object:data load_time:{timer.diff_time_str}")

				if terminal_logger is not None:
					terminal_logger.debug(f"Loaded input and golden ({timer.diff_time_str})")
					terminal_logger.info(f"Finished reloading everything ({timer.to_str(load_data_time + load_model_time)})")

			# Printing timing information
			if terminal_logger:
				wasted_time = iter_compare_time
				iter_time = iter_ker_time + iter_compare_time
				if iter_time != 0:
					wasted_time_ratio = (wasted_time / iter_time) * 100.0
				iteration_out = f"It:{it:<3} inference time:{iter_ker_time:.5f}, "
				iteration_out += f"compare time:{iter_compare_time:.5f}, "
				iteration_out += f"(wasted:{wasted_time_ratio:.1f}%) errors:{iter_err_count} ({iter_err_count/n_images}%)"
				terminal_logger.debug(iteration_out)

		if compute_golden:
			save_golden(golden, gold_file)

	logger.end_log_file()

if __name__ == '__main__':
	try:
		main()
	except Exception as main_function_exception:
		#dnn_log_helper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
		raise main_function_exception
