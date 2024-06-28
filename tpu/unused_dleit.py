class TFDleiTPixelShuffle(keras.layers.Layer):
	"""TF layer implementation of torch.nn.PixelShuffle"""

	def __init__(self, upscale_factor: int, **kwargs) -> None:
		super().__init__(**kwargs)
		if not isinstance(upscale_factor, int) or upscale_factor < 2:
			raise ValueError(f"upscale_factor must be an integer value >= 2 got {upscale_factor}")
		self.upscale_factor = upscale_factor

	def call(self, x: tf.Tensor) -> tf.Tensor:
		hidden_states = x
		batch_size, _, _, num_input_channels = shape_list(hidden_states)
		block_size_squared = self.upscale_factor**2
		output_depth = int(num_input_channels / block_size_squared)
		# When the number of output channels >= 2, PyTorch's PixelShuffle and
		# TF's depth_to_space differ in their output as the order of channels selected for combining
		# is a permutation of the other c.f.
		# https://stackoverflow.com/questions/68272502/tf-depth-to-space-not-same-as-torchs-pixelshuffle-when-output-channels-1
		permutation = tf.constant(
			[[i + j * block_size_squared for i in range(block_size_squared) for j in range(output_depth)]]
		)
		hidden_states = tf.gather(params=hidden_states, indices=tf.tile(permutation, [batch_size, 1]), batch_dims=-1)
		hidden_states = tf.nn.depth_to_space(hidden_states, block_size=self.upscale_factor, data_format="NHWC")
		return hidden_states

class TFDleiTDecoder(keras.layers.Layer):
	def __init__(self, config: DeiTConfig, **kwargs) -> None:
		super().__init__(**kwargs)
		self.conv2d = keras.layers.Conv2D(
			filters=config.encoder_stride**2 * config.num_channels, kernel_size=1, name="0"
		)
		self.pixel_shuffle = TFDleiTPixelShuffle(config.encoder_stride, name="1")
		self.config = config

	def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
		hidden_states = inputs
		hidden_states = self.conv2d(hidden_states)
		hidden_states = self.pixel_shuffle(hidden_states)
		return hidden_states

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "conv2d", None) is not None:
			with tf.name_scope(self.conv2d.name):
				self.conv2d.build([None, None, None, self.config.hidden_size])
		if getattr(self, "pixel_shuffle", None) is not None:
			with tf.name_scope(self.pixel_shuffle.name):
				self.pixel_shuffle.build(None)

@add_start_docstrings(
	"DLeiT Model with a decoder on top for masked image modeling, as proposed in"
	" [SimMIM](https://arxiv.org/abs/2111.09886).",
	DLEIT_START_DOCSTRING,
)
class TFDLeiTForMaskedImageModeling(TFDLeiTPreTrainedModel):
	def __init__(self, config: DeiTConfig) -> None:
		super().__init__(config)

		self.dleit = TFDLeiTMainLayer(config, add_pooling_layer=False, use_mask_token=True, name="dleit")
		self.decoder = TFDleiTDecoder(config, name="decoder")

	@unpack_inputs
	@add_start_docstrings_to_model_forward(DLEIT_INPUTS_DOCSTRING)
	@replace_return_docstrings(output_type=TFMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
	def call(
		self,
		pixel_values: tf.Tensor | None = None,
		bool_masked_pos: tf.Tensor | None = None,
		head_mask: tf.Tensor | None = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		training: bool = False,
	) -> Union[tuple, TFMaskedImageModelingOutput]:
		r"""
		bool_masked_pos (`tf.Tensor` of type bool and shape `(batch_size, num_patches)`):
			Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

		Returns:

		Examples:
		```python
		>>> from transformers import AutoImageProcessor, TFDLeiTForMaskedImageModeling
		>>> import tensorflow as tf
		>>> from PIL import Image
		>>> import requests

		>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		>>> image = Image.open(requests.get(url, stream=True).raw)

		>>> image_processor = AutoImageProcessor.from_pretrained("facebook/dleit-base-distilled-patch16-224")
		>>> model = TFDLeiTForMaskedImageModeling.from_pretrained("facebook/dleit-base-distilled-patch16-224")

		>>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
		>>> pixel_values = image_processor(images=image, return_tensors="tf").pixel_values
		>>> # create random boolean mask of shape (batch_size, num_patches)
		>>> bool_masked_pos = tf.cast(tf.random.uniform((1, num_patches), minval=0, maxval=2, dtype=tf.int32), tf.bool)

		>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
		>>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
		>>> list(reconstructed_pixel_values.shape)
		[1, 3, 224, 224]
		```"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.dleit(
			pixel_values,
			bool_masked_pos=bool_masked_pos,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			training=training,
		)

		sequence_output = outputs[0]

		# Reshape to (batch_size, num_channels, height, width)
		sequence_output = sequence_output[:, 1:-1]
		batch_size, sequence_length, num_channels = shape_list(sequence_output)
		height = width = int(sequence_length**0.5)
		sequence_output = tf.reshape(sequence_output, (batch_size, height, width, num_channels))

		# Reconstruct pixel values
		reconstructed_pixel_values = self.decoder(sequence_output, training=training)
		# TF 2.0 image layers can't use NCHW format when running on CPU, so intermediate layers use NHWC,
		# including the decoder. We transpose to compute the loss against the pixel values
		# (batch_size, height, width, num_channels) -> (batch_size, num_channels, height, width)
		reconstructed_pixel_values = tf.transpose(reconstructed_pixel_values, (0, 3, 1, 2))

		masked_im_loss = None
		if bool_masked_pos is not None:
			size = self.config.image_size // self.config.patch_size
			bool_masked_pos = tf.reshape(bool_masked_pos, (-1, size, size))
			mask = tf.repeat(bool_masked_pos, self.config.patch_size, 1)
			mask = tf.repeat(mask, self.config.patch_size, 2)
			mask = tf.expand_dims(mask, 1)
			mask = tf.cast(mask, tf.float32)

			reconstruction_loss = keras.losses.mean_absolute_error(
				# Swap axes as metric calculation reduces over the final dimension
				tf.transpose(pixel_values, (1, 2, 3, 0)),
				tf.transpose(reconstructed_pixel_values, (1, 2, 3, 0)),
			)
			reconstruction_loss = tf.expand_dims(reconstruction_loss, 0)
			total_loss = tf.reduce_sum(reconstruction_loss * mask)
			num_masked_pixels = (tf.reduce_sum(mask) + 1e-5) * self.config.num_channels
			masked_im_loss = total_loss / num_masked_pixels
			masked_im_loss = tf.reshape(masked_im_loss, (1,))

		if not return_dict:
			output = (reconstructed_pixel_values,) + outputs[1:]
			return ((masked_im_loss,) + output) if masked_im_loss is not None else output

		return TFMaskedImageModelingOutput(
			loss=masked_im_loss,
			reconstruction=reconstructed_pixel_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

	def build(self, input_shape=None):
		if self.built:
			return
		self.built = True
		if getattr(self, "dleit", None) is not None:
			with tf.name_scope(self.dleit.name):
				self.dleit.build(None)
		if getattr(self, "decoder", None) is not None:
			with tf.name_scope(self.decoder.name):
				self.decoder.build(None)

