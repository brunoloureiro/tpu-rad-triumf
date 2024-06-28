# Setup for Evaluation of Coral EdgeTPUs with Radiation Experiments

## Getting Started

This setup is meant to be used on a host with Ubuntu 22.04 and the Coral EdgeTPU.

With a Raspberry Pi 4 host, the easiest way is to download the following image and clone it into your Raspberry's SD card.

- link here

**Note: The current image is missing some model/input files and has slightly outdated code. You can use the bash scripts provided to copy the files into the DUT (if you use a different IP, simply replace the IP in the script file).**

### Manual Installation

For manual installation, follow the guides for each dependency. Some of these include:

- libLogHelper
	- https://github.com/radhelper/libLogHelper
- Coral Software
	- https://coral.ai/docs/accelerator/get-started/
	- https://coral.ai/software/
	- https://github.com/prbodmann/Coral-TPU/tree/main/elementary-ops

## Running a Benchmark

The main script used to run benchmarks is `run_model.py`. For a complete list of arguments, run `./run_model.py --help`. Some common (and important) arguments include:

- **\-\-vit: this flag MUST be enabled when using the ViT model (which you should be)**
- \-\-model (\-m): path for the .tflite file containing the EdgeTPU model
- \-\-inputs (\-i): path for the file containing the inputs (.npy file preferred)
- \-\-golden (\-g): path for the file containing the golden file (.npy file preferred)
- \-\-testsamples (\-n): number of images to use. Each image should be one position of the array loaded from the inputs file passed above.
- \-\-generate (\-gen): use this flag to generate the golden. Only needs to be done once per input (this repo already includes the golden for the inputs provided).
- \-\-log_interval: number of iterations between logging performance metrics. Ideally, this number should be configured so the benchmark sends roughly one log per second to the server. Simpler/faster benchmarks should set higher numbers to avoid flooding the network.
- \-\-enable_console (\-log): this flag enables printing to the console. Useful for debugging, but should not be set during real experiments to reduce overhead.

**In general, you should not have to run commands manually.** Instead, the server (https://github.com/brunoloureiro/rad-setup-tpu) should automatically run these commands. An example of a command that the server will run is:

`/home/carol/tpu-rad/run_model.py -m /home/carol/tpu-rad/models/vit16_im64_ps8_proj256_nlayers3_nheads16_mlphead256_MHA_FROM_START_edgetpu.tflite -i /home/carol/tpu-rad/inputs/vit_base_16_images.npy -g /home/carol/tpu-rad/golden/vit_base_16_golden_MHA_from_image.npy --vit -n 32 --log_interval 20`

As you can see, the commands can get quite lengthy, especially with absolute paths (which are preferred). This is why you should use the server to run them!

- Server
	- https://github.com/radhelper/radiation-setup
- Fork with some configuration files
	- https://github.com/brunoloureiro/rad-setup-tpu