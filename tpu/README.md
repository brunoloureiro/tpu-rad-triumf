# Setup for Evaluation of Coral EdgeTPUs with Radiation Experiments

## Getting Started

This setup is meant to be used on a host with Ubuntu 22.04 and the Coral EdgeTPU.

With a Raspberry Pi 4 host, the easiest way is to download the following image and clone it into your Raspberry's SD card.

- https://drive.google.com/file/d/1L-MKph_MsSxnJEZnNCKUhcrTgHz8jUcJ/view?usp=sharing

**Note: The current image is missing some model/input files and has slightly outdated code. You can use the bash scripts provided to copy the files into the DUT (if you use a different IP, simply replace the IP in the script file).**

### Manual Installation

For manual installation, follow the guides for each dependency. Some of these include:

- libLogHelper
	- https://github.com/radhelper/libLogHelper
- Coral Software
	- https://coral.ai/docs/accelerator/get-started/
	- https://coral.ai/software/
	- https://github.com/prbodmann/Coral-TPU/tree/main/elementary-ops

## Configuring the DUT

After cloning the SD and updating the files with the scripts (e.g., `scp_25.py`), you may need to reconfigure the DUT to have the correct IP and use the correct server port.

### DUT IP

Configuring a static IP for the Raspberry Pi 4 was actually a bit annoying. The easiest way is to use the GUI, but this was disabled in our installation (for performance reasons). Currently, the static IP is configured in `~/.atBoot.sh` (edit this as sudo). This value should be unique for each DUT and must match the configuration of the machines in the server. **Important: the DUT should NOT be connected to an external network, i.e., do not connect it to a DCP server, or the IP will be reset.** There are probably better ways to go about this, but this worked in our experiments.

#### Changing the DUT's IP

To change the IP:

- edit the file as sudo (e.g., `sudo nano ~/.atBoot.sh`) and
- change the IP in `ifconfig eth0 <ip>`.

Tip: number the IPs sequentially to make it easier to remember. In our experiments, the IP 192.168.1.**25** used the server port 1025 (explained next), making it easy to remember.

### Server Port

The server port should be configured by changing the `/etc/radiation-benchmarks.conf` file. Note: you should run the text editor (e.g., nano) as sudo.

The configuration should be relatively intuitive, but the most important values to change are `serverip` and `serverport`. The `serverip` should probably be the same for every DUT (unless you are using multiple servers), but each DUT should use an unique `serverport`. We usually number them sequentially, e.g., starting with 1024, 1025, and so on. **Important: these values MUST match the configuration in the server files.**

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

**In general, you should not have to run commands manually.** Instead, the server (https://github.com/brunoloureiro/tpu-rad-triumf/tree/main/server/) should automatically run these commands. An example of a command that the server will run is:

`/home/carol/tpu-rad/run_model.py -m /home/carol/tpu-rad/models/vit16_im64_ps8_proj256_nlayers3_nheads16_mlphead256_MHA_FROM_START_edgetpu.tflite -i /home/carol/tpu-rad/inputs/vit_base_16_images.npy -g /home/carol/tpu-rad/golden/vit_base_16_golden_MHA_from_image.npy --vit -n 32 --log_interval 20`

As you can see, the commands can get quite lengthy, especially with absolute paths (which are preferred). This is why you should use the server to run them!

- Server
	- https://github.com/radhelper/radiation-setup
- Fork with some configuration files
	- https://github.com/brunoloureiro/tpu-rad-triumf/server/