# Setup for Evaluation of Coral EdgeTPUs with Radiation Experiments

This repository contains scripts to perform radiation experiments with the Coral EdgeTPU.

- The server code is a fork of RadHelper's radiation setup
	- https://github.com/radhelper/radiation-setup

- while the DUT code uses libLogHelper
	- https://github.com/radhelper/libLogHelper

## Getting Started

Before doing anything else, we recommend reading the documentation for

- the server
	- https://github.com/brunoloureiro/tpu-rad-triumf/tree/main/server/README.md

- and for the DUTs
	- https://github.com/brunoloureiro/tpu-rad-triumf/tree/main/tpu/README.md

### Some reminders before setting things up

- Keep the DUTs in an internal network, do not connect it to an external network (DHCP server).
- For each DUT, you must re-configure the static IP and the server port (details in the DUT documentation at https://github.com/brunoloureiro/tpu-rad-triumf/tree/main/tpu/README.md).

## Before the Experiment

**Before starting the experiment, we recommend creating a spreadsheet with the configuration of each DUT, including IP, server port, and power switch port. At first, it might be easier to set up a single DUT and checking everything works.**

## During the Experiment

Thanks to ~~magic~~ Fernando (and many other collaborators), the server should handle *most* things (if configured properly), so you should just run `./server.py` in the host responsible for the server (the server should not be in the beam room).

However, you may at times wish to change the benchmark running on each DUT (or you want to add, remove, or replace DUTs). In this case, you can stop the server (`Control + C`) and manually edit some configuration files

- The configuration files are detailed in the server documentation at https://github.com/brunoloureiro/tpu-rad-triumf/tree/main/server/README.md.

### Stopping DUTs

First thing to notice: when you do this, the **DUTs keep running the benchmarks**. We suggest:

- manually turning off the power switches
- and/or connecting to each DUT via SSH and running `sudo shutdown -h now`
	- instead, you can run `sudo pkill -9 -f run_model.py` to kill the benchmark without turning the DUT off

Hopefully, this will be improved in the future (if you want to help, see: https://github.com/radhelper/radiation-setup/issues/4).

### Changing Benchmarks

Next, you must edit the relevant files (which you obviously remember from the DUT documentation):

- (Doc: https://github.com/brunoloureiro/tpu-rad-triumf/tree/main/tpu/README.md)

- To enable/disable a machine, edit `server_parameters.yaml`

- To change the benchmark of a machine, edit `machines_cfgs/rasp4coral.yaml` (for the first Rasp, `rasp4coral2.yaml` for the next, and so on). Again, we recommend **commenting out everything *EXCEPT* the benchmark you want to run.**