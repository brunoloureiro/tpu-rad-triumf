# Setup for Evaluation of Coral EdgeTPUs with Radiation Experiments

This is a fork of the RadHelper radiation setup (https://github.com/radhelper/radiation-setup). This fork is meant to be used for **Neutron** beam experiments with the Coral EdgeTPU.

## Getting Started

For an in-depth installation guide, refer to the original repository (https://github.com/radhelper/radiation-setup).

## How the Server works (or not)

The server should run in an Ubuntu host. The main script for the server is `server.py`, and it can be run without any additional flags.

The server starts each enabled machine and runs the benchmarks configured for each of the machines. In theory, this means that you should just have to run `./server.py` and the script will do the rest.

In practice, you may want to manually edit some configuration files to change what is being run in each machine.

### Machine Configuration Files

Each machine has a yaml file with some important configuration. Usually, they can be found in `machines_cfgs`, such as the `rasp4coral.yaml` file.

In each configuration file, the most important parameters are:

- **ip**: this is the IP of the DUT. Again, this value must match what is configured in the DUT, otherwise the server will not be able to run benchmarks.

- **receive_port**: this value must match what is configured in the **DUT** (`/etc/radiation-benchmarks.conf`), otherwise the server will not receive logs (and will keep rebooting the benchmark/device).

- **json_files**: this is a list of benchmarks that the server will run. In theory, you can leave them all up and the server will run each of them for about an hour. **In practice, we recommend commenting out every file *EXCEPT* the one you want to run**. You must do this for each of the machines.

### Server Configuration Files

The server reads a configuration file with important parameters (default: `server_parameters.yaml`, but it can be changed with `./server.py -c <config_file>`). Luckily, there are not many parameters (at the moment), of which the most important ones are:

- server_ip: this is the IP the server will use to listen to log files. It must match the `server_ip` configuration of the DUTs (`/etc/radiation-benchmarks.conf` of each DUT).

- machines: this is a list of machines (DUTs), where you must specify the yaml configuration file for each machine in `cfg_file`. You can enable or disable a machine by changing the respective `enabled` field.

### Benchmark Files (JSON)

These files tell the server what command(s) to run for the benchmark. Again, the YAML file for each machine is responsible for configuring *which* benchmarks to run, but the benchmarks themselves are detailed in the JSON files (the ones explained here). These are relatively simple files, and multiple machines can use them (so you only need one per benchmark, not one per machine x benchmark). For this experiment, you should not need to edit any of these, but it is nice to know what is happening:

- killcmd: this command is used to kill any remaining benchmark scripts before starting a new benchmark.

- exec: this is the command that will start the benchmark. Include all the arguments needed here, preferably with absolute paths.

- codename: this is just a name that will appear in the server-side log file. The logs can be found in the directory configured in the server parameters (default: `logs`), with a subdirectory created for each machine (based on the name configured in the machine yaml file).

- header: this will show up in each log file that runs this benchmark. It is nice to include some descriptive information, such as important parameters (e.g., which model we are running).