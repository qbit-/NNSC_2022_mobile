# NNSC_2022_mobile
Running PyTorch models on mobile phone
## Brief description
  - This code provides scripts to profile neural networks on Android devices
  - Compression may be done using [MusCo](https://github.com/juliagusak/musco-pytorch) toolkit.

## Repository structure
  - `./device_profiling` helper code to profile models on phones
  - `./doc` Markdown documentation
  
## Dependencies
- see `env_docker.yaml`. You can install requirements with conda: 
```sh 
conda create -n nnsc_2022_mobile --file env_docker.yaml`
```
- you can also use a docker image `qbit271/mmsc_2022_mobile` or build your own here


### Benchmarking on mobile phones
1. You need to first download or compile an appropriate PyTorch benchmarking binary, instructions can be found [here](https://github.com/qbit-/NNSC_2022_mobile/tree/main/doc/building_arm_benchmark.md). Otherwise, you can download the one we prebuilt for you from <https://github.com/qbit-/NNSC_2022_mobile/tree/main/bin>. For the prebuilt binary to work with your models, you need to use Pytorch 1.7.1 to build your models.

2. Copy the benchmark binary to `/data/local/tmp/speed_benchmark_torch` on the device. Check that the device is accessible:
```sh
adb devices
```
copy the profiler to the device and make it executable
```sh
adb push speed_benchmark_torch-$ANDROID_ABI /data/local/tmp/speed_benchmark_torch
adb shell chmod +x /data/local/tmp/speed_benchmark_torch
```
3. Check that you can execute the benchmarking binary
using ADB:
```sh
adb shell /data/local/tmp/speed_benchmark_torch --help
```
Now you can use functions in `./device_profiling` to automate profiling.
See the [example](https://github.com/qbit-/NNSC_2022_mobile/tree/main/example.ipynb)

	
