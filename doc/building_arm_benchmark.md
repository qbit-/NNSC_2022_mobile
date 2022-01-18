## Building Pytorch mobile benchmarking binary

Follow the instructions on building Pytorch from source on [the official page](https://github.com/pytorch/pytorch#from-source)
The following commands are necessary on Mac OS for successful building.

1. Install Android Studio and `adb`
    ```bash
    brew install --cask android-studio android-platform-tools
    ```
2. Open Android Studio and press `Configure->SDK Manager`. Install necessary SDK and NDK versions.
   For SDK we used `SDK Platforms->Android 9.0, Android 6.0`.
   For NDK, go to `SDK Tools`. Check `Show Package Details`.
   We used `NDK (Side by side)->21.0.6113669`

3. Create environment and install packages
    ```bash
    conda create -n bench
    conda install -n bench numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
    ```
2. Specific for Mac OS:
    ```bash
    conda install -n bench pkg-config libuv
    ```
3. Clone the source repo. We used version `v1.7.1`
    ```bash
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    git checkout v1.7.1
    # if you are updating an existing checkout
    git checkout v1.7.1
    git submodule sync
    git submodule update --init --recursive
    ```
4. Configure environment. The version of ABI can be found from `abd shell cat /proc/cpuinfo`.
    ```bash
    conda activate bench
    export ANDROID_ABI=arm64-v8a # default is armeabi-v7a with neon
    export ANDROID_NDK=/path/to/Android/Sdk/ndk/21.0.6113669/
    ```
5. Compile the TorchScript benchmark
    ```bash
    ./scripts/build_android.sh \
    -DBUILD_BINARY=ON \
    -DBUILD_CAFFE2_MOBILE=OFF \
    -DANDROID_ARM_NEON=ON \
    -DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
    -DPYTHON_EXECUTABLE=$(python -c 'import sys; print(sys.executable)')
    ```
	The executable will be produced at `./build_android/bin/speed_benchmark_torch`

6. Compile the ONNX benchmark. First go to the previous release
   ```bash
   python setup.py clean
   git checkout v1.6.0
   git submodule update --init --recursive
   ```
   Next, compile protobuf by hand due to the bug in build scripts
   ```bash
   ./scripts/build_host_protoc.sh
   ```
   Finally, launch the compilation
   ```bash
   ./scripts/build_android.sh  \
   -DBUILD_BINARY=ON \
   -DBUILD_CAFFE2_MOBILE=ON \
   -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=`pwd`/build_host_protoc/bin/protoc \
   -DONNX_CUSTOM_PROTOC_EXECUTABLE=`pwd`/build_host_protoc/bin/protoc \
   -DANDROID_ARM_NEON=ON \
   -DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
   -DPYTHON_EXECUTABLE=$(python -c 'import sys; print(sys.executable)')
   ```
   Copy the exacultable from `./build_android/bin/speed_benchmark`

6. Prepare example torchscript file for benchmark
    ```python
    #file generate_example_model_pt.py
	import torch
	import torchvision
	from torch.utils.bundled_inputs import (
		augment_model_with_bundled_inputs)
	from torch.utils.mobile_optimizer import optimize_for_mobile

	model = torchvision.models.resnet18(pretrained=True)
	model.eval()
	example = torch.zeros(1, 3, 224, 224)
	script_module = torch.jit.trace(model, example)
	script_module_optimized = optimize_for_mobile(script_module)
    augment_model_with_bundled_inputs(script_module_optimized, [(example,)])
	torch.jit.save(script_module_optimized, "./resnet18.pt")
    ```
Make the binary `python generate_example_model_pt.py`

6. Transfer benchmark executable and the example file to the Android phone
    ```bash
    adb push ./build_android/bin/speed_benchmark_torch /data/local/tmp
    adb push resnet18.pt /data/local/tmp
    ```
7. Run benchmark
    ```bash
    adb shell \
    /data/local/tmp/speed_benchmark_torch \
    --model /data/local/tmp/resnet18.pt \
    --use_bundled_input=0 \
    --input_type=float --warmup=5 --iter 3 \
    --report_pep=true
    ```
    Output:
    ```
    Starting benchmark.
    Running warmup runs.
    Main runs.
    PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "599746"}
    PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "570505"}
    PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "583396"}
    Main run finished. Microseconds per iter: 584558. Iters per second: 1.71069
    ```


