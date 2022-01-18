"""
This module collects profiling information from the
on-device execution of PyTorch models
"""
import os
import subprocess
from subprocess import CalledProcessError
import json
from glob import glob
import numpy as np

DEFAULT_PROF_CONFIG = {
    "iter": 5,
    "caffe2_threadpool_android_cap": 1,
    "warmup": 5,
    "input_type": "float",
    "vulkan": False,
    "report_pep": True,
    "use_caching_allocator": True,
    "caffe2_threadpool_force_inline": False,
}
DEFAULT_ADB_CMD = 'adb-1.0.39'


def check_device(verbose=False, *,
                 adb_cmd=DEFAULT_ADB_CMD,
                 device_serialno=None):
    """
    Checks if Android device is online, possibly verifies serial number
    """
    try:
        if device_serialno is None:
            cmd = f"{adb_cmd} shell getprop ro.serialno"
        else:
            cmd = f"{adb_cmd} -s {device_serialno} shell getprop ro.serialno",
        
        res = subprocess.run(
            cmd,
            shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except CalledProcessError as err:
        print('Command failed with:')
        print(err.stderr)
        raise err

    serialno = res.stdout.decode('utf-8').strip()
    if verbose:
        print(f'Found device: {serialno}')

    return True


def set_bit(arg, index, val):
    """
    Set the index:th bit of arg to 1 if val is truthy,
    else to 0, and return the new value.
    """
    mask = 1 << index  # Compute mask with just bit 'index' set
    arg &= ~mask      # Clear the bit indicated by the mask
    if val:
        arg |= mask   # If val is True, set the bit indicated by the mask
    return arg


def run_on_device(torchscript_filename, res_filename='', *,
                  use_bundled_input=-1,
                  input_dims=None,
                  input_type=None,
                  prof_config=DEFAULT_PROF_CONFIG,
                  cpu_affinity=None,
                  adb_cmd=DEFAULT_ADB_CMD,
                  device_serialno=None,
                  verbose=False):
    '''
    This function runs a TorchScript model on a device an saves the results
    to a file. The options are supplied in a dictionary prof_config.
    Default prof_config is provided by this module in DEFAULT_PROF_CONFIG

    Parameters:
    -----------
    torchscript_filename: str, path to the TorchScript model file
    res_filename: str, default ''. File name to save the **RAW** profiler output.
                  If not provided, no file is written.
    use_bundled_input: int, default -1. The index of the input bundled with the
                       model to use. To add bundled inputs to the model, use
                       torch.utils.bundled_inputs.augment_model_with_bundled_inputs
                       The value may be aken from prof_config
    input_dims: List[List[int]], default None. Input diensions of the model.
                Should be provided in case no bundled inputs are available.
                The value may be taken from prof_config, string format:
                "dim11,dim12,dim13\;dim21,dim22". Dimensions are coma separated,
                different inputs are semicolon separated
    input_type: List[[str]], default None. Input types. Must be provided with
                input_dims in case no bundled inputs are available.
                The value may be taken from prof_config, string format:
                "type1\;type2". Different inputs are semicolon separated
    prof_config: dict, dictionary specifying options for the profiler
    adb_cmd: str, name of the abd command
    cpu_affinity: iterable, default None
                  Specifies on which cores the profiling should run.
                  0-based.
                  Enabling this option will run commands with `taskset`.
                  On some systems this drastically reduces performance.
                  Note also that len(cpu_affinity) should equal
                  to the number of threads in config; otherwise,
                  performance degradation will occur/incorrect profiling
                  results will be reported.

    Returns:
    --------
    output: str, output of the profiler
    '''
    # copy model file to device
    torchscript_basename = os.path.basename(torchscript_filename)
    torchscript_path = os.path.abspath(torchscript_filename)
    if device_serialno is None:
        cmd_prefix = f"{adb_cmd}"
    else:
        cmd_prefix = f"{adb_cmd} -s {device_serialno}"

    command = f"{cmd_prefix} push {torchscript_path} /data/local/tmp/{torchscript_basename}"
    if verbose:
        print(command)
    try:
        res = subprocess.run(command,
                             shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except CalledProcessError as err:
        print('Command failed with:')
        print(err.stderr)
        raise err

    # prepare cpu affinity information
    num_threads = prof_config.get("caffe2_threadpool_android_cap", 1)
    if cpu_affinity is not None:
        affinity_mask = 0
        for cpu_id in cpu_affinity:
            affinity_mask = set_bit(affinity_mask, cpu_id, 1)
        affinity_cmd = f'taskset -a {affinity_mask}'

        if len(cpu_affinity) < num_threads:
            print('Warning: num_threads < len(cpu_affinity).'
                  ' Incorrect results are possible')
    else:
        affinity_cmd = ""

    # update current graph name
    prof_config['model'] = f"/data/local/tmp/{torchscript_basename}"

    # check if bundled inputs were supplied with the model
    if use_bundled_input >= 0:
        prof_config['use_bundled_input'] = use_bundled_input
    # or use values from config
    elif prof_config.get('use_bundled_input') is not None:
        pass
    # use explicitly supplied input dims
    elif input_dims is not None and input_type is not None:
        prof_config['input_dims'] = '\;'.join(
            ','.join(str(dim) for dim in inp) for inp in input_dims)
        prof_config['input_type'] = '\;'.join(typ for typ in input_type)
    # or use the values from config
    elif prof_config.get('input_dims') is not None and prof_config.get('input_type') is not None:
        pass
    else:
        raise ValueError('Either use_bundled_input or input_dims'
                         ' should be specified')

    # prepare profiler options
    run_opts = ''
    for key, val in prof_config.items():
        opt = '--' + key + '=' + json.dumps(val).replace(r'\\\\', r'\\')
        run_opts += ' ' + opt

    command_prefix = f'{cmd_prefix} shell'
    command_exe = '/data/local/tmp/speed_benchmark_torch'

    command = ' '.join((command_prefix, affinity_cmd, command_exe,
                        run_opts))
    # execute the profiling
    if verbose:
        print(command)
    try:
        res = subprocess.run(
            command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except CalledProcessError as err:
        print('Command failed with:')
        print(err.stderr)
        raise err

    if res_filename:
        with open(res_filename, 'w') as fp:
            fp.write(res.stdout.decode('utf-8'))

    command_exe = 'rm'
    command_argument = f"/data/local/tmp/{torchscript_basename}"
    command = ' '.join((command_prefix, command_exe,
                        command_argument))
    if verbose:
        print(command)
    try:
        subprocess.run(
            command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except CalledProcessError as err:
        print('Command failed with:')
        print(err.stderr)
        raise err

    return res.stdout.decode('utf-8')


def parse_profiler_output(out_or_filename, is_file=True):
    """
    Parse the output of the TFLite profiler, return
    the results in a nested dictionary
    Parameters:
    -----------
    out_or_filename: str, either the output of the profiler or a filename
                     of the file containing it
    is_file: bool, default True. Specifies whether out_or_filename is a
             file
    Returns:
    --------
    results: dict, contains keys 'unit', 'avg' and 'std'.
    """
    if is_file:
        data = open(out_or_filename, 'r').read()
    else:
        data = out_or_filename

    # parse ugly data string by mutating it into a JSON string
    # remove caret symbols from possible string.
    # Caret symbols are not written to file
    data = data.replace("\r", "")  

    assert '\nMain runs.\n' in data
    main_data = data.split(
        '\nMain runs.\nPyTorchObserver')[1]
    main_data = main_data.split('\nMain run finished.')[0]
    main_data = ('['
                 + main_data.replace(
                     "\nPyTorchObserver", ",")
                 + ']'
    )
    main_results = json.loads(main_data)

    latencies = []
    unit = main_results[0]['unit']
    for elem in main_results:
        assert elem['unit'] == unit
        latencies.append(float(elem['value']))

    res = {'unit': unit, 'avg': float(np.mean(latencies)),
           'std': float(np.std(latencies))}

    return res


def batch_profile(torchscript_path, results_path,
                  *,
                  use_bundled_input=-1,
                  input_dims=None,
                  prof_config=DEFAULT_PROF_CONFIG,
                  cpu_affinity=(0,), adb_cmd=DEFAULT_ADB_CMD):
    """
    Profiles multiple TorchScript models specified by torchscript_path
    Collects results in results_path in json format
    """
    filenames = glob(torchscript_path + '*.pt')
    result_filenames = [
        os.path.join(
            results_path,
            os.path.basename(filename).split('.')[0] + '.json')
        for filename in filenames]

    for filename, result_filename in zip(filenames, result_filenames):
        print(f'profiling: {filename} => {result_filename}')
        res = run_on_device(
            filename, '',
            use_bundled_input=use_bundled_input,
            input_dims=input_dims,
            prof_config=prof_config,
            cpu_affinity=cpu_affinity, adb_cmd=adb_cmd)
        result = parse_profiler_output(res, is_file=False)
        json.dump(result, open(result_filename, 'w'))

