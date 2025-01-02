"""
Install pytorch, pytorch-geometric and the required packages in requirements.txt.
The focus is on torch and pytorch geometric and the purpose is to guaranty correct CUDA version.
"""

from uuid import uuid4
from argparse import ArgumentParser
from typing import Optional
import platform
import subprocess
import os
import re


def get_cuda_version() -> Optional[int]:
    """
    Get the CUDA version.
    :return: None if CUDA is not found, otherwise the CUDA version as integer.
    Defaults to 118, lowest supported at torch installation page at https://pytorch.org/get-started/locally/ as
    for Aug 2024.
    """
    try:
        output = my_run('nvidia-smi').stdout
        match = re.search(r'CUDA Version: (\d+\.\d+)', output)
        if match:
            version = match.group(1)
            version_int = int(float(version) * 10)
            if version_int > 124:
                version_int = 124
            elif version_int > 121:
                version_int = 121
            else:
                version_int = 118
            print(f'Found CUDA version: {float(version_int) / 10}. Using cu{version_int}')
            return version_int
    except subprocess.CalledProcessError:
        print('nvidia-smi is not installed or not found')
    print('Did not find CUDA version. Defaulting to CPU')
    return None


def torch_command(cuda_version: Optional[int]) -> str:
    """
    Get the command to install torch based on the CUDA version.
    :param cuda_version: CUDA version.
    :return: The command to install torch.
    """

    predefined_torch_version = os.getenv('TORCH_VERSION')

    if cuda_version is None:
        torch_url = 'https://download.pytorch.org/whl/cpu'
    else:
        torch_url = f'https://download.pytorch.org/whl/cu{cuda_version}'

    if predefined_torch_version:
        torch_cmd = (
            f'pip install torch=={predefined_torch_version} torchvision torchaudio'
        )
    else:
        torch_cmd = 'pip install torch torchvision torchaudio'

    if torch_url:
        torch_cmd += f' --index-url {torch_url}'

    return torch_cmd


def get_installed_torch_version() -> str:
    """
    Resolve the torch version from the installed packages.
    :return: The torch version.
    """
    result = my_run('pip freeze')
    for line in result.stdout.split('\n'):
        match = re.match(r'torch==(\d+\.\d+\.\d+)', line)
        if match:
            installed_torch_version = match.group(1)
            break
    else:
        raise ValueError('Could not find torch version')
    print(f'Found installed torch version: {installed_torch_version}')
    return installed_torch_version


def get_pyg_url_for_specific_cuda(cuda_version: int, torch_version: str) -> str:
    """
    Get the URL for PyG based on the CUDA version and torch version.
    :param cuda_version: CUDA version.
    :param torch_version: Torch version.
    :return: PyG URL
    """
    return f'https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html'


def my_run(command: str):
    """
    Run a command and print it.
    :param command: Command to run.
    :return: Command Result.
    """
    print(f'Running command: {command}')
    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    return result


def install_packages():
    """
    Install pytorch, pytorch-geometric and the required packages in requirements.txt.
    The focus is on torch and pytorch geometric and the purpose is to guaranty correct CUDA version.
    """

    # Check operating system
    os_type = platform.system()
    if os_type not in ['Linux', 'Windows']:
        raise OSError(f"Unsupported operating system: {os_type}")

    # Resolve CUDA version
    cuda_version = get_cuda_version()

    # Install torch
    my_run(torch_command(cuda_version))

    # Install PyG
    pyg_url = get_pyg_url_for_specific_cuda(cuda_version, get_installed_torch_version())
    my_run('pip install torch_geometric')
    my_run(f'pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f {pyg_url}')

    if os.path.exists('requirements.txt'):
        # Install other packages
        my_run('pip install -r requirements.txt')


def create_conda_env(conda_env: str):
    """
    Create a new Conda environment.
    :param conda_env: New Conda environment name.
    """
    print(f'Creating conda env: {conda_env}')
    my_run(f'conda create -n {conda_env}')
    my_run(f'conda activate {conda_env}')
    my_run('conda install pip')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--new-conda-env', action='store_false', help='Create a new Conda environment')
    parser.add_argument('--conda-env', type=str, default=f'pyg_env_{uuid4()}', help='New Conda environment name')
    args = parser.parse_args()
    if args.new_conda_env:
        create_conda_env(args.conda_env)
    install_packages()
    if args.new_conda_env:
        print(f'Installed packages to conda env: {args.conda_env}')
    else:
        print('Installed packages to current conda environment')