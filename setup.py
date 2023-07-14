from setuptools import setup, find_packages

setup(
    name='hpc-stan',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'dask'
        'matplotlib',
        'pandas',
        'pystan',
        'cmdstanpy',
        'bridgestan',
        'dask-jobqueue',
    ],
    author='Brian Parbhu',
    author_email='brian.parbhu@gmail.com',
    description='A dask powered HPC/cloud friendly interface for BridgeStan, PyStan, and CmdStanPy',
    license='GPL-3.0',
    keywords='stan hpc BridgeStan CmdStanPy PyStan SLURM Dask',
    url='http://github.com/bparbhu/hpc-stan',
)
