# test_hpc_stan_pystan.py
import pytest
import json
from hpc_stan import PyStanSLURMCluster, PyStanPBSCluster, PyStanSGECluster, \
    PyStanSLURMCluster, PyStanPBSCluster, PyStanSGECluster, PyStanLSFCluster, PyStanOARCluster

# Load the Stan model code from the Bernoulli file
with open("/path/to/bernoulli.stan", 'r') as file:
    stan_model_code = file.read()

# Load the data from the JSON file into a dictionary
with open("/path/to/bernoulli.data.json", 'r') as file:
    stan_data = json.load(file)

def test_run_stan_model_pystan_slurm():
    # Arrange
    cluster = PyStanSLURMCluster(stan_model_code, stan_data)

    # Act
    result = cluster.run_stan_model_pystan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_pystan_pbs():
    # Arrange
    cluster = PyStanPBSCluster(stan_model_code, stan_data)

    # Act
    result = cluster.run_stan_model_pystan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_pystan_sge():
    # Arrange
    cluster = PyStanSGECluster(stan_model_code, stan_data)

    # Act
    result = cluster.run_stan_model_pystan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')


def test_run_stan_model_pystan_slurm():
    # Arrange
    cluster = PyStanSLURMCluster(stan_model_code, stan_data)

    # Act
    result = cluster.run_stan_model_pystan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_pystan_pbs():
    # Arrange
    cluster = PyStanPBSCluster(stan_model_code, stan_data)

    # Act
    result = cluster.run_stan_model_pystan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_pystan_sge():
    # Arrange
    cluster = PyStanSGECluster(stan_model_code, stan_data)

    # Act
    result = cluster.run_stan_model_pystan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_pystan_lsf():
    # Arrange
    cluster = PyStanLSFCluster(stan_model_code, stan_data)

    # Act
    result = cluster.run_stan_model_pystan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_pystan_oar():
    # Arrange
    cluster = PyStanOARCluster(stan_model_code, stan_data)

    # Act
    result = cluster.run_stan_model_pystan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')
