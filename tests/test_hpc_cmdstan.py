# test_hpc_stan_cmdstan.py
import pytest
from hpc_stan import CmdStanSLURMCluster, CmdStanPBSCluster, \
    CmdStanSGECluster,  CmdStanPySLURMCluster, CmdStanPyPBSCluster, \
    CmdStanPySGECluster, CmdStanPyLSFCluster, CmdStanPyOARCluster


def test_run_stan_model_cmdstan_slurm():
    # Arrange
    model_path = "/path/to/bernoulli.stan"
    data_path = "/path/to/bernoulli.data.json"

    cluster = CmdStanSLURMCluster(model_path, data_path)

    # Act
    result = cluster.run_stan_model_cmdstan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_cmdstan_pbs():
    # Arrange
    model_path = "/path/to/bernoulli.stan"
    data_path = "/path/to/bernoulli.data.json"

    cluster = CmdStanPBSCluster(model_path, data_path)

    # Act
    result = cluster.run_stan_model_cmdstan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_cmdstan_sge():
    # Arrange
    model_path = "/path/to/bernoulli.stan"
    data_path = "/path/to/bernoulli.data.json"

    cluster = CmdStanSGECluster(model_path, data_path)

    # Act
    result = cluster.run_stan_model_cmdstan()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')


def test_run_stan_model_cmdstanpy_slurm():
    # Arrange
    stan_file = "/path/to/bernoulli.stan"
    data_file = "/path/to/bernoulli.data.json"
    cluster = CmdStanPySLURMCluster(stan_file, data_file)

    # Act
    result = cluster.run_stan_model_cmdstanpy()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_cmdstanpy_pbs():
    # Arrange
    stan_file = "/path/to/bernoulli.stan"
    data_file = "/path/to/bernoulli.data.json"
    cluster = CmdStanPyPBSCluster(stan_file, data_file)

    # Act
    result = cluster.run_stan_model_cmdstanpy()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_cmdstanpy_sge():
    # Arrange
    stan_file = "/path/to/bernoulli.stan"
    data_file = "/path/to/bernoulli.data.json"
    cluster = CmdStanPySGECluster(stan_file, data_file)

    # Act
    result = cluster.run_stan_model_cmdstanpy()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_cmdstanpy_lsf():
    # Arrange
    stan_file = "/path/to/bernoulli.stan"
    data_file = "/path/to/bernoulli.data.json"
    cluster = CmdStanPyLSFCluster(stan_file, data_file)

    # Act
    result = cluster.run_stan_model_cmdstanpy()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')

def test_run_stan_model_cmdstanpy_oar():
    # Arrange
    stan_file = "/path/to/bernoulli.stan"
    data_file = "/path/to/bernoulli.data.json"
    cluster = CmdStanPyOARCluster(stan_file, data_file)

    # Act
    result = cluster.run_stan_model_cmdstanpy()

    # Assert
    assert len(result) == 10
    for fit in result:
        assert hasattr(fit, 'summary')
        assert hasattr(fit, 'diagnose')
