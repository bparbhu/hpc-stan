import pytest
from bridgestan import StanModel, Sampling
from hpc_bridge_stan import HPCBridgeStanBase, PBSClusterBridgeStan, \
    SLURMClusterBridgeStan, MOABClusterBridgeStan, LSFClusterBridgeStan, \
    SGEClusterBridgeStan, HTCondorClusterBridgeStan

def test_compile_model():
    model = HPCBridgeStanBase('tests/models/bernoulli.stan')
    model.compile_model()
    assert isinstance(model.model, StanModel)

@pytest.mark.parametrize('cluster_class', [
    PBSClusterBridgeStan,
    SLURMClusterBridgeStan,
    MOABClusterBridgeStan,
    LSFClusterBridgeStan,
    SGEClusterBridgeStan,
    HTCondorClusterBridgeStan
])
def test_cluster_classes(cluster_class):
    model = cluster_class('tests/models/bernoulli.stan')
    model.compile_model()
    assert isinstance(model.model, StanModel)

def test_fit_model():
    model = HPCBridgeStanBase('tests/models/bernoulli.stan')
    model.compile_model()
    model.fit_model({'N': 10, 'y': [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]})
    assert isinstance(model.fit, Sampling)

def test_get_summary():
    model = HPCBridgeStanBase('tests/models/bernoulli.stan')
    model.compile_model()
    model.fit_model({'N': 10, 'y': [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]})
    summary = model.get_summary()
    assert 'mean' in summary
    assert 'sd' in summary

def test_get_summary_without_fit():
    model = HPCBridgeStanBase('tests/models/bernoulli.stan')
    model.compile_model()
    with pytest.raises(Exception) as exc_info:
        model.get_summary()
    assert str(exc_info.value) == 'No fit available. Call `fit_model` first.'
