# hpc_stan_pystan.py
from abc import ABC, abstractmethod
from dask.distributed import Client
from dask_jobqueue import SLURMCluster, PBSCluster, \
    SGECluster, LSFCluster, OARCluster, MOABCluster, HTCondorCluster
import pystan


class BaseClusterPyStan(ABC):
    def __init__(self, stan_model_code, stan_data):
        self.stan_model_code = stan_model_code
        self.stan_data = stan_data

    @abstractmethod
    def create_cluster(self):
        pass

    def run_stan_model_pystan(self):
        # Your task (running the Stan model)
        def run_task():
            # compile the model
            model = pystan.StanModel(model_code=self.stan_model_code)

            # fit the model
            fit = model.sampling(data=self.stan_data)

            return fit

        # Connect Dask to the cluster
        client = Client(self.create_cluster())

        # Use Dask to run the task on the cluster
        futures = client.map(run_task, range(10))  # run the task 10 times

        # Gather results (this will block until all tasks are done)
        results = client.gather(futures)

        return results


# Implement specific cluster types

class PyStanSLURMCluster(BaseClusterPyStan):
    def create_cluster(self):
        return SLURMCluster()


class PyStanPBSCluster(BaseClusterPyStan):
    def create_cluster(self):
        return PBSCluster()


class PyStanSGECluster(BaseClusterPyStan):
    def create_cluster(self):
        return SGECluster()


class PyStanLSFCluster(BaseClusterPyStan):
    def create_cluster(self):
        return LSFCluster()


class PyStanOARCluster(BaseClusterPyStan):
    def create_cluster(self):
        return OARCluster()


class PyStanMOABCluster(BaseClusterPyStan):
    def create_cluster(self):
        return MOABCluster()


class PyStanHTCondorCluster(BaseClusterPyStan):
    def create_cluster(self):
        return HTCondorCluster()
