# hpc_stan_cmdstan.py
from abc import ABC, abstractmethod
from dask.distributed import Client
from dask_jobqueue import SLURMCluster, PBSCluster, \
    SGECluster, LSFCluster, OARCluster, MOABCluster, HTCondorCluster
from cmdstanpy import CmdStanModel


class BaseClusterCmdStan(ABC):
    def __init__(self, stan_model_path, stan_data_path):
        self.stan_model_path = stan_model_path
        self.stan_data_path = stan_data_path

    @abstractmethod
    def create_cluster(self):
        pass

    def run_stan_model_cmdstan(self):
        # Your task (running the Stan model)
        def run_task():
            # create model object, compile if necessary
            model = CmdStanModel(stan_file=self.stan_model_path)

            # specify data file
            data = self.stan_data_path

            # fit the model
            fit = model.sample(data=data)

            return fit

        # Connect Dask to the cluster
        client = Client(self.create_cluster())

        # Use Dask to run the task on the cluster
        futures = client.map(run_task, range(10))  # run the task 10 times

        # Gather results (this will block until all tasks are done)
        results = client.gather(futures)

        return results


# Implement specific cluster types

class CmdStanSLURMCluster(BaseClusterCmdStan):
    def create_cluster(self):
        return SLURMCluster()


class CmdStanPBSCluster(BaseClusterCmdStan):
    def create_cluster(self):
        return PBSCluster()


class CmdStanSGECluster(BaseClusterCmdStan):
    def create_cluster(self):
        return SGECluster()


class CmdStanLSFCluster(BaseClusterCmdStan):
    def create_cluster(self):
        return LSFCluster()


class CmdStanOARCluster(BaseClusterCmdStan):
    def create_cluster(self):
        return OARCluster()


class CmdStanMOABCluster(BaseClusterCmdStan):
    def create_cluster(self):
        return MOABCluster()


class CmdStanHTCondorCluster(BaseClusterCmdStan):
    def create_cluster(self):
        return HTCondorCluster()
