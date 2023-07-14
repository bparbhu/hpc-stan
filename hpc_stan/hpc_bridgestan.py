import dask_jobqueue
import dask.distributed
from bridgestan import StanModel, Sampling

class HPCBridgeStanBase:
    def __init__(self, stan_file_path, model_data=None, seed=1234, capture_stan_prints=True,
                 stanc_args=[], make_args=[]):
        self.stan_file_path = stan_file_path
        self.model_data = model_data
        self.seed = seed
        self.capture_stan_prints = capture_stan_prints
        self.stanc_args = stanc_args
        self.make_args = make_args
        self.cluster_class = None
        self.cluster_kwargs = {}
        self.cluster = None
        self.client = None
        self.model = None

    def _setup_hpc(self):
        self.cluster = self.cluster_class(**self.cluster_kwargs)
        self.client = dask.distributed.Client(self.cluster)
        self.cluster.scale(jobs=2)  # scale based on your needs

    def compile_model(self):
        self._setup_hpc()
        self.model = StanModel.from_stan_file(
            self.stan_file_path,
            self.model_data,
            stanc_args=self.stanc_args,
            make_args=self.make_args,
            seed=self.seed,
            capture_stan_prints=self.capture_stan_prints
        )

    def fit_model(self, data, **kwargs):
        if self.model is None:
            raise Exception("Model not compiled. Call `compile_model` first.")
        self.fit = self.model.sample(data, **kwargs)

    def get_summary(self):
        if self.fit is None:
            raise Exception("No fit available. Call `fit_model` first.")
        summary = self.fit.summarize()
        return summary

    def log_density(self, theta_unc, propto=True, jacobian=True):
        future = self.client.submit(self.model.log_density, theta_unc, propto=propto, jacobian=jacobian)
        return future

    def log_lik(self, theta_unc):
        future = self.client.submit(self.model.log_lik, theta_unc)
        return future

    def log_prior(self, theta_unc):
        future = self.client.submit(self.model.log_prior, theta_unc)
        return future


class PBSClusterBridgeStan(HPCBridgeStanBase):
    def __init__(self, stan_file_path, model_data=None, seed=1234, capture_stan_prints=True, stanc_args=[], make_args=[], cluster_kwargs={}):
        super().__init__(stan_file_path, model_data, seed, capture_stan_prints, stanc_args, make_args)
        self.cluster_class = dask_jobqueue.PBSCluster
        self.cluster_kwargs = cluster_kwargs


class SLURMClusterBridgeStan(HPCBridgeStanBase):
    def __init__(self, stan_file_path, model_data=None, seed=1234, capture_stan_prints=True, stanc_args=[], make_args=[], cluster_kwargs={}):
        super().__init__(stan_file_path, model_data, seed, capture_stan_prints, stanc_args, make_args)
        self.cluster_class = dask_jobqueue.SLURMCluster
        self.cluster_kwargs = cluster_kwargs


class MOABClusterBridgeStan(HPCBridgeStanBase):
    def __init__(self, stan_file_path, model_data=None, seed=1234, capture_stan_prints=True, stanc_args=[], make_args=[], cluster_kwargs={}):
        super().__init__(stan_file_path, model_data, seed, capture_stan_prints, stanc_args, make_args)
        self.cluster_class = dask_jobqueue.MOABCluster
        self.cluster_kwargs = cluster_kwargs


class LSFClusterBridgeStan(HPCBridgeStanBase):
    def __init__(self, stan_file_path, model_data=None, seed=1234, capture_stan_prints=True, stanc_args=[], make_args=[], cluster_kwargs={}):
        super().__init__(stan_file_path, model_data, seed, capture_stan_prints, stanc_args, make_args)
        self.cluster_class = dask_jobqueue.LSFCluster
        self.cluster_kwargs = cluster_kwargs


class SGEClusterBridgeStan(HPCBridgeStanBase):
    def __init__(self, stan_file_path, model_data=None, seed=1234, capture_stan_prints=True, stanc_args=[], make_args=[], cluster_kwargs={}):
        super().__init__(stan_file_path, model_data, seed, capture_stan_prints, stanc_args, make_args)
        self.cluster_class = dask_jobqueue.SGECluster
        self.cluster_kwargs = cluster_kwargs


class HTCondorClusterBridgeStan(HPCBridgeStanBase):
    def __init__(self, stan_file_path, model_data=None, seed=1234, capture_stan_prints=True, stanc_args=[], make_args=[], cluster_kwargs={}):
        super().__init__(stan_file_path, model_data, seed, capture_stan_prints, stanc_args, make_args)
        self.cluster_class = dask_jobqueue.HTCondorCluster
        self.cluster_kwargs = cluster_kwargs
