from hpc_stan import SLURMClusterBridgeStan

cluster_kwargs = {
    'cores': 2,
    'memory': '2GB',
    'processes': 1,
    'queue': 'regular',
    'walltime': '02:00:00',
    'interface': 'ib0',
    'local_directory': '/scratch',
    'job_extra': ['-M myemail@my.domain', '-m abe'],
    'env_extra': ['export LANG="en_US.utf8"',
                  'export LC_ALL="en_US.utf8"']
}

stan_file_path = "/path/to/your/model.stan"
data_path = "/path/to/your/data.json"

cluster = SLURMClusterBridgeStan(stan_file_path, cluster_kwargs)
cluster.compile_model()
cluster.fit_model(data_path)

summary = cluster.get_summary()
print(summary)
