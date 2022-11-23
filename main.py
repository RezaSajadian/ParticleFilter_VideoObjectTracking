
import os
from ParticleFilter_Module import ParticleFilter

conf_file_name = "configurations.ini"
cwd = os.getcwd()
config_file = os.path.join(cwd, conf_file_name)

if os.path.isfile(config_file):
    print("Main, Config file exists.")

pf = ParticleFilter(config_file)

pf.run()




