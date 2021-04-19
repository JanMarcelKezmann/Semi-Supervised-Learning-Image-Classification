# Imports
import os
import yaml

from absl import logging


def setup_tf(tf_cpp_min_log_level="2", log_error=logging.ERROR):
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_cpp_min_log_level
	logging.set_verbosity(log_error)



def load_config(args):
	"""
	Adds dataset and dataset size specific configurations to args.

	Args:
		args:   dictionary, containing parser arguments and its values

	Returns
		Updated dictionary of args taking dataset specific changes into account
	"""
	# # Get directory name of real path
	# dir_path = os.path.dirname(os.path.realpath(__file__))
	# # Set Config Path
	# config_path = os.path.join(dir_path, args["config_path"])
	# Set Config Path
	config_path = f"{args['config_path']}/{args['dataset']}@{args['num_lab_samples']}.yaml"

	with open(config_path, "r") as config_file:
		config = yaml.load(config_file, Loader=yaml.FullLoader)

	for key in args.keys():
		if key in config.keys():
			args[key] = config[key]

	return args