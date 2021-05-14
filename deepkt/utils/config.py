import os

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
from easydict import EasyDict
from pprint import pprint

from deepkt.utils.dirs import create_dirs


def setup_logging(log_dir):
    # log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : " \
    #                   "%(message)s in %(pathname)s:%(lineno)d"
    log_file_format = "[%(asctime)s: %(message)s"
    log_console_format = "%(message)s"

    # Main logger
    main_logger = logging.getLogger()
    if main_logger.hasHandlers():
        for handler in main_logger.handlers:
            main_logger.removeHandler(handler)

    main_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir),
                                           maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir),
                                                  maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.ERROR)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def process_config(json_file):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config, _ = get_config_from_json(json_file)

    # making sure that you have provided the exp_name.
    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(config.exp_name))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)

    print(" The Configuration of Current Experiment: ")
    print(config)

    # if new datasets are added, we need to specify the data loader
    if "EdNet" in config.data_name:
        config.data_loader = "EdNetDataLoader"
    elif "ASSISTments" in config.data_name:
        config.data_loader = "ASSISTmentsDataLoader"
    elif "MORF" in config.data_name:
        config.data_loader = "MORFDataLoader"
    elif "Junyi" in config.data_name:
        config.data_loader = "JunyiDataLoader"
    elif "FSAI" in config.data_name:
        config.data_loader = "FSAIDataLoader"
    else:
        raise ValueError("No correct data_name specified in configuration file.")

    # create some important directories to be used for that experiment.
    config.summary_dir = os.path.join("experiments",
                                      config.agent,
                                      config.data_name,
                                      config.exp_name,
                                      "summaries/")
    config.checkpoint_dir = os.path.join("experiments",
                                         config.agent,
                                         config.data_name,
                                         config.exp_name,
                                         "checkpoints/")
    config.out_dir = os.path.join("experiments",
                                  config.agent,
                                  config.data_name,
                                  config.exp_name,
                                  "out/")
    config.log_dir = os.path.join("experiments",
                                  config.agent,
                                  config.data_name,
                                  config.exp_name,
                                  "logs/")
    create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info("Hi, This is root.")
    logging.getLogger().info("After the configurations are successfully processed "
                             "and dirs are created.")
    logging.getLogger().info("The pipeline of the project will begin now.")

    return config
