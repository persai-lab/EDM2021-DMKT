"""
__author__ = "Chunpai W."
__email__ = "cwang25@albany.edu"
Description:
    1. Capture the config file
    2. Process the json config passed
    3. Create an agent instance
    4. Run the agent
"""

import argparse
from deepkt.utils.config import *
from deepkt.agents import *

if __name__ == '__main__':
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="DeepKT")
    arg_parser.add_argument('-c', '--config',
                            type=str, metavar='config_json_file',
                            default='configs/dmkt_morf686_best_0.json',
                            help='the configuration json file')
    args = arg_parser.parse_args()

    # parse the config json file
    # note that, if new data added, we should modify the process_config() accordingly.
    config = process_config(args.config)

    # Create the Agent and pass all the configuration to it then run it...
    # The globals() function returns a dictionary containing the variables
    # and classes defined in the global namespace
    agent_class = globals()[config.agent]

    # initialize the agent or model trainer that contains training and
    # validation, as well as finalizing and storing the checkpoints
    # if a new agent class is added,
    agent = agent_class(config)

    agent.run()
    agent.finalize()
