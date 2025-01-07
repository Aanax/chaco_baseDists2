import json
import logging
import os

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object

def ensure_shared_grads(model, shared_model, gpu=False):
    for name, param in model.named_parameters():
        if param.grad is None:
            print("GRAD NONE ", name, param.grad)

    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()

class Player:
    def __init__(self, args):
        self.args = args
        if self.args["Player"]["action_sample"] == "max":
            self.get_probs = self.get_probs_max
        elif self.args["Player"]["action_sample"] == "multinomial":
            self.get_probs = self.get_probs_multinomial
        else:
            raise ValueError("Invalid action_sample method specified")

    def get_probs_max(self):
        pass

    def get_probs_multinomial(self):
        pass