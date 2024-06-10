import json
import logging
import os


def setup_logger(logger_name, log_file, level=logging.INFO):
    #     if os.path.exists(logger_name):
    #         logger_name=logger_name[:-4]+"_new.log"

    l = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    fileHandler = logging.FileHandler(log_file, mode="w")
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, "r"))
    return json_object


# def ensure_shared_grads(model, shared_model, gpu=False):
#     for param, shared_param in zip(model.parameters(),
#                                    shared_model.parameters()):
#         if shared_param.grad is not None and not gpu:
#             return
#         elif not gpu:
#             shared_param._grad = param.grad
#         else:
#             shared_param._grad = param.grad.cpu()


def ensure_shared_grads(model, shared_model, gpu=False):

    #     for name, param in model.named_parameters():
    # #         if param.grad is not None:
    # #             print(name, param.grad.sum())
    #         if param.grad is None:
    #             print(name, param.grad)

    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            #             print("shared None here ", shared_param)
            # try:
            shared_param._grad = param.grad.cpu()
            # except:
            #   pass
