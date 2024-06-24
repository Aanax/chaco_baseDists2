import configparser

def to_dict(config):
    """
    function converts a ConfigParser structure into a nested dict
    Each section name is a first level key in the the dict, and the key values of the section
    becomes the dict in the second level
    {
        'section_name': {
            'key': 'value'
        }
    }
    :param config:  the ConfigParser with the file already loaded
    :return: a nested dict
    """
    res = {}
    for section_name in config.sections():
        res[section_name]=dict(config[section_name])
    return res
        
args = configparser.ConfigParser()
args.optionxform = lambda option: option
args.read("/s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/configs/a3c_with_vae.ini")

args = to_dict(args)

print(args["Training"]["log_dir"])

