import yaml
import pipes
import itertools
from .tools import DotDict

custom_config = {}

# Merge data structures
def merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        d = dict(a)
        d.update({k: merge(a.get(k, None), b[k]) for k in b})
        return d

    if isinstance(a, list) and isinstance(b, list):
        return [merge(x, y) for x, y in itertools.zip_longest(a, b)]

    return a if b is None else b


# Read config file, keep env
def read_file_keep_env(filename):
    f = open(filename, "r")
    out = ""
    for line in f:
        if "#env" in line:
            out += line + "\n"
    return out


def data_to_env(data, prefix):
    env = ""
    for k, v in data.items():
        if type(v) == dict:
            env += data_to_env(v, prefix + k + "_")
        else:
            k = pipes.quote(prefix + k)
            v = pipes.quote(str(v))
            env += "%s=%s\n" % (k, v)
    return env

# Convert YAML config to .env file
def export_to_env():

    # Load config files
    # config = yaml.load(read_file_keep_env("config.yml"))
    # config_priv = yaml.load(read_file_keep_env("config-private.yml"))
    # config = merge({}, merge(config, config_priv))
    config = get_dict()
    envFile = data_to_env(config, "")
    open(".env", "w").write(envFile)

# return config
def get_dict():
    config = yaml.load(open("config.yml", "r"))
    #config_priv = yaml.load(open("config-private.yml", "r"))

    #out = merge({}, merge(config, config_priv))
    out = merge({}, config)
    out = merge(out, custom_config)
    return out

def set_custom(config):
    global custom_config
    if (isinstance(config, dict)):
        custom_config = config

# return config
def get():
    return DotDict(get_dict())

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "env":
        export_to_env()
