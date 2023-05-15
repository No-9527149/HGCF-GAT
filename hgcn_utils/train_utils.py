import argparse
from utils import log


def add_flags_from_config(_, parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    print(log.set_color('\n' + _, 'pink'))

    for param in config_dict:
        # 循环读取trainning_config/model_config/model_config中的参数、描述
        # default：parameter在config中设定的值
        # description：描述
        default, description = config_dict[param]
        # 检查default是否为dict
        try:
            # 检查default是否为dict
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            # 检查default是否为list
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        type=type(default[0]),
                        default=default,
                        help=description
                    )
                else:
                    pass
                    parser.add_argument(
                        f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(
                    f"--{param}", type=OrNone(default), default=default, help=description)
            param_str = log.set_color(param, 'yellow')
            equal_str = log.set_color(' = ', 'white')
            default_str = log.set_color(str(default), 'blue')
            print(param_str + equal_str + default_str)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser
