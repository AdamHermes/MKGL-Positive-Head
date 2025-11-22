import inspect

# Simple Registry that mirrors the .register decorator used in your code.
class Registry:
    def __init__(self):
        self._dict = {}

    def register(self, name):
        def decorator(cls):
            self._dict[name] = cls
            return cls
        return decorator

    def get(self, name):
        return self._dict.get(name)

R = Registry()

# A very small Configurable helper used by model.py:
class Configurable:
    """
    - instances should implement config_dict() which returns a dict:
        {'class': <class>, 'kwargs': {...}}
    - load_config_dict will re-instantiate the class using those kwargs.
    """
    def config_dict(self):
        # Try to build kwargs from common attribute names used in the repo
        kwargs = {}
        for key in ("input_dim", "output_dim", "num_relation", "query_input_dim",
                    "message_func", "aggregate_func", "layer_norm",
                    "activation", "dependent"):
            if hasattr(self, key):
                kwargs[key] = getattr(self, key)
        return {"class": self.__class__, "kwargs": kwargs}

    @classmethod
    def load_config_dict(cls, config):
        # If already an object, return it
        if not isinstance(config, dict):
            return config
        c = config.get("class")
        kwargs = config.get("kwargs", {})
        # If class has a signature mismatch it's the caller's responsibility.
        return c(**kwargs)
