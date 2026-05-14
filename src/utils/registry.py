class Registry(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_register = True

    def get(self, key, default_value=None):
        if key not in self and default_value is None:
            raise KeyError("please select keys in {}".format(self.keys()))
        return super().get(key, default_value)

    def register(self, module_name: str = None, module=None):
        # used as function call
        if not self._is_register: return
        if module is not None:
            assert module_name is not None
            assert module_name not in self, "The module name has been used. Please select another one"
            self[module_name] = module
            return

        # used as decorator
        def register_fn(fn):  # fn should be module or function call
            if module_name is None:
                name = (fn.__name__).lower()
            else:
                name = module_name
            assert module_name not in self, "The module name has been used. Please select another one"
            self[name] = fn
            return fn

        return register_fn


head_registry = Registry()
embedding_registry = Registry()
pe_registry = Registry()
backbone_registry = Registry()
layer_registry = Registry()
init_registry = Registry()

task_registry = Registry()
paralleltask_registry = Registry()
metric_registry = Registry()

# nn registry
activation_registry = Registry()
norm_registry = Registry()

# data registry
data_registry = Registry()

# optimizer and lr_scheduler
optimizer_registry = Registry()
lr_scheduler_registry = Registry()


def get_all_registries():
    return [
        head_registry,
        embedding_registry,
        backbone_registry,
        layer_registry,
        task_registry,
        metric_registry,
        activation_registry,
        norm_registry,
        data_registry,
        optimizer_registry,
        lr_scheduler_registry,
        pe_registry,
        paralleltask_registry,
    ]
