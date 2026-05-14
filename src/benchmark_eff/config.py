from easydict import EasyDict


class BaseConfig:
    d_model = 1024
    d_head = 64
    num_head = 16
    bias = False
    ln = "rmsnorm"
    init = EasyDict({
        "_name_": "fixed",
        "initializer_range": 0.02
    })

    def to_dict(self):
        return EasyDict(
            {k: getattr(self, k) for k in dir(self) \
                if not k.startswith('__') and not callable(getattr(self, k))})


class Config1(BaseConfig):
    d_model = 1024
    d_head = 64
    num_head = 16
    bias = False
    init = EasyDict({
        "_name_": "fixed",
        "initializer_range": 0.02
    })


class Config2(BaseConfig):
    d_model = 2048
    d_head = 128
    num_head = 16
    bias = False
    init = EasyDict({
        "_name_": "fixed",
        "initializer_range": 0.02
    })
    ln = "rmsnorm"


class Config3(BaseConfig):
    d_model = 4096
    d_head = 128
    num_head = 32
    bias = False
    init = EasyDict({
        "_name_": "fixed",
        "initializer_range": 0.02
    })
    ln = "rmsnorm"
