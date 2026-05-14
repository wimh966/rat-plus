import torch
from easydict import EasyDict
from ..task.task import LMTask


def convert_from_sequence_models(task: LMTask, pretrained_path):
    # for mid training, and eval
    ckpt = torch.load(pretrained_path, map_location="cuda")
    state_dict = ckpt['task']
    new_state_dict = {}
    for k, v in state_dict.items():
        if "in_proj" in k:
            d_model = v.shape[1]
            fgate = k.replace("in_proj", "fgate_proj")
            if v.shape[0] == 4 * d_model and fgate in state_dict and state_dict[fgate].shape[0] == d_model:
                v = torch.cat([v, state_dict[fgate]], dim=0)
        if "fgate_proj" in k:
            continue
        if "_orig_mod." in k:
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v
    task.load_state_dict(new_state_dict)


def convert(task, pretrained_path):
    torch.serialization.add_safe_globals([EasyDict])
    if not isinstance(task, LMTask):
        raise NotImplementedError
    if "sequence_model" in pretrained_path:
        convert_from_sequence_models(task, pretrained_path)
    else:
        raise NotImplementedError