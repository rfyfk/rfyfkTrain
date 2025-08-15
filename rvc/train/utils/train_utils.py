import glob
import os
from collections import OrderedDict

import torch


def replace_keys_in_dict(d, old_key_part, new_key_part):
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        updated_dict[new_key] = replace_keys_in_dict(value, old_key_part, new_key_part) if isinstance(value, dict) else value
    return updated_dict


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_dict = replace_keys_in_dict(
        replace_keys_in_dict(checkpoint_dict, ".weight_v", ".parametrizations.weight.original1"),
        ".weight_g",
        ".parametrizations.weight.original0",
    )

    model_to_load = model.module if hasattr(model, "module") else model
    model_state_dict = model_to_load.state_dict()

    new_state_dict = {k: checkpoint_dict["model"].get(k, v) for k, v in model_state_dict.items()}
    model_to_load.load_state_dict(new_state_dict, strict=False)

    if optimizer and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))

    print(f"Загружена контрольная точка '{checkpoint_path}' (эпоха {checkpoint_dict['iteration']})", flush=True)
    return model, optimizer, checkpoint_dict.get("learning_rate", 0), checkpoint_dict["iteration"]


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }

    torch.save(
        replace_keys_in_dict(
            replace_keys_in_dict(checkpoint_data, ".parametrizations.weight.original1", ".weight_v"),
            ".parametrizations.weight.original0",
            ".weight_g",
        ),
        checkpoint_path,
    )

    print(f"Сохранен чекпоинт '{checkpoint_path}' (эпоха {iteration})", flush=True)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    checkpoints = sorted(glob.glob(os.path.join(dir_path, regex)))
    return checkpoints[-1] if checkpoints else None


def attempt_load_checkpoint_pair(net_g, optim_g, g_path, net_d, optim_d, d_path):
    if not (os.path.exists(g_path) and os.path.exists(d_path)):
        raise FileNotFoundError(f"Один или оба файла чекпоинта не найдены: {g_path}, {d_path}")

    # Загружаем оба чекпоинта и проверяем, совпадают ли эпохи
    _, _, _, epoch_d = load_checkpoint(d_path, net_d, optim_d)
    _, _, _, epoch_g = load_checkpoint(g_path, net_g, optim_g)

    if epoch_d != epoch_g:
        raise ValueError(f"Несоответствие эпох в чекпоинтах: G={epoch_g}, D={epoch_d}")

    return epoch_g


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)
