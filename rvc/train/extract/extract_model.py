import os
import traceback
from collections import OrderedDict

import torch


def replace_keys_in_dict(d, old_key_part, new_key_part):
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = key.replace(old_key_part, new_key_part)
        if isinstance(value, dict):
            value = replace_keys_in_dict(value, old_key_part, new_key_part)
        updated_dict[new_key] = value
    return updated_dict


def extract_model(hps, ckpt, epoch, step, final_save):
    if final_save:
        filename = f"{hps.model_name}.pth"
        filepath = os.path.join(hps.model_dir, filename)
    else:
        weights_dir = os.path.join(hps.model_dir, "weights")
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir, exist_ok=True)

        filename = f"{hps.model_name}_e{epoch}_s{step}.pth"
        filepath = os.path.join(weights_dir, filename)

    try:
        opt = OrderedDict(weight={key: value.half() for key, value in ckpt.items() if "enc_q" not in key})
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sample_rate,
        ]

        # Основные метаданные модели
        opt["model_name"] = hps.model_name
        opt["epoch"] = epoch
        opt["step"] = step
        opt["sr"] = hps.data.sample_rate
        opt["f0"] = True
        opt["version"] = "v2"
        opt["vocoder"] = hps.model.vocoder

        # Дополнительные метаданные
        opt["learning_environment"] = "rfyfkTrain"

        # Сохранение модели
        torch.save(
            replace_keys_in_dict(
                replace_keys_in_dict(opt, ".parametrizations.weight.original1", ".weight_v"),
                ".parametrizations.weight.original0",
                ".weight_g",
            ),
            filepath,
        )

        return f"Модель '{filename}' успешно сохранена!"
    except Exception:
        return f"Ошибка при сохранении модели: {traceback.format_exc()}"
