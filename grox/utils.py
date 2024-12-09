from pathlib import Path

import jax
import orbax.checkpoint as ocp


def convert_str_to_int(d):
    new_d = {}
    for key, value in d.items():
        if isinstance(key, str):
            new_key = int(key) if key.isdigit() else key
        else:
            new_key = key

        if isinstance(value, dict):
            new_value = convert_str_to_int(value)
        else:
            new_value = value

        new_d[new_key] = new_value

    return new_d


def convert_keys_to_arrays(d):
    new_d = {}
    for key, value in d.items():
        if isinstance(value, jax._src.prng.PRNGKeyArray):
            new_value = jax.random.key_data(value)
        elif isinstance(value, dict):
            new_value = convert_keys_to_arrays(value)
        else:
            new_value = value

        new_d[key] = new_value

    return new_d


def initialize_checkpoint_manager(
    ckpt_path: str, metadata: dict, max_to_keep: int = 10
):
    ckpt_dir = Path(ckpt_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=True,
    )
    return ocp.CheckpointManager(ckpt_dir, options=options, metadata=metadata)


def restore_checkpoint(restored):
    rng = jax.random.wrap_key_data(restored["rng"])
    params = restored["params"]
    opt_state = restored["opt_state"]
    total_step = restored["step"]
    epoch = restored["epoch"]

    return params, opt_state, rng, total_step, epoch
