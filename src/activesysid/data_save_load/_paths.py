from pathlib import Path


def pickle_path(
    exp_type, is_noise, system_name="", model_name="", is_scale=1, is_const=0
):
    # Normalize bools and other truthy/falsy flag values so filenames always
    # use the stable public convention ``noise_0/1_scale_0/1_const_0/1``.
    flag_tokens = tuple(
        str(int(bool(value))) for value in (is_noise, is_scale, is_const)
    )
    filename = "_".join(
        [
            str(system_name),
            str(model_name),
            str(exp_type),
            "noise",
            flag_tokens[0],
            "scale",
            flag_tokens[1],
            "const",
            flag_tokens[2],
        ]
    )
    # Keep the historical underscore before the extension so existing files
    # remain readable (the old implementation joined ".pkl" as a name part).
    return (
        Path.cwd()
        / "example"
        / "experiments"
        / "artifacts"
        / "data"
        / f"{filename}_.pkl"
    )
