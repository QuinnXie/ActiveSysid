import dill as pickle
import os

import numpy as np

from ._paths import pickle_path


_FULL_FIELDS = (
    "samples",
    "scores",
    "system",
    "model",
    "pred",
    "exp_type",
    "AL_method_set",
    "delta_set",
    "alpha_set",
    "N_exp",
    "N_set",
    "N_train_init",
    "N_train_max",
    "N_test",
    "rho_x",
    "rho_th",
    "Qx_cov",
    "Qy_cov",
    "Qth_cov",
    "isScale",
    "isConst",
)


def _host_value(value):
    """Convert JAX/array-like leaves to pickle-stable host values."""
    if isinstance(value, dict):
        return {key: _host_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_host_value(item) for item in value]
        return converted if isinstance(value, list) else tuple(converted)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value)
    return value


def _system_snapshot(system, *, system_name):
    """Keep analysis metadata without serializing JIT-compiled callables."""
    return {
        "snapshot_version": 1,
        "system_name": system_name,
        "class_name": type(system).__name__,
        "nx": getattr(system, "nx", None),
        "ny": getattr(system, "ny", None),
        "nu": getattr(system, "nu", None),
        "Ts": getattr(system, "Ts", None),
        "temporality": getattr(system, "temporality", None),
        "qx": getattr(system, "qx", None),
        "qy": getattr(system, "qy", None),
        "const": _host_value(getattr(system, "const", None)),
        "params": _host_value(getattr(system, "params", {})),
        "x0": _host_value(getattr(system, "x0", None)),
        "u_set": _host_value(getattr(system, "u_set", None)),
    }


def _model_snapshot(model, *, model_name):
    """Save model structure and fitted arrays, not Flax/JAX runtime state."""
    fx = getattr(model, "fx", None)
    fy = getattr(model, "fy", None)
    return {
        "snapshot_version": 1,
        "model_name": model_name,
        "class_name": type(model).__name__,
        "nx": getattr(model, "nx", None),
        "ny": getattr(model, "ny", None),
        "nu": getattr(model, "nu", None),
        "Ts": getattr(model, "Ts", None),
        "y_in_x": getattr(model, "y_in_x", None),
        "fx_class": type(fx).__name__ if fx is not None else None,
        "fy_class": type(fy).__name__ if fy is not None else None,
        # The shared model is the experiment template; fitted models are
        # created per run inside simulation and are not retained here.
        "template_params": _host_value(getattr(model, "params", None)),
        "x0": _host_value(getattr(model, "x0", None)),
    }


def save_data_pkl(
    isSave,
    samples,
    scores,
    system,
    model,
    pred,
    exp_type,
    N_train_init,
    N_train_max,
    N_test,
    N_exp,
    N_set,
    AL_method_set=("passive", "idw"),
    delta_set=(1.0,),
    alpha_set=(1.0,),
    rho_x=1e-3,
    rho_th=1e-3,
    Qx_cov=1e-10,
    Qy_cov=1,
    Qth_cov=1e-10,
    system_name="",
    model_name="",
    isScale=1,
    isConst=0,
):
    if not isSave:
        return None

    is_noise = int(not (system.qx == 0.0 and system.qy == 0.0))
    path = pickle_path(
        exp_type, is_noise, system_name, model_name, isScale, isConst
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    values = (
        _host_value(samples),
        _host_value(scores),
        _system_snapshot(system, system_name=system_name),
        _model_snapshot(model, model_name=model_name),
        pred,
        exp_type,
        AL_method_set,
        delta_set,
        alpha_set,
        N_exp,
        N_set,
        N_train_init,
        N_train_max,
        N_test,
        rho_x,
        rho_th,
        Qx_cov,
        Qy_cov,
        Qth_cov,
        isScale,
        isConst,
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("wb") as file:
            for value in values:
                pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        # Do not replace a valid result unless the complete temporary file can
        # be read back using the same schema.
        with temporary_path.open("rb") as file:
            for _ in _FULL_FIELDS:
                pickle.load(file)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print("---- DATA is successfully saved! ----")
    print("Data file path:", path)


def save_data_pkl0(
    isSave,
    samples,
    scores,
    system,
    model,
    pred,
    exp_type,
    N_train_init,
    N_train_max,
    N_test,
    N_exp,
    N_set,
    AL_method_set=("passive", "idw"),
    delta_set=(1.0,),
    alpha_set=(1.0,),
    rho_x=1e-3,
    rho_th=1e-3,
    Qx_cov=1e-10,
    Qy_cov=1,
    Qth_cov=1e-10,
    system_name="",
    model_name="",
    isScale=1,
    isConst=0,
):
    is_noise = int(not (system.qx == 0.0 and system.qy == 0.0))
    del (
        model,
        pred,
        N_train_init,
        N_train_max,
        N_test,
        N_exp,
        N_set,
        AL_method_set,
        delta_set,
        alpha_set,
        rho_x,
        rho_th,
        Qx_cov,
        Qy_cov,
        Qth_cov,
    )
    if not isSave:
        return None

    path = pickle_path(
        exp_type, is_noise, system_name, model_name, isScale, isConst
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(samples, file)
        pickle.dump(scores, file)
    print("---- DATA is successfully saved! ----")
    print("Data file path:", path)


def load_data_pkl(
    isLoad, exp_type, isNoise, system_name="", model_name="", isScale=1, isConst=0
):
    if not isLoad:
        return None

    path = pickle_path(exp_type, isNoise, system_name, model_name, isScale, isConst)
    print(path.stat().st_size)
    with path.open("rb") as file:
        results = {field: pickle.load(file) for field in _FULL_FIELDS}
    print("---- DATA is loaded! ----")
    print("Data file path:", path)
    return results


def load_data_pkl_0(
    isLoad, exp_type, isNoise, system_name="", model_name="", isScale=1, isConst=0
):
    if not isLoad:
        return None

    path = pickle_path(exp_type, isNoise, system_name, model_name, isScale, isConst)
    print(path.stat().st_size)
    with path.open("rb") as file:
        results = {"samples": pickle.load(file), "scores": pickle.load(file)}
    print("---- DATA is loaded! ----")
    print("Data file path:", path)
    return results
