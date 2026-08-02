"""
IDW - Inverse-Distance Weighting for Active Learning

An active learning algorithm for regression
described in the following paper:

The following methods are also implemented for comparison:

- RANDOM:    queries are performed randomly
- GREEDY_x:  queries are performed by selecting the sample with maximum min-distance
             from the samples already selected. This is GSx AL_method in (Wu, Lin, Huang, 2019),
             Algorithm 1 (pool-based sampling only)
- GREEDY_xy: iGS AL_method in (Wu, Lin, Huang, 2019), Algorithm 3
- QBC:       query-by-committee (Burbridge, Rowland, King, 2007), based on N_qbc predictors

"""
from jax_sysid.utils import vec_reshape
from activesysid.utils import standard_scale, unscale
from jax_sysid.models import find_best_model
from activesysid.acquisition.passive import passive, passive_constrained_jax
from activesysid.acquisition.gsu import GSu_fixed_jax
from activesysid.acquisition.gsx import GSx_fixed_jax
from activesysid.acquisition.igs import iGS_fixed_jax
from activesysid.acquisition.gsuy import GSuy_fixed_jax
from activesysid.acquisition.idwuy import IDWuy_fixed_jax
from activesysid.acquisition.idw import (
    idw_fixed_jax, idw_fixed_jax_sequence,
    idw_fixed_jax_plain_constraints,
    idw_fixed_jax_with_margin,
    idw_fixed_jax_filtered_with_asymmetric_margin,
    idw_fixed_jax_asymmetric_margin,
)
from activesysid.predict import predict
from activesysid.extended_kalman_filter import (
    EKF_measurement_update,
    EKF_set,
    EKF_time_update,
)
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental.ode import odeint
from copy import deepcopy
from dataclasses import dataclass
import os
from joblib import Parallel, delayed

from activesysid.timing import time_call as _time_call
from activesysid.timing import timing_stats as _timing_stats

from activesysid.learn_x0 import learn_x0, learn_x0_fixed
from tqdm import tqdm
from activesysid.utils import scale


jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations


from activesysid.evaluation import (
    finite_r2_score as _finite_r2_score,
    regression_scores as _regression_scores,
    compute_prediction_scores as compute_R2_RMSE_scores,
)


_ACQUISITION_ALIASES = {
    "greedy_u": "gsu",
    "greedy-u": "gsu",
    "gsu": "gsu",
    "greedy_uy": "gsuy",
    "greedy-uy": "gsuy",
    "greedy_u_y": "gsuy",
    "gsuy": "gsuy",
    "gs_uy": "gsuy",
    "input_output_greedy": "gsuy",
    "idw_uy": "idwuy",
    "idw-uy": "idwuy",
    "idwuy": "idwuy",
    "input_output_idw": "idwuy",
    "greedy_x": "gsx",
    "greedy-x": "gsx",
    "greedy_ux": "gsx",
    "gs_ux": "gsx",
    "gsx": "gsx",
    "greedy_xy": "igs",
    "greedy-xy": "igs",
    "greedy_uxy": "igs",
    "gs_uxy": "igs",
    "igs": "igs",
    "random": "passive",
    "passive": "passive",
    "idw": "idw",
    "idwz": "idwz",
    "idw_z": "idwz",
    "idw-z": "idwz",
    "idw_uxy": "idw",
    "idw-uxy": "idw",
    "state_aware_idw": "idw",
}


def _canonical_acquisition_method(method):
    """Return one internal name for each supported acquisition method."""
    try:
        return _ACQUISITION_ALIASES[method.lower()]
    except (AttributeError, KeyError) as exc:
        supported = ", ".join(sorted(_ACQUISITION_ALIASES))
        raise ValueError(
            f"Invalid AL_method {method!r}. Supported values: {supported}."
        ) from exc


def _select_idw_input(
    *,
    Us_train,
    Xshat,
    Ys_train,
    m,
    u_set,
    model,
    const,
    delta,
    alpha,
    distance_weights,
    distance_eps,
    uncertainty_weight,
):
    """Select an input using the configured IDW constraint variant."""
    histories = (Us_train, Xshat, Ys_train)
    dynamics = (model.state_fcn, model.output_fcn, model.params)
    distance_weights = {} if distance_weights is None else distance_weights
    distance_kwargs = {
        "input_weight": distance_weights.get("input", 1.0),
        "state_weight": distance_weights.get("state", 1.0),
        "output_weight": distance_weights.get("output", 1.0),
        "next_state_weight": distance_weights.get("next_state", 1.0),
        "distance_eps": distance_eps,
    }
    if const is None:
        return idw_fixed_jax(
            *histories, m, u_set, *dynamics, delta=delta, alpha=alpha,
            uncertainty_weight=uncertainty_weight, **distance_kwargs
        )

    prediction_horizon = int(const.get("prediction_horizon", 1))
    flag = const["flag"]
    idw_args = (*histories, m, u_set, *dynamics)
    idw_kwargs = {
        "delta": delta,
        "alpha": alpha,
    }

    match flag:
        case 1:
            return idw_fixed_jax_plain_constraints(
                *idw_args,
                **idw_kwargs,
                y_min=const["y_min"],
                y_max=const["y_max"],
                penalty_rho=const.get("penalty_rho", 1e12),
                **distance_kwargs,
            )
        case 2:
            return idw_fixed_jax_with_margin(
                *idw_args,
                **idw_kwargs,
                const=const,
                prediction_horizon=prediction_horizon,
                **distance_kwargs,
            )
        case 3:
            sequence_stride = max(int(const.get("sequence_stride", 20)), 1)
            return idw_fixed_jax_sequence(
                *histories,
                m,
                u_set,
                u_set[::sequence_stride],
                *dynamics,
                **idw_kwargs,
                const=const,
                prediction_horizon=prediction_horizon,
                **distance_kwargs,
            )
        case 4:
            return idw_fixed_jax_filtered_with_asymmetric_margin(
                *idw_args,
                **idw_kwargs,
                const=const,
                prediction_horizon=prediction_horizon,
                **distance_kwargs,
            )
        case 5:
            return idw_fixed_jax_asymmetric_margin(
                *idw_args,
                **idw_kwargs,
                const=const,
                prediction_horizon=prediction_horizon,
                **distance_kwargs,
            )
        case _:
            return idw_fixed_jax(
                *idw_args,
                **idw_kwargs,
                const=const,
                prediction_horizon=prediction_horizon,
                **distance_kwargs,
            )


def _select_next_input(
    method,
    *,
    Us_train,
    Xshat,
    Ys_train,
    m,
    u_set,
    model,
    const,
    delta,
    alpha,
    distance_weights,
    distance_eps,
    uncertainty_weight,
):
    """Apply one acquisition rule to the shared posterior model state."""
    dynamics = (model.state_fcn, model.output_fcn, model.params)
    distance_weights = {} if distance_weights is None else distance_weights

    match method:
        case "passive":
            if const is None:
                return passive(u_set)
            key = jax.random.PRNGKey(
                np.random.randint(0, np.iinfo(np.int32).max)
            )
            return passive_constrained_jax(
                u_set,
                key,
                Xshat[m],
                *dynamics,
                const["y_min"],
                const["y_max"],
            )
        case "gsu":
            if const is None:
                return GSu_fixed_jax(Us_train, m, u_set, input_weight=distance_weights.get("input", 1.0), distance_eps=distance_eps)
            return GSu_fixed_jax(Us_train, m, u_set, Xshat[m], *dynamics, const=const, input_weight=distance_weights.get("input", 1.0), distance_eps=distance_eps)
        case "gsuy":
            return GSuy_fixed_jax(Us_train, Ys_train, m, u_set, Xshat[m], *dynamics, const=const, input_weight=distance_weights.get("input", 1.0), output_weight=distance_weights.get("output", 1.0), distance_eps=distance_eps)
        case "idwuy":
            return IDWuy_fixed_jax(Us_train, Ys_train, m, u_set, Xshat[m], *dynamics, delta=delta, const=const, input_weight=distance_weights.get("input", 1.0), output_weight=distance_weights.get("output", 1.0), distance_eps=distance_eps)
        case "gsx":
            args = (Us_train, Xshat, m, u_set)
            if const is None:
                return GSx_fixed_jax(*args, input_weight=distance_weights.get("input", 1.0), state_weight=distance_weights.get("state", 1.0), distance_eps=distance_eps)
            return GSx_fixed_jax(*args, *dynamics, const=const, input_weight=distance_weights.get("input", 1.0), state_weight=distance_weights.get("state", 1.0), distance_eps=distance_eps)
        case "igs":
            return iGS_fixed_jax(
                Us_train,
                Xshat,
                Ys_train,
                m,
                u_set,
                *dynamics,
                input_weight=distance_weights.get("input", 1.0),
                state_weight=distance_weights.get("state", 1.0),
                output_weight=distance_weights.get("output", 1.0),
                next_state_weight=distance_weights.get("next_state", 1.0),
                distance_eps=distance_eps,
                **({"const": const} if const is not None else {}),
            )
        case "idw" | "idwz":
            return _select_idw_input(
                Us_train=Us_train,
                Xshat=Xshat,
                Ys_train=Ys_train,
                m=m,
                u_set=u_set,
                model=model,
                const=const,
                delta=delta,
                alpha=alpha,
                distance_weights=distance_weights,
                distance_eps=distance_eps,
                uncertainty_weight=(0.0 if method == "idwz" else uncertainty_weight),
            )
        case _:
            raise ValueError(
                f"Unsupported canonical acquisition method: {method}"
            )


@dataclass
class _EstimatorInitialization:
    model: object
    xhat0: object
    current_state: object
    covariance: object
    initial_covariance: object
    smoother_covariance: object
    state_history: object
    cache: dict


@dataclass
class _RunData:
    U_train: np.ndarray
    X_train: np.ndarray
    Y_train: np.ndarray
    U_test: np.ndarray
    Y_test: np.ndarray
    Us_train: np.ndarray
    Ys_train: np.ndarray
    Us_test: np.ndarray
    Ys_test: np.ndarray
    u_set: np.ndarray
    u_set_scaled: np.ndarray
    umean: object
    ugain: object
    ymean: object
    ygain: object
    const: object
    isConst: bool
    const_flag: int


@dataclass
class _RunTracking:
    R2_train: np.ndarray
    R2_test: np.ndarray
    BFR_train: np.ndarray
    BFR_test: np.ndarray
    rmse_train: np.ndarray
    rmse_test: np.ndarray
    confidence_margin: np.ndarray
    one_step_prediction: np.ndarray
    ekf_innovation: np.ndarray
    ekf_innovation_covariance: np.ndarray
    ekf_nis: np.ndarray
    ekf_parameter_update_norm: np.ndarray
    ekf_parameter_covariance_trace: np.ndarray
    ekf_covariance_min_eig: np.ndarray
    score_indices: list
    score_index_set: set
    params_history: dict
    xhat0_history: dict


@dataclass
class _RunTimings:
    active_learning: list
    ekf_step: list
    ekf_measurement: list
    ekf_time_update: list
    ekf_set: list
    learn_x0: list
    model_fit: list


def _initialize_ekf_covariance(
    prediction_method, model, P, rho_x, rho_th
):
    """Build the joint state/parameter covariance used by EKF predictors."""
    if prediction_method not in {"EKF_step", "EKF_set"}:
        return P, None
    if P is None:
        flat_params, _ = ravel_pytree(model.params)
        nx = model.nx
        nth = flat_params.shape[0]
        P = jnp.block(
            [
                [
                    np.eye(nx) / rho_x,
                    np.zeros((nx, nth)),
                ],
                [
                    np.zeros((nth, nx)),
                    np.eye(nth) / rho_th,
                ],
            ]
        )
    return P, P


def _initialize_run_data(
    system,
    u_set,
    N_train_init,
    N_train_max,
    *,
    init_set,
    test_set,
    is_scale,
    use_constraints,
    qx,
    qy,
    Ts,
    temporality,
    noise_mode,
):
    """Allocate, generate, and consistently scale training and test data."""
    N_test = N_train_max if test_set is None else test_set["U"].shape[0]
    U_train = np.zeros((N_train_max, system.nu))
    X_train = np.zeros((N_train_max, system.nx))
    Y_train = np.zeros((N_train_max, system.ny))

    u_set = vec_reshape(u_set)
    if init_set is None:
        U_train[:N_train_init] = np.array(
            [np.random.choice(u_set[:, i], N_train_init) for i in range(system.nu)]
        ).T
        Y_train[:N_train_init], X_train[:N_train_init] = predict(
            system.x0,
            U_train[:N_train_init],
            system.state_fcn,
            system.output_fcn,
            system.params,
            qx,
            qy,
            return_X=True,
            Ts=Ts,
            temporality=temporality,
            noise_mode=noise_mode,
        )
    else:
        U_train[:N_train_init] = init_set["U"]
        X_train[:N_train_init] = init_set["X"]
        Y_train[:N_train_init] = init_set["Y"]

    if test_set is None:
        U_test = np.array(
            [np.random.choice(u_set[:, i], N_test) for i in range(system.nu)]
        ).T
        Y_test = predict(
            system.x0,
            U_test,
            system.state_fcn,
            system.output_fcn,
            system.params,
            qx,
            qy,
            Ts=Ts,
            temporality=temporality,
            noise_mode=noise_mode,
        )
    else:
        U_test = test_set["U"]
        Y_test = test_set["Y"]

    if is_scale:
        Us_train = np.zeros((N_train_max, system.nu))
        Ys_train = np.zeros((N_train_max, system.ny))
        Ys_train[:N_train_init], ymean, ygain = standard_scale(
            Y_train[:N_train_init]
        )
        Us_train[:N_train_init], umean, ugain = standard_scale(
            U_train[:N_train_init]
        )
    else:
        Ys_train = Y_train
        Us_train = U_train
        ymean = umean = 0
        ygain = ugain = 1

    const = None
    if use_constraints:
        if system.const is None:
            print("Constraints requested, but system.const is not provided.")
        else:
            const = deepcopy(system.const)
    if const is not None and is_scale:
        const["y_min"] = scale(const["y_min"], ymean, ygain)
        const["y_max"] = scale(const["y_max"], ymean, ygain)
        if "first_step_buffer" in const:
            const["first_step_buffer"] = np.asarray(const["first_step_buffer"]) * ygain
        if "cbf_buffer" in const:
            const["cbf_buffer"] = np.asarray(const["cbf_buffer"]) * ygain
    isConst = const is not None and const.get("flag", 0) > 0
    const_flag = const.get("flag", 0) if isConst else 0

    return _RunData(
        U_train=U_train,
        X_train=X_train,
        Y_train=Y_train,
        U_test=U_test,
        Y_test=Y_test,
        Us_train=Us_train,
        Ys_train=Ys_train,
        Us_test=scale(U_test, umean, ugain),
        Ys_test=scale(Y_test, ymean, ygain),
        u_set=u_set,
        u_set_scaled=scale(u_set, umean, ugain),
        umean=umean,
        ugain=ugain,
        ymean=ymean,
        ygain=ygain,
        const=const,
        isConst=isConst,
        const_flag=const_flag,
    )


def _initialize_run_tracking(
    N_train_init, N_train_max, ny, score_interval, const_flag
):
    """Allocate score outputs and define the model-snapshot checkpoints."""
    score_shape = (N_train_max, ny)
    score_interval = max(int(score_interval), 1)
    score_indices = list(
        range(N_train_init - 1, N_train_max, score_interval)
    )
    if score_indices[-1] != N_train_max - 1:
        score_indices.append(N_train_max - 1)

    return _RunTracking(
        R2_train=np.full(score_shape, np.nan),
        R2_test=np.full(score_shape, np.nan),
        BFR_train=np.full(score_shape, np.nan),
        BFR_test=np.full(score_shape, np.nan),
        rmse_train=np.full(score_shape, np.nan),
        rmse_test=np.full(score_shape, np.nan),
        confidence_margin=np.full(
            (N_train_max, 2 * ny if const_flag in (4, 5) else ny),
            np.nan,
        ),
        one_step_prediction=np.full(score_shape, np.nan),
        ekf_innovation=np.full(score_shape, np.nan),
        ekf_innovation_covariance=np.full((N_train_max, ny, ny), np.nan),
        ekf_nis=np.full(N_train_max, np.nan),
        ekf_parameter_update_norm=np.full(N_train_max, np.nan),
        ekf_parameter_covariance_trace=np.full(N_train_max, np.nan),
        ekf_covariance_min_eig=np.full(N_train_max, np.nan),
        score_indices=score_indices,
        score_index_set=set(score_indices),
        params_history={},
        xhat0_history={},
    )


def _initialize_run_timings():
    """Create independent timing histories for each workflow stage."""
    return _RunTimings([], [], [], [], [], [], [])


def _add_simulation_noise(value, scale, noise_mode="additive"):
    """Apply additive or multiplicative Gaussian simulation noise."""
    value = np.asarray(value)
    noise = scale * np.random.randn(*value.shape)
    if noise_mode == "additive":
        return value + noise
    if noise_mode == "multiplicative":
        return value * (1 + noise)
    raise ValueError("noise_mode must be either 'additive' or 'multiplicative'")


def _tree_all_finite(tree):
    """Return True when every leaf in a parameter pytree is finite."""
    leaves = jax.tree_util.tree_leaves(tree)
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def _array_all_finite(value):
    """Return True when every entry in an array-like value is finite."""
    return np.all(np.isfinite(np.asarray(value)))


def _fit_initial_model_candidate(model_template, Ys_train, Us_train, seed):
    """Fit one initialization without mutating or returning the template."""
    candidate = deepcopy(model_template)
    candidate.init(params=candidate.init_fcn(seed))
    candidate.fit(Ys_train, Us_train)
    return candidate


def _fit_initial_model_candidates(
    model_template, Ys_train, Us_train, *, seeds, n_jobs
):
    """Fit independent model copies with identical serial/parallel semantics."""
    seeds = tuple(seeds)
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    if n_jobs == 1:
        return [
            _fit_initial_model_candidate(
                model_template, Ys_train, Us_train, seed
            )
            for seed in seeds
        ]
    return Parallel(n_jobs=n_jobs)(
        delayed(_fit_initial_model_candidate)(
            model_template, Ys_train, Us_train, seed
        )
        for seed in seeds
    )


def _initialize_estimator(
    model,
    Us_train,
    Ys_train,
    *,
    training_method,
    prediction_method,
    P,
    P0,
    Qx_cov,
    Qy_cov,
    Qth_cov,
    initial_ekf_epochs = 0,
    init_cache,
):
    """Fit or restore the model and construct the initial state history."""
    if init_cache is not None:
        return _EstimatorInitialization(
            model=deepcopy(init_cache["model"]),
            xhat0=deepcopy(init_cache["xhat0_train"]),
            current_state=deepcopy(init_cache["current_state"]),
            covariance=deepcopy(init_cache["P"]),
            initial_covariance=deepcopy(init_cache["P0"]),
            smoother_covariance=deepcopy(init_cache["P0_train"]),
            state_history=deepcopy(init_cache["Xshat_train"]),
            cache=init_cache,
        )

    xhat0 = model.x0
    current_state = None
    if training_method == "jax-sysid":
        # Repeated loky process pools are unreliable once JAX has initialized
        # its CPU runtime (and can also multiply its memory/thread usage).  Use
        # an in-process fit by default; advanced users can opt back into a
        # bounded process pool when their runtime supports it.
        # Six deterministic initialization seeds are fitted below.  Use one
        # worker per candidate by default; Slurm experiment jobs reserve at
        # least eight CPUs so this does not oversubscribe their allocation.
        init_n_jobs = int(os.environ.get("ACTIVESYSID_INIT_N_JOBS", "6"))
        if init_n_jobs < 1:
            raise ValueError("ACTIVESYSID_INIT_N_JOBS must be a positive integer")
        models = _fit_initial_model_candidates(
            model,
            Ys_train,
            Us_train,
            seeds=range(6),
            n_jobs=init_n_jobs,
        )
        model, _ = find_best_model(
            models,
            Ys_train,
            Us_train,
            fit=_finite_r2_score,
        )
        xhat0 = model.x0
        if initial_ekf_epochs > 0:
            model.params, current_state, P, _ = EKF_set(
                jnp.array(model.x0),
                Us_train,
                Ys_train,
                model.state_fcn,
                model.output_fcn,
                model.params,
                P=P,
                Qx_cov=Qx_cov,
                Qy_cov=Qy_cov,
                Qth_cov=Qth_cov,
                N_epoch=initial_ekf_epochs,
            )
    elif training_method == "ekf":
        ekf_epochs = max(int(initial_ekf_epochs), 1)
        model.params, current_state, P, _ = EKF_set(
            jnp.zeros(model.nx),
            Us_train,
            Ys_train,
            model.state_fcn,
            model.output_fcn,
            model.params,
            P=P,
            Qx_cov=Qx_cov,
            Qy_cov=Qy_cov,
            Qth_cov=Qth_cov,
            N_epoch=ekf_epochs,
        )
    else:
        raise ValueError(
            "init_set_trainning must be either 'jax-sysid' or 'ekf'."
        )

    xhat0, smoother_covariance, state_history = learn_x0(
        Us_train,
        Ys_train,
        model.state_fcn,
        model.output_fcn,
        model.params,
        x=xhat0,
        nx=model.nx,
        return_PX=True,
    )
    if current_state is None:
        current_state = state_history[-1]

    if prediction_method == "jax-sysid":
        model.optimization(
            adam_epochs=0, lbfgs_epochs=1000, lbfgs_tol=1.e-6
        )
        model.force_stability(rho_A=1.e3, epsilon_A=1.e-3)

    cache = {
        "model": deepcopy(model),
        "xhat0_train": deepcopy(xhat0),
        "current_state": deepcopy(current_state),
        "P": deepcopy(P),
        "P0": deepcopy(P0),
        "P0_train": deepcopy(smoother_covariance),
        "Xshat_train": deepcopy(state_history),
    }
    return _EstimatorInitialization(
        model=model,
        xhat0=xhat0,
        current_state=current_state,
        covariance=P,
        initial_covariance=P0,
        smoother_covariance=smoother_covariance,
        state_history=state_history,
        cache=cache,
    )


def active_learning_sysid(system, model, u_set, 
            N_train_init, N_train_max, 
            train_interval=1, qx=0., qy=0., Ts = 1, temporality = "discrete",
            delta=1.0, alpha = 1.0, AL_method='idw', 
            init_set = None, test_set = None, pred = "jax-sysid", init_set_trainning = "jax-sysid",
            P = None, rho_x = 1e-3, rho_th = 1e-3,
            Qx_cov=1e-10, Qy_cov=1, Qth_cov=1e-10, isScale = True,
            isConst=False, seed = 1, verbose = False, timing_warmup=1,
            score_interval=1, init_cache=None, return_init_cache=False,
            initial_ekf_epochs=0, noise_mode="additive",
            distance_weights=None, distance_eps=1e-8,
            uncertainty_weight=1.0,
            history_refresh_interval=1,
            record_ekf_diagnostics=False, ekf_nis_gate=None):
    """
    IDW - Inverse-Distance Weighting for Active Learning
    An active learning algorithm for regression and classification (pool-based version).

    The online EKF path assumes that the output has no direct input-to-output
    feedthrough: ``y_k = h(x_k)``.  Consequently, the measurement at step
    ``k`` is assimilated before ``u_k`` is selected and applied.

    Simulation noise uses the same convention as ``predict``: additive noise
    is the default and treats ``qx``/``qy`` as standard deviations, while
    multiplicative noise treats them as relative standard-deviation factors.
    EKF covariance arguments such as ``Qx_cov`` and ``Qy_cov`` are variances
    or covariance matrices in the coordinates passed to the EKF.  With
    ``isScale=True``, that means scaled input/output coordinates.
    
    (C) 2024 K. Xie

    For making comparisons, the AL_method also supports passive methods of random sampling,
    greedy sampling based on distances between feature vectors (Yu, Kim, 2010),
    and the improved representativeness-diversity maximization AL_method
    (Liu, Jiang, Luo, Fang, Liu, Wu, 2021).

    For regression problems only, the AL_method also supports greedy sampling based on feature
    vectors and predicted targets (Wu, Lin, Huang, 2019), and query-by-committee
    for regression (Burbidge, Rowland, King, 2007) based on bootstrap subsets
    (with repetitions).

    The fitting score of each predictor is computed on all training and, if provided,
    test samples to evaluate the performance of the active learning AL_method.

    :param system:

    :param model:

    :param pred: predictor (in scikit-learn format)

    :param pred_type: prediction type, either 'regression' (default) or 'classification'

    :param N_train_init: number of initial samples (queried according to init_method)

    :param N_train_max: number of total samples that can be queried.
                     Active learning is run several times, each time acquiring M samples,
                     M=N_train_init,N_train_init+1,...,N_train_max

    :param train_interval: re-train predictor every 'train_interval' queries (default: 1)

    :param delta: weight on IDW function for pure exploration (default: delta=0.0).
            If equal to 0.0, acquisition is purely based on IDW variance.

    :param AL_method: active learning AL_method used
        'idw'
        'random'    queries are performed randomly
        'greedy_x'  queries are performed by selecting the sample with
                    maximum min-distance from the samples already selected
                    GSx AL_method in (Wu, Lin, Huang, 2019), Algorithm 1
        'greedy_xy' iGS AL_method in (Wu, Lin, Huang, 2019), Algorithm 3
        'qbc'       query-by-committee (Burbridge, Rowland, King, 2007), based on N_qbc predictors

    :param N_qbc: number of predictors used in QBC sampling

    :param qbc_method: QBC AL_method used to create subsets of the current set of k acquired samples:
                'bootstrap': create bootstrap subsets of dimension k (with repetitions) from the existing samples
                'leave-out': leaves out floor(k/N_qbc) samples, where k=number of samples acquired.
                            This AL_method may not perform well when k remains small.

    :param init_method: AL_method used to generate initial N_train_init samples
        'random' generate samples randomly

    :param verbose: verbosity level (0=None)

    :return:
        pred: final predictor after active learning
        samples: queried samples (X_act,Y_act), corresponding index (I_act), and their feasibility (Q_act)
        scores: scores on training and test datasets (if provided)

    """
    np.random.seed(seed) # set random seed for reproducibility

    acquisition_method = _canonical_acquisition_method(AL_method)
    isIDW = acquisition_method in {"idw", "idwz"}
    if history_refresh_interval < 1:
        raise ValueError("history_refresh_interval must be at least 1")

    # Initialize the training and test datasets, scaling factors, and constraints.
    data = _initialize_run_data(system, u_set, N_train_init, N_train_max, init_set=init_set, test_set=test_set, is_scale=isScale, use_constraints=isConst, qx=qx, qy=qy, Ts=Ts, temporality=temporality, noise_mode=noise_mode)
    U_train, X_train, Y_train = data.U_train, data.X_train, data.Y_train
    U_test, Y_test = data.U_test, data.Y_test
    Us_train, Ys_train = data.Us_train, data.Ys_train
    Us_test, Ys_test, u_set, u_set_scaled = data.Us_test, data.Ys_test, data.u_set, data.u_set_scaled
    umean, ugain = data.umean, data.ugain
    ymean, ygain = data.ymean, data.ygain
    const, isConst, const_flag = data.const, data.isConst, data.const_flag

    # Initialize the tracking of scores and model snapshots at specified intervals.
    tracking = _initialize_run_tracking(N_train_init, N_train_max, system.ny, score_interval, const_flag)
    R2_train, R2_test = tracking.R2_train, tracking.R2_test
    BFR_train, BFR_test = tracking.BFR_train, tracking.BFR_test
    rmse_train, rmse_test = tracking.rmse_train, tracking.rmse_test
    confidence_margin = tracking.confidence_margin
    Y_one_step_pred_train = tracking.one_step_prediction
    ekf_innovation = tracking.ekf_innovation
    ekf_innovation_covariance = tracking.ekf_innovation_covariance
    ekf_nis = tracking.ekf_nis
    ekf_parameter_update_norm = tracking.ekf_parameter_update_norm
    ekf_parameter_covariance_trace = tracking.ekf_parameter_covariance_trace
    ekf_covariance_min_eig = tracking.ekf_covariance_min_eig
    score_indices, score_index_set = tracking.score_indices, tracking.score_index_set
    params_hist = tracking.params_history
    xhat0_train_hist = tracking.xhat0_history

    # Initialize timing histories for each stage of the active learning workflow.
    run_timings = _initialize_run_timings()
    al_times = run_timings.active_learning
    ekf_step_times = run_timings.ekf_step
    ekf_measurement_times = run_timings.ekf_measurement
    ekf_time_update_times = run_timings.ekf_time_update
    ekf_set_times = run_timings.ekf_set
    learn_x0_times = run_timings.learn_x0
    model_fit_times = run_timings.model_fit

    # Initialize the estimator from the shared initial training set.
    m = N_train_init
    P, P0 = _initialize_ekf_covariance(pred, model, P, rho_x, rho_th)
    initialization = _initialize_estimator(model, Us_train[:m], Ys_train[:m], training_method=init_set_trainning, prediction_method=pred, P=P, P0=P0, Qx_cov=Qx_cov, Qy_cov=Qy_cov, Qth_cov=Qth_cov, initial_ekf_epochs=initial_ekf_epochs, init_cache=init_cache)
    model = initialization.model
    xhat0_train = initialization.xhat0
    current_state = initialization.current_state
    P = initialization.covariance
    P0 = initialization.initial_covariance
    P0_train = initialization.smoother_covariance
    Xshat_train = initialization.state_history
    init_cache_result = initialization.cache

    # Fixed-shape buffers prevent one JIT compilation for every value of m.
    # The extra state/output row supports the shifted histories used by iGS and IDW: Y[1:] and Xhat[1:].
    # Estimated states belong to the learned model and can have a different dimension from the physical system (for example, Robot Arm uses 4 vs 5).
    Xshat_fixed = jnp.zeros((N_train_max + 1, model.nx))
    Xshat_fixed = Xshat_fixed.at[:Xshat_train.shape[0]].set(Xshat_train)
    Xshat_acquisition = Xshat_fixed
    xhat0_acquisition = xhat0_train
    P0_acquisition = P0_train
    Ys_train_fixed = jnp.zeros((N_train_max + 1, system.ny))
    Ys_train_fixed = Ys_train_fixed.at[:N_train_max].set(Ys_train)
    Us_train_fixed = jnp.asarray(Us_train)

    params_hist[N_train_init - 1] = model.params
    xhat0_train_hist[N_train_init - 1] = xhat0_train

    integration_grid = None
    if temporality == "continuous":
        integration_grid = jnp.linspace(0, Ts, num=2)

    relearn_acquisition_history = acquisition_method in {"gsx", "igs", "idw", "idwz"}
    failure_index = None
    failure_reason = None

    for k in tqdm(range(N_train_init, N_train_max), desc="Active Learning Progress"):
        # print("Iteration        : %2d/%2d \n" % (k,N_train_max))

        # Measure the next output by simulating the system forward in time, using the previous state and input.
        if temporality == "discrete":
            # sample the next output (No direct I/O feedthrough)
            X_train[k] = _add_simulation_noise(system.state_fcn(X_train[k-1], U_train[k-1], system.params), qx, noise_mode)
            Y_train[k] = _add_simulation_noise(system.output_fcn(X_train[k], U_train[k], system.params), qy, noise_mode)
            Ys_train[k] = scale(Y_train[k], ymean, ygain)
        elif temporality == "continuous":
            X_train[k] = _add_simulation_noise(odeint(system.state_fcn, X_train[k-1], integration_grid, U_train[k-1])[-1], qx, noise_mode)
            Y_train[k] = _add_simulation_noise(system.output_fcn(X_train[k]), qy, noise_mode)
            # X_train[k], Y_train[k] = system.process(X_train[k-1], U_train[k-1], qx, qy)
            Ys_train[k] = scale(Y_train[k], ymean, ygain)

        # Check for non-finite values in the simulated state or output, which would indicate a failure in the simulation.
        if not (
            np.all(np.isfinite(X_train[k]))
            and np.all(np.isfinite(Y_train[k]))
            and np.all(np.isfinite(Ys_train[k]))
        ):
            failure_index = k
            failure_reason = "non-finite simulated state or output"
            tqdm.write(
                f"Stopping run at training index {k}: {failure_reason}."
            )
            for array in (U_train, X_train, Y_train, Us_train, Ys_train): array[k:] = np.nan
            break

        Ys_train_fixed = Ys_train_fixed.at[k].set(Ys_train[k])

        x_kk = None
        P_kk = None
        measurement_elapsed = 0.0
        if pred == "EKF_step":
            # Contract: output_fcn has no direct input feedthrough, so the
            # previous input is only a shape-compatible placeholder here.
            if record_ekf_diagnostics:
                (model.params, x_kk, P_kk, ekf_diag), elapsed = _time_call(
                    lambda: EKF_measurement_update(
                        current_state,
                        Us_train[k - 1],
                        Ys_train[k],
                        model.output_fcn,
                        model.params,
                        P=P,
                        Qy_cov=Qy_cov,
                        return_diagnostics=True,
                        nis_gate=ekf_nis_gate,
                    )
                )
            else:
                (model.params, x_kk, P_kk), elapsed = _time_call(
                    lambda: EKF_measurement_update(
                        current_state,
                        Us_train[k - 1],
                        Ys_train[k],
                        model.output_fcn,
                        model.params,
                        P=P,
                        Qy_cov=Qy_cov,
                        return_diagnostics=False,
                        nis_gate=ekf_nis_gate,
                    )
                )
                ekf_diag = None
            measurement_elapsed = elapsed
            ekf_measurement_times.append(elapsed)
            if record_ekf_diagnostics:
                ekf_innovation[k] = np.asarray(ekf_diag["innovation"]).reshape(-1)
                ekf_innovation_covariance[k] = np.asarray(
                    ekf_diag["innovation_covariance"]
                )
                ekf_nis[k] = float(np.asarray(ekf_diag["nis"]))
                ekf_parameter_update_norm[k] = float(
                    np.asarray(ekf_diag["parameter_update_norm"])
                )
                ekf_parameter_covariance_trace[k] = float(
                    np.asarray(ekf_diag["posterior_parameter_covariance_trace"])
                )
                ekf_covariance_min_eig[k] = float(
                    np.asarray(ekf_diag["posterior_covariance_min_eig"])
                )
            if not (
                _tree_all_finite(model.params)
                and _array_all_finite(x_kk)
                and _array_all_finite(P_kk)
                and (
                    not record_ekf_diagnostics
                    or _tree_all_finite(ekf_diag)
                )
            ):
                failure_index = k
                failure_reason = "non-finite EKF measurement update"
                tqdm.write(
                    f"Stopping run at training index {k}: {failure_reason}."
                )
                for array in (U_train, X_train, Y_train, Us_train, Ys_train):
                    array[k + 1:] = np.nan
                break
            Xshat_fixed = Xshat_fixed.at[k].set(x_kk)
            completed_online_samples = k - N_train_init + 1
            refresh_acquisition_history = (
                relearn_acquisition_history
                and completed_online_samples % history_refresh_interval == 0
            )
            if refresh_acquisition_history:
                # Relearn latent history with the posterior parameters for acquisition only; Xshat_fixed remains consistent with EKF P.
                (xhat0_acquisition, P0_acquisition, Xshat_acquisition), elapsed = _time_call(
                    lambda: learn_x0_fixed(Us_train_fixed, Ys_train_fixed[:-1], k+1, model.state_fcn, model.output_fcn, model.params, x=xhat0_acquisition, P=P0_acquisition, return_PX=True))
                learn_x0_times.append(elapsed)
                Xshat_acquisition = Xshat_acquisition.at[k].set(x_kk)
            else:
                Xshat_acquisition = Xshat_acquisition.at[k].set(x_kk)

        # All methods receive the same posterior histories and model.
        selected, elapsed = _time_call(
            lambda: _select_next_input(acquisition_method, Us_train=Us_train_fixed, Xshat=Xshat_acquisition, Ys_train=Ys_train_fixed, m=k, u_set=u_set_scaled, model=model, const=const if isConst else None, delta=delta, alpha=alpha, distance_weights=distance_weights, distance_eps=distance_eps, uncertainty_weight=uncertainty_weight))
        al_times.append(elapsed)
        if isIDW and isConst and const['flag'] in (2, 4, 5):
            next_input, selected_margin = selected
            margin_gain = ygain
            if const['flag'] in (4, 5):
                margin_gain = np.tile(np.asarray(ygain).reshape(-1), 2)
            confidence_margin[k] = np.asarray(unscale(selected_margin, 0, margin_gain)).reshape(-1)
        else:
            next_input = selected
        Us_train[k] = next_input
        Us_train_fixed = Us_train_fixed.at[k].set(next_input)

        if k + 1 < N_train_max:
            x_pred_next = model.state_fcn(Xshat_acquisition[k], jnp.asarray(next_input), model.params)
            y_pred_next = model.output_fcn(x_pred_next, jnp.asarray(next_input), model.params)
            Y_one_step_pred_train[k + 1] = np.asarray(unscale(y_pred_next, ymean, ygain)).reshape(-1)

        U_train[k] = unscale(Us_train[k], umean, ugain)

        # update the model
        if pred == "EKF_step":
            (current_state, P), elapsed = _time_call(
                lambda: EKF_time_update(
                    x_kk, Us_train[k], model.state_fcn, model.params, P_kk,
                    Qx_cov=Qx_cov, Qth_cov=Qth_cov,
                )
            )
            if record_ekf_diagnostics:
                flat_params, _ = ravel_pytree(model.params)
                parameter_count = flat_params.shape[0]
                ekf_parameter_covariance_trace[k] = float(
                    np.asarray(jnp.trace(P[-parameter_count:, -parameter_count:]))
                )
                ekf_covariance_min_eig[k] = float(
                    np.asarray(jnp.min(jnp.linalg.eigvalsh(P)))
                )
            if not (
                _tree_all_finite(model.params)
                and _array_all_finite(current_state)
                and _array_all_finite(P)
                and (
                    not record_ekf_diagnostics
                    or (
                        np.isfinite(ekf_parameter_covariance_trace[k])
                        and np.isfinite(ekf_covariance_min_eig[k])
                    )
                )
            ):
                failure_index = k
                failure_reason = "non-finite EKF time update"
                tqdm.write(
                    f"Stopping run at training index {k}: {failure_reason}."
                )
                for array in (U_train, X_train, Y_train, Us_train, Ys_train):
                    array[k + 1:] = np.nan
                break
            ekf_time_update_times.append(elapsed)
            ekf_step_times.append(measurement_elapsed + elapsed)
            # Use the same online-EKF state policy for every acquisition
            # method. Xhat[k] is posterior; Xhat[k+1] is the next prior.
            Xshat_fixed = Xshat_fixed.at[k+1].set(current_state)
            Xshat_acquisition = Xshat_acquisition.at[k+1].set(
                model.state_fcn(Xshat_acquisition[k], Us_train[k], model.params)
            )
        elif pred == "EKF_set": # use EKF for all training data again 
            (model.params, current_state, _, X_set), elapsed = _time_call(
                lambda: EKF_set(xhat0_train, Us_train[:k+1], Ys_train[:k+1], model.state_fcn, model.output_fcn, model.params, P=P0, Qx_cov=Qx_cov, Qy_cov=Qy_cov, Qth_cov=Qth_cov)
            )
            ekf_set_times.append(elapsed)
            Xshat_fixed = Xshat_fixed.at[:X_set.shape[0]].set(X_set)
            Xshat_acquisition = Xshat_fixed
        elif pred == "jax-sysid":
            print("model.optimization")
            _, elapsed = _time_call(lambda: model.fit(Ys_train[:k+1], Us_train[:k+1]))
            model_fit_times.append(elapsed)
            xhat0_train = model.x0
            (model.x0, P0_train, Xshat_train), elapsed = _time_call(
                lambda: learn_x0_fixed(Us_train_fixed, Ys_train_fixed[:-1], k+1, model.state_fcn, model.output_fcn, model.params, x=xhat0_train, P=P0_train, return_PX=True)
            )
            learn_x0_times.append(elapsed)
            Xshat_fixed = Xshat_fixed.at[:Xshat_train.shape[0]].set(Xshat_train)
            Xshat_acquisition = Xshat_fixed
        else:
            raise ValueError("Prediction method is not recognized")
        
        # Keep only snapshots that will actually be scored.
        if k in score_index_set:
            params_hist[k] = model.params
            xhat0_train_hist[k] = xhat0_train

    print("Final parameters: ")

    # Fixed test arrays are reused at every score checkpoint.
    Us_test_jax = jnp.asarray(Us_test)
    Ys_test_jax = jnp.asarray(Ys_test)

    completed_score_indices = [
        m for m in score_indices if m in params_hist and m in xhat0_train_hist
    ]
    for i in tqdm(completed_score_indices, desc="Test Progress"):
        R2_train[i], R2_test[i], BFR_train[i], BFR_test[i], rmse_train[i], rmse_test[i], msg, Yhat_train, Yhat_test = compute_R2_RMSE_scores(
            xhat0_train_hist[i], Us_train[:i+1], Y_train[:i+1], Us_test_jax, Ys_test_jax, Y_test, ymean, ygain, params_hist[i], model.state_fcn, model.output_fcn,
            us_train_fixed=Us_train_fixed, train_valid_length=i+1)
        # tqdm.write(msg); # print("\n")

    if failure_index is not None:
        Yhat_train_completed = np.asarray(Yhat_train)
        Yhat_train = np.full_like(Y_train, np.nan)
        Yhat_train[:Yhat_train_completed.shape[0]] = Yhat_train_completed

    # m =  N_train_max-1
    # R2_train[m], R2_test[m], BFR_train[m], BFR_test[m], rmse_train[m], rmse_test[m], msg, Yhat_train, Yhat_test = compute_R2_RMSE_scores(
    # xhat0_train, Us_train[:m+1], Y_train[:m+1], Us_test, Ys_test, Y_test, ymean, ygain, params_hist[m-N_train_init+1], model.state_fcn, model.output_fcn)
    # print(msg); print("\n")

    samples = {'Y_train': Y_train, 'Y_test': Y_test, 
               'Yhat_train': Yhat_train, 'Yhat_test': Yhat_test,
               'Y_one_step_pred_train': Y_one_step_pred_train,
               'U_train': U_train, 'U_test': U_test}
    timings = {
        'AL_method': AL_method,
        'pred': pred,
        'warmup_samples': timing_warmup,
        'active_learning': _timing_stats(al_times, timing_warmup),
        'EKF_step': _timing_stats(ekf_step_times, timing_warmup),
        'EKF_measurement_update': _timing_stats(
            ekf_measurement_times, timing_warmup),
        'EKF_time_update': _timing_stats(
            ekf_time_update_times, timing_warmup),
        'EKF_set': _timing_stats(ekf_set_times, timing_warmup),
        'learn_x0': _timing_stats(learn_x0_times, timing_warmup),
        'model_fit': _timing_stats(model_fit_times, timing_warmup),
        'failed': failure_index is not None,
        'failure_index': failure_index,
        'failure_reason': failure_reason,
        'ekf_diagnostics': {
            'innovation': ekf_innovation,
            'innovation_covariance': ekf_innovation_covariance,
            'nis': ekf_nis,
            'parameter_update_norm': ekf_parameter_update_norm,
            'parameter_covariance_trace': ekf_parameter_covariance_trace,
            'covariance_min_eig': ekf_covariance_min_eig,
        },
    }

    scores = {'R2_train': R2_train, 'R2_test': R2_test, 
              'BFR_train': BFR_train, 'BFR_test': BFR_test,
              'rmse_train': rmse_train, 'rmse_test': rmse_test,
              'confidence_margin': confidence_margin,
              'timings': timings}

    if return_init_cache:
        return model, samples, scores, init_cache_result

    return model, samples, scores
