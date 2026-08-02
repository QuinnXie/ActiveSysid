"""Prediction and regression evaluation helpers."""

import numpy as np

from .learn_x0 import learn_x0_fixed
from .predict import predict, predict_fixed
from .utils import compute_scores, unscale


def finite_r2_score(y, yhat):
    score, _, _ = compute_scores(y, yhat, fit="R2")
    score = np.asarray(score, dtype=float)
    return np.where(np.isfinite(score), score, -np.inf)


def regression_scores(y_train, yhat_train, y_test, yhat_test):
    datasets = [
        np.asarray(value).reshape(len(value), -1)
        for value in (y_train, yhat_train, y_test, yhat_test)
    ]
    y_train, yhat_train, y_test, yhat_test = datasets
    if y_train.shape != yhat_train.shape or y_test.shape != yhat_test.shape:
        raise ValueError("Reference and predicted outputs must have equal shapes")

    def scores(reference, prediction):
        error = prediction - reference
        sse = np.sum(error ** 2, axis=0)
        energy = np.sum(
            (reference - np.mean(reference, axis=0)) ** 2, axis=0
        )
        valid = energy > np.finfo(float).eps
        r2 = np.where(valid, 100.0 * (1.0 - sse / energy),
                      np.where(sse == 0.0, 100.0, -np.inf))
        bfr = np.where(valid, 100.0 * (1.0 - np.sqrt(sse / energy)),
                       np.where(sse == 0.0, 100.0, -np.inf))
        rmse = np.sqrt(sse / reference.shape[0])
        return r2, bfr, rmse

    r2_train, bfr_train, rmse_train = scores(y_train, yhat_train)
    r2_test, bfr_test, rmse_test = scores(y_test, yhat_test)
    lines = [
        f"y{i + 1}: R2 score: training = {r2_train[i]: 5.4f}, "
        f"test = {r2_test[i]: 5.4f}"
        for i in range(r2_train.size)
    ]
    message = lines[0].removeprefix("y1: ") if len(lines) == 1 else "\n".join(lines)
    return tuple(np.squeeze(value) for value in (
        r2_train, r2_test, bfr_train, bfr_test, rmse_train, rmse_test
    )) + (message,)


def compute_prediction_scores(
        xhat0_train, us_train, y_train, us_test, ys_test, y_test,
        ymean, ygain, params, state_fcn, output_fcn,
        us_train_fixed=None, train_valid_length=None):
    if us_train_fixed is None:
        yhat_train = np.asarray(unscale(
            predict(xhat0_train, us_train, state_fcn, output_fcn, params),
            ymean, ygain,
        ))
    else:
        if train_valid_length is None:
            raise ValueError(
                "train_valid_length is required with us_train_fixed"
            )
        # Transfer the fixed-size result before slicing/scaling so JAX only
        # sees one training trajectory shape across score checkpoints.
        yhat_train_scaled = np.asarray(predict_fixed(
            xhat0_train,
            us_train_fixed,
            train_valid_length,
            state_fcn,
            output_fcn,
            params,
        ))
        yhat_train = (
            yhat_train_scaled[:train_valid_length] / np.asarray(ygain)
            + np.asarray(ymean)
        )
    xhat0_test = learn_x0_fixed(
        us_test, ys_test, us_test.shape[0], state_fcn, output_fcn,
        params, nx=xhat0_train.shape[0], x=xhat0_train,
    )
    if np.isnan(xhat0_test).any():
        xhat0_test = xhat0_train
    yhat_test = np.asarray(unscale(
        predict(xhat0_test, us_test, state_fcn, output_fcn, params),
        ymean, ygain,
    ))
    return regression_scores(y_train, yhat_train, y_test, yhat_test) + (
        yhat_train, yhat_test,
    )
