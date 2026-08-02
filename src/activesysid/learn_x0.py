"""Initial-state estimation with EKF/RTS smoothing."""

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from functools import partial
from jax_sysid.utils import lbfgs_options
from jax.scipy.linalg import solve
import sys

from .utils import params_to_abcd


def _validate_learn_x0_inputs(
        U, Y, x, nx, P, Q, R, rho_x0, RTS_epochs):
    """Normalize array shapes and validate the state-estimation inputs."""
    U = jnp.asarray(U)
    Y = jnp.asarray(Y)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if U.ndim != 2 or Y.ndim != 2:
        raise ValueError("U and Y must be one- or two-dimensional arrays")
    if U.shape[0] == 0:
        raise ValueError("U and Y must contain at least one sample")
    if U.shape[0] != Y.shape[0]:
        raise ValueError(
            f"U and Y must have the same length, got {U.shape[0]} "
            f"and {Y.shape[0]}"
        )
    if not isinstance(RTS_epochs, int) or RTS_epochs < 1:
        raise ValueError("RTS_epochs must be a positive integer")
    if rho_x0 <= 0:
        raise ValueError("rho_x0 must be positive")

    if x is None:
        if nx is None:
            raise ValueError("Either x or nx must be specified")
        x = jnp.zeros(nx)
    else:
        x = jnp.asarray(x).reshape(-1)
        nx = x.shape[0]

    ny = Y.shape[1]
    Q = 1.e-5 * jnp.eye(nx) if Q is None else jnp.asarray(Q)
    R = jnp.eye(ny) if R is None else jnp.asarray(R)
    P = (
        jnp.eye(nx) / (rho_x0 * U.shape[0])
        if P is None
        else jnp.asarray(P)
    )
    if P.shape != (nx, nx):
        raise ValueError(f"P must have shape {(nx, nx)}, got {P.shape}")
    if Q.shape != (nx, nx):
        raise ValueError(f"Q must have shape {(nx, nx)}, got {Q.shape}")
    if R.shape != (ny, ny):
        raise ValueError(f"R must have shape {(ny, ny)}, got {R.shape}")
    return U, Y, x, nx, P, Q, R


def _symmetrize(matrix):
    return (matrix + matrix.T) / 2


@partial(
    jax.jit,
    static_argnames=("state_fcn", "output_fcn", "isLinear", "RTS_epochs"),
)
def _learn_x0_fixed_rts(
        U, Y, valid_length, x, P, state_fcn, output_fcn, params, Q, R,
        A_LTI, C_LTI, isLinear=False, RTS_epochs=1):
    """Fixed-shape EKF/RTS smoother; only indices below valid_length are used."""
    nx = x.shape[0]
    ny = Y.shape[1]
    indices = jnp.arange(U.shape[0])

    def ekf_update(state, data):
        x_prior, P_prior, mse_loss = state
        index, yk, uk = data

        def valid_step(current_state):
            xk, Pk, loss = current_state
            yhat = output_fcn(xk, uk, params)
            Ck = (
                C_LTI
                if isLinear
                else jax.jacrev(output_fcn)(xk, u=uk, params=params)
            )
            PC = Pk @ Ck.T
            innovation_covariance = _symmetrize(R + Ck @ PC)
            innovation_covariance += (
                1.e-10 * jnp.eye(ny, dtype=innovation_covariance.dtype)
            )
            gain = solve(
                innovation_covariance, PC.T, assume_a="pos"
            ).T
            error = yk - yhat
            x_filtered = xk + gain @ error

            IKH = jnp.eye(nx) - gain @ Ck
            P_filtered = _symmetrize(
                IKH @ Pk @ IKH.T + gain @ R @ gain.T
            )

            Ak = (
                A_LTI
                if isLinear
                else jax.jacrev(state_fcn)(
                    x_filtered, u=uk, params=params
                )
            )
            x_predicted = state_fcn(x_filtered, uk, params)
            P_predicted = _symmetrize(
                Ak @ P_filtered @ Ak.T + Q
            )
            output = (
                x_filtered, P_filtered, x_predicted, P_predicted, Ak
            )
            return (
                x_predicted,
                P_predicted,
                loss + jnp.sum(error ** 2),
            ), output

        def padded_step(current_state):
            xk, Pk, loss = current_state
            output = (xk, Pk, xk, Pk, jnp.eye(nx, dtype=xk.dtype))
            return current_state, output

        return jax.lax.cond(
            index < valid_length, valid_step, padded_step, state
        )

    def rts_update(state, data):
        x_next, P_next = state
        index, P_filtered, P_predicted, x_filtered, x_predicted, Ak = data

        def valid_step(current_state):
            xs, Ps = current_state
            predicted_covariance = _symmetrize(P_predicted)
            predicted_covariance += (
                1.e-10 * jnp.eye(nx, dtype=predicted_covariance.dtype)
            )
            gain = solve(
                predicted_covariance,
                (P_filtered @ Ak.T).T,
                assume_a="pos",
            ).T
            x_smoothed = x_filtered + gain @ (xs - x_predicted)
            P_smoothed = _symmetrize(
                P_filtered + gain @ (Ps - P_predicted) @ gain.T
            )
            return (x_smoothed, P_smoothed), x_smoothed

        def padded_step(current_state):
            return current_state, current_state[0]

        return jax.lax.cond(
            index < valid_length, valid_step, padded_step, state
        )

    X = jnp.zeros((U.shape[0] + 1, nx), dtype=x.dtype)
    mse_loss = jnp.asarray(0.0, dtype=x.dtype)

    for _ in range(RTS_epochs):
        mse_loss = jnp.asarray(0.0, dtype=x.dtype)
        forward_state, output = jax.lax.scan(
            ekf_update, (x, P, mse_loss), (indices, Y, U)
        )
        XX1, PP1, XX2, PP2, AA = output

        terminal_index = jnp.maximum(valid_length - 1, 0)
        terminal_state = (
            XX2[terminal_index],
            PP2[terminal_index],
        )
        reverse_input = (
            indices[::-1],
            PP1[::-1],
            PP2[::-1],
            XX1[::-1],
            XX2[::-1],
            AA[::-1],
        )
        (x, P), X_reverse = jax.lax.scan(
            rts_update, terminal_state, reverse_input
        )
        X = jnp.vstack((X_reverse[::-1], terminal_state[0]))
        mse_loss = forward_state[2]

    return x, P, X, mse_loss / jnp.maximum(valid_length, 1)


def learn_x0_fixed(
        U, Y, valid_length, state_fcn, output_fcn, params, x=None, nx=None,
        x0_min=-10, x0_max=10, isLinear=False, P=None, return_PX=False,
        rho_x0=1e-3, RTS_epochs=1, verbosity=False,
        LBFGS_refinement=False, LBFGS_rho_x0=1.e-4,
        lbfgs_epochs=1000, Q=None, R=None):
    """Fixed-shape variant of learn_x0 for repeated online calls."""
    default_covariance = P is None
    U, Y, x, nx, P, Q, R = _validate_learn_x0_inputs(
        U, Y, x, nx, P, Q, R, rho_x0, RTS_epochs
    )

    valid_length = int(valid_length)
    if not 0 < valid_length <= U.shape[0]:
        raise ValueError("valid_length must be between 1 and U.shape[0]")

    ny = Y.shape[1]
    if default_covariance:
        P = jnp.eye(nx) / (rho_x0 * valid_length)

    if isLinear:
        A_LTI, _, C_LTI, _ = params_to_abcd(params)
        A_LTI = jnp.asarray(A_LTI)
        C_LTI = jnp.asarray(C_LTI)
    else:
        # Shape-correct placeholders keep the compiled argument structure fixed.
        A_LTI = jnp.zeros((nx, nx), dtype=x.dtype)
        C_LTI = jnp.zeros((ny, nx), dtype=x.dtype)

    x_result, P_result, X_result, mse_loss = _learn_x0_fixed_rts(
        U, Y, jnp.asarray(valid_length), x, P, state_fcn, output_fcn,
        params, Q, R, A_LTI, C_LTI, isLinear=isLinear,
        RTS_epochs=RTS_epochs,
    )

    x_numpy = np.asarray(x_result)
    out_of_bounds = (
        (x0_min is not None and np.any(x_numpy < np.asarray(x0_min)))
        or (x0_max is not None and np.any(x_numpy > np.asarray(x0_max)))
    )
    needs_fallback = (
        LBFGS_refinement or np.isnan(x_numpy).any() or out_of_bounds
    )
    if needs_fallback:
        result = learn_x0(
            np.asarray(U)[:valid_length],
            np.asarray(Y)[:valid_length],
            state_fcn,
            output_fcn,
            params,
            x=x,
            nx=nx,
            x0_min=x0_min,
            x0_max=x0_max,
            isLinear=isLinear,
            P=P,
            return_PX=return_PX,
            rho_x0=rho_x0,
            RTS_epochs=RTS_epochs,
            verbosity=verbosity,
            LBFGS_refinement=True,
            LBFGS_rho_x0=LBFGS_rho_x0,
            lbfgs_epochs=lbfgs_epochs,
            Q=Q,
            R=R,
        )
        if not return_PX:
            return result

        x_result, P_result, X_valid = result
        X_result = jnp.zeros((U.shape[0] + 1, nx), dtype=X_valid.dtype)
        X_result = X_result.at[:X_valid.shape[0]].set(X_valid)
        X_result = X_result.at[X_valid.shape[0]:].set(X_valid[-1])

    if verbosity:
        print(
            f"RTS smoothing, valid samples: {valid_length}, "
            f"MSE loss = {float(mse_loss):8.6f}"
        )
    if return_PX:
        return x_result, P_result, X_result
    return x_result


def learn_x0(U, Y, state_fcn, output_fcn, params, x=None, nx=None, 
             x0_min=-10, x0_max=10, isLinear=False, P=None, return_PX=False,
             rho_x0=1e-3, RTS_epochs=1, verbosity=False, LBFGS_refinement=False, 
             LBFGS_rho_x0=1.e-4, lbfgs_epochs=1000, Q=None, R=None):
    
    """Estimate x0 by Rauch-Tung-Striebel smoothing (Sarkka and Svenson, 2023, p.268),
    possibly followed by L-BFGS optimization.
    
    (c) 2023 A. Bemporad

    (c) 2024 K. Xie
        - Modified the inputs and outputs of the function
        - Modified the function with jax.jit to improve performance

    Parameters
    ----------
    U : array
        Input data, U must be a N-by-nu numpy array
    Y : array
        Output data, Y must be a N-by-ny numpy array
    state_fcn : function handle
        Function handle to the state function x(k+1)=state_fcn(x(k),u(k),params).
    output_fcn : function handle
        Function handle to the output function y(k)=output_fcn(x(k),u(k),params).
    params : list of ndarrays
        List of ndarrays of parameters for the state and output functions
    x : array
        Initial state estimate
    nx : int
        Number of states
    x0_min : array
        Lower bound on the initial state
    x0_max : array
        Upper bound on the initial state
    isLinear : bool
        If True, the state and output functions are linear
    P : array
        Initial covariance matrix
    return_P : bool
        If True, return the covariance matrix P
    rho_x0 : float
        L2-regularization on initial state x0, 0.5*rho_x0*||x0||_2^2 (default: model.rho_x0)
    RTS_epochs : int
        Number of forward KF and backward RTS passes
    verbosity : bool
        If false, removes printout of operations
    LBFGS_refinement : bool
        If True, refine solution via L-BFGS optimization. Also used in the case bounds on x0 have been specified and the value of x0 estimated by RTS is not feasible.
    LBFGS_rho_x0 : float
        L2-regularization used by L-BFGS, by default 1.e-8
    lbfgs_epochs : int
        Max number of L-BFGS iterations
    Q : array
        Process noise covariance matrix, by default 1.e-8*I
    R : array
        Measurement noise covariance matrix, by default the identity matrix I

    Returns
    -------
    array
        Optimal initial state x0.
    matrix
        Optimal covariance matrix P0. If return_P is True, P0 is returned as well.
    """

    U, Y, x, nx, P, Q, R = _validate_learn_x0_inputs(
        U, Y, x, nx, P, Q, R, rho_x0, RTS_epochs
    )

    # Keep a single RTS implementation. The legacy body below is retained
    # only for the optional bounded L-BFGS refinement.
    if not LBFGS_refinement:
        return learn_x0_fixed(
            U, Y, U.shape[0], state_fcn, output_fcn, params,
            x=x, nx=nx, x0_min=x0_min, x0_max=x0_max,
            isLinear=isLinear, P=P, return_PX=return_PX,
            rho_x0=rho_x0, RTS_epochs=RTS_epochs,
            verbosity=verbosity, LBFGS_refinement=False,
            LBFGS_rho_x0=LBFGS_rho_x0,
            lbfgs_epochs=lbfgs_epochs, Q=Q, R=R,
        )

    ny = Y.shape[1]
    N = U.shape[0]
    
    if isLinear:
        A_LTI, _, C_LTI, _ = params_to_abcd(params)
        A = jnp.array(A_LTI)
        C = jnp.array(C_LTI)
    else:
        @jax.jit
        def Ck(x, u):
            return jax.jacrev(output_fcn)(x, u=u, params=params)
        @jax.jit
        def Ak(x, u):
            return jax.jacrev(state_fcn)(x, u=u, params=params)
          
    # Forward EKF pass:
    # L2-regularization on initial state x0, 0.5*rho_x0*||x0||_2^2
    @jax.jit
    def EKF_update(state, yuk):
        x, P, mse_loss = state
        yk = yuk[:ny]
        u = yuk[ny:]

        # measurement update
        y = output_fcn(x, u, params)
        if not isLinear:
            Ckk = Ck(x, u)
        else:
            Ckk = C_LTI
        PC = P @ Ckk.T
        # M = PC / (R + C @PC) # this solves the linear system M*(R + C @PC) = PC
        # Note: Matlab's mrdivide A / B = (B'\A')' = np.linalg.solve(B.conj().T, A.conj().T).conj().T
        M = solve((R+Ckk@PC), PC.T, assume_a='pos').T
        e = yk-y
        mse_loss += jnp.sum(e**2)  # just for monitoring purposes
        x1 = x + M@e  # x(k | k)

        # Standard Kalman measurement update
        # P -= M@PC.T
        # P = (P + P.T)/2. # P(k|k)

        # Joseph stabilized covariance update
        IKH = -M@Ckk
        IKH += jnp.eye(nx)
        P1 = IKH@P@IKH.T+M@R@M.T  # P(k|k)

        # Time update
        if not isLinear:
            Akk = Ak(x1, u)
        else:
            Akk = A_LTI
        P2 = Akk@P1@Akk.T+Q
        # P2 = (P2+P2.T)/2.
        x2 = state_fcn(x1, u, params)
        if not isLinear:
            output = (x1, P1, x2, P2, Akk)
        else:
            # Avoid returning the same A matrix everytime
            output = (x1, P1, x2, P2)

        return (x2, P2, mse_loss), output

    # @jax.jit
    def RTS_update(state, input):
        x, P = state
        if not isLinear:
            P1, P2, x1, x2, A = input
        else:
            P1, P2, x1, x2 = input  # The A matrix is the LTI state-update matrix defined earlier
            A = A_LTI

        # G=(PP1[k]@AA[k].T)/PP2[k]
        try:
            G = jax.scipy.linalg.solve(P2, (P1@A.T).T, assume_a='pos').T
        except:
            G = jax.scipy.linalg.solve(P2, (P1@A.T).T, assume_a='gen').T
        x = x1+G@(x-x2)
        P = P1+G@(P-P2)@G.T
        return (x, P), x

    for epoch in range(RTS_epochs):
        mse_loss = 0.

        # Forward EKF pass
        state = (x, P, mse_loss)
        state, output = jax.lax.scan(EKF_update, state, jnp.hstack((Y, U)))
        if not isLinear:
            XX1, PP1, XX2, PP2, AA = output
        else:
            XX1, PP1, XX2, PP2 = output
        # PP1 = P(k | k)
        # PP2 = P(k + 1 | k)
        # XX1 = x(k | k)
        # XX2 = x(k + 1 | k)
        mse_loss = state[2]/N

        # RTS smoother pass:
        x = XX2[N-1]
        P = PP2[N-1]
        state = (x, P)
        if not isLinear:
            input = (PP1[::-1], PP2[::-1], XX1[::-1], XX2[::-1], AA[::-1])
        else:
            input = (PP1[::-1], PP2[::-1], XX1[::-1], XX2[::-1])
        state, X = jax.lax.scan(RTS_update, state, input)
        x, P = state

        if verbosity:
            sys.stdout.write('\033[F')
            print(
                f"\nRTS smoothing, epoch: {epoch+1: 3d}/{RTS_epochs: 3d}, MSE loss = {mse_loss: 8.6f}")
    X = X[::-1]
    X = jnp.vstack((X,XX2[-1]))

    isstatebounded = x0_min is not None or x0_max is not None
    if isstatebounded:
        lb = x0_min
        if lb is None:
            lb = -jnp.inf*jnp.ones(nx)
        ub = x0_max
        if ub is None:
            ub = jnp.inf*jnp.ones(nx)
        # if jnp.any(x < lb) or jnp.any(x > ub):
        #     LBFGS_refinement = True
        def check_bounds(x, lb, ub):
            return jnp.any(x < lb) | jnp.any(x > ub)

        LBFGS_refinement = bool(LBFGS_refinement) or bool(
            np.asarray(check_bounds(x, lb, ub))
        )
    
    # print("LBFGS_refinement:", 1)
    if np.isnan(x).any():
        print("Wrong initial state:", x)
        print("LBFGS refinement is activated for learning x0!")
        LBFGS_refinement = True
        x = jnp.zeros(nx)

    if LBFGS_refinement:
        # Refine via L-BFGS with very small penalty on x0
        options = lbfgs_options(
            iprint=-1, iters=lbfgs_epochs, lbfgs_tol=1.e-10, memory=100)

        @jax.jit
        def SS_step(x, u):
            y = output_fcn(x, u, params)
            x = state_fcn(x, u, params).reshape(-1)
            return x, y
 
        @jax.jit
        def J(x0):
            _, Yhat = jax.lax.scan(SS_step, x0, U)
            return jnp.sum((Yhat - Y) ** 2) / U.shape[0]+.5*LBFGS_rho_x0*jnp.sum(x0**2)
        if not isstatebounded:
            solver = jaxopt.ScipyMinimize(
                fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options)
            x, state = solver.run(x)
        else:
            solver = jaxopt.ScipyBoundedMinimize(
                fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options)
            x, state = solver.run(x, bounds=(lb, ub))

        print("Refined initial state:", x)
        if verbosity:
            mse_loss = state.fun_val-.5*LBFGS_rho_x0*jnp.sum(x**2)
            print(
                f"\nFinal loss MSE (after LBFGS refinement) = {mse_loss: 8.6f}")
        if return_PX:
            mse_loss = 0.

            # Forward EKF pass
            state = (x, P, mse_loss)
            state, output = jax.lax.scan(EKF_update, state, jnp.hstack((Y, U)))
            if not isLinear:
                XX1, PP1, XX2, PP2, AA = output
            else:
                XX1, PP1, XX2, PP2 = output
            X = jnp.vstack((XX1, XX2[-1]))
    if return_PX:
        return x, P, X
    return x
