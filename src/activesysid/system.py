"""Dynamical-system definition used by identification experiments."""


class System:
    """Container describing a dynamical system and its simulation settings.

    Parameters
    ----------
    nx, nu, ny:
        Numbers of states, inputs, and outputs.
    state_fcn, output_fcn:
        State-transition and output callables.
    params:
        Parameters passed to the system callables.
    """

    def __init__(
        self,
        nx,
        nu,
        ny,
        state_fcn=None,
        output_fcn=None,
        params=None,
        x0=None,
        u_set=None,
        Ts=None,
        qx=None,
        qy=None,
        const=None,
        uPWC=None,
        temporality=None,
    ):
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.state_fcn = state_fcn
        self.output_fcn = output_fcn
        self.params = {} if params is None else params
        self.temporality = temporality
        self.x0 = x0
        self.u_set = u_set
        self.Ts = Ts
        self.qx = qx
        self.qy = qy
        self.const = const
        self.uPWC = uPWC
