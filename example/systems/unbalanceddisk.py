import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
# import jax.debug # Import the debug module
## Modeling an Unbalanced Disk
@jax.jit
def unbalanceddisk_state_fcn(x, t, u):
    """
    Continuous-time nonlinear dynamic model of an unbalanced disk
    """

    # Parameters
    omega0 = 11.339846957335382
    delta_th = 0.0
    gamma = 1.3328339309394384
    Ku = 28.136158407237073
    Fc = 6.062729509386865
    coulomb_omega = 0.001

    # Initialization
    dxdt = jnp.zeros_like(x)

    # Use jax.debug.print for printing inside JIT-compiled functions
    # jax.debug.print("unbalanceddisk_state_fcn called with x: {}, t: {}, u: {}", x, t, u)

    # State equations
    dxdt = dxdt.at[0].set(x[1])
    dxdt = dxdt.at[1].set(-omega0**2 * jnp.sin(x[0] + delta_th) - (gamma * x[1] + Fc * jnp.tanh(x[1] / coulomb_omega)) + Ku * u[0])

    return dxdt

@jax.jit
def unbalanceddisk_output_fcn(x, u = 0.0, params={}):
    return x[1]

# @jax.jit
def unbalanceddisk_process(x0, u, qx=0., qy=0., key=jax.random.PRNGKey(0)):
    Ts = 0.025  # Time step size
    t = jnp.linspace(0, Ts, num=2)

    x1 = odeint(unbalanceddisk_state_fcn, x0, t, u)  # Integrate the ODE

    y1 = unbalanceddisk_output_fcn(x1[-1])
    noise = jax.random.normal(key) * qy
    y1 = y1 + y1 * noise  # noise - 10% of measurement

    return x1[-1], y1

def test_unbalanceddisk_process(seed=0):
    key = jax.random.PRNGKey(seed)

    x0 = jnp.zeros((2,1))  # Initial state

    num_steps = 1000  # Number of time steps

    U = jnp.arange(-1., 1, 0.01)  # Input
    qx = 0.01  # state noise
    qy = 0.01  # output noise
    y_sequence = []
    
    for _ in range(num_steps):
        key, subkey = jax.random.split(key)
        u = jnp.array([U[jax.random.randint(subkey, (), 0, len(U))]])
        x0, y = unbalanceddisk_process(x0, u, qx, qy, subkey)

        y_sequence.append(y)

    y_sequence = jnp.stack(y_sequence)

    print("Final state:", x0)
    print("Output sequence with noise:", y)

    # Plot the output sequence
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(y_sequence, label='Output sequence')
    plt.xlabel('Time step')
    plt.ylabel('Output value')
    plt.title('Output sequence of the robot-arm process')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_unbalanceddisk_process()

    x_10 = jnp.array([0., 0])
    u = jnp.array([0.5])
    p_10 = 0
    @jax.jit
    def Cx_fcn(x, u, p):
        return jax.jacrev(lambda x: unbalanceddisk_output_fcn(x, u, p))(x)
    Cx = Cx_fcn(x_10, u, p_10)
    print(Cx)
