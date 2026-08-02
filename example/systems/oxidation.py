import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental.ode import odeint
from jax import debug
@jit
def oxidation_state_fcn(x, t, u):
    """
    Continuous-time nonlinear dynamic model of an ethylene oxidation plant
    """
    gam1 = -8.13
    gam2 = -7.12
    gam3 = -11.07
    A1 = 92.80
    A2 = 12.66
    A3 = 2412.71
    B1 = 7.32
    B2 = 10.39
    B3 = 2170.57
    B4 = 7.02
    Tc = 1.0

    u = jnp.array([u[0], 0.5])

    def safe_power(base, exp):
        return jnp.where(base >= 0, jnp.power(base, exp), 0.0)

    r1 = jnp.exp(gam1 / x[3]) * safe_power(x[1] * x[3], 0.5)
    r2 = jnp.exp(gam2 / x[3]) * safe_power(x[1] * x[3], 0.25)
    r3 = jnp.exp(gam3 / x[3]) * safe_power(x[2] * x[3], 0.5)

    dxdt0 = u[0] * (1 - x[0] * x[3])
    dxdt1 = u[0] * (u[1] - x[1] * x[3]) - A1 * r1 - A2 * r2
    dxdt2 = -u[0] * x[2] * x[3] + A1 * r1 - A3 * r3
    dxdt3 = (u[0] * (1 - x[3]) + B1 * r1 + B2 * r2 + B3 * r3 - B4 * (x[3] - Tc)) / x[0]

    return jnp.array([dxdt0, dxdt1, dxdt2, dxdt3])

@jit
def oxidation_output_fcn(x, u = 0, params = {}):
    return x[2]

@jit
def oxidation_process(x0, u, qx = 0., qy = 0., key = jax.random.PRNGKey(0)):
    """
    Continuous-time nonlinear dynamic model of an ethylene oxidation plant
    (c) 2024 K. Xie
    """
    Ts = 0.5  # Time step size
    t = jnp.linspace(0, Ts, num=2)

    x1 = odeint(oxidation_state_fcn, x0, t, u)[-1] # Integrate the ODE

    y1 = oxidation_output_fcn(x1)
    noise = jax.random.normal(key) * qy
    y1 = y1 + y1 * noise  # noise - 10% of measurement

    return x1, y1

# @jit
def test_oxidation_process(seed=0):
    import matplotlib.pyplot as plt
    key = jax.random.PRNGKey(seed)

    x0 = jnp.array([0.9981, 0.4291, 0.0303, 1.0019])  # Initial state
    U = jnp.arange(0.0704, 0.7042, 0.01)  # Input
    qx = 0.01  # state noise
    qy = 0.1  # output noise

    num_steps = 100  # Number of time steps
    Ts = 5  # Time step size
    x = x0
    y_sequence = []

    for _ in range(num_steps):
        key, subkey = jax.random.split(key)
        u = jnp.array([U[jax.random.randint(subkey, (), 0, len(U))]])
        print("u",u)
        x, y = oxidation_process(x, u, qx, qy, subkey)
        y_sequence.append(y)

    y_sequence = jnp.stack(y_sequence)

    print("Final state:", x)
    print("Output sequence with noise:", y_sequence)

    # Plot the output sequence
    plt.figure()
    plt.plot(y_sequence, label='Output sequence')
    plt.xlabel('Time step')
    plt.ylabel('Output value')
    plt.title('Output sequence of the oxidation process')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_oxidation_process()

    import jax
    import jax.numpy as jnp
    
    x_10 = jnp.array([0.9981, 0.4291, 0.0303, 1.0019])
    u = jnp.array([0.5])
    p_10 = 0
    @jax.jit
    def Cx_fcn(x, u, p):
        return jax.jacrev(lambda x: oxidation_output_fcn(x, u, p))(x)
    Cx = Cx_fcn(x_10, u, p_10)
    print(Cx)
