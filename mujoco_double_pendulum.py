import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import math
import mujoco
from brax import envs
from brax.io import mjcf
from brax.mjx import pipeline

# Mujoco XML definition for the double pendulum
PENDULUM_XML = """
<mujoco>
    <option iterations="50" timestep="0.001" integrator="RK4">
    </option>
    <worldbody>
        <camera name="side" pos="0 5 2" euler="-90 0 180"/>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
        <body name="link1" pos="0 0 2.5">
            <joint name="pin1" type="hinge" axis="0 -1 0" pos="0 0 -0.5"/>
            <geom type="cylinder" size="0.05 0.5" rgba="0 .9 0 1" mass="1"/>
            <body name="link2" pos="0 0 0.5">
                <joint name="pin2" type="hinge" axis="0 -1 0" pos="0 0 0"/>
                <geom type="cylinder" size="0.05 0.5" rgba="0 0 .9 1" mass="1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor joint="pin1" name="motor1" gear="1" ctrlrange="-10 10"/>
        <motor joint="pin2" name="motor2" gear="1" ctrlrange="-10 10"/>
    </actuator>
</mujoco>
"""

# Load Mujoco model and set up system
mj_model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
sys = mjcf.load_model(mj_model)

# Constants
u_upper = 10
u_lower = -10
epsilon = 0.1
dimension = 4  # State dimension: [q1, q2, qd1, qd2]
eta = 0.01
gamma = 0.01
loss_epsilon = 5e-6

# Define state space generation function
def state_space_d(left_bound, right_bound, epsilon, dimension):
    og = jnp.arange(left_bound, right_bound, epsilon)
    grids = jnp.meshgrid(*[og] * dimension, indexing='ij')
    res = jnp.stack(grids, axis=-1).reshape(-1, dimension)
    return res

# Generate state space sets
full = state_space_d(-math.pi/4, math.pi/4, epsilon, dimension)
max_abs = jnp.max(jnp.abs(full), axis=1)  # Infinity norm
init = full[max_abs < math.pi / 20]  # Initial set
unsafe = full[max_abs >= math.pi / 6]  # Unsafe set
safe = full[max_abs < math.pi / 6]  # Safe set

# Neural network definitions
class Barrier(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class Controller(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)  # Output 2D control for two actuators
        return x

# Create jitted apply functions for neural networks
def make_jitted_apply(clasz):
    @jax.jit
    def jitted_apply(params, x):
        return clasz.apply(params, x)
    return jitted_apply

# Initialize neural networks
barrier = Barrier()
controller = Controller()
rng = jax.random.PRNGKey(1337)
x_dummy = jnp.ones((1, 4))  # 4D input: [q1, q2, qd1, qd2]
barrier_params = barrier.init(rng, x_dummy)
controller_params = controller.init(rng, x_dummy)
barrier = make_jitted_apply(barrier)
controller = make_jitted_apply(controller)

# Set up optimizers
lr = 3e-4
barrier_opt = optax.adam(lr)
controller_opt = optax.adam(lr)
barrier_opt_state = barrier_opt.init(barrier_params)
controller_opt_state = controller_opt.init(controller_params)

def make_jitted_update(optimizer):
    @jax.jit
    def jitted_update(grads, opt_state):
        return optimizer.update(grads, opt_state)
    return jitted_update

barrier_opt = make_jitted_update(barrier_opt)
controller_opt = make_jitted_update(controller_opt)

# Dynamics function with custom VJP
@jax.custom_vjp
def dynamics(q, qd, u):
    q = jnp.atleast_1d(q)
    qd = jnp.atleast_1d(qd)
    u = jnp.atleast_1d(u)
    state = pipeline.init(sys, q, qd)
    next_state = pipeline.step(sys, state, u)
    return next_state.qpos, next_state.qvel

def dynamics_fwd(q, qd, u):
    next_q, next_qd = dynamics(q, qd, u)
    return (next_q, next_qd), (q, qd, u)

def dynamics_bwd(res, g):
    q, qd, u = res
    g_next_q, g_next_qd = g
    delta = 1e-5

    # Jacobians w.r.t. q
    dnext_q_dq = jnp.zeros((2, 2))
    dnext_qd_dq = jnp.zeros((2, 2))
    for i in range(2):
        q_plus = q.at[i].add(delta)
        q_minus = q.at[i].add(-delta)
        next_q_plus, next_qd_plus = dynamics(q_plus, qd, u)
        next_q_minus, next_qd_minus = dynamics(q_minus, qd, u)
        dnext_q_dq = dnext_q_dq.at[:, i].set((next_q_plus - next_q_minus) / (2 * delta))
        dnext_qd_dq = dnext_qd_dq.at[:, i].set((next_qd_plus - next_qd_minus) / (2 * delta))

    # Jacobians w.r.t. qd
    dnext_q_dqd = jnp.zeros((2, 2))
    dnext_qd_dqd = jnp.zeros((2, 2))
    for i in range(2):
        qd_plus = qd.at[i].add(delta)
        qd_minus = qd.at[i].add(-delta)
        next_q_plus, next_qd_plus = dynamics(q, qd_plus, u)
        next_q_minus, next_qd_minus = dynamics(q, qd_minus, u)
        dnext_q_dqd = dnext_q_dqd.at[:, i].set((next_q_plus - next_q_minus) / (2 * delta))
        dnext_qd_dqd = dnext_qd_dqd.at[:, i].set((next_qd_plus - next_qd_minus) / (2 * delta))

    # Jacobians w.r.t. u
    dnext_q_du = jnp.zeros((2, 2))
    dnext_qd_du = jnp.zeros((2, 2))
    for j in range(2):
        u_plus = u.at[j].add(delta)
        u_minus = u.at[j].add(-delta)
        next_q_plus, next_qd_plus = dynamics(q, qd, u_plus)
        next_q_minus, next_qd_minus = dynamics(q, qd, u_minus)
        dnext_q_du = dnext_q_du.at[:, j].set((next_q_plus - next_q_minus) / (2 * delta))
        dnext_qd_du = dnext_qd_du.at[:, j].set((next_qd_plus - next_qd_minus) / (2 * delta))

    # Compute gradients
    dL_dq = dnext_q_dq.T @ g_next_q + dnext_qd_dq.T @ g_next_qd
    dL_dqd = dnext_q_dqd.T @ g_next_q + dnext_qd_dqd.T @ g_next_qd
    dL_du = dnext_q_du.T @ g_next_q + dnext_qd_du.T @ g_next_qd

    return dL_dq, dL_dqd, dL_du

dynamics.defvjp(dynamics_fwd, dynamics_bwd)

# Loss functions and training step
@jax.jit
def apply_models(barrier_params, controller_params, init, unsafe, safe, eta, gamma):
    def loss1_fn(barrier_params):
        bvalues = barrier(barrier_params, init)
        loss1 = jnp.mean((bvalues + eta) ** 2)
        return loss1

    def loss2_fn(barrier_params):
        bvalues = barrier(barrier_params, unsafe)
        loss2 = jnp.mean((bvalues - eta) ** 2)
        return loss2

    def single_loss3(barrier_params, controller_params, safe_single):
        q = safe_single[:2]
        qd = safe_single[2:]
        u = controller(controller_params, safe_single)
        next_q, next_qd = dynamics(q, qd, u)
        next_obs = jnp.concatenate([next_q, next_qd])
        bvalue = barrier(barrier_params, next_obs)
        loss3_single = (bvalue + eta) ** 2
        return loss3_single

    batched_loss3 = jax.vmap(single_loss3, in_axes=(None, None, 0))

    def loss3_fn(barrier_params, controller_params):
        bvalues_og = barrier(barrier_params, safe)
        mask = bvalues_og <= gamma
        loss3s = batched_loss3(barrier_params, controller_params, safe)
        masked_loss3s = jnp.where(mask, loss3s, 0.0)
        num_valid = jnp.sum(mask)
        loss3 = jnp.sum(masked_loss3s) / (num_valid + 1e-8)
        return loss3

    def total_loss_fn(barrier_params, controller_params):
        loss1 = loss1_fn(barrier_params)
        loss2 = loss2_fn(barrier_params)
        loss3 = loss3_fn(barrier_params, controller_params)
        return loss1 + loss2 + loss3

    grad_fn = jax.value_and_grad(total_loss_fn, argnums=(0, 1))
    loss, (barrier_grads, controller_grads) = grad_fn(barrier_params, controller_params)
    return barrier_grads, controller_grads, loss

# Update function for model parameters
def update_model(grads, opt, opt_state, params):
    updates, opt_state = opt(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Function to check barrier certificate conditions
def get_combettes_pesquet_lipschitz(params):
    weights = [params['params'][f'Dense_{i}']['kernel'] for i in range(3)]
    nm = len(weights)
    lip = 0.0
    test = weights[0]
    for k in range(1, nm):
        if k > 1:
            test = jnp.matmul(test, weights[k-1])
        if k < nm - 1:
            test2 = weights[k]
            for j in range(k + 1, nm):
                test2 = jnp.matmul(test2, weights[j])
            norm2 = jnp.linalg.svdvals(test2)[0]
        else:
            norm2 = 1.0
        norm1 = jnp.linalg.svdvals(test)[0]
        lip += norm1 * norm2
    return lip / (2 ** (nm - 1))

lx = 1.098  # Lipschitz constant for dynamics w.r.t. state (adjust if necessary)
lu = 0.39   # Lipschitz constant for dynamics w.r.t. control (adjust if necessary)

def check_conditions(barrier, barrier_params, controller, controller_params, init, unsafe, safe):
    barrier_init = barrier(barrier_params, init)
    barrier_unsafe = barrier(barrier_params, unsafe)
    eta_val = jnp.minimum(-jnp.max(barrier_init), jnp.min(barrier_unsafe))
    print(f"found eta is {eta_val}")

    b_lip = get_combettes_pesquet_lipschitz(barrier_params)
    c_lip = get_combettes_pesquet_lipschitz(controller_params)

    lip1 = b_lip * epsilon
    lip2 = lip1 * (lx + c_lip * lu)
    print(f"b_lip is {b_lip}")
    print(f"c_lip is {c_lip}")
    print(f"the small lipschitz is {lip1}")
    print(f"the big lipschitz is {lip2}")

    gamma = lip1
    biggest_epsilon = eta_val / (lip2 / epsilon)
    print(f"the biggest epsilon where it would work is {biggest_epsilon}")

    def single_loss3(barrier_params, controller_params, safe_single):
        q = safe_single[:2]
        qd = safe_single[2:]
        u = controller(controller_params, safe_single)
        next_q, next_qd = dynamics(q, qd, u)
        next_obs = jnp.concatenate([next_q, next_qd])
        bvalue = barrier(barrier_params, next_obs)
        loss3_single = bvalue <= -eta_val
        return loss3_single

    batched_loss3 = jax.vmap(single_loss3, in_axes=(None, None, 0))

    def loss3_fn(barrier_params, controller_params):
        bvalues_og = barrier(barrier_params, safe)
        mask = bvalues_og <= gamma
        loss3s = batched_loss3(barrier_params, controller_params, safe)
        masked_loss3s = jnp.where(mask, loss3s, 0.0)
        num_valid = jnp.sum(mask)
        loss3 = jnp.sum(masked_loss3s) / (num_valid + 1e-8)
        return loss3

    loss3 = loss3_fn(barrier_params, controller_params)
    print(f"loss3 goodness with that eta is {loss3 * 100}%")

    return gamma

# Training loop
def train(barrier_stuff, controller_stuff, init, unsafe, safe, eta, gamma):
    barrier, barrier_params, barrier_opt, barrier_opt_state = barrier_stuff
    controller, controller_params, controller_opt, controller_opt_state = controller_stuff
    i = 0
    while True:
        barrier_grads, controller_grads, loss = apply_models(barrier_params, controller_params, init, unsafe, safe, eta, gamma)

        barrier_params, barrier_opt_state = update_model(barrier_grads, barrier_opt, barrier_opt_state, barrier_params)
        controller_params, controller_opt_state = update_model(controller_grads, controller_opt, controller_opt_state, controller_params)

        if i % 100 == 0:
            print(f"Iteration {i}, loss: {loss}")
            gamma = check_conditions(barrier, barrier_params, controller, controller_params, init, unsafe, safe)
        i += 1

# Package model components and start training
barrier_stuff = (barrier, barrier_params, barrier_opt, barrier_opt_state)
controller_stuff = (controller, controller_params, controller_opt, controller_opt_state)
train(barrier_stuff, controller_stuff, init, unsafe, safe, eta, gamma)
