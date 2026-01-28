"""
MuJoCo Cart-Pole Environment

Based on the classic cart-pole problem from Barto et al. (1983).
State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
Action: continuous force applied to cart
"""

import numpy as np
import mujoco


CART_POLE_XML = """
<mujoco model="cart_pole">
  <!-- Barto et al. (1983) parameters -->
  <option timestep="0.02" integrator="Euler" gravity="0 0 -9.8">
    <flag contact="disable"/>
  </option>

  <default>
    <joint armature="0"/>
    <geom contype="0" conaffinity="0" rgba="0.7 0.7 0.7 1"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="fixed" pos="0 -4 1.5" xyaxes="1 0 0 0 0 1"/>

    <!-- Track/rail for the cart -->
    <geom type="box" size="2.4 0.05 0.05" pos="0 0 -0.1" rgba="0.3 0.3 0.3 1"/>

    <!-- Cart body: m_c = 1.0 kg -->
    <body name="cart" pos="0 0 0">
      <joint name="cart_joint" type="slide" axis="1 0 0" limited="true" range="-2.4 2.4"
             frictionloss="0.0005"/>
      <geom name="cart_geom" type="box" size="0.2 0.1 0.05" mass="1.0" rgba="0.2 0.6 0.8 1"/>

      <!-- Pole attached to cart: m = 0.1 kg, half-length l = 0.5 m (total = 1.0 m) -->
      <body name="pole" pos="0 0 0.05">
        <joint name="pole_joint" type="hinge" axis="0 1 0" limited="false"
               frictionloss="0.000002"/>
        <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 1.0" size="0.02" mass="0.1" rgba="0.9 0.4 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- F_t = +/- 10.0 N applied to cart -->
    <motor name="cart_motor" joint="cart_joint" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


class CartPoleEnv:
    """
    MuJoCo Cart-Pole Environment

    Physical parameters from Barto et al. (1983):
        - Cart mass: 1.0 kg
        - Pole mass: 0.1 kg
        - Pole half-length: 0.5 m (total length 1.0 m)
        - Gravity: -9.8 m/s²
        - Cart friction: 0.0005
        - Pole hinge friction: 0.000002
        - Timestep: 0.02 s
        - Integrator: Euler

    State space (4D):
        [0] cart_position: position of cart on track (meters)
        [1] cart_velocity: velocity of cart (m/s)
        [2] pole_angle: angle of pole from vertical (radians, 0 = upright)
        [3] pole_angular_velocity: angular velocity of pole (rad/s)

    Action space (1D continuous):
        Force applied to cart, clipped to [-10, 10] Newtons

    Reward (cost formulation):
        -0.01 per timestep (cost to minimize; full 1000-step episode = -10)

    Termination:
        - Pole angle exceeds ±12 degrees (±0.209 radians)
        - Cart position exceeds ±2.4 meters
        - Episode length exceeds max_steps
    """

    def __init__(self, max_steps=1000):
        self.max_steps = max_steps

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_string(CART_POLE_XML)
        self.data = mujoco.MjData(self.model)

        # Environment bounds
        self.x_threshold = 2.4  # cart position limit (meters)
        self.theta_threshold = 12 * np.pi / 180  # pole angle limit (radians, ~12 degrees)

        # Dimension info
        self.state_dim = 4
        self.action_dim = 1

        # Episode tracking
        self.steps = 0
        self.renderer = None

    def _get_state(self):
        """Extract state from MuJoCo data."""
        cart_pos = self.data.qpos[0]  # cart position
        pole_angle = self.data.qpos[1]  # pole angle (0 = upright)
        cart_vel = self.data.qvel[0]  # cart velocity
        pole_angular_vel = self.data.qvel[1]  # pole angular velocity

        return np.array([cart_pos, cart_vel, pole_angle, pole_angular_vel], dtype=np.float32)

    def reset(self, seed=None):
        """
        Reset environment to initial state.

        Returns:
            state: initial state vector
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Small random initial state (as per Barto et al.)
        self.data.qpos[0] = np.random.uniform(-0.05, 0.05)  # cart position
        self.data.qpos[1] = np.random.uniform(-0.05, 0.05)  # pole angle
        self.data.qvel[0] = np.random.uniform(-0.05, 0.05)  # cart velocity
        self.data.qvel[1] = np.random.uniform(-0.05, 0.05)  # pole angular velocity

        # Forward dynamics to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        return self._get_state()

    def step(self, action):
        """
        Take one environment step.

        Args:
            action: force to apply to cart (scalar or 1D array)

        Returns:
            state: new state vector
            reward: reward for this step
            done: whether episode has ended
            info: additional information dict
        """
        # Handle action input
        if isinstance(action, np.ndarray):
            action = action.flatten()[0]
        action = np.clip(action, -10.0, 10.0)

        # Apply action and step simulation
        self.data.ctrl[0] = action
        mujoco.mj_step(self.model, self.data)

        self.steps += 1

        # Get new state
        state = self._get_state()
        cart_pos, cart_vel, pole_angle, pole_angular_vel = state

        # Check termination conditions
        done = bool(
            abs(cart_pos) > self.x_threshold or
            abs(pole_angle) > self.theta_threshold or
            self.steps >= self.max_steps
        )

        # Cost: -0.01 per step (so max episode of 1000 steps = -10 cost)
        # This matches the GAE paper's Figure 2 scale
        reward = -0.01

        info = {
            'cart_position': cart_pos,
            'pole_angle': pole_angle,
            'steps': self.steps,
            'truncated': self.steps >= self.max_steps and not (
                abs(cart_pos) > self.x_threshold or
                abs(pole_angle) > self.theta_threshold
            )
        }

        return state, reward, done, info

    def render(self):
        """
        Render current frame.

        Returns:
            pixels: RGB array of rendered frame
        """
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.renderer.update_scene(self.data, camera="fixed")
        return self.renderer.render()

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    @property
    def dt(self):
        """Simulation timestep in seconds."""
        return self.model.opt.timestep


def test_random_policy():
    """Test the environment with random actions."""
    env = CartPoleEnv(max_steps=200)

    print("Cart-Pole Environment")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Timestep: {env.dt}s")
    print()

    # Run a few episodes
    for episode in range(3):
        state = env.reset()
        total_cost = 0

        while True:
            action = np.random.uniform(-10, 10)  # random force
            state, cost, done, info = env.step(action)
            total_cost += cost

            if done:
                break

        print(f"Episode {episode + 1}: {info['steps']} steps, cost={total_cost:.2f}, "
              f"cart_pos={info['cart_position']:.3f}, "
              f"pole_angle={np.degrees(info['pole_angle']):.1f}°")

    env.close()
    print("\nEnvironment test passed")


if __name__ == "__main__":
    test_random_policy()
