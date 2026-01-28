import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from envs.cart_pole_env import CartPoleEnv

class LinearGaussianPolicy:
    """
    Linear policy with Gaussian exploration.
    π(a|s) = N(W @ s + b, σ²)
    """

    def __init__(self, state_dim, action_dim, init_std=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Initialize weights to zero (as is common for policy gradient)
        self.W = np.zeros((action_dim, state_dim))
        self.b = np.zeros(action_dim)
        self.log_std = np.log(init_std) * np.ones(action_dim)

    def forward(self, state):
        """Compute action mean for a given state."""
        return self.W @ state + self.b

    def sample(self, state):
        """Sample action from π(a|s)."""
        mean = self.forward(state)
        std = np.exp(self.log_std)
        return mean + std * np.random.randn(*mean.shape)

    def log_prob(self, state, action):
        """Compute log π(a|s) for given state-action pair."""
        mean = self.forward(state)
        std = np.exp(self.log_std)
        var = std ** 2
        # Gaussian log probability
        log_p = -0.5 * np.sum(((action - mean) ** 2) / var + 2 * self.log_std + np.log(2 * np.pi))
        return log_p

    def log_prob_batch(self, states, actions):
        """Compute log π(a|s) for batch of state-action pairs."""
        means = states @ self.W.T + self.b  # (N, action_dim)
        std = np.exp(self.log_std)
        var = std ** 2
        # Gaussian log probability for each sample
        log_probs = -0.5 * np.sum(((actions - means) ** 2) / var + 2 * self.log_std + np.log(2 * np.pi), axis=1)
        return log_probs

    def kl_divergence(self, states, old_means, old_log_std):
        """Compute mean KL divergence between old and current policy."""
        new_means = states @ self.W.T + self.b
        old_std = np.exp(old_log_std)
        new_std = np.exp(self.log_std)

        # KL(old || new) for diagonal Gaussians
        # = sum[ log(new_std/old_std) + (old_std^2 + (old_mean - new_mean)^2) / (2*new_std^2) - 0.5 ]
        kl = np.sum(
            self.log_std - old_log_std +
            (old_std ** 2 + (old_means - new_means) ** 2) / (2 * new_std ** 2) - 0.5,
            axis=1
        )
        return np.mean(kl)

    def get_params(self):
        """Return flat parameter vector θ."""
        return np.concatenate([self.W.flatten(), self.b, self.log_std])

    def set_params(self, params):
        """Set parameters from flat vector θ."""
        w_size = self.action_dim * self.state_dim
        b_size = self.action_dim
        self.W = params[:w_size].reshape(self.action_dim, self.state_dim)
        self.b = params[w_size:w_size + b_size]
        self.log_std = params[w_size + b_size:]

    def get_param_shapes(self):
        """Return shapes for unflattening."""
        return [
            ('W', (self.action_dim, self.state_dim)),
            ('b', (self.action_dim,)),
            ('log_std', (self.action_dim,))
        ]

class ValueFunction:
    """
    Neural network value function with one hidden layer.
    V(s) = W2 @ tanh(W1 @ s + b1) + b2
    Architecture from paper: 20-unit hidden layer
    """

    def __init__(self, state_dim, hidden_size=20):
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        # Xavier initialization
        self.W1 = np.random.randn(hidden_size, state_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(1, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(1)

    def forward(self, state):
        """Compute V(s) for a single state."""
        hidden = np.tanh(self.W1 @ state + self.b1)
        return (self.W2 @ hidden + self.b2)[0]

    def forward_batch(self, states):
        """Compute V(s) for a batch of states. states: (N, state_dim)"""
        hidden = np.tanh(states @ self.W1.T + self.b1)  # (N, hidden_size)
        return (hidden @ self.W2.T + self.b2).flatten()  # (N,)

    def get_params(self):
        """Return flat parameter vector φ."""
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def set_params(self, params):
        """Set parameters from flat vector φ."""
        idx = 0
        w1_size = self.hidden_size * self.state_dim
        self.W1 = params[idx:idx + w1_size].reshape(self.hidden_size, self.state_dim)
        idx += w1_size
        self.b1 = params[idx:idx + self.hidden_size]
        idx += self.hidden_size
        w2_size = self.hidden_size
        self.W2 = params[idx:idx + w2_size].reshape(1, self.hidden_size)
        idx += w2_size
        self.b2 = params[idx:idx + 1]

def compute_gae(rewards, values, gamma, lam, done):
    """
    Compute Generalized Advantage Estimation.
    GAE(γ,λ): Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
    where δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0

    # Work backwards
    for t in reversed(range(T)):
        if done[t]:
            # Terminal state: no bootstrapping
            delta = rewards[t] - values[t]
            gae = delta
        else:
            # TD residual: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            # GAE: Â_t = δ_t + (γλ) * Â_{t+1}
            gae = delta + gamma * lam * gae

        advantages[t] = gae

    return advantages


def compute_returns(rewards, gamma, done):
    """
    Compute discounted returns (for value function targets).
    G_t = Σ_{l=0}^{∞} γ^l r_{t+l}
    """
    T = len(rewards)
    returns = np.zeros(T)
    G = 0

    for t in reversed(range(T)):
        if done[t]:
            G = rewards[t]
        else:
            G = rewards[t] + gamma * G
        returns[t] = G

    return returns

def compute_log_prob_grad(policy, state, action):
    """
    Compute ∇_θ log π(a|s) for a single state-action pair.
    For Gaussian: ∇_θ log π = (a - μ) / σ² * ∇_θ μ  for mean params
                            = ((a - μ)² / σ² - 1)  for log_std params
    """
    mean = policy.forward(state)
    std = np.exp(policy.log_std)
    var = std ** 2

    diff = action - mean  # (action_dim,)

    # Gradient w.r.t. W: ∂log_p/∂W = (a - μ) / σ² ⊗ s
    # For linear policy: μ = W @ s + b, so ∂μ/∂W = s (outer product structure)
    grad_W = (diff / var).reshape(-1, 1) @ state.reshape(1, -1)  # (action_dim, state_dim)

    # Gradient w.r.t. b: ∂log_p/∂b = (a - μ) / σ²
    grad_b = diff / var  # (action_dim,)

    # Gradient w.r.t. log_std: ∂log_p/∂log_std = (a - μ)² / σ² - 1
    grad_log_std = (diff ** 2) / var - 1  # (action_dim,)

    return np.concatenate([grad_W.flatten(), grad_b, grad_log_std])


def compute_policy_gradient(policy, states, actions, advantages):
    """
    Compute vanilla policy gradient.
    g = (1/N) Σ_n ∇_θ log π(a_n|s_n) * Â_n
    """
    N = len(states)
    grad = np.zeros_like(policy.get_params())

    for i in range(N):
        log_prob_grad = compute_log_prob_grad(policy, states[i], actions[i])
        grad += log_prob_grad * advantages[i]

    return grad / N

def compute_fisher_vector_product(policy, states, vector, damping=0.1):
    """
    Compute Fisher-vector product: (F + damping * I) @ v
    F = (1/N) Σ_n ∇_θ log π(a_n|s_n) ∇_θ log π(a_n|s_n)^T

    We compute Fv without forming F explicitly.
    """
    N = len(states)

    # For each state, compute ∇_θ log π (evaluated at mean action for FIM)
    # The Fisher information matrix uses the expected outer product
    Fv = np.zeros_like(vector)

    for i in range(N):
        state = states[i]
        mean = policy.forward(state)
        std = np.exp(policy.log_std)
        var = std ** 2

        # Gradient structure for Gaussian policy at the mean
        # ∇_θ log π structure (without the (a-μ) term, which averages to identity under expectation)

        # For W: each row of W has gradient = s / σ²
        # For b: gradient = 1 / σ²
        # For log_std: gradient contribution from variance term

        # Simpler: compute the gradient of KL divergence (Hessian of KL = FIM)
        # Actually, let's use the analytic FIM for Gaussian

        # FIM for diagonal Gaussian w.r.t. (W, b, log_std):
        # Block diagonal structure
        # For mean parameters: F_μ = 1/σ² * E[∇μ ∇μ^T]
        # For W: F_W[i,j,k,l] = s_j s_l / σ_i² δ_ik
        # For log_std: F_σ = 2

        # Let's extract the vector components
        w_size = policy.action_dim * policy.state_dim
        b_size = policy.action_dim
        v_W = vector[:w_size].reshape(policy.action_dim, policy.state_dim)
        v_b = vector[w_size:w_size + b_size]
        v_log_std = vector[w_size + b_size:]

        # FIM @ v for W component: (1/σ²) * (s s^T) @ v_W
        # = (1/σ²) * s * (s^T @ v_W) for each action dim
        s_outer_v = state * (state @ v_W.T)  # (action_dim,) - one per action dim
        Fv_W = (s_outer_v / var.reshape(-1, 1))

        # FIM @ v for b component: (1/σ²) * v_b
        Fv_b = v_b / var

        # FIM @ v for log_std: 2 * v_log_std (Fisher info for variance param)
        Fv_log_std = 2 * v_log_std

        Fv += np.concatenate([Fv_W.flatten(), Fv_b, Fv_log_std])

    Fv = Fv / N

    # Add damping for numerical stability
    Fv += damping * vector

    return Fv

def conjugate_gradient(f_Ax, b, max_iters=10, residual_tol=1e-10):
    """
    Conjugate gradient algorithm to solve Ax = b.
    Returns approximate solution x ≈ A^{-1} b
    """
    x = np.zeros_like(b)
    r = b.copy()  # residual = b - Ax, and x=0 initially
    p = r.copy()  # search direction
    r_dot_r = r @ r

    for i in range(max_iters):
        Ap = f_Ax(p)
        alpha = r_dot_r / (p @ Ap + 1e-8)
        x += alpha * p
        r -= alpha * Ap

        r_dot_r_new = r @ r
        if r_dot_r_new < residual_tol:
            break

        beta = r_dot_r_new / r_dot_r
        p = r + beta * p
        r_dot_r = r_dot_r_new

    return x

def trpo_update(policy, states, actions, advantages, max_kl=0.01):
    """
    Trust Region Policy Optimization update.

    1. Compute policy gradient g
    2. Compute natural gradient direction: s = F^{-1} g (using CG)
    3. Compute step size to satisfy KL constraint: β = sqrt(2δ / s^T F s)
    4. Line search to ensure improvement
    5. Update: θ_new = θ_old + β * s
    """
    # Save old policy parameters and compute old action distributions
    old_params = policy.get_params().copy()
    old_means = states @ policy.W.T + policy.b
    old_log_std = policy.log_std.copy()

    # 1. Compute policy gradient
    g = compute_policy_gradient(policy, states, actions, advantages)

    # 2. Compute natural gradient direction using CG: s ≈ F^{-1} g
    def fvp(v):
        return compute_fisher_vector_product(policy, states, v)

    step_dir = conjugate_gradient(fvp, g, max_iters=10)

    # 3. Compute step size: β = sqrt(2δ / s^T F s)
    sFs = step_dir @ fvp(step_dir)
    if sFs <= 0:
        print("Warning: sFs <= 0, skipping update")
        return

    beta = np.sqrt(2 * max_kl / (sFs + 1e-8))

    # 4. Line search with backtracking
    # Compute old surrogate loss for comparison
    old_log_probs = policy.log_prob_batch(states, actions)
    old_surrogate = np.mean(old_log_probs * advantages)

    # Try full step, then backtrack if needed
    for fraction in [1.0, 0.5, 0.25, 0.125]:
        new_params = old_params + fraction * beta * step_dir
        policy.set_params(new_params)

        # Check KL constraint
        kl = policy.kl_divergence(states, old_means, old_log_std)

        # Check surrogate improvement
        new_log_probs = policy.log_prob_batch(states, actions)
        new_surrogate = np.mean(new_log_probs * advantages)

        # Accept if KL constraint satisfied and surrogate improves
        if kl <= max_kl * 1.5 and new_surrogate >= old_surrogate:
            return

    # If no step accepted, revert to old parameters
    policy.set_params(old_params)

def update_value_function(value_fn, states, returns, epochs=10, lr=0.01):
    """
    Update value function via gradient descent on MSE loss.
    minimize Σ_n ||V_φ(s_n) - G_n||²
    """
    N = len(states)

    for epoch in range(epochs):
        # Forward pass
        predictions = value_fn.forward_batch(states)
        errors = predictions - returns  # (N,)

        # Compute gradients via backprop
        # V(s) = W2 @ tanh(W1 @ s + b1) + b2
        hidden_pre = states @ value_fn.W1.T + value_fn.b1  # (N, hidden)
        hidden = np.tanh(hidden_pre)  # (N, hidden)

        # Gradient of loss w.r.t. output: 2 * (pred - target) / N
        d_output = 2 * errors / N  # (N,)

        # Gradient w.r.t. W2: d_output @ hidden
        grad_W2 = d_output.reshape(-1, 1).T @ hidden  # (1, hidden)

        # Gradient w.r.t. b2
        grad_b2 = np.sum(d_output, keepdims=True)  # (1,)

        # Backprop through tanh
        d_hidden = d_output.reshape(-1, 1) * value_fn.W2  # (N, hidden)
        d_hidden_pre = d_hidden * (1 - hidden ** 2)  # (N, hidden)

        # Gradient w.r.t. W1
        grad_W1 = d_hidden_pre.T @ states  # (hidden, state_dim)

        # Gradient w.r.t. b1
        grad_b1 = np.sum(d_hidden_pre, axis=0)  # (hidden,)

        # Update parameters
        value_fn.W1 -= lr * grad_W1
        value_fn.b1 -= lr * grad_b1
        value_fn.W2 -= lr * grad_W2
        value_fn.b2 -= lr * grad_b2

def collect_trajectories(env, policy, num_trajectories, max_steps):
    """
    Collect batch of trajectories using current policy.
    """
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    episode_returns = []
    episodes = []

    for _ in range(num_trajectories):
        states = []
        actions = []
        rewards = []
        dones = []

        state = env.reset()
        episode_return = 0

        for t in range(max_steps):
            action = policy.sample(state)
            action = np.clip(action, -10, 10)  # Clip to action bounds

            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            episode_return += reward

            if done:
                dones.append(True)
                break
            else:
                dones.append(False)
                state = next_state

        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_dones.extend(dones)
        episode_returns.append(episode_return)
        episodes.append(states)

    return {
        'states': np.array(all_states),
        'actions': np.array(all_actions),
        'rewards': np.array(all_rewards),
        'dones': np.array(all_dones),
        'episode_returns': np.array(episode_returns),
        'episodes': episodes
    }

def train(
    gamma=0.99,
    lam=0.97,
    num_iterations=50,
    trajectories_per_batch=20,
    max_steps_per_trajectory=1000,
    max_kl=0.01,
    seed=None
):
    """
    Main training loop for GAE + TRPO on cart-pole.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize environment
    env = CartPoleEnv(max_steps=max_steps_per_trajectory)

    # Initialize policy (linear) and value function (NN with 20 hidden units)
    policy = LinearGaussianPolicy(state_dim=4, action_dim=1)
    value_fn = ValueFunction(state_dim=4, hidden_size=20)

    # Training history
    history = {
        'iteration': [],
        'mean_cost': [],
        'mean_episode_length': [],
    }

    pbar = tqdm(range(num_iterations), desc="Training", leave=False)
    for iteration in pbar:
        # ----- Collect trajectories -----
        batch = collect_trajectories(
            env, policy,
            num_trajectories=trajectories_per_batch,
            max_steps=max_steps_per_trajectory
        )

        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']

        # Compute value estimates
        # We need V(s_t) for all t, plus V(s_T) for bootstrapping
        values = value_fn.forward_batch(states)
        # Append a zero for terminal states (or could bootstrap if not terminal)
        values = np.append(values, 0)

        # Compute GAE advantages
        advantages = compute_gae(rewards, values, gamma, lam, dones)

        # Compute returns for value function targets
        returns = compute_returns(rewards, gamma, dones)

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Negate advantages for cost minimization
        advantages = -advantages

        # TRPO policy update
        trpo_update(policy, states, actions, advantages, max_kl)

        # Value function update
        update_value_function(value_fn, states, returns, epochs=10, lr=0.01)

        # Logging
        mean_cost = np.mean(batch['episode_returns'])
        mean_length = np.mean([len(ep) for ep in batch['episodes']])

        history['iteration'].append(iteration)
        history['mean_cost'].append(mean_cost)
        history['mean_episode_length'].append(mean_length)

        pbar.set_postfix({'cost': f'{mean_cost:.2f}', 'len': f'{mean_length:.0f}'})

    env.close()
    return policy, value_fn, history


def plot_results(results, num_iterations=50, whisker_interval=10):
    """
    Plot cost vs policy iterations for different lambda values. (Figure 2 in Schulman et al).

    Args:
        results: dict mapping lambda -> list of histories (one per seed)
        num_iterations: number of policy iterations
        whisker_interval: interval at which to show whiskers
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map for different lambda values
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for (lam, histories), color in zip(sorted(results.items()), colors):
        # Stack costs from all seeds: (num_seeds, num_iterations)
        costs = np.array([h['mean_cost'] for h in histories])
        iterations = np.arange(num_iterations)

        # Compute mean across seeds and smooth with Gaussian filter
        mean_cost = np.mean(costs, axis=0)
        mean_cost_smooth_data = gaussian_filter1d(mean_cost, sigma=2)

        # Create smooth spline interpolation
        iterations_smooth = np.linspace(0, num_iterations - 1, 300)
        spline = make_interp_spline(iterations, mean_cost_smooth_data, k=3)
        mean_cost_smooth = spline(iterations_smooth)

        # Plot smooth line
        ax.plot(iterations_smooth, mean_cost_smooth, color=color, label=f'λ={lam}', linewidth=1.5)

        # Add whiskers at specified intervals
        whisker_iters = np.arange(0, num_iterations, whisker_interval)
        for it in whisker_iters:
            if it < len(mean_cost):
                min_cost = np.min(costs[:, it])
                max_cost = np.max(costs[:, it])
                # Draw vertical whisker line
                ax.plot([it, it], [min_cost, max_cost], color=color, linewidth=1.5)
                # Draw whisker caps
                cap_width = 0.8
                ax.plot([it - cap_width, it + cap_width], [min_cost, min_cost], color=color, linewidth=1.5)
                ax.plot([it - cap_width, it + cap_width], [max_cost, max_cost], color=color, linewidth=1.5)

    ax.set_xlabel('number of policy iterations')
    ax.set_ylabel('cost')
    ax.set_xlim(0, num_iterations)
    ax.set_ylim(-2, 0)
    ax.legend(loc='lower left')
    ax.set_title('Cart-pole learning curves (at γ=0.99)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/figure_2.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    # Run experiment matching GAE paper Figure 2
    # Test different lambda values at gamma=0.99
    LAMBDAS = [0, 0.36, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99, 1.0]

    cache_path = Path("cache/cart_pole.pkl")

    if cache_path.exists():
        print(f"Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

        for lam in tqdm(LAMBDAS, desc="Lambda values"):
            # Average over multiple seeds (paper uses 21 seeds)
            all_histories = []
            for seed in tqdm(range(21), desc=f"λ={lam}", leave=False):  # Use fewer seeds for testing; paper uses 21
                policy, value_fn, history = train(
                    gamma=0.99,
                    lam=lam,
                    num_iterations=50,
                    seed=seed
                )
                all_histories.append(history)

            results[lam] = all_histories

        print("\nTraining complete!")

        # Save to cache
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {cache_path}")

    # Plot results
    plot_results(results, num_iterations=50, whisker_interval=10)
