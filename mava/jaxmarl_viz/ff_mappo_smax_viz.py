import pathlib
import hydra
import jax
import jax.numpy as jnp
from typing import Any, Dict, List, Tuple
import chex
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.network_utils import get_action_head
from mava.wrappers.jaxmarl import JaxMarlState, JaxMarlWrapper
from mava.networks import FeedForwardActor as Actor
from mava.systems.ppo.types import Params
from jaxmarl.viz.visualizer import SMAXVisualizer

def unbatchify(x: chex.Array, agents: List[str]) -> Dict[str, chex.Array]:
    """Split array into dictionary entries."""
    return {agent: x[i] for i, agent in enumerate(agents)}

def play_episode(
    env: JaxMarlWrapper,
    actor_apply_fn: callable,
    actor_params: FrozenDict,
    key: chex.PRNGKey,
    max_steps: int = 100
) -> List[Tuple[chex.PRNGKey, Any, chex.Array]]:
    """
    Play a single episode and collect a state sequence.

    Args:
        env: The multi-agent environment (MarlEnv).
        actor_apply_fn: The actor network's apply function.
        actor_params: The trained actor parameters.
        key: PRNG key for action sampling.
        max_steps: Maximum steps to prevent infinite loops.

    Returns:
        state_seq: List of (key_s, state, actions) tuples, where:
            - key_s: PRNG key used for action sampling.
            - state: Environment state.
            - actions: Actions taken by agents.
    """
    # Initialize environment
    env_state, timestep = env.reset(key)
    env_state: JaxMarlState = env_state
    state_seq = []
    step = 0
    done = jnp.zeros((env.num_agents,), dtype=bool)

    while not jnp.all(done) and step < max_steps:
        # Split key for action sampling
        key, policy_key = jax.random.split(key)

        # Select action using actor policy
        actor_policy = actor_apply_fn(actor_params, timestep.observation)
        actions = actor_policy.sample(seed=policy_key)

        # Store the step data
        _, step_key = jax.random.split(env_state.key)
        state_seq.append((step_key, env_state.env_state.state, unbatchify(actions, env.agents)))

        # Step the environment
        env_state, timestep = env.step(env_state, actions)

        # Check if episode is done (assuming timestep.last() is per-agent)
        done = timestep.last().repeat(env.num_agents).reshape(-1)
        step += 1

    return state_seq

@hydra.main(
    config_path="../configs/default",
    config_name="ff_mappo.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    """Load checkpoint and print actor parameter dimensions."""
    OmegaConf.set_struct(cfg, False)
    
    # Initialize PRNG key
    key = jax.random.PRNGKey(cfg.system.seed)
    
    # Initialize actor network
    actor_torso = hydra.utils.instantiate(cfg.network.actor_network.pre_torso)
    env, _ = environments.make(config=cfg, add_global_state=True)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    
    # Initialize dummy params for checkpoint restoration
    obs = env.observation_spec.generate_value()
    init_x = jax.tree.map(lambda x: x[jnp.newaxis, ...], obs)
    key, actor_key = jax.random.split(key)
    actor_params = actor_network.init(actor_key, init_x)
    critic_params = actor_params  # Dummy critic params
    params = Params(actor_params, critic_params)
    
    # Load checkpoint
    checkpointer = Checkpointer(
        model_name=cfg.logger.system_name,
        **cfg.logger.checkpointing.load_args
    )
    restored_params, _ = checkpointer.restore_params(input_params=params)
    
    # Get actor parameters
    actor_params = restored_params.actor_params

    # play an episode
    key, episode_key = jax.random.split(key)
    state_seq = play_episode(
        env=env,
        actor_apply_fn=actor_network.apply,
        actor_params=actor_params,
        key=episode_key,
    )

    # visualize the episode
    viz = SMAXVisualizer(env, state_seq)
    viz.animate(view=False, save_fname= pathlib.Path(cfg.logger.base_exp_path) / "smax_viz.gif")
    

if __name__ == "__main__":
    main()