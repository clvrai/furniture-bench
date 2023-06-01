from typing import Callable, Sequence, Tuple, Dict

import jax.numpy as jnp
from flax import linen as nn

from common import MLP, Encoder, Encoder


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    use_encoder: bool = False

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        features = []
        for k, v in observations.items():
            if self.use_encoder and (k == 'image1' or k == 'image2'):
                features.append(Encoder()(v))
            else:
                features.append(v)
        obs = jnp.concatenate(features, axis=-1)

        critic = MLP((*self.hidden_dims, 1))(obs)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_encoder: bool = False
    training: bool = False,

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray], actions: jnp.ndarray, training:bool=False) -> jnp.ndarray:
        features = []
        for k, v in observations.items():
            if self.use_encoder and (k == 'image1' or k == 'image2'):
                features.append(Encoder()(v))
            else:
                features.append(v)
        if len(actions.shape) == 3:
            # Reduce the redundant dimension
            actions = jnp.squeeze(actions, 1)

        obs = jnp.concatenate(features, axis=-1)
        inputs = jnp.concatenate([obs, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_encoder: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations,
                         use_encoder=self.use_encoder)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations,
                         use_encoder=self.use_encoder)(observations, actions)
        return critic1, critic2
