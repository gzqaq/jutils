from .types import PRNGKey

from functools import partial
from jax import jit, random


@jit
def default_rng() -> PRNGKey:
  return random.PRNGKey(42)


@partial(jit, static_argnames=("n",))
def split_rng(rng: PRNGKey, n: int = 2) -> PRNGKey:
  return random.split(rng, n)


def rng_wrapper(func):
  def wrapped(rng, *args, **kwargs):
    rng, _rng = split_rng(rng)
    return rng, func(_rng, *args, **kwargs)

  return wrapped
