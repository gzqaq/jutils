from .types import Array, Scalar
from .random import rng_wrapper

import chex
import jax.numpy as np
from functools import partial
from jax import jit, value_and_grad, random, tree_map, lax


@partial(jit, static_argnames=("axis",))
def flatten(a: Array, axis: int = -1) -> Array:
  """Flatten ndarray starting from the given axis."""
  return a.reshape(-1, *a.shape[axis:])

@partial(jit, static_argnames=("axis", "size"))
def tile_over_axis(a: Array, size: int, axis: int = 0) -> Array:
  return np.repeat(np.expand_dims(a, axis), size, axis=axis)

@jit
def concat(*args) -> Array:
  """Flatten all ndarrays and concatenate them. Like c() in R."""
  return np.concatenate(args, axis=None)

@jit
def cbind(*args) -> Array:
  """Concatenate args along the last axis. Like cbind() in R."""
  return np.concatenate(args, axis=-1)

def type2dtype(t):
  if t == int:
    return np.int_
  elif t == float:
    return np.float_
  elif t == bool:
    return np.bool_
  else:
    raise ValueError(f"Unsupported type: {t}")

def scalar2jax(scalar: Scalar):
  chex.assert_scalar(scalar)
  return np.asarray(scalar, dtype=type2dtype(type(scalar)))

def make_supervised_train(n_epochs: int, bs: int, loss_fn):
  def train(rng, state, inp, labels):
    ds_size = inp.shape[0] // bs * bs
    inp, labels = inp[:ds_size], labels[:ds_size]

    def one_epoch(run_state, _):
      def one_batch(state, ds):
        inp, labels = ds
        loss, grads = value_and_grad(loss_fn)(state.params, inp, labels)
        return state.apply_gradients(grads=grads), loss

      rng, state, inp, labels = run_state
      rng, inds = rng_wrapper(random.permutation)(rng, np.arange(ds_size))
      shuffled = tree_map(lambda x: x[inds].reshape(ds_size // bs, bs, *x.shape[1:]), (inp, labels))
      state, losses = lax.scan(one_batch, state, shuffled)

      return (rng, state, inp, labels), losses

    (_, state, *_), losses = lax.scan(one_epoch, (rng, state, inp, labels), None, n_epochs)
    return state, losses

  return jit(train)
