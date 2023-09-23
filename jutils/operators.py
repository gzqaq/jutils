from .types import Array

from functools import partial
from jax import jit
from jax import numpy as np
from typing import Optional, Union, Sequence


@partial(jit, static_argnames=("axis",))
def flatten(x: Array, axis: int = -1) -> Array:
  return x.reshape(-1, *x.shape[axis:])


@jit
def matinv(mat: Array) -> Array:
  inv = np.linalg.inv(mat)
  return np.where(
      np.isclose(0, np.linalg.det(mat))[..., None, None], np.inf, inv)


@jit
def bop(v1: Array, v2: Array) -> Array:
  batch_shape = v1.shape[:-1]
  return np.einsum("bi,bj->bij", flatten(v1),
                   flatten(v2)).reshape(*batch_shape, v1.shape[-1],
                                        v2.shape[-1])


@jit
def bdot(v1: Array, v2: Array) -> Array:
  batch_shape = v1.shape[:-1]
  return np.einsum("bi,bi->b", flatten(v1), flatten(v2)).reshape(*batch_shape)


@jit
def bmm(mat1: Array, mat2: Array) -> Array:
  batch_shape = mat1.shape[:-2]
  return np.einsum("bik,bkj->bij", flatten(mat1, -2),
                   flatten(mat2, -2)).reshape(*batch_shape, mat1.shape[-2],
                                              mat2.shape[-1])


norm = np.linalg.norm


@partial(jit, static_argnames=("axis", "keepdims"))
def sum_of_square(x: Array,
                  axis: Optional[Union[int, Sequence[int]]] = None,
                  keepdims: bool = False) -> Array:
  return np.sum(np.square(x), axis=axis, keepdims=keepdims)
