from .types import Array, PRNGKey, DType, Shape, DataDict, Initializer
from .random import default_rng, split_rng, rng_wrapper
from .operators import matinv, bop, bdot, bmm, norm, sum_of_square, mean_of_square
from .utils import flatten, tile_over_axis, type2dtype, scalar2jax, make_supervised_train, concat, cbind

import jax
from jax import lax, jit, vmap, random
from jax import numpy as np

__version__ = "0.1.1"
