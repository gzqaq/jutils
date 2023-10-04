from .types import Array, PRNGKey, DType, Shape, VariableDict, DataDict, Initializer, ApplyFunction
from .random import default_rng, split_rng, rng_wrapper
from .operators import flatten, matinv, bop, bdot, bmm, norm, sum_of_square

import jax
from jax import lax, jit, vmap, random
from jax import numpy as np

__version__ = "0.1.0"
