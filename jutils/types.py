import chex
from typing import Callable, Any, Dict

Array = chex.Array
Scalar = chex.Scalar
PRNGKey = chex.PRNGKey
DType = chex.ArrayDType
Shape = chex.Shape

DataDict = Dict[str, Array]
Initializer = Callable[[PRNGKey, Shape, DType], Any]
