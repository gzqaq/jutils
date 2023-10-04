import chex
from flax.core.frozen_dict import FrozenDict
from typing import Callable, Any, Dict, Tuple, Union

Array = chex.Array
PRNGKey = chex.PRNGKey
DType = chex.ArrayDType
Shape = chex.Shape

VariableDict = Union[FrozenDict[str, Any], Dict[str, Any]]
DataDict = Union[FrozenDict[str, Array], Dict[str, Array]]

Initializer = Callable[[PRNGKey, Shape, DType], Any]
ApplyFunction = Callable[[VariableDict, Any], Any]
