import re
from tensorflow.python.framework import tensor_util

def node_name_from_input(node_name):
    """Strips off ports and other decorations to get the underlying node name."""
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        node_name = m.group(1)
    return node_name


def node_from_map(node_map, name):
    """Pulls a node def from a dictionary for a given name.

    Args:
      node_map: Dictionary containing an entry indexed by name for every node.
      name: Identifies the node we want to find.

    Returns:
      NodeDef of the node with the given name.

    Raises:
      ValueError: If the node isn't present in the dictionary.
    """
    stripped_name = node_name_from_input(name)
    if stripped_name not in node_map:
        raise ValueError("No node named '%s' found in map." % name)
    return node_map[stripped_name]

def values_from_const(node_def):
    """Extracts the values from a const NodeDef as a numpy ndarray.

    Args:
      node_def: Const NodeDef that has the values we want to access.

    Returns:
      Numpy ndarray containing the values.

    Raises:
      ValueError: If the node isn't a Const.
    """
    if node_def.op != "Const":
        raise ValueError(
            "Node named '%s' should be a Const op for values_from_const." %
            node_def.name)
    input_tensor = node_def.attr["value"].tensor
    tensor_value = tensor_util.MakeNdarray(input_tensor)
    return tensor_value