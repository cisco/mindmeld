"""NotebookNode - adding attribute access to dicts"""

from ipython_genutils.ipstruct import Struct

class NotebookNode(Struct):
    """A dict-like node with attribute-access"""
    pass

def from_dict(d):
    """Convert dict to dict-like NotebookNode
    
    Recursively converts any dict in the container to a NotebookNode.
    This does not check that the contents of the dictionary make a valid
    notebook or part of a notebook.
    """
    if isinstance(d, dict):
        return NotebookNode({k:from_dict(v) for k,v in d.items()})
    elif isinstance(d, (tuple, list)):
        return [from_dict(i) for i in d]
    else:
        return d


