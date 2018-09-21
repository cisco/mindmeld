from copy import deepcopy

class SelectiveDeepcopyDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)
        self.copy_self = False
        self.parent = None

    def enable_deepcopy(self):
        parent = self
        while isinstance(parent, SelectiveDeepcopyDict) and not parent.copy_self:
            parent.copy_self = True
            parent = parent.parent

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if isinstance(value, SelectiveDeepcopyDict):
            value.parent = self

    def __deepcopy__(self, memo):
        if self.copy_self:
            result = { k: (deepcopy(v, memo) if isinstance(v, SelectiveDeepcopyDict) else v) for k, v in self.items() }
            self.copy_self = False
        else:
            result = self
        memo[id(self)] = result
        return result

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
