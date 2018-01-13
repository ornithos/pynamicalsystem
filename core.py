class pynamicalsystem:
    """pynamicalsystems: a class implementing linear and nonlinear (discrete time)
    dynamical systems. These are also known as 'state space models' in the machine
    learning literature."""
    import numpy as np

    # attributes of pynamicalsystems for reference:
    _defaults = {
        "opts": {"warnings", True, "verbose", True},
        "stack": [],
        "par": {
            "x0": {
                "mu": None,"sigma": None,
                },
            "A": None, "H": None, "B": None, "C": None,
            "Q": None, "R": None, "c": None, "f": None,
            "Df": None, "h": None, "Dh": None,
            "evoNLParams": None, "emiNLParams": None
            },
        "x": None,
        "y": None,
        "yhat": None,
        "d": None,
        "u": None,
        "evoLinear": None,
        "evoNLhasParams": False,
        "emiLinear": None,
        "emiNLhasParams": False,
        "hasControl": np.zeros([2, 1]),
        "stackptr": 0,
        "infer": {
            "filter": None,
            "smooth": None,
            "llh": None,
            "fType": None,
            "sType": None,
            "fpHash": None
        }
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)

