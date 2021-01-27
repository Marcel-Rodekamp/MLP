import torch
import functools

HANDLED_FUNCTIONS = {}

def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator
