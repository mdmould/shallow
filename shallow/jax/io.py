import json
import equinox


def save(filename, model):
    with open(filename, 'wb') as f:
        equinox.tree_serialise_leaves(f, model)


def load(filename, model):
    with open(filename, 'rb') as f:
        return equinox.tree_deserialise_leave(f, model)

