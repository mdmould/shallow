import equinox


def save(filename, model):
    equinox.tree_serialise_leaves(filename, model)


def load(filename, model):
    return equinox.tree_deserialise_leave(filename, model)
