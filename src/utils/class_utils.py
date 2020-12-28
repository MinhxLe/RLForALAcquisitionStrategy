import inspect


def get_class_kwargs(class_obj, top_dict):
    """
    constructs an object from a class with a top level dict
    """
    args = inspect.getfullargspec(class_obj.__init__)[0]
    class_kwargs = {}
    for arg in args:
        if arg != 'self' and arg in top_dict:
            class_kwargs[arg] = top_dict[arg]
    return class_kwargs



def construct_object(class_obj, top_dict):
    """
    constructs an object from a class with a top level dict
    """
    args = inspect.getfullargspec(class_obj.__init__)[0]
    class_kwargs = {}
    for arg in args :
        if arg != 'self' and arg in top_dict:
            class_kwargs[arg] = top_dict[arg]
    return class_obj(**class_kwargs)
