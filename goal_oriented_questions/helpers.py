from torch import Tensor


def ns(iterable):
    # if isinstance(iterable, Tensor):
    # return iterable.shape
    try:
        return iterable.shape
    except:
        pass
    # if isinstance(iterable, (list, tuple)):
    try:
        return [len(iterable)] + [
            ns(item) for item in iterable  # if isinstance(item, (list, tuple))
        ]
        print("hi")
    except TypeError:
        return 1
    # else:
    # return 1
