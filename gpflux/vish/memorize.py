import atexit
from pathlib import Path
from typing import Callable

from json_tricks import dump, load


class memorize:
    """
    Decorator that caches the return value of a method.

    The return values are cached in a dictorionary where the key is
    made out of the arguments of the method and the value is the return
    value of the method. The decorator will save the dictionary to disk,
    and load it back in memory at the start of the program.
    """

    def __init__(self, path: str, args_to_key: Callable):
        """
        :param path: Path to persist the return values on disk.
        :param args_to_key: function that takes the arguments of 
            the dectorated function and returns a unique key to
            be used in the dictionary.
        """
        self.path = Path(path)
        self.args_to_key = args_to_key
        if self.path.exists():
            with open(self.path, "r") as fp:
                self.cache = load(fp)
        else:
            self.cache = {}

        # at exit write the content of our cache back to disk.
        atexit.register(self.write_cache_to_disk)

    def __call__(self, function):
        def cached_function(*args, **kwargs):
            key_for_args = self.args_to_key(*args)
            if not key_for_args in self.cache:
                print("Computing function value and caching for", key_for_args)
                self.cache[key_for_args] = function(*args, **kwargs)
            return self.cache[key_for_args]

        return cached_function

    def write_cache_to_disk(self):
        with open(self.path, "w") as fp:
            # the cache dict is stored in json format
            dump(self.cache, fp)
