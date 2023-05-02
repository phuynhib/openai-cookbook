import argparse

class Arguments():
    args_kwargs: list = []
    args: argparse.Namespace = None

    def add(self, *args, **kwargs):
        """
        Add an argument.
        """
        
        for idx, arg in enumerate(self.args_kwargs):
            # if an argument already exists, then replace it
            if arg[0][0] == args[0]:
                self.args_kwargs[idx] = [args, kwargs]
                return

        self.args_kwargs.append(
            [args, kwargs]
        )

    def all(self, *args, **kwargs) -> argparse.Namespace:
        """
        Returns a namespace containing all the arguments.
        """
        if self.args:
            return self.args

        default = {
            'formatter_class': argparse.ArgumentDefaultsHelpFormatter
        }
        kwargs = {**default, **kwargs}        

        parser = argparse.ArgumentParser(*args, **kwargs)
        for a in self.args_kwargs:
            parser.add_argument(*a[0], **a[1])

        self.args = parser.parse_args()
        return self.args
    
    def get(self, key):
        """
        Returns a value for the given argument key
        """
        return getattr(self.all(), key)