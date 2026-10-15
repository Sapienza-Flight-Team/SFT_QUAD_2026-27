class Options:
    """Choices (tuple/index or dict/key), an integer range, or editable text."""

    def __init__(self, arg, default, alias=()):
        self.items = arg
        self.default = default
        self.alias = alias or (
            tuple(map(str, arg)) if isinstance(arg, (tuple, dict)) else ()
        )

    def get_default(self):
        if isinstance(self.items, (tuple, dict)):
            return self.items[self.default]
        return self.default

    def display_default(self):
        if isinstance(self.items, tuple):
            return self.alias[self.default]
        if isinstance(self.items, dict):
            return self.alias[list(self.items).index(self.default)]
        return str(self.default)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return str(self.alias or self.items)

    __str__ = __repr__
