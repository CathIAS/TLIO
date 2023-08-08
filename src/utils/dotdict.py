# Copyright 2004-present Facebook. All Rights Reserved.


class dotdict(dict):
    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    
    def __repr__(self):
        return super().__repr__()

# Testing code
if __name__ == "__main__":
    d = dotdict()
    d.x = 7
    print(d)
    print(d.x)
    print(d['x'])
