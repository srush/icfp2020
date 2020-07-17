
class Item:
    def __init__(self, x, y, x_size, y_size, content):
        self.xy = (x, y)
        self.size = (x_size, y_size)
        self.content = content

    def render(self, svg):
        svg.annotation(*self.xy, *self.size, self.content.render())

class Number:
    def __init__(self, val):
        self.val = val

    def render(self):
        return self.val

class Var:
    def __init__(self, name):
        pass


class Sym:
    def __init__(self, name):
        self.name = name

    def render(self):
        return self.name
