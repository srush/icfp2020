
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

    def write(self):
        return self.val



class Sym:
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def render(self):
        if self.name == "[":
            return "[" + str(self.content) + "]"
        if self.content is not None:
            return "%s %s"%(self.name, self.content)
        return self.name 

    def write(self):
        if self.name == "inc" and self.content == -1:
            return "dec"
        if self.name == "var":
            return "x" + str(self.content)
        if self.name == "[":
            return "[" + str(self.content) + "]"
        return self.name
