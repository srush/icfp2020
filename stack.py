from base import syms
import sys

lookup = {}
d = {}
reduced = {}



    
def run_stack(symbols, name):
    cur = None
    print("NAME", name)
    stack = []
    debug = False
    # if name == ":1079":
    #     debug = True
    
    buffer = list(reversed(symbols))
    while buffer:
        s = buffer[0]
        buffer = buffer[1:]
        if debug:
            print("eating", s)
        
        if s[0] == "ap":
            fun = stack[0]
            arg = stack[1]
            new = fun()(arg)
            stack = [new] + stack[2:]
        else:
            if debug:
                print("pushing", s[1])
            stack = [s[1]] + stack
        if debug:
            print("stack is", stack)
    print(stack)
    return stack[0]



def main(txt):
    for name, code, _ in syms:
        lookup[name] = code
        
    for l in open(txt):
        name, code = l.split("=")
        def make(l):
            global reduced
            if l not in lookup:
                if  l[0] == ":":
                    def lazy():
                        if l in reduced:                          
                            return reduced[l]
                        else:
                            reduced[l] = run_stack(d[l], l)()
                            return reduced[l]
                    return lazy
                else:
                    return lambda : int(l)
            else:
                return None
                    
        cmds = [(l, lookup.get(l, make(l)))
                for l in code.split()]
        d[name.strip()] = cmds
        
    def expand(x):
        if isinstance(x, tuple) or isinstance(x, int):
            return x            
        else:
            x2 = x()

        if isinstance(x2, tuple):
            return tuple((expand(t) for t in x2))
        else:
            return x2


    for i in range(1029, 1030):
        key = ":%s"%i
        print(key)
        if key in d:
            print(d[key])
            out = run_stack(d[key], key)
        print(expand(out))
            
    # for l in open(txt):
    #     name, code = l.split("=")
    #     print("REDUCING", name)
    #     val = 
    #     print(name)
    #     d[name.strip()] = val
    #     assert(val is not None)

    
main(sys.argv[1])
