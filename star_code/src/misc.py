class Pipeline:

    def __init__(self, *funcs):
        self.funcs = funcs

    def fit(X):
        for e in X:
            res = e
            for f in funcs:
                e = f(res)
