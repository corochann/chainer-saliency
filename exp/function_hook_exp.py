import numpy as np

import chainer
from chainer import links as L
from chainer import functions as F
from chainer.function_hooks.timer import TimerHook


class PrintHook(chainer.FunctionHook):

    def __init__(self, message):
        self.message = message

    def forward_preprocess(self, function, in_data):
        print('From {}'.format(type(function)), self.message)


function_hooks = chainer.get_function_hooks()
function_hooks['PrintHook'] = PrintHook('hogehoge')


class Model(chainer.Chain):
    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 10)
            self.l2 = L.Linear(None, 10)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = self.l2(h)
        return h


if __name__ == '__main__':
    model = Model()
    x = np.random.rand(1, 10).astype(np.float32)
    # y = model(x)

    hook = TimerHook()
    with hook:
        y = model(x)
    hook.print_report()
