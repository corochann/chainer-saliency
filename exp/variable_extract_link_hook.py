import numpy as np

import chainer
from chainer import links as L
from chainer import functions as F


def _to_list(s):
    if s is None:
        return []
    elif isinstance(s, str):
        return [s]
    else:
        return list(s)


class VariableExtractLinkHook(chainer.LinkHook):
    """Monitor by link reference"""

    # This LinkHook maybe instantiated multiple times.
    # So it is allowed to change name by argument.
    # name = 'VariableExtractLinkHook'

    def __init__(self, target_link, name='VariableExtractLinkHook',
                 timing='output', postprocess_fn=None):
        assert isinstance(target_link, chainer.Link)
        super(VariableExtractLinkHook, self).__init__()
        self.target_link = target_link
        self.name = name
        self.postprocess_fn = postprocess_fn
        self.timing = timing
        self.result = None

    def added(self, link):
        print('added called for link {}'.format(link))

    def deleted(self, link):
        print('deleted called for link {}'.format(link))

    def forward_preprocess(self, args):
        print('forward_preprocess')
        if self.timing == 'input' and args.link is self.target_link:
            print('matched at {}'.format(args.link.name))
            self.result = args.args
        # print('link', args.link, 'name', args.link.name, 'stack', self.stack)
        # print('forward_name', args.forward_name)
        # out_data.creator.parent_link = link

    def forward_postprocess(self, args):
        if self.timing == 'output' and args.link is self.target_link:
            print('matched at {}'.format(args.link.name))
            self.result = args.out

    def get_variable(self):
        if self.postprocess_fn is not None:
            res = self.postprocess_fn(self.result)
        else:
            res = self.result
        return res


class HogeModel(chainer.Chain):
    def __init__(self):
        super(HogeModel, self).__init__()
        with self.init_scope():
            self.l3 = L.Linear(None, 10)

    def forward(self, x):
        return self.l3(x)

    # def __call__(self, x):
    #     return self.l1(x)


class Model(chainer.Chain):
    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 10)
            # self.l2 = L.Linear(None, 10)
            self.l2 = HogeModel()

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = self.l2(h)
        return h


if __name__ == '__main__':
    print('chainer version', chainer.__version__)
    model = Model()
    x = np.random.rand(1, 10).astype('f')
    # with RecordParend():
    hook = VariableExtractLinkHook(model.l2.l3)
    with hook:
        y = model(x)

    print(type(hook.result), hook.result)

