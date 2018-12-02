import numpy as np

import chainer
from chainer import links as L
from chainer import functions as F


class RecordParentLink(chainer.LinkHook):

    name = 'RecordParentLink'

    def __init__(self):
        super(RecordParentLink, self).__init__()
        self.stack = []

    def added(self, link):
        print('added called for link {}'.format(link))

    def deleted(self, link):
        print('deleted called for link {}'.format(link))

    def forward_preprocess(self, args):

        if args.link.name is None and len(self.stack) == 0:
            print('forward_preprocess')
            print('skip root link')
        else:
            self.stack.append(args.link.name)
            print('forward_preprocess', '/'.join(self.stack))
        # print('link', args.link, 'name', args.link.name, 'stack', self.stack)
        # print('forward_name', args.forward_name)
        # out_data.creator.parent_link = link

    def forward_postprocess(self, args):
        print('forward_postprocess', '/'.join(self.stack))
        # print('link', args.link)
        if args.link.name is None and len(self.stack) == 0:
            print('skip root link')
            return
        if self.stack[-1] == args.link.name:
            self.stack.pop()
        else:
            print('[WARNING] link {} is not in stack {}'.format(args.link.name, self.stack))
        # print('link', args.link, 'name', args.link.name, 'stack', self.stack)
        # print('forward_name', args.forward_name)


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
    with RecordParentLink():
        y = model(x)

