import chainer
import chainer.links as L
import chainer.functions as F
from chainer_saliency.calculator.gradient_calculator import GradientCalculator
from chainer_saliency.variable_extract_link_hook import VariableExtractLinkHook


class Model(chainer.Chain):
    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(10, 10)
            # self.l2 = L.Linear(None, 10)
            self.l2 = L.Linear(10, 1)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = self.l2(h)
        return h


if __name__ == '__main__':
    model = Model()
    gradient_calculator = GradientCalculator(
        model, target_extractor=VariableExtractLinkHook(model.l1),
    )
