import numpy
from chainer import Variable

from saliency.calculator.base_calculator import BaseCalculator
from saliency.calculator.gradient_calculator import GradientCalculator


class IntegratedGradientsCalculator(GradientCalculator):

    def __init__(self, model, eval_fun=None, eval_key=None, target_key=0,
                 baseline=None, steps=25):
        super(IntegratedGradientsCalculator, self).__init__(
            model, eval_fun=eval_fun, eval_key=eval_key, target_key=target_key,
            multiply_target=False
        )
        # self.target_key = target_key
        self.baseline = baseline or 0.
        self.steps = steps

    def get_target_var(self, inputs):
        if self.target_key is None:
            target_var = inputs
        elif isinstance(self.target_key, int):
            target_var = inputs[self.target_key]
        else:
            raise TypeError('Unexpected type {} for target_key'
                            .format(type(self.target_key)))
        return target_var

    def _compute_core(self, *inputs):

        total_grads = 0.
        self.model.cleargrads()
        self.eval_fun(*inputs)  # Need to forward once to get target_var
        target_var = self.target_extractor.get_variable()
        # output_var = self.output_extractor.get_variable()

        base = self.baseline
        diff = target_var.array - base

        for alpha in numpy.linspace(0., 1., self.steps):
            def interpolate_target_var(hook, args, target_var):
                # target_var = args.out
                # diff = target_var.array - base
                interpolated_inputs = base + alpha * diff
                target_var.array[:] = interpolated_inputs

            self.target_extractor.set_process(interpolate_target_var)
            total_grads += super(
                IntegratedGradientsCalculator, self)._compute_core(
                *inputs)[0]
        saliency = total_grads * diff / self.steps
        return saliency,
