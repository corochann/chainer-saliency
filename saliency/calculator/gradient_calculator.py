from saliency.calculator.base_calculator import BaseCalculator


class GradientCalculator(BaseCalculator):

    def __init__(self, model, eval_fun=None, eval_key=None, target_key=0,
                 multiply_target=False, device=None):
        super(GradientCalculator, self).__init__(model, device=device)
        # self.model = model
        # self._device = cuda.get_array_module(model)
        self.eval_fun = eval_fun or model.__call__
        self.eval_key = eval_key
        self.target_key = target_key

        self.multiply_target = multiply_target

    def _compute_core(self, *inputs):
        self.model.cleargrads()
        self.eval_fun(*inputs)
        target_var = self.target_extractor.get_variable()
        output_var = self.output_extractor.get_variable()
        # TODO: Consider how deal with the case when eval_var is not scalar,
        # 1. take sum
        # 2. raise error (default behavior)
        # I think option 1 "take sum" is better, since gradient is calculated
        # automatically independently in that case.
        output_var.backward(retain_grad=True)
        saliency = target_var.grad
        if self.multiply_target:
            saliency *= target_var.data
        outputs = (saliency,)
        return outputs
