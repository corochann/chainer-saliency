# chainer-saliency

Saliency calculation examples.

Article
 - [Library release: visualize saliency map of deep neural network](http://corochann.com/library-release-visualize-saliency-map-of-deep-neural-network-1478.html)
 - [NNの予測根拠可視化をライブラリ化する](https://qiita.com/corochann/items/066dcbfbe04a6bd447a3)

<div float="left" align="middle">
  <img src="https://qiita-image-store.s3.amazonaws.com/0/25635/963ca7b5-9f88-d7c7-4b6e-30afa35d5573.png" width="250" /> 
  <img src="https://qiita-image-store.s3.amazonaws.com/0/25635/56171a48-920a-f346-a581-6e0116f333e6.png" width="250" />
  <img src="https://qiita-image-store.s3.amazonaws.com/0/25635/1d2585f8-7cc9-7ddc-cdf1-6f1094ced2b3.png" width="250" />
</div>

From left: 1. Classification saliency map visualization of VGG16, CNN model. 2. iris dataset feature importance calculation of MLP model. 3. Water solubility contribution visualization of Graph convolutional network model.

# setup

chainer>=5.0.0 : To use `LinkHook` 
chainer-chemistry>=0.5.0 : saliency module is introduced from 0.5.0

```bash
# please update your chainer (version>=5.0.0 is necessary)
pip install -U chainer

# install chainer-chemistry
pip install -U chainer-chemistry
```

# saliency module usage

```python
# model is chainer.Chain, x is dataset
calculator = GradientCalculator(model)
saliency_samples = calculator.compute(x)
saliency = calculator.aggregate(saliency_samples)
 
visualizer = ImageVisualizer()
visualizer.visualize(saliency)
```

Basically that's all to show saliency plot like top figure!
