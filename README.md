BC learning for images
=========================

Implementation of [Between-class Learning for Image Classification](https://arxiv.org/abs/1711.10284) by Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada.

Our preliminary experimental results on CIFAR-10 and ImageNet-1K were already presented in ILSVRC2017 on July 26, 2017.

#### Between-class (BC) learning:
- We generate between-class examples by mixing two training examples belonging to different classes with a random ratio.
- We then input the mixed data to the model and
train the model to output the mixing ratio.
- Original paper: [Learning from Between-class Examples for Deep Sound Recognition](https://arxiv.org/abs/1711.10282) by us ([github](https://github.com/mil-tokyo/bc_learning_sound))

## Contents
- BC learning for images
	- BC: mix two images simply using internal divisions.
	- BC+: mix two images treating them as waveforms.
- Training of 11-layer CNN on CIFAR datasets


## Setup
- Install [Chainer](https://chainer.org/) v1.24 on a machine with CUDA GPU.
- Prepare CIFAR datasets.


## Training
- Template:

		python main.py --dataset [cifar10 or cifar100] --netType convnet --data path/to/dataset/directory/ (--BC) (--plus)
 
- Recipes:
	- Standard learning on CIFAR-10 (around 6.1% error):

			python main.py --dataset cifar10 --netType convnet --data path/to/dataset/directory/
	

	- BC learning on CIFAR-10 (around 5.4% error):

			python main.py --dataset cifar10 --netType convnet --data path/to/dataset/directory/ --BC
	
	- BC+ learning on CIFAR-10 (around 5.2% error):

			python main.py --dataset cifar10 --netType convnet --data path/to/dataset/directory/ --BC --plus
	
- Notes:
	- It uses the same data augmentation scheme as [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).
	- By default, it runs training 10 times. You can specify the number of trials by using --nTrials command.
	- Please check [opts.py](https://github.com/mil-tokyo/bc_learning_image/blob/master/opts.py) for other command line arguments.

## Results

Error rate (average of 10 trials)

| Learning | CIFAR-10 | CIFAR-100 |
|:--|:-:|:-:|
| Standard | 6.07  | 26.68 |
| BC (ours) | 5.40 | 24.28 |
| BC+ (ours) | **5.22** | **23.68** |

- Other results (please see [paper](https://arxiv.org/abs/1711.10284)):
	- The performance of [Shake-Shake Regularization](https://github.com/xgastaldi/shake-shake) [[1]](#1) on CIFAR-10 was improved from 2.86% to 2.26%.
	- The performance of [ResNeXt](https://github.com/facebookresearch/ResNeXt) [[2]](#2) on ImageNet-1K was improved from 20.4% to 19.4% (single-crop top-1 validation error).

---

#### Reference
<i id=1></i>[1] X. Gastaldi. Shake-shake regularization. In *ICLR Workshop*, 2017.

<i id=2></i>[2] S. Xie, R. Girshick, P. Dollar, Z. Tu, and K. He. Aggregated residual transformations for deep neural networks. In *CVPR*, 2017.
