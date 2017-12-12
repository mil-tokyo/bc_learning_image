import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import Uniform
from convbnrelu import ConvBNReLU


class ConvNet(chainer.Chain):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__(
            conv11=ConvBNReLU(3, 64, 3, pad=1),
            conv12=ConvBNReLU(64, 64, 3, pad=1),
            conv21=ConvBNReLU(64, 128, 3, pad=1),
            conv22=ConvBNReLU(128, 128, 3, pad=1),
            conv31=ConvBNReLU(128, 256, 3, pad=1),
            conv32=ConvBNReLU(256, 256, 3, pad=1),
            conv33=ConvBNReLU(256, 256, 3, pad=1),
            conv34=ConvBNReLU(256, 256, 3, pad=1),
            fc4=L.Linear(256 * 4 * 4, 1024, initialW=Uniform(1. / math.sqrt(256 * 4 * 4))),
            fc5=L.Linear(1024, 1024, initialW=Uniform(1. / math.sqrt(1024))),
            fc6=L.Linear(1024, n_classes, initialW=Uniform(1. / math.sqrt(1024)))
        )
        self.train = True

    def __call__(self, x):
        h = self.conv11(x, self.train)
        h = self.conv12(h, self.train)
        h = F.max_pooling_2d(h, 2)

        h = self.conv21(h, self.train)
        h = self.conv22(h, self.train)
        h = F.max_pooling_2d(h, 2)

        h = self.conv31(h, self.train)
        h = self.conv32(h, self.train)
        h = self.conv33(h, self.train)
        h = self.conv34(h, self.train)
        h = F.max_pooling_2d(h, 2)

        h = F.dropout(F.relu(self.fc4(h)), train=self.train)
        h = F.dropout(F.relu(self.fc5(h)), train=self.train)

        return self.fc6(h)
