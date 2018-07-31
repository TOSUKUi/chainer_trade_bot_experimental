from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class MyChain():
    
    def __init__(self):
        super(MyChain, self).__init__(
            l1 = L.liner(4,3)
            l2 = L.liner(3,2)
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        o = self.l2(h)




