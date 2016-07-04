import ipt
import mxnet as mx
import my_utils as u
import logging
from cnn import cnn_net
import jiter as j

Adam15 = ('/home/zijia/HeartDeepLearning/Net/CNN/Result/<-1>[E15]/[ACC-0.47735 E8]', 8) 
Adam200  = ('/home/zijia/HeartDeepLearning/Net/CNN/Result/<0Save>/-200 epoch/0.48642', 4)
Good30   = ('/home/zijia/HeartDeepLearning/Net/CNN/Result/<0Save>/<1-12:30:48>[E30]/[ACC-0.90725 E29]',29)
Good80   = ('/home/zijia/HeartDeepLearning/Net/CNN/Result/<0Save>/<1-17:12:45>[E40]/[ACC-0.92900 E39]',39) 

def main():
    net = cnn_net()
    out = j.get(3, small=True, shift=True)
    train = out['train']
    val = out['val']

    model = mx.model.FeedForward.load(*Good80,
        ctx=u.gpu(1),
        learning_rate=6,
        num_epoch=10,
        optimizer='sgd',
        initializer=mx.initializer.Xavier(rnd_type='gaussian')
    )
        
    u.predict_draw(model,val, folder='MoveCheck')

    normal = u.get(3, small=True)
    val = normal['val']
    u.predict_draw(model,val, folder='MoveCheck/Truth')

if __name__ == '__main__':
    main()