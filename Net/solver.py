import ipt
import time
import os
import logging
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
import os
import pickle as pk
from PIL import Image
import copy
import json
import RNN
import CNN
from utils import *


class Solver():

    def __init__(self, net, train_data, **kwargs):
        
        k = kwargs.copy()
        self.net = net
        self.train_data = train_data
        if isinstance(train_data, mx.io.DataIter):
            self.batch_size = train_data.batch_size
        else:
            self.batch_size = k.pop(
                'numpy_batch_size', min(train_data.shape[0], 128))
            k['numpy_batch_size'] = self.batch_size

        self.num_epoch = k['num_epoch']
        self.acc_hist = {}
        self.arg = {}
        self.best_acc = 0
        self.best_param = None
        self.nbatch = -1
        self.nepoch = -1
        self.count  = 0

        self.block_bn = k.pop('block_bn', False)
        # draw outputs of every forwards
        self.draw_each = k.pop('draw_each', False)
        # save prediction to pk files
        self.save_pred = k.pop('save_pred', False)
        self.save_best = k.pop('save_best', True)
        self.is_rnn = k.pop('is_rnn', False)

        now = time.ctime(int(time.time()))
        now = now.split(' ')
        name = k.pop('name', None)
        self.name = now[2] + '-' + now[3]
        if name is not None:
            self.name += name

        self.path = 'Result/' + self.name + '/'
        try:
            os.mkdir(self.path)
        except OSError, e:
            print e, 'ecountered'
        
        logging.basicConfig(format='%(levelname)s:%(message)s')
        if k.pop('save_log', True):
            logging.basicConfig(filename='log.txt')

        if k.pop('debug_level', False):
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        logging.info(self.name)

        with open(self.path + "SolverParam.json", 'w') as f:
            kwargs.pop('eval_data')
            kwargs.pop('ctx')
            kwargs.pop('initializer')
            json.dump(kwargs, f)

        self.kwargs = k

    def reset(self):
        self.acc_hist = {}
        self.arg = {}
        self.best_acc = 0
        self.best_param = None
        self.nbatch = -1
        self.nepoch = -1

    def _draw_together(self, preds, labels, perfix):
        gap = np.ones((256, 5))

        if isinstance(preds, mx.ndarray.NDArray):
            preds = preds.asnumpy()
        if isinstance(labels, mx.ndarray.NDArray):
            labels = labels.asnumpy()

        N = preds.shape[0]
        for i in range(N):
            pic = np.hstack([preds[i, 0], gap, labels[i, 0]])
            plt.imsave(self.path + perfix + '~N%d.png' % i, pic)
            plt.close('all')

    def eval(self, label, pred):
        pred = copy.deepcopy(pred)
        conjunct = pred * label
        union = pred + label

        out = np.sum(conjunct * 2) / np.sum(union)
        logging.debug('EVAL, mean of prediciton %f, truth %f, iou %f' %
                      (pred.mean(), label.mean(), out))

        if self.draw_each:
            self._draw_together(
                pred, label, 'IOU[E%d-B%d]-#%d' % (self.nepoch, self.nbatch, self.count))

        if self.save_pred:
            with open(self.path + 'pk[E%d-B%d]-#%d.pk' % (self.nepoch, self.nbatch, self.count), 'w') as f:
                pk.dump(pred, f)
                pk.dump(label, f)

        self.count += 1

        if not 0 <= out <= 1:
            logging.warning('eval error >>%f %f %f' %
                            (out, np.sum(conjunct), np.sum(union)))

        return out

    def batch(self, params):
        """epoch, nbatch, eval_metric, locals """
        self.nbatch = params[1]
        for pairs in zip(params[3]['executor_manager'].param_names, params[3]['executor_manager'].param_arrays):
            n, p = pairs
            if 'beta' in n and self.block_bn:
                # print 'in batch', n , p[0].asnumpy().mean()
                shape = p[0].shape
                conttx = p[0].context
                p[0] = mx.ndarray.zeros(shape, ctx=conttx)
            if 'weight' in n:
                logging.debug('[BATCH Parm %s]> %f', n, p[0].asnumpy().mean())

    def eval_batch(self, params):
        local = params[3]
        preds = local['executor_manager'].curr_execgrp.train_execs[0].outputs[0]
        labels = local['eval_batch'].label[0]
        self._draw_together(
            preds, labels, 'EVAL[E%d-B%d]' % (params[0], params[1]))

    def epoch(self, epoch, symbol, arg_params, aux_params, acc):
        self.acc_hist[epoch] = acc
        self.arg[epoch] = arg_params
        self.nepoch = epoch
        # print 'Epoch[%d] Train accuracy: %f' % (epoch, np.sum(acc) /
        this_acc = np.sum(acc) / float(len(acc))
        logging.info('Epoch[%d] Train accuracy: %f', epoch, this_acc)

        if self.save_best and \
                (self.best_param is None or this_acc > self.best_acc):
            self.best_param = (epoch, symbol, arg_params, aux_params)
            self.best_acc = this_acc

    def save_best_model(self):
        if self.best_param is None or self.best_acc == 0:
            print 'No Best Model'
            return

        from mxnet.model import save_checkpoint
        save_checkpoint("%s[%0.5f]" %
                        (self.path, self.best_acc), *self.best_param)

    def get_dict(self):
        return self.acc_hist

    def get_list(self):
        l = []
        for k in sorted(self.acc_hist.keys()):
            l += self.acc_hist[k]
        return l

    def each_to_png(self):
        for k in sorted(self.acc_hist.keys()):
            plt.plot(self.acc_hist[k])
            path = os.path.join(self.path, 'acc_his-' + str(k) + '.png')
            plt.savefig(path)
            plt.close()

    def all_to_png(self):
        l = self.get_list()
        plt.plot(l)
        path = os.path.join(self.path, 'acc_his-all.png')
        plt.savefig(path)
        plt.close()

    def _init_model(self):

        if self.is_rnn:
            self.model = RNN.rnn_feed.Feed(self.net, **self.kwargs)
        else:
            self.model = mx.model.FeedForward(self.net, **self.kwargs)

        if self.kwargs.pop('load',False):
            perfix = self.kwargs['load_perfix']
            epoch = self.kwargs['load_epoch']
            raise NotImplemented('Load is not supported')

    def train(self):
        kwords = {
            'kvstore': 'local',
            'eval_metric': self.eval,
            'epoch_end_callback': self.epoch,
            'batch_end_callback': self.batch,
            'eval_batch_end_callback': self.eval_batch,
        }

        for term in ['y', 'eval_data', 'logger', 'work_load_list', 'monitor']:
            if term in self.kwargs.keys():
                kwords[term] = self.kwargs.pop(term) 

        self._init_model()

        if self.is_rnn:
            kwords['e_marks'] = self.kwargs['e_marks']
            marks = self.kwargs.pop('marks')
            self.model.fit(self.train_data, marks, **kwords)
        else:
            self.model.fit(self.train_data, **kwords)
