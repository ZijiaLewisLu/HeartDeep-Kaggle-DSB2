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
import sys
from my_utils import *


class Solver():

    def __init__(self, net, train_data, sks, **kwargs):
        """sks is the dict of config of solver whereas other args and kwargs will be passed into net""" 
        k = kwargs.copy()
        self.net = net

        # prepare Train_data
        self.train_data = train_data
        if isinstance(train_data, mx.io.DataIter):
            self.batch_size = train_data.batch_size
        else:
            self.batch_size = k.pop(
                'numpy_batch_size', min(train_data.shape[0], 128))
            k['numpy_batch_size'] = self.batch_size

        # init params
        self.num_epoch = k['num_epoch']
        self.reset()

        sks_bk = sks.copy()
        self.sks = sks
        # whether draw outputs of every forward step
        # self.draw_each = k.pop('draw_each', False)
        # whether save prediction to pk files
        # self.save_pred = k.pop('save_pred', False)
        # self.save_best = k.pop('save_best', True)
        self.block_bn = self.sks.pop('block_bn', False)
        self.is_rnn = self.sks.pop('is_rnn', False)
        self.lgr    = self.sks.pop('logger', None)

        # make name and save_dir
        now = time.ctime(int(time.time()))
        now = now.split(' ')
        name = self.sks.pop('name', None)
        #t = now[3].split(':')
        #t = ':'.join(t[:2])
        self.name = '<' + now[-3] + '-' + now[-2] + '>'
        if name is not None:
            self.name += name

        self.path = 'Result/' + self.name + '[E%d]/'%self.num_epoch
        try:
            os.mkdir(self.path)
        except OSError, e:
            print e, 'ecountered'

        # config logging
        if self.lgr is None:
            self._init_log()

        self.lgr.info(self.name)

        # save kwargs to file
        self.save_kwargs(sks_bk, kwargs)

        # store kwargs
        self.kwargs = k
        self.origin_k = kwargs

    def _init_log(self):
        logging.basicConfig(level=logging.DEBUG, filename=self.path+'LOG.txt', format='%(levelname)s:%(message)s')
        logger = logging.getLogger('')
        formatter = logging.Formatter('%(levelname)s:%(message)s')

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        self.lgr = logger

    def reset(self):
        self.acc_hist = {}
        self.arg = {}
        self.best_acc = 0
        self.best_param = None
        self.nbatch = -1
        self.nepoch = -1
        self.param_grad = {}
        self.param_name = None
        self.count = 0
        self.model = None

    def save_kwargs(self, s,k):
        save_k = k.copy()
        with open(self.path + "SolverParam.json", 'w') as f:
            save_k.pop('eval_data', None)
            ctx = save_k['ctx']
            ctx = [ctx] if not isinstance(ctx, list) else ctx
            save_k['ctx'] = ctx.__str__()
            save_k.pop('initializer', None)
            save_k.pop('logger', None)

            if self.is_rnn:
                save_k['marks'] = save_k['marks'].__str__()
                if 'e_marks' in save_k:
                    save_k['e_marks'] = save_k['e_marks'].__str__()  

            json.dump(s, f, indent=4, sort_keys=True)
            json.dump(save_k, f, indent=4, sort_keys=True)



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
        self.lgr.debug('-------------------------EVAL, mean of prediciton %f, truth %f, iou %f----------------------' %
                       (pred.mean(), label.mean(), out))

        if self.sks.pop('draw_each', False):
            self._draw_together(
                pred, label, 'Evaluation[E%d-B%d]-#%d' % (self.nepoch, self.nbatch, self.count))

        if self.sks.pop('save_pred', False):
            with open(self.path + 'pk[E%d-B%d]-#%d.pk' % (self.nepoch, self.nbatch, self.count), 'w') as f:
                pk.dump(pred, f)
                pk.dump(label, f)

        self.count += 1

        if not 0 <= out <= 1:
            self.lgr.warning('eval error >>%f %f %f' %
                             (out, np.sum(conjunct), np.sum(union)))

        return out

    def batch(self, params):
        """epoch, nbatch, eval_metric, locals """
        self.nbatch = params[1]

        manager = params[3]['executor_manager']

        if self.param_name is None:
            self.param_name = manager.param_names

        for i, n in enumerate(self.param_name):
            ps = params[3]['executor_manager'].param_arrays[i]
            gs = params[3]['executor_manager'].grad_arrays[i]

            psum = None
            gsum = None

            # for the same param on different gpu
            # operation for param
            for j, p in enumerate(ps):
                # if necessary, fix beta in batch norm
                if 'beta' in n and self.block_bn:
                    # p = 0*p
                    # params[3]['executor_manager'].param_arrays[i][j] = 0*p
                    print 'check mean', params[3]['executor_manager'].param_arrays[i][j].asnumpy().mean()
                
                if psum is None:
                    psum = p.asnumpy()
                else:
                    psum += p.asnumpy()

            # print params' means
            self.lgr.debug('[B%d %s]> %f', self.nbatch, n, psum.mean())

            # save param
            if n not in self.param_grad.keys():
                self.param_grad[n] = [[psum.mean()],[]]
            else:
                self.param_grad[n][0].append(psum.mean())

            # operation for grad
            for g in gs:
                if gsum is None:
                    gsum = g.asnumpy()
                else:
                    gsum+= g.asnumpy()
            
            # save grad
            self.param_grad[n][1].append(gsum.mean())

    def eval_batch(self, params):
        local = params[3]
        preds = local['executor_manager'].curr_execgrp.train_execs[
            0].outputs[0]
        labels = local['eval_batch'].label[0]
        self._draw_together(
            preds, labels, 'EVAL[E%d-B%d]' % (params[0], params[1]))

    def epoch(self, epoch, symbol, arg_params, aux_params, acc):
        self.acc_hist[epoch] = acc
        self.arg[epoch] = arg_params
        self.nepoch = epoch
        # print 'Epoch[%d] Train accuracy: %f' % (epoch, np.sum(acc) /
        this_acc = np.sum(acc) / float(len(acc))
        self.lgr.info('E[%d] T acc: %f', epoch, this_acc)

        if self.sks.pop('save_best',True) and \
                (self.best_param is None or this_acc > self.best_acc):
            self.best_param = (epoch, symbol, arg_params, aux_params)
            self.best_acc = this_acc

    def plot_process(self):
        """
        self.param_grad is a dict containning a list of each params
        for each list, the first item is a list of all params, the second item is a list of a grad
        """
        names = self.param_name

        path = self.path + 'Insight/'
        os.mkdir(path)

        for i, n in enumerate(names):
            fig = plt.figure()
            param, grad = self.param_grad[n]

            # when using more than one gpu, weight are in differnt gpus
            mean_param = param # [ x.mean() for x in param ]
            mean_grad  = grad  # [ x.mean() for x in grad]

            fig.add_subplot(1,2,1).plot(mean_param, marker='o')
            fig.add_subplot(1,2,2).plot(mean_grad,  marker='o')
            fig.suptitle(n+' Param:Grad')

            fig.savefig(path+n+'.png')
            fig.clear()
            plt.close('all')

    def save_best_model(self):
        if self.best_param is None or self.best_acc == 0:
            print 'No Best Model'
            return

        from mxnet.model import save_checkpoint
        save_checkpoint("%s[ACC-%0.5f E%d]" %
                        (self.path, self.best_acc, self.best_param[0]), *self.best_param)

    def get_acc_list(self):
        l = []
        for k in sorted(self.acc_hist.keys()):
            l += self.acc_hist[k]
        return l

    def each_to_png(self):
        for k in sorted(self.acc_hist.keys()):
            plt.plot(self.acc_hist[k], marker='o')
            path = os.path.join(self.path, 'acc_his-' + str(k) + '.png')
            plt.savefig(path)
            plt.close()

    def all_to_png(self):
        l = []
        for k in sorted(self.acc_hist.keys()):
            average = np.mean(self.acc_hist[k])
            l.append(average)

        plt.plot(l, marker='o')
        path = os.path.join(self.path, 'acc_his-all.png')
        plt.savefig(path)
        plt.close()

    def _load(self, perfix, epoch):
        """
        ``prefix-symbol.json`` will be saved for symbol.
        ``prefix-epoch.params`` will be saved for parameters.
        """
        from mx.model import load_checkpoint
        return load_checkpoint(prefix, epoch)

    def _init_model(self):

        if self.is_rnn:
            from RNN import rnn_feed

            if self.sks.pop('load', False):
                perfix = self.sks['load_perfix']
                epoch = self.sks['load_epoch']
                self.model = rnn_feed.Feed.load(perfix, epoch, **self.kwargs)
                self.model.begin_epoch=0

            elif self.sks.pop('load_from_cnn',False):
                perfix = self.sks['load_perfix']
                epoch = self.sks['load_epoch']
                shape = dict(self.train_data.provide_data+self.train_data.provide_label)
                self.model = rnn_feed.Feed.load_from_cnn(perfix, epoch, self.net, shape, **self.kwargs)
                self.model.begin_epoch=0
                
            else:
                self.model = rnn_feed.Feed(self.net, **self.kwargs)

        else:
            if self.sks.pop('load', False):
                perfix = self.sks['load_perfix']
                epoch = self.sks['load_epoch']
                self.model = mx.model.FeedForward.load(perfix, epoch, **self.kwargs)
                self.model.begin_epoch=0
            else:
                self.model = mx.model.FeedForward(self.net, **self.kwargs)

    def train(self):

        kwords = {
            'kvstore': 'local',
            'eval_metric': self.eval,
            'epoch_end_callback': self.epoch,
            'batch_end_callback': self.batch,
            #'eval_batch_end_callback': self.eval_batch,
        }

        for term in ['y', 'eval_data', 'logger', 'work_load_list', 'monitor']:
            if term in self.kwargs.keys():
                kwords[term] = self.kwargs.pop(term)

        if self.is_rnn:
            kwords['e_marks'] = self.kwargs.pop('e_marks',None)
            marks = self.kwargs.pop('marks')
            from RNN import rnn_metric
            kwords['eval_metric'] = rnn_metric.RnnM(self.eval)

        # prepare and train
        self._init_model()

        if self.is_rnn:
            self.model.fit(self.train_data, marks, logger = self.lgr, **kwords)
        else:
            self.model.fit(self.train_data, logger=self.lgr, **kwords)

    def predict(self):
        if 'eval_data' in self.origin_k.keys():
            X = self.origin_k['eval_data']
        else:
            X = self.train_data
            self.lgr.warning('No Eval Data, Using Training Data')

        # if not train, directly predict, -> init model
        if self.model is None:
            for term in ['y', 'eval_data', 'logger', 'work_load_list', 'monitor', 'marks','e_marks']:
                if term in self.kwargs.keys():
                   self.kwargs.pop(term)
            self._init_model()

        if self.model.arg_params is None:
            d = X.provide_data
            l = X.provide_label
            self.model._init_params(dict(d + l))

        out = self.model.predict(X, return_data=True)
        out=list(out)
        if self.is_rnn:
            self.lgr.debug('Prediction Done, reshape rnn outputs')
            for idx, array in enumerate(out):
                out[idx] = array.reshape((-1,1,256,256))
        
        N = out[0].shape[0]

        for idx in range(N):
            gap = np.ones((256, 5))
            pred = out[0][idx, 0]
            img = out[1][idx, 0]
            label = out[2][idx, 0]
            png = np.hstack([pred, gap, label])

            self.lgr.debug('Prediction mean>>%f Label mean>>%f', pred.mean(), label.mean())

            fig = plt.figure()
            fig.add_subplot(121).imshow(png)
            fig.add_subplot(122).imshow(img)
            fig.savefig(self.path+'Pred[%d].png'%(idx))
            fig.clear()
            plt.close('all')
