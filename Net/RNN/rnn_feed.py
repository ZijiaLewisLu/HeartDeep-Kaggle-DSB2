# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""MXNet model module"""
from __future__ import absolute_import

import ipt
import mxnet as mx
import numpy as np
import time, logging
from mxnet import io
from mxnet import nd
from mxnet import symbol as sym
from mxnet import optimizer as opt
from mxnet import metric
from mxnet import kvstore as kvs
from mxnet.context import Context, cpu
from mxnet.initializer import Uniform
from collections import namedtuple
from mxnet.optimizer import get_updater
from mxnet.executor_manager import DataParallelExecutorManager, _check_arguments, _load_data
from mxnet import ndarray as nd
from mxnet.model import *
from mxnet.model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore
from mxnet.executor_manager import _load_general

from utils import parse_time, plot_save

import matplotlib.pyplot as plt

RNN_HIDDEN = 250

def _run_sax(data_batch_zoo, marks, executor_manager, eval_metric, updater, ctx, kvstore, acc_hist,
                                    monitor=None,
                                    update_on_kvstore=None,
                                    is_train = False):
    
    for t in range(len(marks)):
        data_batch = data_batch_zoo[t]
        m = marks[t]

        # print 'T>> M>>', t, m
        assert isinstance(m,int), 'Marks Type Error'

        executor_manager.load_data_batch(data_batch)

        executor_manager.forward(is_train=is_train)
        # assume only using one gpu

        import copy
        out = copy.deepcopy(executor_manager.curr_execgrp.train_execs[0].outputs)
        c = out[1]
        h = out[2]
        pred = out[0]
        if m > 0:
            plot_save(pred.asnumpy()[0,0], 'whatever.png')

        # print type(c), c.asnumpy().mean()
        # print type(h), h.asnumpy().mean()

        if is_train and m>0:
            # print 'is_train and m>0', m
            executor_manager.backward()

            ######force gradient of bn to be zero
            # for i in range(len(executor_manager.param_names)):
            #     n = executor_manager.param_names[i]
            #     if 'beta' in n:
            #         executor_manager.grad_arrays[i][0] = 0*executor_manager.grad_arrays[i][0]
                # if 'weight' in n:
                    # executor_manager.grad_arrays[i][0] += 2*executor_manager.param_arrays[i][0]


            if update_on_kvstore:
                _update_params_on_kvstore(executor_manager.param_arrays,
                                          executor_manager.grad_arrays,
                                          kvstore)
            else:
                _update_params(executor_manager.param_arrays,
                               executor_manager.grad_arrays,
                               updater=updater,
                               num_device=len(ctx),
                               kvstore=kvstore)


        if monitor is not None:
            monitor.toc_print()

        if is_train:
            eval_metric.reset()

        if m == 1:
            print 'here'
            executor_manager.update_metric(eval_metric, data_batch.label)
            name_value = eval_metric.get_name_value()

            for name, value in name_value:
                acc_hist.append(value)


        if False and m>0:
            img = data_batch.data[0]
            ll = data_batch.label[0]
            pred = out[0]
            fig = plt.figure()

            o = [pred, img, ll]
            for num in range(3):
                fig.add_subplot(131+num).imshow(o[num].asnumpy()[0,0])
            fig.savefig('Pred/rnn-%d.png' % t)
            fig.clear()
            # plt.imshow(pred.asnumpy()[0,0])
            # plt.savefig('Pred/pred-%d.png' % t)
            plt.close('all')
            del fig

    return executor_manager, eval_metric, acc_hist



def _train_rnn(
    symbol, 
    ctx, 
    marks, 
                        arg_names, param_names, aux_names,
                        arg_params, aux_params,
                        begin_epoch, end_epoch, epoch_size, optimizer,
                        kvstore, update_on_kvstore,train_data,
    e_marks=None,
                        eval_data=None, eval_metric=None,
                        epoch_end_callback=None, batch_end_callback=None,
                        logger=None, work_load_list=None, monitor=None,
                        eval_batch_end_callback=None, sym_gen=None,
                        mutable_data_shape=False, max_data_shape=None):
    
    """Mark should be a list of #SeriesLength, annotating if image has label by 1 , 0"""
    #TODO check mark shape
    #TODO marks not working if label of SAX is different in one batch

    if logger is None:
        logger = logging
    executor_manager = DataParallelExecutorManager(symbol=symbol,
                                                   sym_gen=sym_gen,
                                                   ctx=ctx,
                                                   train_data=train_data,
                                                   param_names=param_names,
                                                   arg_names=arg_names,
                                                   aux_names=aux_names,
                                                   work_load_list=work_load_list,
                                                   logger=logger,
                                                   mutable_data_shape=mutable_data_shape,
                                                   max_data_shape=max_data_shape)
    if monitor:
        executor_manager.install_monitor(monitor)

    executor_manager.set_params(arg_params, aux_params)

    if not update_on_kvstore:
        updater = get_updater(optimizer)

    if kvstore:
        _initialize_kvstore(kvstore=kvstore,
                            param_arrays=executor_manager.param_arrays,
                            arg_params=arg_params,
                            param_names=executor_manager.param_names,
                            update_on_kvstore=update_on_kvstore)

    if update_on_kvstore:
        kvstore.set_optimizer(optimizer)

    # Now start training
    train_data.reset()

    N = train_data.batch_size
    c = h = mx.nd.zeros((N,RNN_HIDDEN))

    for epoch in range(begin_epoch, end_epoch):
        # Training phase
        tic = time.time()
        eval_metric.reset()
        nbatch = 0
        # Iterate over training data.

        #Into Epoch
        #########################
        #record acc
        acc_hist = []

        while True:                        
            do_reset = True

            for data_batch_zoo in train_data:
                assert isinstance(data_batch_zoo, list), "Iter Error"
                if monitor is not None:
                    monitor.tic()

                # load in C and H
                # In future, execgrp should become curr_execgrp!!
                data_targets = [[ e.arg_dict[name] 
                            for i, e in enumerate(executor_manager.execgrp.train_execs)]
                            for name in ['c','h']] 
                _load_general([c],data_targets[0])
                _load_general([h],data_targets[1])  


                # Start to iter on Time steps
                executor_manager, eval_metric, acc_hist= _run_sax(
                    data_batch_zoo, marks, executor_manager, eval_metric, updater, ctx, kvstore, acc_hist,
                                        monitor=monitor,
                                        update_on_kvstore=update_on_kvstore,
                                        is_train=True,
                                        )


                nbatch += 1
                # batch callback (for print purpose)
                if batch_end_callback != None:
                    batch_end_params = BatchEndParam(epoch=epoch,
                                                     nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    if isinstance(batch_end_callback, list):
                        for call in batch_end_callback:
                            call(batch_end_params)
                    else:
                        batch_end_callback(batch_end_params)

                # this epoch is done possibly earlier
                if epoch_size is not None and nbatch >= epoch_size:
                    do_reset = False
                    break

            if do_reset is True:
                logger.info('Epoch[%d] Resetting Data Iterator', epoch)
                train_data.reset()
                logger.info('Epoch[%d] Resetting Eval Metric', epoch)
                eval_metric.reset()

            # this epoch is done
            if epoch_size is None or nbatch >= epoch_size:
                break

        toc = time.time()
        logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

        # print epoch ,'<<<<<<<<<<<<'
        print 'should print epoch end call back'
        print epoch_end_callback
        if epoch_end_callback or epoch + 1 == end_epoch:
            executor_manager.copy_to(arg_params, aux_params)

        if epoch_end_callback != None:
            if isinstance(epoch_end_callback, list):
                for call in epoch_end_callback:
                    call(epoch, symbol, arg_params, aux_params, 
                        acc_hist)
            else:
                epoch_end_callback(epoch, symbol, arg_params, aux_params, 
                        acc_hist)

        # evaluation
        # print 'enter evaluation'
        if eval_data:
            assert e_marks is not None, 'e marks cannot be None'
            eval_metric.reset()
            eval_data.reset()
            for eval_zoo in eval_data:
                
                executor_manager, eval_metric, acc_hist = _run_sax(
                        eval_zoo, e_marks, executor_manager, eval_metric, updater, ctx, kvstore, acc_hist,
                        update_on_kvstore=update_on_kvstore,
                        is_train=False)

                # executor_manager.load_data_batch(eval_batch)
                # executor_manager.forward(is_train=False)
                # executor_manager.update_metric(eval_metric, eval_batch.label)

                if eval_batch_end_callback != None:
                    batch_end_params = BatchEndParam(epoch=epoch,
                                                     nbatch=i,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    if isinstance(eval_batch_end_callback, list):
                        for call in eval_batch_end_callback:
                            call(batch_end_params)
                    else:
                        eval_batch_end_callback(batch_end_params)
            name_value = eval_metric.get_name_value()
            for name, value in name_value:
                logger.info('Epoch[%d] Validation-%s=%f', epoch, name, value)
                print 'Epoch[%d] Validation=%f' % (epoch, value)
    # end of all epochs
    return


class Feed(FeedForward):
    def __init__(self, symbol, rnn_hidden=RNN_HIDDEN,**kwargs):
        """Overwrite"""
        super(Feed, self).__init__(symbol,**kwargs)
        self.rnn_hidden = rnn_hidden

    def fit(self, 
        X, 
        marks, 
        e_marks=None,
            y=None, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local', logger=None,
            work_load_list=None, monitor=None, eval_batch_end_callback=None):
        """Overwrite"""

        data = self._init_iter(X, y, is_train=True)
        eval_data = self._init_eval_iter(eval_data)

        if self.sym_gen:
            self.symbol = self.sym_gen(data.default_bucket_key) # pylint: disable=no-member
            self._check_arguments()
        self.kwargs["sym"] = self.symbol

        # fixed
        N = data.batch_size
        param_dict = dict(data.provide_data+data.provide_label)
        param_dict['c'] = param_dict['h'] = (N, self.rnn_hidden)

        arg_names, param_names, aux_names = self._init_params(param_dict)

        # setup metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        # create kvstore
        (kvstore, update_on_kvstore) = _create_kvstore(
            kvstore, len(self.ctx), self.arg_params)

        param_idx2name = {}
        if update_on_kvstore:
            param_idx2name.update(enumerate(param_names))
        else:
            for i, n in enumerate(param_names):
                for k in range(len(self.ctx)):
                    param_idx2name[i*len(self.ctx)+k] = n
        self.kwargs["param_idx2name"] = param_idx2name

        # init optmizer
        if isinstance(self.optimizer, str):
            batch_size = data.batch_size
            if kvstore and kvstore.type == 'dist_sync':
                batch_size *= kvstore.num_workers
            optimizer = opt.create(self.optimizer,
                                   rescale_grad=(1.0/batch_size),
                                   **(self.kwargs))
        elif isinstance(self.optimizer, opt.Optimizer):
            optimizer = self.optimizer

        # do training
        # print 'before _train_rnn self.arg_params',self.arg_params.keys()
        _train_rnn(self.symbol, self.ctx, 
            marks,
            arg_names, param_names, aux_names,
                            self.arg_params, self.aux_params,
                            begin_epoch=self.begin_epoch, end_epoch=self.num_epoch,
                            epoch_size=self.epoch_size,
                            optimizer=optimizer,
                            train_data=data, eval_data=eval_data,
                            eval_metric=eval_metric,
                            epoch_end_callback=epoch_end_callback,
                            batch_end_callback=batch_end_callback,
                            kvstore=kvstore, update_on_kvstore=update_on_kvstore,
                            logger=logger, work_load_list=work_load_list, monitor=monitor,
                            eval_batch_end_callback=eval_batch_end_callback,
                            sym_gen=self.sym_gen, e_marks=e_marks)





    @staticmethod
    def load(prefix, epoch, ctx=None, **kwargs):
        """Overwrite"""
        symbol, arg_params, aux_params = load_checkpoint(prefix, epoch)
        return Feed(symbol, ctx=ctx,
                           arg_params=arg_params, aux_params=aux_params,
                           begin_epoch=epoch,
                           **kwargs)

    @staticmethod
    def create(symbol, X, y=None, ctx=None,
               num_epoch=None, epoch_size=None, optimizer='sgd', initializer=Uniform(0.01),
               eval_data=None, eval_metric='acc',
               epoch_end_callback=None, batch_end_callback=None,
               kvstore='local', logger=None, work_load_list=None,
               eval_batch_end_callback=None, **kwargs):
        """Overwrite"""
        model = Feed(symbol, ctx=ctx, num_epoch=num_epoch,
                            epoch_size=epoch_size,
                            optimizer=optimizer, initializer=initializer, **kwargs)
        model.fit(X, y, eval_data=eval_data, eval_metric=eval_metric,
                  epoch_end_callback=epoch_end_callback,
                  batch_end_callback=batch_end_callback,
                  kvstore=kvstore,
                  logger=logger,
                  work_load_list=work_load_list,
                  eval_batch_end_callback=eval_batch_end_callback)
        return model

    def simple_pred(self, X):
        if not isinstance(X, mx.nd.NDArray):
            X = mx.nd.array(X)

        N = X.shape[0]
        c = h = mx.nd.zeros((N, self.rnn_hidden))

        data_shapes ={ 'data': X.shape, 'c': (N,self.rnn_hidden), 'h':(N,self.rnn_hidden)}


        # for now only use the first device
        pred_exec = self.symbol.simple_bind(self.ctx[0], grad_req='null', **dict(data_shapes))
        pred_exec.copy_params_from(self.arg_params, self.aux_params)


        _load_general([X], pred_exec.arg_dict['data'])
        _load_general([c], pred_exec.arg_dict['c'])
        _load_general([h], pred_exec.arg_dict['h'])

        pred_exec.forward(is_train=False)

        return pred_exec.outputs
