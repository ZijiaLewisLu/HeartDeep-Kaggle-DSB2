# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""MXNet model module"""
from __future__ import absolute_import

import ipt
import mxnet as mx
import numpy as np
import time
import logging
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
from mxnet.executor_manager import _load_general, _load_data

from HeartDeepLearning.my_utils import parse_time, save_img

import matplotlib.pyplot as plt
import pickle as pk

RNN_HIDDEN = 250


def _run_sax(data_batch_zoo, marks, executor_manager, eval_metric, updater, ctx, kvstore, acc_hist,
             logger=None,
             callback=None,
             monitor=None,
             update_on_kvstore=None,
             is_train=False):

    if logger is None:
        logger = logging

    data_targets = [[e.arg_dict[name] for i, e in enumerate(executor_manager.execgrp.train_execs)]
                    for name in ['c', 'h']]
    # for idx in range(len(data_targets[0])):
    #     print 'data_targets c mean', data_targets[0][idx].asnumpy().mean()
    #     print 'data_targets h mean', data_targets[1][idx].asnumpy().mean()
    #     _load_general([c[idx]], [data_targets[0][idx]])
    #     _load_general([h[idx]], [data_targets[1][idx]])

    for i, tg in enumerate(executor_manager.execgrp.data_arrays[1:]):
        for j, slice_array in enumerate(tg):
            assert isinstance(slice_array, tuple)
            array = slice_array[1]
            replace = 0 * array
            _load_general([replace], [array])

    # Start Looping
    for t in range(len(marks)):
        m = marks[t]
        logger.debug('Time Step %d M %d', t, m)
        data_batch = data_batch_zoo[t]

        assert isinstance(m, int), 'Marks Type Error, %s provided' % type(m)

        #load in data
        executor_manager.load_data_batch(data_batch)

        executor_manager.forward(is_train=is_train)

        for num, ex in enumerate(executor_manager.curr_execgrp.train_execs):
            out = ex.outputs
            ccc = out[1]
            hhh = out[2]
            _load_general([ccc], [data_targets[0][num]])
            _load_general([hhh], [data_targets[1][num]])

        if is_train and m > 0:
            executor_manager.backward()

            logger.debug('Updateing weight...')
            #logger.debug('--------before update | grad check-------------')
            #for pari in zip(executor_manager.param_names, executor_manager.grad_arrays):
            #    logger.debug('%s-%f', pari[0], pari[1][0].asnumpy().mean())

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

            logger.debug('Done update')

        if monitor is not None:
            monitor.toc_print()

        if is_train:
            eval_metric.reset()

        if m == 1:
            executor_manager.update_metric(eval_metric, data_batch.label)
            name_value = eval_metric.get_name_value()

            for name, value in name_value:
                acc_hist.append(value)
                if is_train:
                    logger.debug('[%02dth Step] %s:%f', t, name, value)

        # Time Step Callback
        if callback:
            callback = [callback] if not isinstance(callback, list) else callback
            for cb in callback:
                assert callable(cb), cb
                params = (t, m, eval_metric, locals())
                cb(params)

    # end of all T

    return executor_manager, eval_metric, acc_hist


def _train_rnn(
        symbol,
        ctx,
        marks,
        arg_names, param_names, aux_names,
        arg_params, aux_params,
        begin_epoch, end_epoch, epoch_size, optimizer,
        kvstore, update_on_kvstore, train_data,
        e_marks=None,
        eval_data=None, eval_metric=None,
        epoch_end_callback=None, batch_end_callback=None, time_step_callback=None,
        logger=None, work_load_list=None, monitor=None,
        eval_batch_end_callback=None, sym_gen=None,
        mutable_data_shape=False, max_data_shape=None):
    """Mark should be a list of #SeriesLength, annotating if image has label by 1 , 0"""
    # TODO marks not working if label of SAX is different in one batch

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

    #if not update_on_kvstore:
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

    for epoch in range(begin_epoch, end_epoch):
        # Training phase
        tic = time.time()
        eval_metric.reset()
        nbatch = 0
        # Iterate over training data.

        # Into Epoch
        #########################
        # record acc
        acc_hist = []

        logger.info('Starting New Epoch...')
        while True:
            do_reset = True

            # iter on batch_size
            for data_batch_zoo in train_data:
                assert isinstance(data_batch_zoo, list), "Iter Error"
                if monitor is not None:
                    monitor.tic()

                # Start to iter on Time steps
                if isinstance(marks[nbatch], list):
                    M = marks[nbatch]
                else:
                    M = marks

                executor_manager, eval_metric, acc_hist = _run_sax(
                    data_batch_zoo, M, executor_manager, eval_metric, updater, ctx, kvstore, acc_hist,
                    monitor=monitor,
                    logger=logger,
                    update_on_kvstore=update_on_kvstore,
                    is_train=True,
                    callback= time_step_callback
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

            # end on batch_size
            if do_reset is True:
                logger.debug('Epoch[%d] Resetting Data Iterator', epoch)
                train_data.reset()
                logger.debug('Epoch[%d] Resetting Eval Metric', epoch)
                eval_metric.reset()

            # this epoch is done
            if epoch_size is None or nbatch >= epoch_size:
                break

        toc = time.time()
        logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

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
            for b, eval_zoo in enumerate(eval_data):

                if isinstance(e_marks[b], list):
                    M = e_marks[b]
                else:
                    M = e_marks

                executor_manager, eval_metric, acc_hist = _run_sax(
                    eval_zoo, M, executor_manager, eval_metric, updater, ctx, kvstore, acc_hist,
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
    # end of all epochs
    return


class Feed(FeedForward):
    def __init__(self, symbol, rnn_hidden=RNN_HIDDEN, **kwargs):
        """Overwrite"""
        super(Feed, self).__init__(symbol, **kwargs)
        self.rnn_hidden = rnn_hidden

    def fit(self,
            X,
            marks,
            e_marks=None,
            y=None, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, time_step_callback=None,
            kvstore='local', logger=None,
            work_load_list=None, monitor=None, eval_batch_end_callback=None):
        """Overwrite"""

        data = self._init_iter(X, y, is_train=True)
        eval_data = self._init_eval_iter(eval_data)

        if self.sym_gen:
            self.symbol = self.sym_gen(
                data.default_bucket_key)  # pylint: disable=no-member
            self._check_arguments()
        self.kwargs["sym"] = self.symbol

        param_dict = dict(data.provide_data + data.provide_label)
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
                    param_idx2name[i * len(self.ctx) + k] = n
        self.kwargs["param_idx2name"] = param_idx2name

        # init optmizer
        if isinstance(self.optimizer, str):
            batch_size = data.batch_size
            if kvstore and kvstore.type == 'dist_sync':
                batch_size *= kvstore.num_workers
            optimizer = opt.create(self.optimizer,
                                   rescale_grad=(1.0 / batch_size),
                                   **(self.kwargs))
        elif isinstance(self.optimizer, opt.Optimizer):
            optimizer = self.optimizer

        # do training
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
                   time_step_callback=time_step_callback,
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
    def create(symbol, X, marks,
               e_marks=None,
               y=None, ctx=None,
               num_epoch=None, epoch_size=None, optimizer='sgd', initializer=Uniform(0.01),
               eval_data=None, eval_metric='acc',
               epoch_end_callback=None, batch_end_callback=None, time_step_callback=None,
               kvstore='local', logger=None, work_load_list=None,
               eval_batch_end_callback=None, **kwargs):
        """Overwrite"""
        model = Feed(symbol, ctx=ctx, num_epoch=num_epoch,
                     epoch_size=epoch_size,
                     optimizer=optimizer, initializer=initializer, **kwargs)
        model.fit(X, y, marks, e_marks=e_marks, eval_data=eval_data, eval_metric=eval_metric,
                  epoch_end_callback=epoch_end_callback,
                  batch_end_callback=batch_end_callback,
                  kvstore=kvstore,
                  logger=logger,
                  work_load_list=work_load_list,
                  eval_batch_end_callback=eval_batch_end_callback)
        return model

    @staticmethod
    def load_from_cnn(perfix, epoch, net, shape, ctx=None, **kwargs):
        symbol, arg_params, aux_params = load_checkpoint(perfix, epoch)
        for rm in ['full1_bias', 'full1_weight', 'full2_weight', 'full2_bias']:
            arg_params.pop(rm)
        model = Feed(net, ctx=ctx, begin_epoch=epoch, **kwargs)
        model._init_params(shape)
        model.arg_params.update(arg_params)
        model.aux_params.update(aux_params)
        return model

    def predict(self, X, num_batch=None, return_data=False, reset=True):
        """Overwrite"""
        X = self._init_iter(X, None, is_train=False)

        if reset:
            X.reset()
        data_shapes = X.provide_data
        data_names = [x[0] for x in data_shapes]
        self._init_predictor(data_shapes)
        batch_size = X.batch_size
        data_arrays = [self._pred_exec.arg_dict[name] for name in data_names]

        if return_data:
            data_list = [[] for _ in X.provide_data[:1]]
            label_list = [[] for _ in X.provide_label]

        pred_list = [[] for _ in self._pred_exec.outputs]
        lists = [pred_list, data_list,
                 label_list] if return_data else [pred_list]

        i = 0
        for batch_zoo in X:
            preds = [[] for _ in self._pred_exec.outputs]
            if return_data:
                datas = [[] for _ in X.provide_data[:1]]
                labels = [[] for _ in X.provide_label]

            for t, data_batch in enumerate(batch_zoo):

                _load_data(data_batch, [data_arrays[0]])
                self._pred_exec.forward(is_train=False)

                pred = self._pred_exec.outputs
                real_size = batch_size - data_batch.pad

                for i, p in enumerate(pred):
                    preds[i].append(p[:real_size].asnumpy())
                if return_data:
                    for j, d in enumerate(data_batch.data):
                        datas[j].append(d[:real_size].asnumpy())
                    for z, l in enumerate(data_batch.label):
                        labels[z].append(l[:real_size].asnumpy())

            combine = [preds, datas, labels] if return_data else [preds]
            for c in combine:
                for i, ps in enumerate(c):
                    for p in ps:
                        p = p.reshape((1,) + p.shape)
                    c[i] = np.concatenate(ps, axis=0)

            for to, small in zip(lists, combine):
                for idx, target in enumerate(to):
                    to[idx].append(small[idx])

            i += 1
            if num_batch is not None and i == num_batch:
                break

        for i, l in enumerate(lists):
            for idx, item in enumerate(l):
                l[idx] = np.concatenate(item, axis=1)
            if len(l) == 1:
                lists[i] = l[0]

        if return_data:
            return lists
        else:
            return lists[0]
