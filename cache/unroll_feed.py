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

class Unroll_Feed(FeedForward):
    def __init__(self, symbol, **kwargs):
        """Overwrite"""
        super(Unroll_Feed, self).__init__(symbol, **kwargs)

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
