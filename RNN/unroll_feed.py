import ipt
import mxnet as mx

from mxnet.model import FeedForward
from mxnet.executor_manager import DataParallelExecutorGroup, _split_input_slice

class ExecuterGroup(DataParallelExecutorGroup):
    
    def __init__(self, sym, arg_names, param_names,
                 ctx, slices, train_data, 
                 max_data_shape=None, shared_group=None):
        _check_arguments(sym)

        def in_name(name_list, k):
            for n in name_list:
                if n in k:
                    return True
            return False

        if shared_group is None:
            self.shared_data_arrays = [{} for _ in ctx]
        else:
            self.shared_data_arrays = shared_group.shared_data_arrays

        self.data_names = [x[0] for x in train_data.provide_data]
        self.label_names = [x[0] for x in train_data.provide_label]
        self.aux_names = sym.list_auxiliary_states()
        self.param_idx = [i for i in range(len(arg_names)) if arg_names[i] in param_names]
        self.param_names = [arg_names[i] for i in self.param_idx]

        self.train_execs = []
        batch_size = train_data.batch_size
        for i in range(len(ctx)):
            data_shapes = {}
            for k, v in train_data.provide_data + train_data.provide_label:
                if in_name(['data', 'label'], k):
                    if shared_group is None and max_data_shape is not None:
                        # init first executor group
                        # data size is set to max possible size of input data
                        data_shapes[k] = tuple([slices[i].stop - slices[i].start] + max_data_shape)
                    else:
                        data_shapes[k] = tuple([slices[i].stop - slices[i].start] + list(v[1:]))
                elif in_name(['weight','bias','gamma','beta'], k):
                    data_shapes[k] = v
                else:
                    data_shapes[k] = tuple([int((slices[i].stop - slices[i].start) * v[0] \
                                           / batch_size)] + list(v[1:]))

            shared_exec = None if shared_group is None else shared_group.train_execs[i]
            train_exec = _bind_exec(sym, ctx[i], data_shapes, self.param_names,
                                    need_grad=True, base_exec=shared_exec,
                                    shared_data_arrays=self.shared_data_arrays[i])
            self.train_execs.append(train_exec)

        # data structure
        self.data_arrays = [[(slices[i], e.arg_dict[name]) for i, e in enumerate(self.train_execs)]
                            for name in self.data_names]
        self.label_arrays = [[(slices[i], e.arg_dict[name]) for i, e in enumerate(self.train_execs)]
                             for name in self.label_names]

        self.param_arrays = [[e.arg_arrays[i] for e in self.train_execs]
                             for i in self.param_idx]
        self.grad_arrays = [[e.grad_arrays[i] for e in self.train_execs]
                            for i in self.param_idx]

        self.aux_arrays = [[e.aux_arrays[i] for e in self.train_execs]
                           for i in range(len(self.aux_names))]

        self.slices = slices


class UnrollFeed(FeedForward):

    def __init__(*args, **kwargs):
        super(UnrollFeed, self).__init__(*args, **kwargs)


    def fit(self, X, y=None, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local', logger=None,
            work_load_list=None, monitor=None, eval_batch_end_callback=None):

        train_data = self._init_iter(X, y, is_train=True)
        eval_data = self._init_eval_iter(eval_data)

        # prepare sym
        if self.sym_gen:
            self.symbol = self.sym_gen(train_data.default_bucket_key) # pylint: disable=no-member
            self._check_arguments()
        self.kwargs["sym"] = self.symbol

        # prepare args name and shape
        arg_names, param_names, aux_names = \
                self._init_params(dict(train_data.provide_data+train_data.provide_label))

        # setup metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        # create kvstore
        (kvstore, update_on_kvstore) = _create_kvstore(
            kvstore, len(self.ctx), self.arg_params)

        # param_idx2name = {}
        # if update_on_kvstore:
        #     param_idx2name.update(enumerate(param_names))
        # else:
        #     for i, n in enumerate(param_names):
        #         for k in range(len(self.ctx)):
        #             param_idx2name[i*len(self.ctx)+k] = n
        # self.kwargs["param_idx2name"] = param_idx2name

        # init optmizer
        if isinstance(self.optimizer, str):
            batch_size = train_data.batch_size
            if kvstore and kvstore.type == 'dist_sync':
                batch_size *= kvstore.num_workers
            optimizer = opt.create(self.optimizer,
                                   rescale_grad=(1.0/batch_size),
                                   **(self.kwargs))
        elif isinstance(self.optimizer, opt.Optimizer):
            optimizer = self.optimizer


        num_device = len(self.ctx)
        logger.info('Start training with %s', str(self.ctx))

        if work_load_list is None:
            work_load_list = [1] * num_device
        assert isinstance(work_load_list, list) and len(work_load_list) == num_device, \
            "Invalid settings for work load. "

        from mxnet.executor_manager import _split_input_slice
        self.slices = _split_input_slice(train_data.batch_size, work_load_list)

        execgrp = ExecuterGroup(self.symbol, arg_names, param_names, self.ctx,
                                                 self.slices, train_data)

        if self.sym_gen is not None:
            self.execgrp_bucket = {train_data.default_bucket_key: self.execgrp}

        # executor_manager.set_params(arg_params, aux_params)

        train_data.reset()
        for epoch in range(self.begin_epoch, self.num_epoch):
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
                for data_batch in train_data:
                    executor_manager.load_data_batch(data_batch)

                    if monitor is not None:
                        monitor.tic()

                    executor_manager.forward(is_train=True)
                    executor_manager.backward()
                    
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

                    eval_metric.reset()
                    executor_manager.update_metric(eval_metric, data_batch.label)
                    name_value = eval_metric.get_name_value()

                    for name, value in name_value:
                        acc_hist.append(value)
                        # logger.info('Epoch[%d] Training-%s=%f', epoch, name, value)


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
                    logger.debug('Epoch[%d] Resetting Data Iterator', epoch)
                    train_data.reset()
                    logger.debug('Epoch[%d] Resetting Eval Metric', epoch)
                    eval_metric.reset()

                # this epoch is done
                if epoch_size is None or nbatch >= epoch_size:
                    break

            toc = time.time()
            logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

            # print epoch ,'<<<<<<<<<<<<'
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
                eval_metric.reset()
                eval_data.reset()
                for i, eval_batch in enumerate(eval_data):
                    executor_manager.load_data_batch(eval_batch)
                    executor_manager.forward(is_train=False)
                    executor_manager.update_metric(eval_metric, eval_batch.label)
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
                    logger.info('E[%d] V %s:%f', epoch, name, value)
                    # print 'Epoch[%d] Validation=%f' % (epoch, value)
        # end of all epochs
        return




