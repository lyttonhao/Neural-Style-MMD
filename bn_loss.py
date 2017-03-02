# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np
import random


class bnLoss(mx.operator.CustomOp):
    def __init__(self, grad_scale):
        '''
        BN statistic matching
        '''
        self.grad_scale = grad_scale

    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        n = data.shape[0]
        p = data.shape[1]
        x, y = data[:n / 2], data[n / 2:]

        self.ux = mx.nd.sum(x, axis=0) * 2.0 / n
        self.uy = mx.nd.sum(y, axis=0) * 2.0 / n
        diffu = mx.nd.sum(mx.nd.square(self.ux - self.uy))
        self.vx = mx.nd.sqrt(
            mx.nd.sum(mx.nd.square(x - self.ux.reshape((1, p))), axis=0) * 2.0 / n)
        self.vy = mx.nd.sqrt(
            mx.nd.sum(mx.nd.square(y - self.uy.reshape((1, p))), axis=0) * 2.0 / n)
        diffv = mx.nd.sum(mx.nd.square(self.vx - self.vy))

        diff = (diffu + diffv) / p

        self.assign(out_data[0], req[0], diff)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data = in_data[0]
        n = data.shape[0]
        p = data.shape[1]
        ctx = data.context
        x, y = data[:n / 2], data[n / 2:]

        dux = 2.0 / n * mx.nd.dot(
            mx.nd.ones((n / 2, 1), ctx=ctx),
            (self.ux - self.uy).reshape((1, p)))
        duy = 2.0 / n * mx.nd.dot(
            mx.nd.ones((n / 2, 1), ctx=ctx),
            (self.uy - self.ux).reshape((1, p)))

        dvx = 2.0 / n * mx.nd.dot(
            mx.nd.ones((n / 2, 1), ctx=ctx),
            ((self.vx - self.vy) / self.vx).reshape((1, p)))
        dvy = 2.0 / n * mx.nd.dot(
            mx.nd.ones((n / 2, 1), ctx=ctx),
            ((self.vy - self.vx) / self.vy).reshape((1, p)))
        dvx *= x - self.ux.reshape((1, p))
        dvy *= y - self.uy.reshape((1, p))

        dx = dux + dvx
        dy = duy + dvy
        d = mx.nd.concatenate([dx, dy], axis=0)

        d *= self.grad_scale / p
        self.assign(in_grad[0], req[0], d)


@mx.operator.register("bnloss")
class bnLossProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0):
        super(bnLossProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)


    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (1, )
  
        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return bnLoss(self.grad_scale)
