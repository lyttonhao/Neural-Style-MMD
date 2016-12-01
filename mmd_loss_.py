# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np


class mmdLoss(mx.operator.CustomOp):
    def __init__(self, kernel, c, grad_scale):
        self.kernel = kernel
        self.c = c
        self.grad_scale = grad_scale

    def dis(self, diff):
        return np.sum(diff)

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        n = x.shape[0]
        n0 = n / 2 - ((n / 2) & 1) # first domain (even number of sampels)
        x0, x1 = x[range(0, n0, 2)], x[range(1, n0, 2)]
        y0, y1 = x[range(n - n0, n, 2)], x[range(n - n0 + 1, n, 2)]

        #print n, x0.shape, x1.shape, y0.shape, y1.shape

        diff0 = np.sum(np.square(x0 - x1), axis=0)
        diff1 = np.sum(np.square(y0 - y1), axis=0)
        diff2 = np.sum(np.square(x0 - y1), axis=0)
        diff3 = np.sum(np.square(x1 - y0), axis=0)

        self.assign(
            out_data[0], req[0],
            self.dis(diff0) + self.dis(diff1) - self.dis(diff2) - self.dis(diff3))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0].asnumpy()
        n = x.shape[0]
        n0 = n / 2 - ((n / 2) & 1) # first domain (even number of sampels)
        x0, x1 = x[range(0, n0, 2)], x[range(1, n0, 2)]
        y0, y1 = x[range(n - n0, n, 2)], x[range(n - n0 + 1, n, 2)]

        dx = np.zeros(x.shape)
        dx[range(0, n0, 2)] = 2 * x0 - x1 - y1
        dx[range(1, n0, 2)] = 2 * x1 - x0 - y0
        dx[range(n - n0, n, 2)] = 2 * y0 - x1 - y1
        dx[range(n - n0 + 1, n, 2)] = 2 * y1 - x0 - y0
        dx *= self.grad_scale
        self.assign(in_grad[0], req[0], mx.nd.array(dx))


@mx.operator.register("mmdloss")
class mmdLossProp(mx.operator.CustomOpProp):
    def __init__(self, kernel='poly', c=0.0, grad_scale=1.0):
        super(mmdLossProp, self).__init__(need_top_grad=True)
        self.kernel = kernel
        self.c = float(c)
        self.grad_scale = float(grad_scale)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (1,)

        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return mmdLoss(self.kernel, self.c, self.grad_scale)
