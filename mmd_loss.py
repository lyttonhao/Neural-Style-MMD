# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np
import random


class mmdLoss(mx.operator.CustomOp):
    def __init__(self, kernel, c, grad_scale, gaussian_multi):
        '''
        MMD loss for neural style transfer
        :param kernel: kind of mmd kernel ('linear', 'poly' or 'Gaussian')
        :param c: param c for poly kernel
        :param grad_scale: gradient scale
        :param gaussian_multi: param of gamma in Gaussian kernel
        '''
        self.kernel = kernel
        self.c = c
        self.grad_scale = grad_scale
        self.gaussian_multi = gaussian_multi

    def forward(self, is_train, req, in_data, out_data, aux):
        # print "forward"
        data = in_data[0]
        n = data.shape[0]
        x, y = data[:n / 2], data[n / 2:]
        ctx = data.context

        # we skip the forward computation of poly and linear kernel to save time
        if self.kernel == 'poly':
            ## speed up version
            # diffx = mx.nd.sum(mx.nd.square(mx.nd.dot(x.T, x))) + 2 * self.c * mx.nd.sum(mx.nd.dot(x.T, x))
            # diffy = mx.nd.sum(mx.nd.square(mx.nd.dot(y.T, y))) + 2 * self.c * mx.nd.sum(mx.nd.dot(y.T, y))
            # diffxy = mx.nd.sum(mx.nd.square(mx.nd.dot(x, y.T))) + 2 * self.c * mx.nd.sum(mx.nd.dot(x, y.T))
            # diff = diffx + diffy - 2 * diffxy
            diff = 0
        elif self.kernel == 'linear':
            # print y.shape
            # idx, idy = np.argmax(x.asnumpy()), np.argmax(y.asnumpy())
            # print idx, idy, x.asnumpy()[:, idx], y.asnumpy()[:, idy]
            # diffx = mx.nd.sum(mx.nd.dot(x, x.T))
            # diffy = mx.nd.sum(mx.nd.dot(y, y.T))
            # diffxy = mx.nd.sum(mx.nd.dot(x, y.T))
            # print diffx.asnumpy(), diffy.asnumpy(), diffxy.asnumpy()
            # diff = diffx + diffy - 2 * diffxy
            diff = 0
        else:
            # gaussian
            x, y = x.asnumpy(), y.asnumpy()
            idx, idy = np.arange(n / 2), np.arange(n / 2)
            random.shuffle(idx)
            random.shuffle(idy)
            x, y = x[idx], y[idy]
            n0 = n / 2 - ((n / 2) & 1)
            x0, x1 = mx.nd.array(x[range(0, n0, 2)], ctx=ctx), mx.nd.array(x[range(1, n0, 2)], ctx=ctx)
            y0, y1 = mx.nd.array(y[range(0, n0, 2)], ctx=ctx), mx.nd.array(y[range(1, n0, 2)], ctx=ctx)
            # print x0.shape, x1.shape, y0.shape, y1.shape
            diffx = mx.nd.sum(mx.nd.square(x0 - x1), axis=1)
            diffy = mx.nd.sum(mx.nd.square(y0 - y1), axis=1)
            diffxy = mx.nd.sum(mx.nd.square(x0 - y1), axis=1)
            diffyx = mx.nd.sum(mx.nd.square(y0 - x1), axis=1)
            # print mx.nd.sum(diffx).asnumpy()
            self.gamma = self.gaussian_multi * n0 * 2 /\
                (mx.nd.sum(diffx) + mx.nd.sum(diffy) +
                 mx.nd.sum(diffxy) + mx.nd.sum(diffyx))
            self.diffx = mx.nd.exp(-self.gamma * diffx)
            self.diffy = mx.nd.exp(-self.gamma * diffy)
            self.diffxy = mx.nd.exp(-self.gamma * diffxy)
            self.diffyx = mx.nd.exp(-self.gamma * diffyx)

            diff = mx.nd.sum(self.diffx) + mx.nd.sum(self.diffy) - mx.nd.sum(self.diffxy) - mx.nd.sum(self.diffyx)

            self.idx = idx
            self.idy = idy

        self.assign(out_data[0], req[0], diff)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data = in_data[0]
        n = data.shape[0]
        f = data.shape[1]
        x, y = data[:n / 2], data[n / 2:]
        ctx = data.context

        if self.kernel == 'poly':
            dx = mx.nd.dot(x, mx.nd.dot(x.T, x) - mx.nd.dot(y.T, y))\
                + self.c * mx.nd.dot(mx.nd.ones((n / 2, 1), ctx=ctx), mx.nd.sum(x - y, axis=0).reshape((1, f)))
            dy = mx.nd.dot(y, mx.nd.dot(y.T, y) - mx.nd.dot(x.T, x))\
                + self.c * mx.nd.dot(mx.nd.ones((n / 2, 1), ctx=ctx), mx.nd.sum(y - x, axis=0).reshape((1, f)))

            d = mx.nd.concatenate([dx, dy], axis=0)
        elif self.kernel == 'linear':
            dx = mx.nd.dot(mx.nd.ones((n / 2, 1), ctx=ctx), mx.nd.sum(x - y, axis=0).reshape((1, f)))
            dy = mx.nd.dot(mx.nd.ones((n / 2, 1), ctx=ctx), mx.nd.sum(y - x, axis=0).reshape((1, f)))

            d = mx.nd.concatenate([dx, dy], axis=0)
        else:
            x, y = x.asnumpy(), y.asnumpy()

            idx, idy = self.idx, self.idy

            x, y = x[idx], y[idy]
            n0 = n / 2 - ((n / 2) & 1)
            x0, x1 = mx.nd.array(x[range(0, n0, 2)], ctx=ctx), mx.nd.array(x[range(1, n0, 2)], ctx=ctx)
            y0, y1 = mx.nd.array(y[range(0, n0, 2)], ctx=ctx), mx.nd.array(y[range(1, n0, 2)], ctx=ctx)

            k = self.diffx.shape[0]

            diffx = self.diffx.reshape((k, 1))
            diffy = self.diffy.reshape((k, 1))
            diffxy = self.diffxy.reshape((k, 1))
            diffyx = self.diffyx.reshape((k, 1))

            dx0 = self.gamma * (-diffx * (x0 - x1) + diffxy * (x0 - y1))
            dx1 = self.gamma * (-diffx * (x1 - x0) + diffyx * (x1 - y0))
            dy0 = self.gamma * (-diffy * (y0 - y1) + diffyx * (y0 - x1))
            dy1 = self.gamma * (-diffy * (y1 - y0) + diffxy * (y1 - x0))

            d = np.zeros(data.shape)
            d[idx[range(0, n0, 2)]] = dx0.asnumpy()
            d[idx[range(1, n0, 2)]] = dx1.asnumpy()
            d[n / 2 + idy[range(0, n0, 2)]] = dy0.asnumpy()
            d[n / 2 + idy[range(1, n0, 2)]] = dy1.asnumpy()

            d = mx.nd.array(d).as_in_context(ctx)

        d *= self.grad_scale
        self.assign(in_grad[0], req[0], d)


@mx.operator.register("mmdloss")
class mmdLossProp(mx.operator.CustomOpProp):
    def __init__(self, kernel='poly', c=0.0, grad_scale=1.0, gaussian_multi=1.0):
        super(mmdLossProp, self).__init__(need_top_grad=False)
        self.kernel = kernel
        self.c = float(c)
        self.grad_scale = float(grad_scale)
        self.gaussian_multi = float(gaussian_multi)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (1, )

        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return mmdLoss(self.kernel, self.c, self.grad_scale, self.gaussian_multi)
