# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
#os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np
import random

class mmdLoss(mx.operator.CustomOp):
    def __init__(self, kernel, c, grad_scale):
        self.kernel = kernel
        self.c = c
        self.grad_scale = grad_scale

    def forward(self, is_train, req, in_data, out_data, aux):
        # print "forward"
        data = in_data[0]
        n = data.shape[0]
        x, y = data[:n / 2], data[n / 2:]
        ctx = data.context

        # print data
        if self.kernel == 'poly':
            # diffx = mx.nd.sum(mx.nd.square(mx.nd.dot(x, x.T) + self.c))
            # diffy = mx.nd.sum(mx.nd.square(mx.nd.dot(y, y.T) + self.c))
            # diffxy = mx.nd.sum(mx.nd.square(mx.nd.dot(x, y.T) + self.c))
            ## speed up version
            diffx = mx.nd.sum(mx.nd.square(mx.nd.dot(x.T, x))) + 2 * self.c * mx.nd.sum(mx.nd.dot(x.T, x))
            diffy = mx.nd.sum(mx.nd.square(mx.nd.dot(y.T, y))) + 2 * self.c * mx.nd.sum(mx.nd.dot(y.T, y))
            diffxy = mx.nd.sum(mx.nd.square(mx.nd.dot(x.T, y))) + 2 * self.c * mx.nd.sum(mx.nd.dot(x.T, y))
            #diff = mx.nd.array([diffx + diffy - 2 * diffxy], ctx=ctx)
            diff = diffx + diffy - 2 * diffxy
            # print "pre"
            # diff = mx.nd.concatenate([diff, mx.nd.array(range(n)).as_in_context(ctx)])
            #print diff
        elif self.kernel == 'linear':
            diffx = mx.nd.sum(mx.nd.dot(x.T, x))
            diffy = mx.nd.sum(mx.nd.dot(y.T, y))
            diffxy = mx.nd.sum(mx.nd.dot(x.T, y))
            diff = diffx + diffy - 2 * diffxy
        else:
            # gaussian
            x, y = x.asnumpy(), y.asnumpy()
            idx, idy = np.arange(n / 2), np.arange(n / 2)
            random.shuffle(idx)
            random.shuffle(idy)
            x, y = x[idx], y[idy]
            n0 = n / 2 - ((n / 2) & 1)
            x0, x1 = mx.nd.array(x[range(0, n0, 2)]).as_in_context(ctx), mx.nd.array(x[range(1, n0, 2)]).as_in_context(ctx)
            y0, y1 = mx.nd.array(y[range(0, n0, 2)]).as_in_context(ctx), mx.nd.array(y[range(1, n0, 2)]).as_in_context(ctx)
            # print x0.shape, x1.shape, y0.shape, y1.shape
            diffx = mx.nd.sum(mx.nd.square(x0 - x1), axis=1)
            diffy = mx.nd.sum(mx.nd.square(y0 - y1), axis=1)
            diffxy = mx.nd.sum(mx.nd.square(x0 - y1), axis=1)
            diffyx = mx.nd.sum(mx.nd.square(y0 - x1), axis=1)
            # print mx.nd.sum(diffx).asnumpy()
            self.gamma = n0 * 2 / (mx.nd.sum(diffx) + mx.nd.sum(diffy) + mx.nd.sum(diffxy) + mx.nd.sum(diffyx))
            self.diffx = mx.nd.exp(-self.gamma * diffx)
            self.diffy = mx.nd.exp(-self.gamma * diffy)
            self.diffxy = mx.nd.exp(-self.gamma * diffxy)
            self.diffyx = mx.nd.exp(-self.gamma * diffyx)
            #diff = 0
            #print diffx.asnumpy(), diffy.asnumpy(), diffxy.asnumpy(), diffyx.asnumpy()
            
            diff = mx.nd.sum(self.diffx) + mx.nd.sum(self.diffy) - mx.nd.sum(self.diffxy) - mx.nd.sum(self.diffyx)
            # diff = mx.nd.concatenate([diff, mx.nd.array(idx + idy).as_in_context(ctx)])#.as_in_context(ctx)

            # print "last"
            # print diff
            #self.assign(out_data[1], req[1], idxy)

            #diff = 
            self.idx = idx
            self.idy = idy



        self.assign(out_data[0], req[0], diff)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # print "backward"
        data = in_data[0]
        n = data.shape[0]
        f = data.shape[1]
        x, y = data[:n / 2], data[n / 2:]
        ctx = data.context

        if self.kernel == 'poly':
            # dx = mx.nd.dot(mx.nd.dot(x, x.T) + self.c, x - y)
            # dy = mx.nd.dot(mx.nd.dot(y, y.T) + self.c, y - x)
            ## speed up version
            dx = mx.nd.dot(x, mx.nd.dot(x.T, x - y)) + self.c * (x - y)
            dy = mx.nd.dot(y, mx.nd.dot(y.T, y - x)) + self.c * (y - x)

            # dx = mx.nd.dot(mx.nd.ones((n / 2, 1), ctx=ctx), mx.nd.sum(dx, axis=1).reshape((1, f)))
            # dy = mx.nd.dot(mx.nd.ones((n / 2, 1), ctx=ctx), mx.nd.sum(dy, axis=1).reshape((1, f)))

            d = mx.nd.concatenate([dx, dy], axis=0)
        elif self.kernel == 'linear':
            # dx = mx.nd.dot(mx.nd.ones((n / 2, 1), ctx=ctx), mx.nd.sum(x - y, axis=1).reshape((1, f)))
            # dy = mx.nd.dot(mx.nd.ones((n / 2, 1), ctx=ctx), mx.nd.sum(y - x, axis=1).reshape((1, f)))
            # print mx.nd.sum(x - y, axis=0).shape, n/ 2
            dx = mx.nd.dot(mx.nd.sum(x - y, axis=1).reshape((n / 2, 1)), mx.nd.ones((1, f), ctx=ctx))
            dy = mx.nd.dot(mx.nd.sum(y - x, axis=1).reshape((n / 2, 1)), mx.nd.ones((1, f), ctx=ctx))

            d = mx.nd.concatenate([dx, dy], axis=0)
        else:
            x, y = x.asnumpy(), y.asnumpy()
            # idxy = out_data[0].asnumpy()
            # idxy = idxy[1:].astype(np.int32)
            # idx, idy = idxy[:n / 2], idxy[n / 2:]
            idx, idy = self.idx, self.idy
            # print idx[0], self.idx[0]
            x, y = x[idx], y[idy]
            n0 = n / 2 - ((n / 2) & 1)
            x0, x1 = mx.nd.array(x[range(0, n0, 2)]).as_in_context(ctx), mx.nd.array(x[range(1, n0, 2)]).as_in_context(ctx)
            y0, y1 = mx.nd.array(y[range(0, n0, 2)]).as_in_context(ctx), mx.nd.array(y[range(1, n0, 2)]).as_in_context(ctx)
            # diffx = mx.nd.sum(mx.nd.square(x0 - x1), axis=1)
            # diffy = mx.nd.sum(mx.nd.square(y0 - y1), axis=1)
            # diffxy = mx.nd.sum(mx.nd.square(x0 - y1), axis=1)
            # diffyx = mx.nd.sum(mx.nd.square(y0 - x1), axis=1)
            # gamma = n0 * 2 / (mx.nd.sum(diffx) + mx.nd.sum(diffy) + mx.nd.sum(diffxy) + mx.nd.sum(diffyx))
            k = self.diffx.shape[0]
            # diffx = mx.nd.exp(-gamma * diffx).reshape((k, 1))
            # diffy = mx.nd.exp(-gamma * diffy).reshape((k, 1))
            # diffxy = mx.nd.exp(-gamma * diffxy).reshape((k, 1))
            # diffyx = mx.nd.exp(-gamma * diffyx).reshape((k, 1))
            diffx = self.diffx.reshape((k, 1))
            diffy = self.diffy.reshape((k, 1))
            diffxy = self.diffxy.reshape((k, 1))
            diffyx = self.diffyx.reshape((k, 1))

            
            dx0 = self.gamma * (-diffx * (x0 - x1) + diffxy * (x0 - y1))
            dx1 = self.gamma * (-diffx * (x1 - x0) + diffyx * (x1 - y0))
            dy0 = self.gamma * (-diffy * (y0 - y1) + diffyx * (y0 - x1))
            dy1 = self.gamma * (-diffy * (y1 - y0) + diffxy * (y1 - x0))
            #dy0 = gamma * (-mx.nd.broadcast_mul(diffy, (y0 - y1)) + mx.nd.broadcast_mul(diffyx, (y0 - x1)))
            #dy1 = gamma * (-mx.nd.broadcast_mul(diffy, (y1 - y0)) + mx.nd.broadcast_mul(diffxy, (y1 - x0)))

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
    def __init__(self, kernel='poly', c=0.0, grad_scale=1.0):
        super(mmdLossProp, self).__init__(need_top_grad=False)
        self.kernel = kernel
        self.c = float(c)
        self.grad_scale = float(grad_scale)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    # # def NumOutputs(self):
    # #     return 2;

    # def NumVisibleOutputs(self):
    #     return 1;

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (1, )
        # idx_shape = (in_shape[0][0], )

        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return mmdLoss(self.kernel, self.c, self.grad_scale)
