import sys
sys.path.insert(0, 'mxnet/python')
import mxnet as mx
import numpy as np
import logging
import importlib
logging.basicConfig(level=logging.DEBUG)
import argparse
from collections import namedtuple
from skimage import io, transform, color
from skimage.restoration import denoise_tv_chambolle
import mmd_loss
import bn_loss
import os

def get_args(arglist=None):
    parser = argparse.ArgumentParser(description='mmd neural style')

    parser.add_argument('--model', type=str, default='vgg19',
                        choices=['vgg'],
                        help='the pretrained model to use')
    parser.add_argument('--content-image', type=str, default='input/IMG_4343.jpg',
                        help='the content image')
    parser.add_argument('--style-image', type=str, default='input/starry_night.jpg',
                        help='the style image')
    parser.add_argument('--stop-eps', type=float, default=.005,
                        help='stop if the relative chanage is less than eps')
    parser.add_argument('--content-weight', type=float, default=1.0,
                        help='the weight for the content image')
    parser.add_argument('--style-weight', type=float, default=5.0,
                        help='the weight for the style image')
    parser.add_argument('--tv-weight', type=float, default=1e-2,
                        help='the magtitute on TV loss')
    parser.add_argument('--max-num-epochs', type=int, default=1000,
                        help='the maximal number of training epochs')
    parser.add_argument('--max-long-edge', type=int, default=600,
                        help='resize the content image')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='the initial learning rate')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu card to use, -1 means using cpu')
    parser.add_argument('--output', type=str, default='out',
                        help='the output image')
    parser.add_argument('--output-folder', type=str, default='output',
                        help='the output folder')
    parser.add_argument('--save-epochs', type=int, default=100,
                        help='save the output every n epochs')
    parser.add_argument('--remove-noise', type=float, default=.02,
                        help='the magtitute to remove noise')
    parser.add_argument('--lr-sched-delay', type=int, default=80,
                        help='how many epochs between decreasing learning rate')
    parser.add_argument('--lr-sched-factor', type=int, default=0.9,
                        help='factor to decrease learning rate on schedule')
    parser.add_argument('--mmd-kernel', type=str, default='',
                        help='kernel type of mmd')
    parser.add_argument('--mmd-gaussian-multi', type=float, default=1.0,
                        help='the gaussian-multiplication in mmd kernels')
    parser.add_argument('--mmd-poly-c', type=float, default=0.0,
                        help='the poly-c in mmd kernels')
    parser.add_argument('--bn-loss', action='store_true',
                        default=False, help='if use bn loss instead of mmd loss')
    parser.add_argument('--style-layer', type=int, default=5,
                        help='number of layers used for style loss (VGG-net)')
    parser.add_argument('--init', type=str, default='random',
                        help='initialization mode. (random, content)')
    parser.add_argument('--multi-weight', type=str, default='1.0',
                        help='the balance weight when using multiple methdos, e.g. "0.5,0.5"\
                            The sum of weights should be 1.0')

    return parser.parse_args()


def PreprocessContentImage(path, long_edge):
    img = io.imread(path)
    if len(img.shape) == 2:
        img = color.gray2rgb(img)
    logging.info("load the content image, size = %s", img.shape[:2])
    factor = float(long_edge) / max(img.shape[:2])
    new_size = (int(round(img.shape[0] * factor)),
                int(round(img.shape[1] * factor)))
    resized_img = transform.resize(img, new_size)
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    logging.info("resize the content image to %s", new_size)
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))


def PreprocessStyleImage(path, shape):
    img = io.imread(path)
    if len(img.shape) == 2:
        img = color.gray2rgb(img)
    resized_img = transform.resize(img, (shape[2], shape[3]))
    sample = np.asarray(resized_img) * 256
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))


def PostprocessImage(img):
    img = np.resize(img, (3, img.shape[2], img.shape[3]))
    img[0, :] += 123.68
    img[1, :] += 116.779
    img[2, :] += 103.939
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    img = np.clip(img, 0, 255)
    return img.astype('uint8')


def SaveImage(img, filename):
    logging.info('save output to %s', filename)
    out = PostprocessImage(img)
    if args.remove_noise != 0.0:
        out = denoise_tv_chambolle(
            out, weight=args.remove_noise, multichannel=True)
    out = np.clip(out, 0, 1.0)
    io.imsave(filename, out)


def style_symbol(input_size, style):
    _, output_shapes, _ = style.infer_shape(
        data=(1, 3, input_size[0], input_size[1]))
    sym_list = []
    grad_scale = []
    for i in range(len(style.list_outputs())):
        shape = output_shapes[i]

        x = mx.sym.Reshape(style[i], target_shape=(
            int(shape[1]), int(np.prod(shape[2:]))))
        x = mx.sym.SwapAxis(data=x, dim1=0, dim2=1)

        sym_list.append(x)

        grad_scale.append((np.prod(shape[1:]), shape[1]))
    return mx.sym.Group(sym_list), grad_scale


def get_loss(style_sym, content, gscale, w_style):
    """generate style loss and content loss
    """
    style_loss = []
    tmp = []
    for i in range(len(style_sym.list_outputs())):
        gvar = mx.sym.Variable("target_gram_%d_label" % i)
        data = mx.sym.Concat(*[style_sym[i], gvar], dim=0)

        loss = None
        j = 0
        if args.bn_loss:
            weight = args.style_weight * \
                w_style[0] * gscale[i][0] / gscale[i][1]
            loss = mx.symbol.Custom(
                data=data, grad_scale=weight, op_type='bnloss')
            j += 1
        for kernel in args.mmd_kernel:
            weight = args.style_weight * w_style[j]
            weight /= gscale[i][0] * \
                gscale[i][1] if kernel == 'poly' else gscale[i][0]
            sym = mx.sym.Custom(
                data=data, grad_scale=weight,
                kernel=kernel, gaussian_multi=args.mmd_gaussian_multi,
                c=args.mmd_poly_c,
                op_type='mmdloss')
            loss = sym if loss is None else loss + sym
            j += 1

        style_loss.append(loss)

        tmp.append(gvar + style_sym[i])

    cvar = mx.sym.Variable("target_content")
    content_loss = mx.sym.sum(mx.sym.square(cvar - content))
    return mx.sym.Group(style_loss), content_loss, mx.sym.Group(tmp)


def get_tv_grad_executor(img, ctx, tv_weight):
    """create TV gradient executor with input binded on img
    """
    if tv_weight <= 0.0:
        return None
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg, num_outputs=nchannel)
    out = mx.sym.Concat(*[
        mx.sym.Convolution(data=channels[i], weight=skernel,
                           num_filter=1,
                           kernel=(3, 3), pad=(1, 1),
                           no_bias=True, stride=(1, 1))
        for i in range(nchannel)])
    kernel = mx.nd.array(np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
                         .reshape((1, 1, 3, 3)),
                         ctx) / 8.0
    out = out * tv_weight
    return out.bind(ctx, args={"img": img,
                               "kernel": kernel})


def train_nstyle(args):
    """Train a neural style network based on MMD loss or BN loss.
    """
    args.mmd_kernel = [x for x in args.mmd_kernel.split(',') if len(x) > 0]
    # Set predefined weight to balance each style loss according
    # The value is determined by the scale of gradients of each style loss
    w_style = []
    if args.bn_loss:
        w_style.append(1e3)
    for kernel in args.mmd_kernel:
        if kernel == 'poly':
            w_style.append(1e-1)
        elif kernel == 'gaussian':
            w_style.append(3e14)
        else:
            w_style.append(2e3)

    # Weights for multiple style loss
    multi_weight = [
        float(x) for x in args.multi_weight.split(',') if len(x) > 0]
    if len(multi_weight) > 0:
        for i in range(len(multi_weight)):
            w_style[0] *= multi_weight[i]
    w_content = 1.0

    dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    content_np = PreprocessContentImage(args.content_image, args.max_long_edge)
    style_np = PreprocessStyleImage(args.style_image, shape=content_np.shape)
    size = content_np.shape[2:]

    # Executor = namedtuple('Executor', ['executor', 'data', 'data_grad'])

    model_module = importlib.import_module('model_' + args.model)
    style, content = model_module.get_symbol(args.style_layer)
    style_sym, gscale = style_symbol(size, style)
    model_executor = model_module.get_executor(style_sym, content, size, dev)

    model_executor.data[:] = content_np
    model_executor.executor.forward()
    content_array = model_executor.content.copyto(mx.cpu())

    model_executor.data[:] = style_np
    model_executor.executor.forward()
    style_array = []
    for i in range(len(model_executor.style)):
        style_array.append(model_executor.style[i].copyto(mx.cpu()))

    # delete the executor
    del model_executor

    style_loss, content_loss, tmp_forshape = get_loss(
        style_sym, content, gscale, w_style)
    model_executor = model_module.get_executor(
        style_loss, content_loss, size, dev, tmp_forshape)

    grad_array = []
    for i in range(len(style_array)):
        style_array[i].copyto(model_executor.arg_dict["target_gram_%d_label" % i])
        grad_array.append(mx.nd.ones((1,), dev)) 
    print np.prod(content_array[0].shape)
    # / np.prod(content_array[0].shape))
    grad_array.append(
        mx.nd.ones((1,), dev) * float(args.content_weight) * w_content)

    print([x.asscalar() for x in grad_array])
    content_array.copyto(model_executor.arg_dict["target_content"])

    # train
    img = mx.nd.zeros(content_np.shape, ctx=dev)
    if args.init == 'random':
        img[:] = mx.rnd.uniform(-0.1, 0.1, img.shape)
    else:
        img[:] = mx.nd.array(content_np, ctx=dev)

    lr = mx.lr_scheduler.FactorScheduler(
        step=args.lr_sched_delay, factor=args.lr_sched_factor)

    optimizer = mx.optimizer.SGD(
        learning_rate=args.lr,
        wd=0.0005,
        momentum=0.9,
        lr_scheduler=lr)
    optim_state = optimizer.create_state(0, img)

    logging.info('start training arguments %s', args)
    old_img = img.copyto(dev)
    clip_norm = 1 * np.prod(img.shape)
    tv_grad_executor = get_tv_grad_executor(img, dev, args.tv_weight)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    for e in range(args.max_num_epochs):
        img.copyto(model_executor.data)
        model_executor.executor.forward()
        model_executor.executor.backward(grad_array)
        gnorm = mx.nd.norm(model_executor.data_grad).asscalar()
        # print np.mean(np.abs(model_executor.data_grad.asnumpy()))
        # print model_executor.style[0].asnumpy()[0], model_executor.content.asnumpy()[0]
        if gnorm > clip_norm:
            model_executor.data_grad[:] *= clip_norm / gnorm
    #    print [x.asnumpy() for x in model_executor.style]
        if tv_grad_executor is not None:
            tv_grad_executor.forward()
            optimizer.update(0, img,
                             model_executor.data_grad +
                             tv_grad_executor.outputs[0],
                             optim_state)
        else:
            optimizer.update(0, img, model_executor.data_grad, optim_state)
        new_img = img
        eps = (mx.nd.norm(old_img - new_img) / mx.nd.norm(new_img)).asscalar()

        old_img = new_img.copyto(dev)
        logging.info('epoch %d, relative change %f', e, eps)
        if eps < args.stop_eps:
            logging.info('eps < args.stop_eps, training finished')
            break
        if (e + 1) % args.save_epochs == 0:
            sav_img = new_img.asnumpy()
            SaveImage(sav_img, '%s/tmp_%d.jpg' % (args.output_folder, e + 1))

    method = ','.join(args.mmd_kernel)
    if args.bn_loss:
        if len(method) > 0:
            method = 'bn,' + method
        else:
            method = 'bn'
    if 'poly' in method:
        method += '-c%.2f' % args.mmd_poly_c

    sav_img = new_img.asnumpy()
    sav_name = '%s/%s-%s-%.2f-%.2f-w%s.jpg' % (
        args.output_folder, args.output, method,
        args.style_weight, args.content_weight, args.multi_weight)
    SaveImage(sav_img, sav_name)


if __name__ == '__main__':
    args = get_args()
    train_nstyle(args)
