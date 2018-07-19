#encoding:utf-8
import os
import sys
from PIL import Image
import numpy as np
import paddle
import paddle.fluid as fluid

def convolutional_neural_network():
    img = fluid.layers.data(name = 'img', shape = [1,28,28], dtype = 'float32')
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input = img,
        filter_size = 5,
        num_filters = 20,
        pool_size = 2,
        pool_stride = 2,
        act = "relu"
    )
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)

    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input = conv_pool_1,
        filter_size = 5,
        num_filters = 50,
        pool_size = 2,
        pool_stride = 2,
        act = "relu"
    )

    prediction = fluid.layers.fc(input = conv_pool_2, size = 10, act = 'softmax')

    return prediction

def train_program():
    label = fluid.layers.data(name = 'label', shape = [1], dtype = 'int64')

    predit = convolutional_neural_network()
    cost = fluid.layers.cross_entropy(input = predit, label = label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input = predit, label = label)
    
    return[avg_cost, acc] 

def optimizer_program():
    return fluid.optimizer.Adam(learning_rate = 0.001)


def main():
    tarin_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size = 500), batch_size = 64)

    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size = 64)

    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    trainer = fluid.Trainer(
        train_func = train_program,
        place = place,
        optimizer_func = optimizer_program
    )

    params_dirname = "recognize_digits_network.inference.model"

    lists = []

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            if event.step % 100 == 0:
                print "Pass %d, Batch %d, Cost %f" %(event.step,event.epoch, event.metrics[0])

        if isinstance(event, fluid.EndEpochEvent):
            avg_cost ,acc = trainer.test(
                reader = test_reader,
                feed_order = ['img','label']
            )

            print("Test with Epoch %d, avg_cost = %s, acc: %s" %(event.epoch, avg_cost, acc))
            trainer.save_params(params_dirname)
            lists.append((event.epoch, avg_cost, acc))

    trainer.train(
        num_epochs = 5,
        event_handler = event_handler,
        reader = tarin_reader,
        feed_order = ['img', 'label']
    )            

    best = sorted(lists, key = lambda list: float(list[1]))[0]
    print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
    print 'The classification accuracy is %.2f%%' % (float(best[2]) * 100)

    def lood_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = np.array(im).reshape(1,1,28,28).astype(np.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img = lood_image(cur_dir + '/image/infer_3.png')
    inferencer = fluid.Inferencer(
        infer_func = convolutional_neural_network,
        param_path = params_dirname,
        place = place
    )

    results = inferencer.infer({'img': img})

    lab = np.argsort(results)

    print "Label of image/infer_3.png is: %d" % lab[0][0][-1]

if __name__ == '__main__':
    main()


