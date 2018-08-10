import os
import time
import argparse

import cv2
import numpy as np
import tensorflow as tf

import library.model_loader as model_loader




example_text = '''example:

python3 compare_pb2engine.py \
 --compare_images_dir <compare_images_dir>\
 --pb_path        <pb_path> \
 --engine_path    <path to engine> \
 --output_node   <name of output node> \
 --input_node    <name of input node> \
 
'''

def parse_args():
        
    parser = argparse.ArgumentParser( epilog=example_text,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--compare_images_dir', dest="compare_images_dir", type=str, 
                        help="dir of comparation images")


    parser.add_argument('--pb_path', required=True, dest="pb_path", type=str,
                        help="path to pb")

    parser.add_argument('--engine_path', required=True, dest="engine_path", type=str,
                        help="path to engine")

    parser.add_argument('--output_node', required=True, dest="output_node", type=str, 
                        help="name of output node , split from ','")

    parser.add_argument('--input_node', required=True, dest="input_node", type=str, 
                        help="name of input node")

    args = parser.parse_args()

    return args


def preprocessing(path , h , w):
    print(path)
    img = cv2.imread(path)
    img = cv2.resize(img , (h , w))
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img = img/255.0

    
    return img


def main():
    args = parse_args()

    pb_name = os.path.basename(args.pb_path).replace('.','_')
    engine_name = os.path.basename(args.engine_path).replace('.','_')
    report_name = 'benchmark_{}_{}.txt'.format(pb_name,engine_name)
    report = open(report_name , 'w')
    report.write('Benchmark Table From {} vs {} : \n'.format(pb_name,engine_name))


    # Compare pb and engine plan 
    # Prediction : mse or classes prediction error
    # Benchmark : fps

    engine = model_loader.load_engine(args.engine_path)

    prefix_in = 'prefix/' + args.input_node + ':0'
    prefix_out = 'prefix/' + args.output_node + ':0'
    graph , x , y = model_loader.load_pb(args.pb_path , prefix_in ,prefix_out)
    _ , h , w , c = x.get_shape().as_list()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sample_time = 1000

    
    sess = tf.Session(graph=graph,config = config)
    
    if args.compare_images_dir:
        img_list = [os.path.join(args.compare_images_dir , i)
                    for i in os.listdir(args.compare_images_dir) 
                    if '.jpg' in i]
    else:
        img_list = []
    

    for batch_size in [1,5,10,20,40]:
        if len(img_list) == 0:
            test_img = np.random.rand(h , w , c)
            test_batch = [test_img for i in range(batch_size)]
        else:
            indices = np.random.choice(len(img_list), batch_size).tolist()
            test_batch = [preprocessing(img_list[i],h , w )
                            for i in sorted(indices)]
                            
        test_batch = np.array(test_batch)
        # test pb speed
        feed_dict = {x : test_batch}
        start = time.time()
        for i in range(int(sample_time/batch_size)):
            pb_out = sess.run(y,feed_dict = feed_dict)
            
        fps = sample_time /(time.time() - start)
        report.write('TF-PB \t batch : {} \t fps : {} \n'.format(batch_size , fps))
        print ('TF-PB batch : {} , fps : {} '.format(batch_size , fps))

        # HWC -> CHW
        test_batch = test_batch.transpose((0,3,1,2))
        
        # test engine speed
        start = time.time()
        for i in range(int(sample_time/batch_size)):
            out = engine.infer(test_batch)
        fps = sample_time /(time.time() - start)
        report.write('Engine \t batch : {} \t fps : {} \n'.format(batch_size , fps))
        print ('Engine batch : {} , fps : {}'.format(batch_size , fps))
    
    report.write('\n\n')
    # test mse or classes prediction error
    mse = 0
    same = 0
    total = 0
    for i in range(1000):
        batch_size = 1
        if len(img_list) == 0:
            test_img = np.random.rand(h , w , c)
            test_batch = [test_img for i in range(batch_size)]
        else:
            indices = np.random.choice(len(img_list), batch_size).tolist()
            test_batch = [preprocessing(img_list[i],h , w )
                            for i in sorted(indices)]
        
        test_batch = np.array(test_batch)
        feed_dict = {x : test_batch}
        pb_out = sess.run(y,feed_dict = feed_dict)
        pb_pred = np.argmax(pb_out , axis=-1)

        # HWC -> CHW
        test_batch = test_batch.transpose((0,3,1,2))

        plan_out = engine.infer(test_batch)
        plan_out = np.array(plan_out).reshape(pb_out.shape)
        plan_pred = np.argmax(plan_out , axis=-1)


        mse += np.sum(np.abs(pb_out - plan_out))
        same += np.sum(pb_pred == plan_pred)
        total += pb_pred.shape[0]
        
    report.write('MSE : {} , Same Predictions : {} ({}/{}) \n'.format(
        mse/total,same/float(total),same,total
    ))
    print ('MSE : {} , Same Predictions : {} ({}/{})'.format(
        mse/total,same/float(total),same,total
    ))


if __name__ == "__main__" :
    main()