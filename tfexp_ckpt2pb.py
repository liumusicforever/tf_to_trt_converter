import os
import sys
import argparse

path_to_tfexp = 'tf_models_experiment_framework/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_tfexp)))

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

import lib.utils as utils

example_text = '''example:

python3 tfexp_ckp2pb.py \
 --ckpt_path   <ckpt_path> \
 --pb_path     <pb_path> \
 --output_node <name of output node> \
 --network  <path to network> \
 --image_size 224
'''
def parse_args():
    
    parser = argparse.ArgumentParser( epilog=example_text,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)

    
    parser.add_argument('--ckpt_path', required=True, dest="ckpt_path", type=str,
                        help="path to ckpt")
    parser.add_argument('--pb_path', required=True, dest="pb_path", type=str, 
                        help="path to pb")
    parser.add_argument('--output_node', required=True, dest="output_node", type=str, 
                        help="name of output node , split from ','")
    parser.add_argument('--network', required=True, dest="network", type=str, 
                        help="path to network")

    parser.add_argument('--image_size', required=True, dest="image_size", type=int, 
                        help="size of network")                   
    
    args = parser.parse_args()


    return args

def main():
    args = parse_args()
    
    ckpt_path = args.ckpt_path
    pb_path = args.pb_path
    output_node = args.output_node
    network = args.network
    
    mod_graph , data_iter , params = utils.load_network(args.network)

    # Define the model
    tf.logging.info("Creating the model...")    
    image_size = int(args.image_size)
    image = tf.placeholder(name='input', dtype=tf.float32,
                                 shape=[None, image_size,
                                        image_size, 3])
    
    net_tensors  = mod_graph.model_fn(image,None,tf.estimator.ModeKeys.PREDICT,params)

    mode , pred_ops = net_tensors[0:2]

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # restore checkpoint 
    if tf.gfile.IsDirectory(ckpt_path):
        ckpt_path = tf.train.latest_checkpoint(ckpt_path)
    else:
        ckpt_path = ckpt_path
    saver.restore(sess, ckpt_path)

    
    # get all var 
    var_list = [n for n in tf.get_default_graph().as_graph_def().node]
    # var_list = [n for n in tf.global_variables()]
    out_dir = os.path.dirname(os.path.abspath(pb_path))
    with open(os.path.join(out_dir , 'graph.txt'), "w") as output:
            output.write("\n".join([str(i) for i in var_list]))

            
    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    # We use a built-in TF helper to export variables to constant
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, 
        input_graph_def, 
        [output_node]
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(pb_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == "__main__":
    main()