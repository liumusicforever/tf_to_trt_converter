import os
import sys
import argparse

path_to_slim = 'models/research/slim/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_slim)))

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

from nets import nets_factory

example_text = '''example:

python slim_ckp2pb.py \
 --ckpt_path   <ckpt_path> \
 --pb_path     <pb_path> \
 --output_node <name of output node> \
 --model_name  <name of network> \
 --num_classes <number of classes>

'''
def parse_args():
    
    parser = argparse.ArgumentParser( epilog=example_text,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)

    
    parser.add_argument('--ckpt_path', required=True, dest="ckpt_path", type=str,
                        help="path to ckpt")
    parser.add_argument('--pb_path', required=True, dest="pb_path", type=str, 
                        help="path to pb")
    parser.add_argument('--output_nodes', required=True, dest="output_nodes", type=str, 
                        help="name of output node , split from ','")
    parser.add_argument('--model_name', required=True, dest="model_name", type=str, 
                        help="name of network")
    parser.add_argument('--num_classes', required=True, dest="num_classes", type=int, 
                        help="number of classes")
    
    args = parser.parse_args()


    return args

def main():
    
    args = parse_args()
    
    ckpt_path = args.ckpt_path
    pb_path = args.pb_path
    output_nodes = args.output_nodes
    model_name = args.model_name
    num_classes = args.num_classes

    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes,
        is_training=False)
    image_size = network_fn.default_image_size
    
    image = tf.placeholder(name='input', dtype=tf.float32,
        shape=[None, image_size,image_size, 3])
    logits, end_points = network_fn(image)
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
        output_nodes.split(",") # We split on comma for convenience
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(pb_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == "__main__":
    main()