import os
import time
import argparse

import cv2
import uff
import ctypes
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from tensorrt.parsers import uffparser
import tensorflow as tf


example_text = '''example:

python3 pb2engine.py \
 --pb_path        <pb_path> \
 --engine_path    <path to engine> \
 --output_node   <name of output node> \
 --input_node    <name of input node> \
 --image_size     <size of network> \
 --max_batch_size <max size of batch> \
 --max_workspace  <max workspace> \
 --int8 <type anything if need>
'''

def parse_args():
    
    parser = argparse.ArgumentParser( epilog=example_text,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)

    
    parser.add_argument('--pb_path', required=True, dest="pb_path", type=str,
                        help="path to pb")

    parser.add_argument('--engine_path', required=True, dest="engine_path", type=str,
                        help="path to engine")

    parser.add_argument('--output_node', required=True, dest="output_node", type=str, 
                        help="name of output node , split from ','")

    parser.add_argument('--input_node', required=True, dest="input_node", type=str, 
                        help="name of input node")

    parser.add_argument('--image_size', required=True, dest="image_size", type=int, 
                        help="size of network")   

    parser.add_argument('--max_batch_size', required=True, dest="max_batch_size", type=int, 
                        help="max size of batch")   

    parser.add_argument('--max_workspace', required=True, dest="max_workspace", type=int, 
                        help="max workspace")   
    
    parser.add_argument('--calib_images_dir', dest="calib_images_dir", type=str, 
                        help="(optional) dir of calibration images")
    
    parser.add_argument('--int8', dest="int8", type=str, 
                        help="(optional) convert with int8")
    
    args = parser.parse_args()


    return args

class PythonEntropyCalibrator(trt.infer.EntropyCalibrator):
    def __init__(self, input_layers, stream):
        trt.infer.EntropyCalibrator.__init__(self)       
        self.input_layers = input_layers
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, bindings, names):
        batch = self.stream.next_batch()
        if not batch.size:   
            return None

        cuda.memcpy_htod(self.d_input, batch)
        for i in self.input_layers[0]:
            assert names[0] != i

        bindings[0] = int(self.d_input)
        return bindings

    def read_calibration_cache(self, length):
        return None

    def write_calibration_cache(self, ptr, size):
        cache = ctypes.c_char_p(int(ptr))
        with open('calibration_cache.bin', 'wb') as f:
            f.write(cache.value)
        return None

class ImageBatchStream():
    def __init__(self, batch_size, calibration_files , INPUT_SIZE):
        self.batch_size = batch_size
        # TODO : is the time of calibration effect result?
        self.max_batches = 10
        self.files = calibration_files
        self.CHANNEL, self.HEIGHT, self.WIDTH = INPUT_SIZE
        self.calibration_data = np.zeros((batch_size, self.CHANNEL, 
                                         self.HEIGHT, self.WIDTH) 
                                         , dtype=np.float32)

        self.batch = 0

    def read_image_chw(self,path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        img = cv2.resize(img , (self.HEIGHT , self.WIDTH))
        img = img/255.0
        img = img.transpose((2,0,1))
        return img

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            if self.files:
                indices = np.random.choice(len(self.files), self.batch_size).tolist()
                files_for_batch = [self.files[i] for i in sorted(indices)]
                print (files_for_batch)
                for f in files_for_batch:
                    img = self.read_image_chw(f)
                    imgs.append(img)
            else:
                for i in range(self.batch_size):
                    img = np.random.rand(3,self.HEIGHT , self.WIDTH)
                    imgs.append(img)    

            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

def main():
    args = parse_args()
    
    # Convert pb to uff
    uff_model = uff.from_tensorflow_frozen_model(args.pb_path, [args.output_node])

    # Create UFF parser and logger
    parser = uffparser.create_uff_parser()

    INPUT_SIZE = [3 , args.image_size , args.image_size]

    parser.register_input(args.input_node,INPUT_SIZE , 0)
    parser.register_output(args.output_node)
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

    # Convert uff to plan
    if args.calib_images_dir:
        calibration_files = [os.path.join(args.calib_images_dir,i)
                        for i in os.listdir(args.calib_images_dir)]
    else:
        calibration_files = []
    batchstream = ImageBatchStream(args.max_batch_size, calibration_files,INPUT_SIZE)
    int8_calibrator = PythonEntropyCalibrator([args.input_node], batchstream)

    if args.int8:
        engine = trt.utils.uff_to_trt_engine(
            G_LOGGER, uff_model, 
            parser, 
            args.max_batch_size, args.max_workspace,
            datatype = trt.infer.DataType.INT8,
            calibrator = int8_calibrator
        )
    else:
        engine = trt.utils.uff_to_trt_engine(
            G_LOGGER, uff_model, 
            parser, 
            args.max_batch_size, args.max_workspace
        )
    
    trt.utils.write_engine_to_file(args.engine_path, engine.serialize())

if __name__ == "__main__" :
    main()