# tf_to_trt_converter
* Convert model implementation from Tensorflow to TensorRT engine.
* You can convert the model training from [tensorflow slim framwork](https://github.com/tensorflow/models/tree/master/research/slim) or [tf experiment framework](https://github.com/liumusicforever/tf_models_experiment_framework).
* But also convert your custom froozen model (.pb) , by given input and output names.

# Requirement
* Tensorflow >= 1.5
* TensorRT == 4.0
* Numpy 
* Opencv
* Python3

# Quick Start

* Convert .pb to tensorrt engine
```bash
python3 pb2engine.py  
--pb_path        <pb_path>  \
--engine_path    <path to engine> \
--output_node    <name of output node>  \
--input_node     <name of input node>  \
--image_size     <size of network>  \
--max_batch_size <max size of batch>  \
--max_workspace  <max workspace>  \
--int8           <type anything if need>
```
example : 
```bash
python3 pb2engine.py  --pb_path model.pb  --engine_path engine.plan  --output_node InceptionResnetV2/Logits/Predictions  --input_node input  --image_size  299  --max_batch_size 40  --max_workspace 20
```

* Compare .pb with tensorRT engine
```bash
python3 compare_pb2engine.py \
 --compare_images_dir <compare_images_dir>\
 --pb_path            <pb_path> \
 --engine_path        <path to engine> \
 --output_node        <name of output node> \
 --input_node         <name of input node> \
```
example : 
```bash
python3 compare_pb2engine.py --pb_path model.pb --engine_path engine.plan --output_node InceptionResnetV2/Logits/Predictions  --input_node input --compare_images_dir /data/
```
`compare_pb2engine.py` will generate benchmark report in your current dir which named report_<pb>_<engine.txt>.The benchmark report include fps , mse , diffrent of prediction .


* Convert slim model to pb
```bash
python3 slim_ckp2pb.py  \
--ckpt_path   <ckpt_path>  \
--pb_path     <pb_path>  \
--output_node <name of output node>  \
--model_name  <name of network>  \
--num_classes <number of classes>
```
example:
```bash
python3 slim_ckpt2pb.py  --ckpt_path  my-models  --pb_path model.pb  --output_node InceptionResnetV2/Logits/Predictions  --model_name  inception_resnet_v2  --num_classes 1000
```
* Convert tf experiment model to pb
```bash
python3 tfexp_ckp2pb.py  \
--ckpt_path   <ckpt_path>  \
--pb_path     <pb_path>  \
--output_node <name of output node>  \
--network  <path to network>  \
--image_size <size of network>
```

# Environment
Provide the docker file to create same environment
* Download TensorRT Deb into `env` and **replace file name in `env/Dockerfile` line 14 , 16**

* Build Docker Image
```bash
nvidia-docker build -t liu/tensorrt:4.0 env
```
* RUN docker
```bash
nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /sharedfolder:/root/sharedfolder liu/tensorrt:4.0 bash
```

