## Build image

```
docker build -t 'dist-training' .
```

## Run container

```
docker run -it --gpus=all --rm --ipc=host --net=host dist-training bash
```

### Run single-GPU FP32 training:
```
python main.py --batch-size 96
```

### Run single-GPU FP16 training:
```
python main.py --batch-size 96 --use-mixed-precision O1
```

### Run single-GPU FP32 distributed training:
```
Master node:

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="10.2.2.19" --master_port=8884 main.py --batch-size 96
```

```
Worker node:

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.2.2.19" --master_port=8884 main.py --batch-size 96
```

### Run single-GPU FP16 distributed training:

```
Master node:

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="10.2.2.19" --master_port=8884 main.py --batch-size 96 --use-mixed-precision O1
```

```
Worker node:

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.2.2.19" --master_port=8884 main.py --batch-size 96 --use-mixed-precision O1
```
