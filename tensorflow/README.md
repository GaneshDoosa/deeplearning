# Tensorflow installation using anaconda
```bash
conda create -n tensorflow2
conda activate tensorflow2
conda install tensorflow-gpu
pip install jupyter
pip install matplotlib
```

## Diagnosing Problems:
1) InternalError:  Blas GEMM launch failed : a.shape=(32, 784), b.shape=(784, 128)
Solution: 
a) Make sure you have no other processes using the GPU running. Run nvidia-smi to check this.
