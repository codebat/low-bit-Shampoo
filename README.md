# 4-bit Shampoo

## Preparation

1. Requirements:

   ​	For running python code: pytorch+torchvision+timm+numpy.

   ​	For compiling C code: gcc+nvcc+cmake (Linux), msvc+nvcc+cmake (Windows).

2. Compilation of the dynamic library used for quantization:

   ​	See "README.md" in path "./cudaC_python for the compilation process.
   
   ​	After compilation, move the dynamic library to path "./qtensor".



## Usage

File "main_demo.py" shows the basic usage of our optimizer codes, and gives the models used in our paper.

File "shampoo1.py" in path "./optimizers" implements naive 4-bit Shampoo.

File "shampoo2.py" in path "./optimizers" implements our 4-bit Shampoo.



# Citation

```latex
@article{Wang_NeurIPS_2024,
    author = {Sike Wang and Pan Zhou and Jia Li and Hua Huang},
    title = {4-bit Shampoo for Memory-Efficient Network Training},
    journal = {Advances in Neural Information Processing Systems},
    year = {2024},
    url = {https://arxiv.org/abs/2405.18144},
}
```

