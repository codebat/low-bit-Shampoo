from models import vgg, resnet, densenet, vit, swin

# shampoo1: implements naive 4-bit Shampoo (without SVD)
from optimizers.shampoo1 import ShampooSGD as Shampoo1SGD, ShampooAdamW as Shampoo1AdamW
# shampoo2: implements our 4-bit Shampoo (with SVD)
from optimizers.shampoo2 import ShampooSGD as Shampoo2SGD, ShampooAdamW as Shampoo2AdamW


#model = vgg.vgg19(num_classes=100)
#model = resnet.resnet34(num_classes=100)
#model = vit.vit_small(img_size=32, num_classes=100, drop_path_rate=0.1)
model = swin.swin_tiny(img_size=32, num_classes=100, drop_path_rate=0.1)
#model = resnet.resnet50(num_classes=1000)
#model = vit.vit_base_32_224(num_classes=1000, drop_path_rate=0.1)
model = model.cuda()

print("model:", sum(p.numel() for p in model.parameters()))


# start_prec_step: the step starting preconditioning
# stat_compute_steps: interval of updating preconditioners (T_1)
# prec_compute_steps: interval of updating inverse roots of preconditioners (T_2)
# stat_decay: exponential decay rate for preconditioners (beta)
# matrix_eps: dampening term (epsilon)
# prec_maxorder: maximum order for preconditioners
# prec_bits: bitwidth of a preconditioner
# min_lowbit_size: minimum tensor size required for quantization
# quan_blocksize: block size for block-wise quantization 
# rect_t1: number of iterations t1 for rectification
# rect_t2: number of iterations t2 for rectification
# inv_root_mode: "0" for SVD computation and "1" for Schur-Newton iteration

#optimizer = Shampoo2SGD(model.parameters(),
#                        lr=0.1,
#                        momentum=0.9,
#                        weight_decay=0.0005,
#                        nesterov=False,
#                        start_prec_step=1,
#                        stat_compute_steps=100,
#                        prec_compute_steps=500,
#                        stat_decay=0.95,
#                        matrix_eps=1e-6,
#                        prec_maxorder=1200,
#                        prec_bits=4,
#                        min_lowbit_size=4096,
#                        quan_blocksize=64)

optimizer = Shampoo2AdamW(model.parameters(),
                          lr=0.001,
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=0.05,
                          start_prec_step=1,
                          stat_compute_steps=100,
                          prec_compute_steps=500,
                          stat_decay=0.95,
                          matrix_eps=1e-6,
                          prec_maxorder=1200,
                          prec_bits=4,
                          min_lowbit_size=4096,
                          quan_blocksize=64,
                          rect_t1=1,
                          rect_t2=4,
                          inv_root_mode=0)

print(optimizer)
