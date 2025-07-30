import torch
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_randint,
    device_tensor,
    device_zeros,
)
import numpy as np
import iree.runtime as rt
from wave_lang.runtime.launch import Launchable
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.compile_utils import compile_to_vmfb


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = device_tensor(mxfp4_list, dtype=torch.float32)
    return mxfp4_in_f32[x.long()]

def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32

def generate_gemm_afp4wfp4_inputs(shape):
    M, N, K = shape
    torch.manual_seed(5)
    # 34 is two packed e2m1 values 0010 which is 1.0.
    x_low = device_randint(0, 16, (M, K // 2), dtype=torch.uint8)
    x_high = device_randint(0, 16, (M, K // 2), dtype=torch.uint8)
    x = x_low | x_high << 4
    w_low = device_randint(0, 16, (N, K // 2), dtype=torch.uint8)
    w_high = device_randint(0, 16, (N, K // 2), dtype=torch.uint8)
    w = w_low | w_high << 4
    w = w.T
    # Scale of 1.0 in e8m0, bias 127.
    x_scales = device_randint(124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8)
    w_scales = device_randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8)
    x_scales = x_scales.T.contiguous()
    w_scales = w_scales.T.contiguous()

    return x, w, x_scales, w_scales

def torchScaledGemmMXFP4(x, w, x_scales, w_scales):
    # First convert the x and w inputs to f32.
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w.T)
    w_f32 = w_f32.T
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32.T
    return torch.mm(x_f32, w_f32)

SCALE_GROUP_SIZE = 32
shape = (16384,16384,16384)
x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)

out = device_zeros(x.shape[0], w.shape[1], dtype=torch.float32)

w_t = w.T.contiguous()

torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)
# print(torch_out)

with open("kernel_f32.mlir", "rb") as f:
    asm = f.read()
    options = WaveCompileOptions(
        backend="rocm",
        target="gfx950",
        dump_intermediates = True,
    )
    try:
        vmfb = compile_to_vmfb(asm, options)
    except:
        print("Compile Error !!!")
     
    def loader(device):
        vm_instance = device.vm_instance
        return rt.VmModule.copy_buffer(vm_instance, vmfb)

    launchable = Launchable.from_vm_module(loader, entry_point="isolated_benchmark")
    kernel_inputs = [x, x_scales ,w_t, w_scales,out]
    kernel_outputs = [out]
    res = launchable(*kernel_inputs, outputs=kernel_outputs)
    print(res)
    torch.testing.assert_close(torch_out, res)

