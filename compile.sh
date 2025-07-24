iree-compile kernel.mlir \
    --iree-hip-target=gfx950 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-codegen-enable-default-tuning-specs=true \
    -o tmp/kernel.vmfb

iree-benchmark-module \
  --module=tmp/kernel.vmfb \
  --device=hip \
  --function=matmul \
  --input=16384x8192xi8=@a.bin \
  --input=16384x512xi8=@a_scale.bin \
  --input=16384x8192xi8=@b.bin \
  --input=16384x512xi8=@b_scale.bin \
  --input=16384x16384xbf16=@out.bin
