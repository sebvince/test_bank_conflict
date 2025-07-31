FILENAME=kernel_f32_async_ref.mlir #Reference version
FILENAME=kernel_f32_async.mlir     #Swizzling

iree-compile $FILENAME \
    --iree-hip-target=gfx950 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-hal-dump-executable-files-to . \
    -o tmp/kernel.vmfb 
    #--mlir-print-ir-after-all \
    #--mlir-disable-threading 2> out.mlir

PROFILER="rocprofv3 --att --att-perfcounter-ctrl 3 --att-perfcounters "SQ_LDS_BANK_CONFLICT" --"
PROFILER="rocprofv3 --pmc LDSBankConflict --output-format csv --stats --output-file res.csv -- "
PROFILER=
$PROFILER iree-benchmark-module \
  --module=tmp/kernel.vmfb \
  --device=hip \
  --function=isolated_benchmark \
  --input=16384x8192xi8=@a.bin \
  --input=16384x512xi8=@a_scale.bin \
  --input=16384x8192xi8=@b.bin \
  --input=16384x512xi8=@b_scale.bin \
  --input=16384x16384xf32=@out.bin

