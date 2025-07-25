iree-compile kernel512.mlir \
    --iree-hip-target=gfx950 \
    --iree-hal-target-backends=rocm \
    --mlir-disable-threading \
    --iree-hal-dump-executable-files-to . \
    --iree-codegen-enable-default-tuning-specs=true \
    -o tmp/kernel.vmfb

PROFILER="rocprofv3 --att --att-perfcounter-ctrl 3 --att-perfcounters "SQ_LDS_BANK_CONFLICT" --"
PROFILER="rocprofv3 --pmc LDSBankConflict --output-format csv --stats --output-file res.csv -- "
# PROFILER=
$PROFILER iree-benchmark-module \
  --module=tmp/kernel.vmfb \
  --device=hip \
  --function=matmul \
  --input=16384x8192xi8=@a.bin \
  --input=16384x512xi8=@a_scale.bin \
  --input=16384x8192xi8=@b.bin \
  --input=16384x512xi8=@b_scale.bin \
  --input=16384x16384xbf16=@out.bin


cat res.csv_counter_collection.csv 