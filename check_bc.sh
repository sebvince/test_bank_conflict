#PROFILER="rocprofv3 --kernel-include-regex=gemm --att --att-perfcounter-ctrl 3 --att-perfcounters "SQ_LDS_BANK_CONFLICT" --"
PROFILER="rocprofv3 --kernel-include-regex=gemm --pmc LDSBankConflict --output-format csv --stats --output-file res.csv -- "
# PROFILER=
rm res.csv_counter_collection.csv 
$PROFILER python3 test_gemm.py
python3 rocprof_decode.py res.csv_counter_collection.csv 



