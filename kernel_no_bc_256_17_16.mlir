#map = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 256 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16) floordiv 256) * 256)>
#map1 = affine_map<()[s0, s1] -> ((s1 * 16 + s0 floordiv 16) mod 256)>

// #map2 = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256)>
#map2 = affine_map<()[s0] -> (s0 - (s0 floordiv 16) * 16)>

#map3 = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 256 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 32) floordiv 256) * 256 + 32)>
#map4 = affine_map<()[s0, s1] -> (s1 * 16 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 32) floordiv 256) * 256 + 32)>
#map5 = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 256 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 64) floordiv 256) * 256 + 64)>
#map6 = affine_map<()[s0, s1] -> (s1 * 16 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 64) floordiv 256) * 256 + 64)>
#map7 = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 256 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 96) floordiv 256) * 256 + 96)>
#map8 = affine_map<()[s0, s1] -> (s1 * 16 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 96) floordiv 256) * 256 + 96)>
#map9 = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 256 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 128) floordiv 256) * 256 + 128)>
#map10 = affine_map<()[s0, s1] -> (s1 * 16 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 128) floordiv 256) * 256 + 128)>
#map11 = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 256 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 160) floordiv 256) * 256 + 160)>
#map12 = affine_map<()[s0, s1] -> (s1 * 16 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 160) floordiv 256) * 256 + 160)>
#map13 = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 256 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 192) floordiv 256) * 256 + 192)>
#map14 = affine_map<()[s0, s1] -> (s1 * 16 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 192) floordiv 256) * 256 + 192)>
#map15 = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 256 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 224) floordiv 256) * 256 + 224)>
#map16 = affine_map<()[s0, s1] -> (s1 * 16 + s0 floordiv 16 - ((s1 * 16 + s0 floordiv 16 + 224) floordiv 256) * 256 + 224)>

#map18 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
#map19 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>
#map20 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 128)>
#map21 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 192)>

#map17 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
#map22 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
#map23 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
#map24 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>

// #map17 = affine_map<()[s0] -> ((s0 floordiv 64) * 64)>
// #map22 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + 16)>
// #map23 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + 32)>
// #map24 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + 48)>

#map25 = affine_map<()[s0, s1] -> (s0 * 256 + s1 * 16 - (s1 floordiv 16) * 256)>
#map26 = affine_map<()[s0] -> (s0 * 256)>
#map27 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
#map28 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map32 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map33 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map34 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map35 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map36 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map37 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map38 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
#map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
#map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
#map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
#map46 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
#map47 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
#map48 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
#map49 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
#map50 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm_afp4_wfp4_wave {
    stream.executable.export public @gemm_afp4_wfp4_wave workgroups() -> (index, index, index) {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      stream.return %c64, %c64, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm_afp4_wfp4_wave(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
        %cst = arith.constant 1.000000e+00 : f8E8M0FNU
        %c-8192_i14 = arith.constant -8192 : i14
        %c2147483645_i32 = arith.constant 2147483645 : i32
        %c1073741822 = arith.constant 1073741822 : index
        %c16384 = arith.constant 16384 : index
        %cst_0 = arith.constant dense<1.000000e+00> : vector<32xf4E2M1FN>
        %c2147483646_i32 = arith.constant 2147483646 : i32
        %c2147483646 = arith.constant 2147483646 : index
        %c8192 = arith.constant 8192 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %cst_1 = arith.constant dense<0.000000e+00> : vector<4xf32>
        %block_id_x = gpu.block_id  x upper_bound 64
        %block_id_y = gpu.block_id  y upper_bound 64
        %thread_id_x = gpu.thread_id  x upper_bound 256
        %thread_id_y = gpu.thread_id  y upper_bound 2
        %alloc = memref.alloc() : memref<256x17x16xi8, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x8192xi8, strided<[8192, 1], offset: ?>>
        %1 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
        %2 = arith.muli %1, %c8192 overflow<nsw> : index
        %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x8192xi8, strided<[8192, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %3 = amdgpu.fat_raw_buffer_cast %reinterpret_cast validBytes(%c2147483646_i32) cacheSwizzleStride(%c-8192_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %4 = affine.apply #map1()[%thread_id_x, %thread_id_y]
        %5 = affine.apply #map2()[%thread_id_x]
        %6 = affine.apply #map3()[%thread_id_x, %thread_id_y, %block_id_x]
        %7 = arith.muli %6, %c8192 overflow<nsw> : index
        %8 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %9 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
        %10 = arith.muli %9, %c8192 overflow<nsw> : index
        %11 = affine.apply #map6()[%thread_id_x, %thread_id_y]
        %12 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
        %13 = arith.muli %12, %c8192 overflow<nsw> : index
        %14 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        %15 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x]
        %16 = arith.muli %15, %c8192 overflow<nsw> : index
        %17 = affine.apply #map10()[%thread_id_x, %thread_id_y]
        %18 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_x]
        %19 = arith.muli %18, %c8192 overflow<nsw> : index
        %20 = affine.apply #map12()[%thread_id_x, %thread_id_y]
        %21 = affine.apply #map13()[%thread_id_x, %thread_id_y, %block_id_x]
        %22 = arith.muli %21, %c8192 overflow<nsw> : index
        %23 = affine.apply #map14()[%thread_id_x, %thread_id_y]
        %24 = affine.apply #map15()[%thread_id_x, %thread_id_y, %block_id_x]
        %25 = arith.muli %24, %c8192 overflow<nsw> : index
        %26 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %27 = affine.apply #map17()[%thread_id_x]
        %28 = affine.apply #map18()[%thread_id_x]
        %29 = affine.apply #map19()[%thread_id_x]
        %30 = affine.apply #map20()[%thread_id_x]
        %31 = affine.apply #map21()[%thread_id_x]
        %32 = affine.apply #map22()[%thread_id_x]
        %33 = affine.apply #map23()[%thread_id_x]
        %34 = affine.apply #map24()[%thread_id_x]
        %35:32 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %cst_1, %arg7 = %cst_1, %arg8 = %cst_1, %arg9 = %cst_1, %arg10 = %cst_1, %arg11 = %cst_1, %arg12 = %cst_1, %arg13 = %cst_1, %arg14 = %cst_1, %arg15 = %cst_1, %arg16 = %cst_1, %arg17 = %cst_1, %arg18 = %cst_1, %arg19 = %cst_1, %arg20 = %cst_1, %arg21 = %cst_1, %arg22 = %cst_1, %arg23 = %cst_1, %arg24 = %cst_1, %arg25 = %cst_1, %arg26 = %cst_1, %arg27 = %cst_1, %arg28 = %cst_1, %arg29 = %cst_1, %arg30 = %cst_1, %arg31 = %cst_1, %arg32 = %cst_1, %arg33 = %cst_1, %arg34 = %cst_1, %arg35 = %cst_1, %arg36 = %cst_1, %arg37 = %cst_1) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %370 = affine.apply #map25()[%arg5, %thread_id_x]
          %371 = arith.addi %2, %370 overflow<nsw> : index
          %372 = vector.load %3[%371] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          amdgpu.lds_barrier
          vector.store %372, %alloc[%4, %5,%c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %373 = arith.addi %7, %370 overflow<nsw> : index
          %374 = vector.load %3[%373] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %374, %alloc[%8, %5,%c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %375 = arith.addi %10, %370 overflow<nsw> : index
          %376 = vector.load %3[%375] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %376, %alloc[%11, %5,%c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %377 = arith.addi %13, %370 overflow<nsw> : index
          %378 = vector.load %3[%377] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %378, %alloc[%14, %5,%c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %379 = arith.addi %16, %370 overflow<nsw> : index
          %380 = vector.load %3[%379] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %380, %alloc[%17, %5,%c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %381 = arith.addi %19, %370 overflow<nsw> : index
          %382 = vector.load %3[%381] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %382, %alloc[%20, %5,%c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %383 = arith.addi %22, %370 overflow<nsw> : index
          %384 = vector.load %3[%383] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %384, %alloc[%23, %5,%c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %385 = arith.addi %25, %370 overflow<nsw> : index
          %386 = vector.load %3[%385] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %386, %alloc[%26, %5,%c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          amdgpu.lds_barrier
          %387 = vector.load %alloc[%27, %28, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %388 = vector.load %alloc[%27, %29, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %389 = vector.load %alloc[%27, %30, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %390 = vector.load %alloc[%27, %31, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %391 = vector.load %alloc[%32, %28, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %392 = vector.load %alloc[%32, %29, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %393 = vector.load %alloc[%32, %30, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %394 = vector.load %alloc[%32, %31, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %395 = vector.load %alloc[%33, %28, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %396 = vector.load %alloc[%33, %29, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %397 = vector.load %alloc[%33, %30, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %398 = vector.load %alloc[%33, %31, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %399 = vector.load %alloc[%34, %28, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %400 = vector.load %alloc[%34, %29, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %401 = vector.load %alloc[%34, %30, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %402 = vector.load %alloc[%34, %31, %c0] : memref<256x17x16xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %403 = vector.bitcast %387 : vector<16xi8> to vector<32xf4E2M1FN>
          %404 = vector.bitcast %388 : vector<16xi8> to vector<32xf4E2M1FN>
          %405 = vector.bitcast %389 : vector<16xi8> to vector<32xf4E2M1FN>
          %406 = vector.bitcast %390 : vector<16xi8> to vector<32xf4E2M1FN>
          %407 = vector.bitcast %391 : vector<16xi8> to vector<32xf4E2M1FN>
          %408 = vector.bitcast %392 : vector<16xi8> to vector<32xf4E2M1FN>
          %409 = vector.bitcast %393 : vector<16xi8> to vector<32xf4E2M1FN>
          %410 = vector.bitcast %394 : vector<16xi8> to vector<32xf4E2M1FN>
          %411 = vector.bitcast %395 : vector<16xi8> to vector<32xf4E2M1FN>
          %412 = vector.bitcast %396 : vector<16xi8> to vector<32xf4E2M1FN>
          %413 = vector.bitcast %397 : vector<16xi8> to vector<32xf4E2M1FN>
          %414 = vector.bitcast %398 : vector<16xi8> to vector<32xf4E2M1FN>
          %415 = vector.bitcast %399 : vector<16xi8> to vector<32xf4E2M1FN>
          %416 = vector.bitcast %400 : vector<16xi8> to vector<32xf4E2M1FN>
          %417 = vector.bitcast %401 : vector<16xi8> to vector<32xf4E2M1FN>
          %418 = vector.bitcast %402 : vector<16xi8> to vector<32xf4E2M1FN>
          %419 = amdgpu.scaled_mfma(%cst[0] * %403) * (%cst[0] * %cst_0) + %arg6 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %420 = amdgpu.scaled_mfma(%cst[0] * %404) * (%cst[0] * %cst_0) + %419 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %421 = amdgpu.scaled_mfma(%cst[0] * %405) * (%cst[0] * %cst_0) + %420 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %422 = amdgpu.scaled_mfma(%cst[0] * %406) * (%cst[0] * %cst_0) + %421 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %423 = amdgpu.scaled_mfma(%cst[0] * %403) * (%cst[0] * %cst_0) + %arg7 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %424 = amdgpu.scaled_mfma(%cst[0] * %404) * (%cst[0] * %cst_0) + %423 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %425 = amdgpu.scaled_mfma(%cst[0] * %405) * (%cst[0] * %cst_0) + %424 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %426 = amdgpu.scaled_mfma(%cst[0] * %406) * (%cst[0] * %cst_0) + %425 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %427 = amdgpu.scaled_mfma(%cst[0] * %403) * (%cst[0] * %cst_0) + %arg8 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %428 = amdgpu.scaled_mfma(%cst[0] * %404) * (%cst[0] * %cst_0) + %427 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %429 = amdgpu.scaled_mfma(%cst[0] * %405) * (%cst[0] * %cst_0) + %428 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %430 = amdgpu.scaled_mfma(%cst[0] * %406) * (%cst[0] * %cst_0) + %429 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %431 = amdgpu.scaled_mfma(%cst[0] * %403) * (%cst[0] * %cst_0) + %arg9 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %432 = amdgpu.scaled_mfma(%cst[0] * %404) * (%cst[0] * %cst_0) + %431 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %433 = amdgpu.scaled_mfma(%cst[0] * %405) * (%cst[0] * %cst_0) + %432 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %434 = amdgpu.scaled_mfma(%cst[0] * %406) * (%cst[0] * %cst_0) + %433 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %435 = amdgpu.scaled_mfma(%cst[0] * %403) * (%cst[0] * %cst_0) + %arg10 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %436 = amdgpu.scaled_mfma(%cst[0] * %404) * (%cst[0] * %cst_0) + %435 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %437 = amdgpu.scaled_mfma(%cst[0] * %405) * (%cst[0] * %cst_0) + %436 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %438 = amdgpu.scaled_mfma(%cst[0] * %406) * (%cst[0] * %cst_0) + %437 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %439 = amdgpu.scaled_mfma(%cst[0] * %403) * (%cst[0] * %cst_0) + %arg11 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %440 = amdgpu.scaled_mfma(%cst[0] * %404) * (%cst[0] * %cst_0) + %439 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %441 = amdgpu.scaled_mfma(%cst[0] * %405) * (%cst[0] * %cst_0) + %440 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %442 = amdgpu.scaled_mfma(%cst[0] * %406) * (%cst[0] * %cst_0) + %441 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %443 = amdgpu.scaled_mfma(%cst[0] * %403) * (%cst[0] * %cst_0) + %arg12 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %444 = amdgpu.scaled_mfma(%cst[0] * %404) * (%cst[0] * %cst_0) + %443 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %445 = amdgpu.scaled_mfma(%cst[0] * %405) * (%cst[0] * %cst_0) + %444 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %446 = amdgpu.scaled_mfma(%cst[0] * %406) * (%cst[0] * %cst_0) + %445 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %447 = amdgpu.scaled_mfma(%cst[0] * %403) * (%cst[0] * %cst_0) + %arg13 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %448 = amdgpu.scaled_mfma(%cst[0] * %404) * (%cst[0] * %cst_0) + %447 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %449 = amdgpu.scaled_mfma(%cst[0] * %405) * (%cst[0] * %cst_0) + %448 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %450 = amdgpu.scaled_mfma(%cst[0] * %406) * (%cst[0] * %cst_0) + %449 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %451 = amdgpu.scaled_mfma(%cst[0] * %407) * (%cst[0] * %cst_0) + %arg14 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %452 = amdgpu.scaled_mfma(%cst[0] * %408) * (%cst[0] * %cst_0) + %451 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %453 = amdgpu.scaled_mfma(%cst[0] * %409) * (%cst[0] * %cst_0) + %452 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %454 = amdgpu.scaled_mfma(%cst[0] * %410) * (%cst[0] * %cst_0) + %453 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %455 = amdgpu.scaled_mfma(%cst[0] * %407) * (%cst[0] * %cst_0) + %arg15 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %456 = amdgpu.scaled_mfma(%cst[0] * %408) * (%cst[0] * %cst_0) + %455 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %457 = amdgpu.scaled_mfma(%cst[0] * %409) * (%cst[0] * %cst_0) + %456 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %458 = amdgpu.scaled_mfma(%cst[0] * %410) * (%cst[0] * %cst_0) + %457 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %459 = amdgpu.scaled_mfma(%cst[0] * %407) * (%cst[0] * %cst_0) + %arg16 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %460 = amdgpu.scaled_mfma(%cst[0] * %408) * (%cst[0] * %cst_0) + %459 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %461 = amdgpu.scaled_mfma(%cst[0] * %409) * (%cst[0] * %cst_0) + %460 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %462 = amdgpu.scaled_mfma(%cst[0] * %410) * (%cst[0] * %cst_0) + %461 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %463 = amdgpu.scaled_mfma(%cst[0] * %407) * (%cst[0] * %cst_0) + %arg17 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %464 = amdgpu.scaled_mfma(%cst[0] * %408) * (%cst[0] * %cst_0) + %463 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %465 = amdgpu.scaled_mfma(%cst[0] * %409) * (%cst[0] * %cst_0) + %464 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %466 = amdgpu.scaled_mfma(%cst[0] * %410) * (%cst[0] * %cst_0) + %465 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %467 = amdgpu.scaled_mfma(%cst[0] * %407) * (%cst[0] * %cst_0) + %arg18 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %468 = amdgpu.scaled_mfma(%cst[0] * %408) * (%cst[0] * %cst_0) + %467 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %469 = amdgpu.scaled_mfma(%cst[0] * %409) * (%cst[0] * %cst_0) + %468 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %470 = amdgpu.scaled_mfma(%cst[0] * %410) * (%cst[0] * %cst_0) + %469 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %471 = amdgpu.scaled_mfma(%cst[0] * %407) * (%cst[0] * %cst_0) + %arg19 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %472 = amdgpu.scaled_mfma(%cst[0] * %408) * (%cst[0] * %cst_0) + %471 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %473 = amdgpu.scaled_mfma(%cst[0] * %409) * (%cst[0] * %cst_0) + %472 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %474 = amdgpu.scaled_mfma(%cst[0] * %410) * (%cst[0] * %cst_0) + %473 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %475 = amdgpu.scaled_mfma(%cst[0] * %407) * (%cst[0] * %cst_0) + %arg20 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %476 = amdgpu.scaled_mfma(%cst[0] * %408) * (%cst[0] * %cst_0) + %475 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %477 = amdgpu.scaled_mfma(%cst[0] * %409) * (%cst[0] * %cst_0) + %476 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %478 = amdgpu.scaled_mfma(%cst[0] * %410) * (%cst[0] * %cst_0) + %477 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %479 = amdgpu.scaled_mfma(%cst[0] * %407) * (%cst[0] * %cst_0) + %arg21 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %480 = amdgpu.scaled_mfma(%cst[0] * %408) * (%cst[0] * %cst_0) + %479 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %481 = amdgpu.scaled_mfma(%cst[0] * %409) * (%cst[0] * %cst_0) + %480 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %482 = amdgpu.scaled_mfma(%cst[0] * %410) * (%cst[0] * %cst_0) + %481 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %483 = amdgpu.scaled_mfma(%cst[0] * %411) * (%cst[0] * %cst_0) + %arg22 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %484 = amdgpu.scaled_mfma(%cst[0] * %412) * (%cst[0] * %cst_0) + %483 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %485 = amdgpu.scaled_mfma(%cst[0] * %413) * (%cst[0] * %cst_0) + %484 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %486 = amdgpu.scaled_mfma(%cst[0] * %414) * (%cst[0] * %cst_0) + %485 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %487 = amdgpu.scaled_mfma(%cst[0] * %411) * (%cst[0] * %cst_0) + %arg23 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %488 = amdgpu.scaled_mfma(%cst[0] * %412) * (%cst[0] * %cst_0) + %487 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %489 = amdgpu.scaled_mfma(%cst[0] * %413) * (%cst[0] * %cst_0) + %488 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %490 = amdgpu.scaled_mfma(%cst[0] * %414) * (%cst[0] * %cst_0) + %489 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %491 = amdgpu.scaled_mfma(%cst[0] * %411) * (%cst[0] * %cst_0) + %arg24 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %492 = amdgpu.scaled_mfma(%cst[0] * %412) * (%cst[0] * %cst_0) + %491 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %493 = amdgpu.scaled_mfma(%cst[0] * %413) * (%cst[0] * %cst_0) + %492 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %494 = amdgpu.scaled_mfma(%cst[0] * %414) * (%cst[0] * %cst_0) + %493 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %495 = amdgpu.scaled_mfma(%cst[0] * %411) * (%cst[0] * %cst_0) + %arg25 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %496 = amdgpu.scaled_mfma(%cst[0] * %412) * (%cst[0] * %cst_0) + %495 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %497 = amdgpu.scaled_mfma(%cst[0] * %413) * (%cst[0] * %cst_0) + %496 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %498 = amdgpu.scaled_mfma(%cst[0] * %414) * (%cst[0] * %cst_0) + %497 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %499 = amdgpu.scaled_mfma(%cst[0] * %411) * (%cst[0] * %cst_0) + %arg26 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %500 = amdgpu.scaled_mfma(%cst[0] * %412) * (%cst[0] * %cst_0) + %499 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %501 = amdgpu.scaled_mfma(%cst[0] * %413) * (%cst[0] * %cst_0) + %500 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %502 = amdgpu.scaled_mfma(%cst[0] * %414) * (%cst[0] * %cst_0) + %501 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %503 = amdgpu.scaled_mfma(%cst[0] * %411) * (%cst[0] * %cst_0) + %arg27 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %504 = amdgpu.scaled_mfma(%cst[0] * %412) * (%cst[0] * %cst_0) + %503 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %505 = amdgpu.scaled_mfma(%cst[0] * %413) * (%cst[0] * %cst_0) + %504 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %506 = amdgpu.scaled_mfma(%cst[0] * %414) * (%cst[0] * %cst_0) + %505 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %507 = amdgpu.scaled_mfma(%cst[0] * %411) * (%cst[0] * %cst_0) + %arg28 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %508 = amdgpu.scaled_mfma(%cst[0] * %412) * (%cst[0] * %cst_0) + %507 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %509 = amdgpu.scaled_mfma(%cst[0] * %413) * (%cst[0] * %cst_0) + %508 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %510 = amdgpu.scaled_mfma(%cst[0] * %414) * (%cst[0] * %cst_0) + %509 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %511 = amdgpu.scaled_mfma(%cst[0] * %411) * (%cst[0] * %cst_0) + %arg29 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %512 = amdgpu.scaled_mfma(%cst[0] * %412) * (%cst[0] * %cst_0) + %511 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %513 = amdgpu.scaled_mfma(%cst[0] * %413) * (%cst[0] * %cst_0) + %512 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %514 = amdgpu.scaled_mfma(%cst[0] * %414) * (%cst[0] * %cst_0) + %513 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %515 = amdgpu.scaled_mfma(%cst[0] * %415) * (%cst[0] * %cst_0) + %arg30 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %516 = amdgpu.scaled_mfma(%cst[0] * %416) * (%cst[0] * %cst_0) + %515 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %517 = amdgpu.scaled_mfma(%cst[0] * %417) * (%cst[0] * %cst_0) + %516 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %518 = amdgpu.scaled_mfma(%cst[0] * %418) * (%cst[0] * %cst_0) + %517 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %519 = amdgpu.scaled_mfma(%cst[0] * %415) * (%cst[0] * %cst_0) + %arg31 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %520 = amdgpu.scaled_mfma(%cst[0] * %416) * (%cst[0] * %cst_0) + %519 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %521 = amdgpu.scaled_mfma(%cst[0] * %417) * (%cst[0] * %cst_0) + %520 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %522 = amdgpu.scaled_mfma(%cst[0] * %418) * (%cst[0] * %cst_0) + %521 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %523 = amdgpu.scaled_mfma(%cst[0] * %415) * (%cst[0] * %cst_0) + %arg32 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %524 = amdgpu.scaled_mfma(%cst[0] * %416) * (%cst[0] * %cst_0) + %523 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %525 = amdgpu.scaled_mfma(%cst[0] * %417) * (%cst[0] * %cst_0) + %524 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %526 = amdgpu.scaled_mfma(%cst[0] * %418) * (%cst[0] * %cst_0) + %525 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %527 = amdgpu.scaled_mfma(%cst[0] * %415) * (%cst[0] * %cst_0) + %arg33 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %528 = amdgpu.scaled_mfma(%cst[0] * %416) * (%cst[0] * %cst_0) + %527 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %529 = amdgpu.scaled_mfma(%cst[0] * %417) * (%cst[0] * %cst_0) + %528 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %530 = amdgpu.scaled_mfma(%cst[0] * %418) * (%cst[0] * %cst_0) + %529 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %531 = amdgpu.scaled_mfma(%cst[0] * %415) * (%cst[0] * %cst_0) + %arg34 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %532 = amdgpu.scaled_mfma(%cst[0] * %416) * (%cst[0] * %cst_0) + %531 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %533 = amdgpu.scaled_mfma(%cst[0] * %417) * (%cst[0] * %cst_0) + %532 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %534 = amdgpu.scaled_mfma(%cst[0] * %418) * (%cst[0] * %cst_0) + %533 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %535 = amdgpu.scaled_mfma(%cst[0] * %415) * (%cst[0] * %cst_0) + %arg35 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %536 = amdgpu.scaled_mfma(%cst[0] * %416) * (%cst[0] * %cst_0) + %535 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %537 = amdgpu.scaled_mfma(%cst[0] * %417) * (%cst[0] * %cst_0) + %536 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %538 = amdgpu.scaled_mfma(%cst[0] * %418) * (%cst[0] * %cst_0) + %537 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %539 = amdgpu.scaled_mfma(%cst[0] * %415) * (%cst[0] * %cst_0) + %arg36 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %540 = amdgpu.scaled_mfma(%cst[0] * %416) * (%cst[0] * %cst_0) + %539 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %541 = amdgpu.scaled_mfma(%cst[0] * %417) * (%cst[0] * %cst_0) + %540 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %542 = amdgpu.scaled_mfma(%cst[0] * %418) * (%cst[0] * %cst_0) + %541 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %543 = amdgpu.scaled_mfma(%cst[0] * %415) * (%cst[0] * %cst_0) + %arg37 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %544 = amdgpu.scaled_mfma(%cst[0] * %416) * (%cst[0] * %cst_0) + %543 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %545 = amdgpu.scaled_mfma(%cst[0] * %417) * (%cst[0] * %cst_0) + %544 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %546 = amdgpu.scaled_mfma(%cst[0] * %418) * (%cst[0] * %cst_0) + %545 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          scf.yield %422, %426, %430, %434, %438, %442, %446, %450, %454, %458, %462, %466, %470, %474, %478, %482, %486, %490, %494, %498, %502, %506, %510, %514, %518, %522, %526, %530, %534, %538, %542, %546 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %36 = arith.truncf %35#0 : vector<4xf32> to vector<4xbf16>
        %37 = arith.truncf %35#1 : vector<4xf32> to vector<4xbf16>
        %38 = arith.truncf %35#2 : vector<4xf32> to vector<4xbf16>
        %39 = arith.truncf %35#3 : vector<4xf32> to vector<4xbf16>
        %40 = arith.truncf %35#4 : vector<4xf32> to vector<4xbf16>
        %41 = arith.truncf %35#5 : vector<4xf32> to vector<4xbf16>
        %42 = arith.truncf %35#6 : vector<4xf32> to vector<4xbf16>
        %43 = arith.truncf %35#7 : vector<4xf32> to vector<4xbf16>
        %44 = arith.truncf %35#8 : vector<4xf32> to vector<4xbf16>
        %45 = arith.truncf %35#9 : vector<4xf32> to vector<4xbf16>
        %46 = arith.truncf %35#10 : vector<4xf32> to vector<4xbf16>
        %47 = arith.truncf %35#11 : vector<4xf32> to vector<4xbf16>
        %48 = arith.truncf %35#12 : vector<4xf32> to vector<4xbf16>
        %49 = arith.truncf %35#13 : vector<4xf32> to vector<4xbf16>
        %50 = arith.truncf %35#14 : vector<4xf32> to vector<4xbf16>
        %51 = arith.truncf %35#15 : vector<4xf32> to vector<4xbf16>
        %52 = arith.truncf %35#16 : vector<4xf32> to vector<4xbf16>
        %53 = arith.truncf %35#17 : vector<4xf32> to vector<4xbf16>
        %54 = arith.truncf %35#18 : vector<4xf32> to vector<4xbf16>
        %55 = arith.truncf %35#19 : vector<4xf32> to vector<4xbf16>
        %56 = arith.truncf %35#20 : vector<4xf32> to vector<4xbf16>
        %57 = arith.truncf %35#21 : vector<4xf32> to vector<4xbf16>
        %58 = arith.truncf %35#22 : vector<4xf32> to vector<4xbf16>
        %59 = arith.truncf %35#23 : vector<4xf32> to vector<4xbf16>
        %60 = arith.truncf %35#24 : vector<4xf32> to vector<4xbf16>
        %61 = arith.truncf %35#25 : vector<4xf32> to vector<4xbf16>
        %62 = arith.truncf %35#26 : vector<4xf32> to vector<4xbf16>
        %63 = arith.truncf %35#27 : vector<4xf32> to vector<4xbf16>
        %64 = arith.truncf %35#28 : vector<4xf32> to vector<4xbf16>
        %65 = arith.truncf %35#29 : vector<4xf32> to vector<4xbf16>
        %66 = arith.truncf %35#30 : vector<4xf32> to vector<4xbf16>
        %67 = arith.truncf %35#31 : vector<4xf32> to vector<4xbf16>
        %68 = vector.extract_strided_slice %36 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %69 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<16384x16384xbf16, strided<[16384, 1], offset: ?>>
        %70 = affine.apply #map26()[%block_id_x]
        %71 = affine.apply #map26()[%block_id_y]
        %72 = affine.apply #map27()[%thread_id_x]
        %73 = affine.apply #map28()[%thread_id_x, %thread_id_y]
        %74 = arith.muli %70, %c16384 overflow<nsw> : index
        %75 = arith.muli %72, %c16384 overflow<nsw> : index
        %76 = arith.addi %74, %71 overflow<nsw> : index
        %77 = arith.addi %75, %73 overflow<nsw> : index
        %reinterpret_cast_2 = memref.reinterpret_cast %69 to offset: [%76], sizes: [%c1073741822], strides: [1] : memref<16384x16384xbf16, strided<[16384, 1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
        %78 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_2 validBytes(%c2147483645_i32) : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        vector.store %68, %78[%77] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %79 = vector.extract_strided_slice %36 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %80 = affine.apply #map29()[%thread_id_x]
        %81 = arith.muli %80, %c16384 overflow<nsw> : index
        %82 = arith.addi %81, %73 overflow<nsw> : index
        vector.store %79, %78[%82] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %83 = vector.extract_strided_slice %36 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %84 = affine.apply #map30()[%thread_id_x]
        %85 = arith.muli %84, %c16384 overflow<nsw> : index
        %86 = arith.addi %85, %73 overflow<nsw> : index
        vector.store %83, %78[%86] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %87 = vector.extract_strided_slice %36 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %88 = affine.apply #map31()[%thread_id_x]
        %89 = arith.muli %88, %c16384 overflow<nsw> : index
        %90 = arith.addi %89, %73 overflow<nsw> : index
        vector.store %87, %78[%90] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %91 = vector.extract_strided_slice %37 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %92 = affine.apply #map32()[%thread_id_x, %thread_id_y]
        %93 = arith.addi %75, %92 overflow<nsw> : index
        vector.store %91, %78[%93] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %94 = vector.extract_strided_slice %37 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %95 = arith.addi %81, %92 overflow<nsw> : index
        vector.store %94, %78[%95] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %96 = vector.extract_strided_slice %37 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %97 = arith.addi %85, %92 overflow<nsw> : index
        vector.store %96, %78[%97] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %98 = vector.extract_strided_slice %37 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %99 = arith.addi %89, %92 overflow<nsw> : index
        vector.store %98, %78[%99] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %100 = vector.extract_strided_slice %38 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %101 = affine.apply #map33()[%thread_id_x, %thread_id_y]
        %102 = arith.addi %75, %101 overflow<nsw> : index
        vector.store %100, %78[%102] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %103 = vector.extract_strided_slice %38 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %104 = arith.addi %81, %101 overflow<nsw> : index
        vector.store %103, %78[%104] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %105 = vector.extract_strided_slice %38 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %106 = arith.addi %85, %101 overflow<nsw> : index
        vector.store %105, %78[%106] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %107 = vector.extract_strided_slice %38 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %108 = arith.addi %89, %101 overflow<nsw> : index
        vector.store %107, %78[%108] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %109 = vector.extract_strided_slice %39 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %110 = affine.apply #map34()[%thread_id_x, %thread_id_y]
        %111 = arith.addi %75, %110 overflow<nsw> : index
        vector.store %109, %78[%111] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %112 = vector.extract_strided_slice %39 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %113 = arith.addi %81, %110 overflow<nsw> : index
        vector.store %112, %78[%113] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %114 = vector.extract_strided_slice %39 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %115 = arith.addi %85, %110 overflow<nsw> : index
        vector.store %114, %78[%115] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %116 = vector.extract_strided_slice %39 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %117 = arith.addi %89, %110 overflow<nsw> : index
        vector.store %116, %78[%117] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %118 = vector.extract_strided_slice %40 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %119 = affine.apply #map35()[%thread_id_x, %thread_id_y]
        %120 = arith.addi %75, %119 overflow<nsw> : index
        vector.store %118, %78[%120] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %121 = vector.extract_strided_slice %40 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %122 = arith.addi %81, %119 overflow<nsw> : index
        vector.store %121, %78[%122] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %123 = vector.extract_strided_slice %40 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %124 = arith.addi %85, %119 overflow<nsw> : index
        vector.store %123, %78[%124] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %125 = vector.extract_strided_slice %40 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %126 = arith.addi %89, %119 overflow<nsw> : index
        vector.store %125, %78[%126] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %127 = vector.extract_strided_slice %41 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %128 = affine.apply #map36()[%thread_id_x, %thread_id_y]
        %129 = arith.addi %75, %128 overflow<nsw> : index
        vector.store %127, %78[%129] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %130 = vector.extract_strided_slice %41 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %131 = arith.addi %81, %128 overflow<nsw> : index
        vector.store %130, %78[%131] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %132 = vector.extract_strided_slice %41 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %133 = arith.addi %85, %128 overflow<nsw> : index
        vector.store %132, %78[%133] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %134 = vector.extract_strided_slice %41 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %135 = arith.addi %89, %128 overflow<nsw> : index
        vector.store %134, %78[%135] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %136 = vector.extract_strided_slice %42 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %137 = affine.apply #map37()[%thread_id_x, %thread_id_y]
        %138 = arith.addi %75, %137 overflow<nsw> : index
        vector.store %136, %78[%138] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %139 = vector.extract_strided_slice %42 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %140 = arith.addi %81, %137 overflow<nsw> : index
        vector.store %139, %78[%140] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %141 = vector.extract_strided_slice %42 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %142 = arith.addi %85, %137 overflow<nsw> : index
        vector.store %141, %78[%142] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %143 = vector.extract_strided_slice %42 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %144 = arith.addi %89, %137 overflow<nsw> : index
        vector.store %143, %78[%144] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %145 = vector.extract_strided_slice %43 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %146 = affine.apply #map38()[%thread_id_x, %thread_id_y]
        %147 = arith.addi %75, %146 overflow<nsw> : index
        vector.store %145, %78[%147] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %148 = vector.extract_strided_slice %43 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %149 = arith.addi %81, %146 overflow<nsw> : index
        vector.store %148, %78[%149] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %150 = vector.extract_strided_slice %43 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %151 = arith.addi %85, %146 overflow<nsw> : index
        vector.store %150, %78[%151] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %152 = vector.extract_strided_slice %43 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %153 = arith.addi %89, %146 overflow<nsw> : index
        vector.store %152, %78[%153] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %154 = vector.extract_strided_slice %44 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %155 = affine.apply #map39()[%thread_id_x]
        %156 = arith.muli %155, %c16384 overflow<nsw> : index
        %157 = arith.addi %156, %73 overflow<nsw> : index
        vector.store %154, %78[%157] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %158 = vector.extract_strided_slice %44 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %159 = affine.apply #map40()[%thread_id_x]
        %160 = arith.muli %159, %c16384 overflow<nsw> : index
        %161 = arith.addi %160, %73 overflow<nsw> : index
        vector.store %158, %78[%161] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %162 = vector.extract_strided_slice %44 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %163 = affine.apply #map41()[%thread_id_x]
        %164 = arith.muli %163, %c16384 overflow<nsw> : index
        %165 = arith.addi %164, %73 overflow<nsw> : index
        vector.store %162, %78[%165] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %166 = vector.extract_strided_slice %44 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %167 = affine.apply #map42()[%thread_id_x]
        %168 = arith.muli %167, %c16384 overflow<nsw> : index
        %169 = arith.addi %168, %73 overflow<nsw> : index
        vector.store %166, %78[%169] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %170 = vector.extract_strided_slice %45 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %171 = arith.addi %156, %92 overflow<nsw> : index
        vector.store %170, %78[%171] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %172 = vector.extract_strided_slice %45 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %173 = arith.addi %160, %92 overflow<nsw> : index
        vector.store %172, %78[%173] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %174 = vector.extract_strided_slice %45 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %175 = arith.addi %164, %92 overflow<nsw> : index
        vector.store %174, %78[%175] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %176 = vector.extract_strided_slice %45 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %177 = arith.addi %168, %92 overflow<nsw> : index
        vector.store %176, %78[%177] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %178 = vector.extract_strided_slice %46 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %179 = arith.addi %156, %101 overflow<nsw> : index
        vector.store %178, %78[%179] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %180 = vector.extract_strided_slice %46 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %181 = arith.addi %160, %101 overflow<nsw> : index
        vector.store %180, %78[%181] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %182 = vector.extract_strided_slice %46 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %183 = arith.addi %164, %101 overflow<nsw> : index
        vector.store %182, %78[%183] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %184 = vector.extract_strided_slice %46 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %185 = arith.addi %168, %101 overflow<nsw> : index
        vector.store %184, %78[%185] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %186 = vector.extract_strided_slice %47 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %187 = arith.addi %156, %110 overflow<nsw> : index
        vector.store %186, %78[%187] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %188 = vector.extract_strided_slice %47 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %189 = arith.addi %160, %110 overflow<nsw> : index
        vector.store %188, %78[%189] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %190 = vector.extract_strided_slice %47 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %191 = arith.addi %164, %110 overflow<nsw> : index
        vector.store %190, %78[%191] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %192 = vector.extract_strided_slice %47 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %193 = arith.addi %168, %110 overflow<nsw> : index
        vector.store %192, %78[%193] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %194 = vector.extract_strided_slice %48 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %195 = arith.addi %156, %119 overflow<nsw> : index
        vector.store %194, %78[%195] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %196 = vector.extract_strided_slice %48 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %197 = arith.addi %160, %119 overflow<nsw> : index
        vector.store %196, %78[%197] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %198 = vector.extract_strided_slice %48 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %199 = arith.addi %164, %119 overflow<nsw> : index
        vector.store %198, %78[%199] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %200 = vector.extract_strided_slice %48 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %201 = arith.addi %168, %119 overflow<nsw> : index
        vector.store %200, %78[%201] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %202 = vector.extract_strided_slice %49 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %203 = arith.addi %156, %128 overflow<nsw> : index
        vector.store %202, %78[%203] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %204 = vector.extract_strided_slice %49 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %205 = arith.addi %160, %128 overflow<nsw> : index
        vector.store %204, %78[%205] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %206 = vector.extract_strided_slice %49 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %207 = arith.addi %164, %128 overflow<nsw> : index
        vector.store %206, %78[%207] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %208 = vector.extract_strided_slice %49 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %209 = arith.addi %168, %128 overflow<nsw> : index
        vector.store %208, %78[%209] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %210 = vector.extract_strided_slice %50 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %211 = arith.addi %156, %137 overflow<nsw> : index
        vector.store %210, %78[%211] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %212 = vector.extract_strided_slice %50 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %213 = arith.addi %160, %137 overflow<nsw> : index
        vector.store %212, %78[%213] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %214 = vector.extract_strided_slice %50 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %215 = arith.addi %164, %137 overflow<nsw> : index
        vector.store %214, %78[%215] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %216 = vector.extract_strided_slice %50 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %217 = arith.addi %168, %137 overflow<nsw> : index
        vector.store %216, %78[%217] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %218 = vector.extract_strided_slice %51 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %219 = arith.addi %156, %146 overflow<nsw> : index
        vector.store %218, %78[%219] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %220 = vector.extract_strided_slice %51 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %221 = arith.addi %160, %146 overflow<nsw> : index
        vector.store %220, %78[%221] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %222 = vector.extract_strided_slice %51 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %223 = arith.addi %164, %146 overflow<nsw> : index
        vector.store %222, %78[%223] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %224 = vector.extract_strided_slice %51 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %225 = arith.addi %168, %146 overflow<nsw> : index
        vector.store %224, %78[%225] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %226 = vector.extract_strided_slice %52 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %227 = affine.apply #map43()[%thread_id_x]
        %228 = arith.muli %227, %c16384 overflow<nsw> : index
        %229 = arith.addi %228, %73 overflow<nsw> : index
        vector.store %226, %78[%229] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %230 = vector.extract_strided_slice %52 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %231 = affine.apply #map44()[%thread_id_x]
        %232 = arith.muli %231, %c16384 overflow<nsw> : index
        %233 = arith.addi %232, %73 overflow<nsw> : index
        vector.store %230, %78[%233] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %234 = vector.extract_strided_slice %52 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %235 = affine.apply #map45()[%thread_id_x]
        %236 = arith.muli %235, %c16384 overflow<nsw> : index
        %237 = arith.addi %236, %73 overflow<nsw> : index
        vector.store %234, %78[%237] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %238 = vector.extract_strided_slice %52 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %239 = affine.apply #map46()[%thread_id_x]
        %240 = arith.muli %239, %c16384 overflow<nsw> : index
        %241 = arith.addi %240, %73 overflow<nsw> : index
        vector.store %238, %78[%241] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %242 = vector.extract_strided_slice %53 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %243 = arith.addi %228, %92 overflow<nsw> : index
        vector.store %242, %78[%243] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %244 = vector.extract_strided_slice %53 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %245 = arith.addi %232, %92 overflow<nsw> : index
        vector.store %244, %78[%245] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %246 = vector.extract_strided_slice %53 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %247 = arith.addi %236, %92 overflow<nsw> : index
        vector.store %246, %78[%247] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %248 = vector.extract_strided_slice %53 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %249 = arith.addi %240, %92 overflow<nsw> : index
        vector.store %248, %78[%249] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %250 = vector.extract_strided_slice %54 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %251 = arith.addi %228, %101 overflow<nsw> : index
        vector.store %250, %78[%251] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %252 = vector.extract_strided_slice %54 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %253 = arith.addi %232, %101 overflow<nsw> : index
        vector.store %252, %78[%253] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %254 = vector.extract_strided_slice %54 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %255 = arith.addi %236, %101 overflow<nsw> : index
        vector.store %254, %78[%255] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %256 = vector.extract_strided_slice %54 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %257 = arith.addi %240, %101 overflow<nsw> : index
        vector.store %256, %78[%257] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %258 = vector.extract_strided_slice %55 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %259 = arith.addi %228, %110 overflow<nsw> : index
        vector.store %258, %78[%259] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %260 = vector.extract_strided_slice %55 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %261 = arith.addi %232, %110 overflow<nsw> : index
        vector.store %260, %78[%261] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %262 = vector.extract_strided_slice %55 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %263 = arith.addi %236, %110 overflow<nsw> : index
        vector.store %262, %78[%263] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %264 = vector.extract_strided_slice %55 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %265 = arith.addi %240, %110 overflow<nsw> : index
        vector.store %264, %78[%265] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %266 = vector.extract_strided_slice %56 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %267 = arith.addi %228, %119 overflow<nsw> : index
        vector.store %266, %78[%267] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %268 = vector.extract_strided_slice %56 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %269 = arith.addi %232, %119 overflow<nsw> : index
        vector.store %268, %78[%269] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %270 = vector.extract_strided_slice %56 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %271 = arith.addi %236, %119 overflow<nsw> : index
        vector.store %270, %78[%271] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %272 = vector.extract_strided_slice %56 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %273 = arith.addi %240, %119 overflow<nsw> : index
        vector.store %272, %78[%273] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %274 = vector.extract_strided_slice %57 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %275 = arith.addi %228, %128 overflow<nsw> : index
        vector.store %274, %78[%275] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %276 = vector.extract_strided_slice %57 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %277 = arith.addi %232, %128 overflow<nsw> : index
        vector.store %276, %78[%277] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %278 = vector.extract_strided_slice %57 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %279 = arith.addi %236, %128 overflow<nsw> : index
        vector.store %278, %78[%279] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %280 = vector.extract_strided_slice %57 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %281 = arith.addi %240, %128 overflow<nsw> : index
        vector.store %280, %78[%281] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %282 = vector.extract_strided_slice %58 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %283 = arith.addi %228, %137 overflow<nsw> : index
        vector.store %282, %78[%283] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %284 = vector.extract_strided_slice %58 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %285 = arith.addi %232, %137 overflow<nsw> : index
        vector.store %284, %78[%285] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %286 = vector.extract_strided_slice %58 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %287 = arith.addi %236, %137 overflow<nsw> : index
        vector.store %286, %78[%287] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %288 = vector.extract_strided_slice %58 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %289 = arith.addi %240, %137 overflow<nsw> : index
        vector.store %288, %78[%289] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %290 = vector.extract_strided_slice %59 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %291 = arith.addi %228, %146 overflow<nsw> : index
        vector.store %290, %78[%291] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %292 = vector.extract_strided_slice %59 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %293 = arith.addi %232, %146 overflow<nsw> : index
        vector.store %292, %78[%293] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %294 = vector.extract_strided_slice %59 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %295 = arith.addi %236, %146 overflow<nsw> : index
        vector.store %294, %78[%295] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %296 = vector.extract_strided_slice %59 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %297 = arith.addi %240, %146 overflow<nsw> : index
        vector.store %296, %78[%297] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %298 = vector.extract_strided_slice %60 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %299 = affine.apply #map47()[%thread_id_x]
        %300 = arith.muli %299, %c16384 overflow<nsw> : index
        %301 = arith.addi %300, %73 overflow<nsw> : index
        vector.store %298, %78[%301] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %302 = vector.extract_strided_slice %60 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %303 = affine.apply #map48()[%thread_id_x]
        %304 = arith.muli %303, %c16384 overflow<nsw> : index
        %305 = arith.addi %304, %73 overflow<nsw> : index
        vector.store %302, %78[%305] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %306 = vector.extract_strided_slice %60 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %307 = affine.apply #map49()[%thread_id_x]
        %308 = arith.muli %307, %c16384 overflow<nsw> : index
        %309 = arith.addi %308, %73 overflow<nsw> : index
        vector.store %306, %78[%309] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %310 = vector.extract_strided_slice %60 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %311 = affine.apply #map50()[%thread_id_x]
        %312 = arith.muli %311, %c16384 overflow<nsw> : index
        %313 = arith.addi %312, %73 overflow<nsw> : index
        vector.store %310, %78[%313] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %314 = vector.extract_strided_slice %61 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %315 = arith.addi %300, %92 overflow<nsw> : index
        vector.store %314, %78[%315] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %316 = vector.extract_strided_slice %61 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %317 = arith.addi %304, %92 overflow<nsw> : index
        vector.store %316, %78[%317] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %318 = vector.extract_strided_slice %61 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %319 = arith.addi %308, %92 overflow<nsw> : index
        vector.store %318, %78[%319] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %320 = vector.extract_strided_slice %61 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %321 = arith.addi %312, %92 overflow<nsw> : index
        vector.store %320, %78[%321] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %322 = vector.extract_strided_slice %62 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %323 = arith.addi %300, %101 overflow<nsw> : index
        vector.store %322, %78[%323] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %324 = vector.extract_strided_slice %62 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %325 = arith.addi %304, %101 overflow<nsw> : index
        vector.store %324, %78[%325] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %326 = vector.extract_strided_slice %62 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %327 = arith.addi %308, %101 overflow<nsw> : index
        vector.store %326, %78[%327] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %328 = vector.extract_strided_slice %62 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %329 = arith.addi %312, %101 overflow<nsw> : index
        vector.store %328, %78[%329] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %330 = vector.extract_strided_slice %63 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %331 = arith.addi %300, %110 overflow<nsw> : index
        vector.store %330, %78[%331] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %332 = vector.extract_strided_slice %63 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %333 = arith.addi %304, %110 overflow<nsw> : index
        vector.store %332, %78[%333] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %334 = vector.extract_strided_slice %63 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %335 = arith.addi %308, %110 overflow<nsw> : index
        vector.store %334, %78[%335] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %336 = vector.extract_strided_slice %63 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %337 = arith.addi %312, %110 overflow<nsw> : index
        vector.store %336, %78[%337] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %338 = vector.extract_strided_slice %64 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %339 = arith.addi %300, %119 overflow<nsw> : index
        vector.store %338, %78[%339] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %340 = vector.extract_strided_slice %64 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %341 = arith.addi %304, %119 overflow<nsw> : index
        vector.store %340, %78[%341] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %342 = vector.extract_strided_slice %64 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %343 = arith.addi %308, %119 overflow<nsw> : index
        vector.store %342, %78[%343] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %344 = vector.extract_strided_slice %64 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %345 = arith.addi %312, %119 overflow<nsw> : index
        vector.store %344, %78[%345] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %346 = vector.extract_strided_slice %65 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %347 = arith.addi %300, %128 overflow<nsw> : index
        vector.store %346, %78[%347] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %348 = vector.extract_strided_slice %65 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %349 = arith.addi %304, %128 overflow<nsw> : index
        vector.store %348, %78[%349] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %350 = vector.extract_strided_slice %65 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %351 = arith.addi %308, %128 overflow<nsw> : index
        vector.store %350, %78[%351] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %352 = vector.extract_strided_slice %65 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %353 = arith.addi %312, %128 overflow<nsw> : index
        vector.store %352, %78[%353] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %354 = vector.extract_strided_slice %66 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %355 = arith.addi %300, %137 overflow<nsw> : index
        vector.store %354, %78[%355] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %356 = vector.extract_strided_slice %66 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %357 = arith.addi %304, %137 overflow<nsw> : index
        vector.store %356, %78[%357] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %358 = vector.extract_strided_slice %66 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %359 = arith.addi %308, %137 overflow<nsw> : index
        vector.store %358, %78[%359] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %360 = vector.extract_strided_slice %66 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %361 = arith.addi %312, %137 overflow<nsw> : index
        vector.store %360, %78[%361] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %362 = vector.extract_strided_slice %67 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %363 = arith.addi %300, %146 overflow<nsw> : index
        vector.store %362, %78[%363] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %364 = vector.extract_strided_slice %67 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %365 = arith.addi %304, %146 overflow<nsw> : index
        vector.store %364, %78[%365] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %366 = vector.extract_strided_slice %67 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %367 = arith.addi %308, %146 overflow<nsw> : index
        vector.store %366, %78[%367] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %368 = vector.extract_strided_slice %67 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %369 = arith.addi %312, %146 overflow<nsw> : index
        vector.store %368, %78[%369] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<16384x8192xi8>
    %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<16384x512xi8>
    %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<16384x8192xi8>
    %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<16384x512xi8>
    %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<16384x16384xbf16>
    %5 = flow.dispatch @gemm_afp4_wfp4_wave::@gemm_afp4_wfp4_wave(%0, %1, %2, %3, %4) : (tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x16384xbf16>) -> %4
    %6 = hal.tensor.barrier join(%5 : tensor<16384x16384xbf16>) => %arg6 : !hal.fence
    %7 = hal.tensor.export %6 : tensor<16384x16384xbf16> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
    func.func @matmul(%arg0: tensor<16384x8192xi8>, %arg1: tensor<16384x512xi8>, %arg2: tensor<16384x8192xi8>, %arg3: tensor<16384x512xi8>, %arg4: tensor<16384x16384xbf16>) -> tensor<16384x16384xbf16> {
    %0 = flow.dispatch @gemm_afp4_wfp4_wave::@gemm_afp4_wfp4_wave(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x16384xbf16>) -> %arg4
    return %0 : tensor<16384x16384xbf16>
  }
}
