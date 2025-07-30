#map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
#map1 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
#map2 = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 8) * 128)>
#map3 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map4 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map6 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
#map8 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>

#map10 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
#map11 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>

#map9 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
#map12 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
#map13 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
#map14 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>

#map15 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 - (s1 floordiv 8) * 128)>
#map16 = affine_map<()[s0] -> (s0 * 256)>
#map17 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
#map18 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map19 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map20 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map21 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map22 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map23 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map24 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map25 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map26 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map27 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map28 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
#map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
#map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
#map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
#map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
#map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
#map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
#map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
#map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
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
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %cst_1 = arith.constant dense<0.000000e+00> : vector<4xf32>
        %block_id_x = gpu.block_id  x upper_bound 64
        %block_id_y = gpu.block_id  y upper_bound 64
        %thread_id_x = gpu.thread_id  x upper_bound 256
        %thread_id_y = gpu.thread_id  y upper_bound 2
        %alloc = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
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
        %15 = affine.apply #map9()[%thread_id_x]
        %16 = affine.apply #map10()[%thread_id_x]
        %17 = affine.apply #map11()[%thread_id_x]
        %18 = affine.apply #map12()[%thread_id_x]
        %19 = affine.apply #map13()[%thread_id_x]
        %20 = affine.apply #map14()[%thread_id_x]
        %21:32 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %cst_1, %arg7 = %cst_1, %arg8 = %cst_1, %arg9 = %cst_1, %arg10 = %cst_1, %arg11 = %cst_1, %arg12 = %cst_1, %arg13 = %cst_1, %arg14 = %cst_1, %arg15 = %cst_1, %arg16 = %cst_1, %arg17 = %cst_1, %arg18 = %cst_1, %arg19 = %cst_1, %arg20 = %cst_1, %arg21 = %cst_1, %arg22 = %cst_1, %arg23 = %cst_1, %arg24 = %cst_1, %arg25 = %cst_1, %arg26 = %cst_1, %arg27 = %cst_1, %arg28 = %cst_1, %arg29 = %cst_1, %arg30 = %cst_1, %arg31 = %cst_1, %arg32 = %cst_1, %arg33 = %cst_1, %arg34 = %cst_1, %arg35 = %cst_1, %arg36 = %cst_1, %arg37 = %cst_1) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %356 = affine.apply #map15()[%arg5, %thread_id_x]
          %357 = arith.addi %2, %356 overflow<nsw> : index
          %358 = vector.load %3[%357] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          amdgpu.lds_barrier
          vector.store %358, %alloc[%4, %5] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %359 = arith.addi %7, %356 overflow<nsw> : index
          %360 = vector.load %3[%359] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %360, %alloc[%8, %5] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %361 = arith.addi %10, %356 overflow<nsw> : index
          %362 = vector.load %3[%361] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %362, %alloc[%11, %5] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %363 = arith.addi %13, %356 overflow<nsw> : index
          %364 = vector.load %3[%363] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %364, %alloc[%14, %5] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          amdgpu.lds_barrier

          // Get c in element (not byte offet)
          %c_0 = affine.apply affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>()[%thread_id_x]
          %c_1 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 16)+4)>()[%thread_id_x]

          // #map9 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
          // #map12 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
          // #map13 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
          // #map14 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>

          // R mod 8 (maxPhase)
          %r0 = affine.apply affine_map<()[s0] -> ((s0 mod 16 + (s0 floordiv 64) * 64) mod 8)>()[%thread_id_x]
          %r1 = affine.apply affine_map<()[s0] -> ((s0 mod 16 + (s0 floordiv 64) * 64 + 16 ) mod 8)>()[%thread_id_x]
          %r2 = affine.apply affine_map<()[s0] -> ((s0 mod 16 + (s0 floordiv 64) * 64 + 32 ) mod 8)>()[%thread_id_x]
          %r3 = affine.apply affine_map<()[s0] -> ((s0 mod 16 + (s0 floordiv 64) * 64 + 48 ) mod 8)>()[%thread_id_x]

          //XOR swizzling
          %r0_c0_swizzle = arith.xori %r0,%c_0 : index 
          %r0_c1_swizzle = arith.xori %r0,%c_1 : index 
          %r1_c0_swizzle = arith.xori %r1,%c_0 : index 
          %r1_c1_swizzle = arith.xori %r1,%c_1 : index 
          %r2_c0_swizzle = arith.xori %r2,%c_0 : index 
          %r2_c1_swizzle = arith.xori %r2,%c_1 : index 
          %r3_c0_swizzle = arith.xori %r3,%c_0 : index 
          %r3_c1_swizzle = arith.xori %r3,%c_1 : index 

          //Back to byte offset
          %c16 = arith.constant 16 : index
          %r0_c0_swizzle_b = arith.muli %r0_c0_swizzle,%c16 : index 
          %r0_c1_swizzle_b = arith.muli %r0_c1_swizzle,%c16 : index 
          %r1_c0_swizzle_b = arith.muli %r1_c0_swizzle,%c16 : index 
          %r1_c1_swizzle_b = arith.muli %r1_c1_swizzle,%c16 : index 
          %r2_c0_swizzle_b = arith.muli %r2_c0_swizzle,%c16 : index 
          %r2_c1_swizzle_b = arith.muli %r2_c1_swizzle,%c16 : index 
          %r3_c0_swizzle_b = arith.muli %r3_c0_swizzle,%c16 : index 
          %r3_c1_swizzle_b = arith.muli %r3_c1_swizzle,%c16 : index 
          
          %365 = vector.load %alloc[%15, %r0_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %366 = vector.load %alloc[%15, %r0_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %367 = vector.load %alloc[%18, %r1_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %368 = vector.load %alloc[%18, %r1_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %369 = vector.load %alloc[%19, %r2_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %370 = vector.load %alloc[%19, %r2_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %371 = vector.load %alloc[%20, %r3_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %372 = vector.load %alloc[%20, %r3_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>

          //old version
          // %365 = vector.load %alloc[%15, %16] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %366 = vector.load %alloc[%15, %17] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %367 = vector.load %alloc[%18, %16] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %368 = vector.load %alloc[%18, %17] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %369 = vector.load %alloc[%19, %16] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %370 = vector.load %alloc[%19, %17] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %371 = vector.load %alloc[%20, %16] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %372 = vector.load %alloc[%20, %17] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %373 = vector.bitcast %365 : vector<16xi8> to vector<32xf4E2M1FN>
          %374 = vector.bitcast %366 : vector<16xi8> to vector<32xf4E2M1FN>
          %375 = vector.bitcast %367 : vector<16xi8> to vector<32xf4E2M1FN>
          %376 = vector.bitcast %368 : vector<16xi8> to vector<32xf4E2M1FN>
          %377 = vector.bitcast %369 : vector<16xi8> to vector<32xf4E2M1FN>
          %378 = vector.bitcast %370 : vector<16xi8> to vector<32xf4E2M1FN>
          %379 = vector.bitcast %371 : vector<16xi8> to vector<32xf4E2M1FN>
          %380 = vector.bitcast %372 : vector<16xi8> to vector<32xf4E2M1FN>
          %381 = amdgpu.scaled_mfma(%cst[0] * %373) * (%cst[0] * %cst_0) + %arg6 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %382 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %cst_0) + %381 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %383 = amdgpu.scaled_mfma(%cst[0] * %373) * (%cst[0] * %cst_0) + %arg7 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %384 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %cst_0) + %383 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %385 = amdgpu.scaled_mfma(%cst[0] * %373) * (%cst[0] * %cst_0) + %arg8 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %386 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %cst_0) + %385 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %387 = amdgpu.scaled_mfma(%cst[0] * %373) * (%cst[0] * %cst_0) + %arg9 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %388 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %cst_0) + %387 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %389 = amdgpu.scaled_mfma(%cst[0] * %373) * (%cst[0] * %cst_0) + %arg10 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %390 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %cst_0) + %389 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %391 = amdgpu.scaled_mfma(%cst[0] * %373) * (%cst[0] * %cst_0) + %arg11 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %392 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %cst_0) + %391 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %393 = amdgpu.scaled_mfma(%cst[0] * %373) * (%cst[0] * %cst_0) + %arg12 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %394 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %cst_0) + %393 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %395 = amdgpu.scaled_mfma(%cst[0] * %373) * (%cst[0] * %cst_0) + %arg13 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %396 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %cst_0) + %395 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %397 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %cst_0) + %arg14 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %398 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %cst_0) + %397 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %399 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %cst_0) + %arg15 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %400 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %cst_0) + %399 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %401 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %cst_0) + %arg16 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %402 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %cst_0) + %401 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %403 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %cst_0) + %arg17 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %404 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %cst_0) + %403 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %405 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %cst_0) + %arg18 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %406 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %cst_0) + %405 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %407 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %cst_0) + %arg19 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %408 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %cst_0) + %407 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %409 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %cst_0) + %arg20 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %410 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %cst_0) + %409 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %411 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %cst_0) + %arg21 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %412 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %cst_0) + %411 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %413 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %cst_0) + %arg22 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %414 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %cst_0) + %413 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %415 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %cst_0) + %arg23 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %416 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %cst_0) + %415 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %417 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %cst_0) + %arg24 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %418 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %cst_0) + %417 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %419 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %cst_0) + %arg25 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %420 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %cst_0) + %419 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %421 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %cst_0) + %arg26 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %422 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %cst_0) + %421 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %423 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %cst_0) + %arg27 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %424 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %cst_0) + %423 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %425 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %cst_0) + %arg28 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %426 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %cst_0) + %425 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %427 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %cst_0) + %arg29 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %428 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %cst_0) + %427 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %429 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %cst_0) + %arg30 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %430 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %cst_0) + %429 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %431 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %cst_0) + %arg31 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %432 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %cst_0) + %431 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %433 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %cst_0) + %arg32 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %434 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %cst_0) + %433 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %435 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %cst_0) + %arg33 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %436 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %cst_0) + %435 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %437 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %cst_0) + %arg34 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %438 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %cst_0) + %437 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %439 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %cst_0) + %arg35 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %440 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %cst_0) + %439 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %441 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %cst_0) + %arg36 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %442 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %cst_0) + %441 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %443 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %cst_0) + %arg37 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %444 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %cst_0) + %443 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          scf.yield %382, %384, %386, %388, %390, %392, %394, %396, %398, %400, %402, %404, %406, %408, %410, %412, %414, %416, %418, %420, %422, %424, %426, %428, %430, %432, %434, %436, %438, %440, %442, %444 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %22 = arith.truncf %21#0 : vector<4xf32> to vector<4xbf16>
        %23 = arith.truncf %21#1 : vector<4xf32> to vector<4xbf16>
        %24 = arith.truncf %21#2 : vector<4xf32> to vector<4xbf16>
        %25 = arith.truncf %21#3 : vector<4xf32> to vector<4xbf16>
        %26 = arith.truncf %21#4 : vector<4xf32> to vector<4xbf16>
        %27 = arith.truncf %21#5 : vector<4xf32> to vector<4xbf16>
        %28 = arith.truncf %21#6 : vector<4xf32> to vector<4xbf16>
        %29 = arith.truncf %21#7 : vector<4xf32> to vector<4xbf16>
        %30 = arith.truncf %21#8 : vector<4xf32> to vector<4xbf16>
        %31 = arith.truncf %21#9 : vector<4xf32> to vector<4xbf16>
        %32 = arith.truncf %21#10 : vector<4xf32> to vector<4xbf16>
        %33 = arith.truncf %21#11 : vector<4xf32> to vector<4xbf16>
        %34 = arith.truncf %21#12 : vector<4xf32> to vector<4xbf16>
        %35 = arith.truncf %21#13 : vector<4xf32> to vector<4xbf16>
        %36 = arith.truncf %21#14 : vector<4xf32> to vector<4xbf16>
        %37 = arith.truncf %21#15 : vector<4xf32> to vector<4xbf16>
        %38 = arith.truncf %21#16 : vector<4xf32> to vector<4xbf16>
        %39 = arith.truncf %21#17 : vector<4xf32> to vector<4xbf16>
        %40 = arith.truncf %21#18 : vector<4xf32> to vector<4xbf16>
        %41 = arith.truncf %21#19 : vector<4xf32> to vector<4xbf16>
        %42 = arith.truncf %21#20 : vector<4xf32> to vector<4xbf16>
        %43 = arith.truncf %21#21 : vector<4xf32> to vector<4xbf16>
        %44 = arith.truncf %21#22 : vector<4xf32> to vector<4xbf16>
        %45 = arith.truncf %21#23 : vector<4xf32> to vector<4xbf16>
        %46 = arith.truncf %21#24 : vector<4xf32> to vector<4xbf16>
        %47 = arith.truncf %21#25 : vector<4xf32> to vector<4xbf16>
        %48 = arith.truncf %21#26 : vector<4xf32> to vector<4xbf16>
        %49 = arith.truncf %21#27 : vector<4xf32> to vector<4xbf16>
        %50 = arith.truncf %21#28 : vector<4xf32> to vector<4xbf16>
        %51 = arith.truncf %21#29 : vector<4xf32> to vector<4xbf16>
        %52 = arith.truncf %21#30 : vector<4xf32> to vector<4xbf16>
        %53 = arith.truncf %21#31 : vector<4xf32> to vector<4xbf16>
        %54 = vector.extract_strided_slice %22 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %55 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<16384x16384xbf16, strided<[16384, 1], offset: ?>>
        %56 = affine.apply #map16()[%block_id_x]
        %57 = affine.apply #map16()[%block_id_y]
        %58 = affine.apply #map17()[%thread_id_x]
        %59 = affine.apply #map18()[%thread_id_x, %thread_id_y]
        %60 = arith.muli %56, %c16384 overflow<nsw> : index
        %61 = arith.muli %58, %c16384 overflow<nsw> : index
        %62 = arith.addi %60, %57 overflow<nsw> : index
        %63 = arith.addi %61, %59 overflow<nsw> : index
        %reinterpret_cast_2 = memref.reinterpret_cast %55 to offset: [%62], sizes: [%c1073741822], strides: [1] : memref<16384x16384xbf16, strided<[16384, 1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
        %64 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_2 validBytes(%c2147483645_i32) : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        vector.store %54, %64[%63] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %65 = vector.extract_strided_slice %22 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %66 = affine.apply #map19()[%thread_id_x]
        %67 = arith.muli %66, %c16384 overflow<nsw> : index
        %68 = arith.addi %67, %59 overflow<nsw> : index
        vector.store %65, %64[%68] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %69 = vector.extract_strided_slice %22 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %70 = affine.apply #map20()[%thread_id_x]
        %71 = arith.muli %70, %c16384 overflow<nsw> : index
        %72 = arith.addi %71, %59 overflow<nsw> : index
        vector.store %69, %64[%72] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %73 = vector.extract_strided_slice %22 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %74 = affine.apply #map21()[%thread_id_x]
        %75 = arith.muli %74, %c16384 overflow<nsw> : index
        %76 = arith.addi %75, %59 overflow<nsw> : index
        vector.store %73, %64[%76] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %77 = vector.extract_strided_slice %23 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %78 = affine.apply #map22()[%thread_id_x, %thread_id_y]
        %79 = arith.addi %61, %78 overflow<nsw> : index
        vector.store %77, %64[%79] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %80 = vector.extract_strided_slice %23 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %81 = arith.addi %67, %78 overflow<nsw> : index
        vector.store %80, %64[%81] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %82 = vector.extract_strided_slice %23 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %83 = arith.addi %71, %78 overflow<nsw> : index
        vector.store %82, %64[%83] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %84 = vector.extract_strided_slice %23 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %85 = arith.addi %75, %78 overflow<nsw> : index
        vector.store %84, %64[%85] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %86 = vector.extract_strided_slice %24 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %87 = affine.apply #map23()[%thread_id_x, %thread_id_y]
        %88 = arith.addi %61, %87 overflow<nsw> : index
        vector.store %86, %64[%88] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %89 = vector.extract_strided_slice %24 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %90 = arith.addi %67, %87 overflow<nsw> : index
        vector.store %89, %64[%90] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %91 = vector.extract_strided_slice %24 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %92 = arith.addi %71, %87 overflow<nsw> : index
        vector.store %91, %64[%92] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %93 = vector.extract_strided_slice %24 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %94 = arith.addi %75, %87 overflow<nsw> : index
        vector.store %93, %64[%94] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %95 = vector.extract_strided_slice %25 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %96 = affine.apply #map24()[%thread_id_x, %thread_id_y]
        %97 = arith.addi %61, %96 overflow<nsw> : index
        vector.store %95, %64[%97] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %98 = vector.extract_strided_slice %25 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %99 = arith.addi %67, %96 overflow<nsw> : index
        vector.store %98, %64[%99] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %100 = vector.extract_strided_slice %25 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %101 = arith.addi %71, %96 overflow<nsw> : index
        vector.store %100, %64[%101] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %102 = vector.extract_strided_slice %25 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %103 = arith.addi %75, %96 overflow<nsw> : index
        vector.store %102, %64[%103] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %104 = vector.extract_strided_slice %26 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %105 = affine.apply #map25()[%thread_id_x, %thread_id_y]
        %106 = arith.addi %61, %105 overflow<nsw> : index
        vector.store %104, %64[%106] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %107 = vector.extract_strided_slice %26 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %108 = arith.addi %67, %105 overflow<nsw> : index
        vector.store %107, %64[%108] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %109 = vector.extract_strided_slice %26 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %110 = arith.addi %71, %105 overflow<nsw> : index
        vector.store %109, %64[%110] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %111 = vector.extract_strided_slice %26 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %112 = arith.addi %75, %105 overflow<nsw> : index
        vector.store %111, %64[%112] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %113 = vector.extract_strided_slice %27 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %114 = affine.apply #map26()[%thread_id_x, %thread_id_y]
        %115 = arith.addi %61, %114 overflow<nsw> : index
        vector.store %113, %64[%115] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %116 = vector.extract_strided_slice %27 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %117 = arith.addi %67, %114 overflow<nsw> : index
        vector.store %116, %64[%117] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %118 = vector.extract_strided_slice %27 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %119 = arith.addi %71, %114 overflow<nsw> : index
        vector.store %118, %64[%119] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %120 = vector.extract_strided_slice %27 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %121 = arith.addi %75, %114 overflow<nsw> : index
        vector.store %120, %64[%121] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %122 = vector.extract_strided_slice %28 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %123 = affine.apply #map27()[%thread_id_x, %thread_id_y]
        %124 = arith.addi %61, %123 overflow<nsw> : index
        vector.store %122, %64[%124] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %125 = vector.extract_strided_slice %28 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %126 = arith.addi %67, %123 overflow<nsw> : index
        vector.store %125, %64[%126] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %127 = vector.extract_strided_slice %28 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %128 = arith.addi %71, %123 overflow<nsw> : index
        vector.store %127, %64[%128] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %129 = vector.extract_strided_slice %28 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %130 = arith.addi %75, %123 overflow<nsw> : index
        vector.store %129, %64[%130] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %131 = vector.extract_strided_slice %29 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %132 = affine.apply #map28()[%thread_id_x, %thread_id_y]
        %133 = arith.addi %61, %132 overflow<nsw> : index
        vector.store %131, %64[%133] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %134 = vector.extract_strided_slice %29 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %135 = arith.addi %67, %132 overflow<nsw> : index
        vector.store %134, %64[%135] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %136 = vector.extract_strided_slice %29 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %137 = arith.addi %71, %132 overflow<nsw> : index
        vector.store %136, %64[%137] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %138 = vector.extract_strided_slice %29 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %139 = arith.addi %75, %132 overflow<nsw> : index
        vector.store %138, %64[%139] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %140 = vector.extract_strided_slice %30 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %141 = affine.apply #map29()[%thread_id_x]
        %142 = arith.muli %141, %c16384 overflow<nsw> : index
        %143 = arith.addi %142, %59 overflow<nsw> : index
        vector.store %140, %64[%143] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %144 = vector.extract_strided_slice %30 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %145 = affine.apply #map30()[%thread_id_x]
        %146 = arith.muli %145, %c16384 overflow<nsw> : index
        %147 = arith.addi %146, %59 overflow<nsw> : index
        vector.store %144, %64[%147] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %148 = vector.extract_strided_slice %30 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %149 = affine.apply #map31()[%thread_id_x]
        %150 = arith.muli %149, %c16384 overflow<nsw> : index
        %151 = arith.addi %150, %59 overflow<nsw> : index
        vector.store %148, %64[%151] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %152 = vector.extract_strided_slice %30 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %153 = affine.apply #map32()[%thread_id_x]
        %154 = arith.muli %153, %c16384 overflow<nsw> : index
        %155 = arith.addi %154, %59 overflow<nsw> : index
        vector.store %152, %64[%155] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %156 = vector.extract_strided_slice %31 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %157 = arith.addi %142, %78 overflow<nsw> : index
        vector.store %156, %64[%157] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %158 = vector.extract_strided_slice %31 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %159 = arith.addi %146, %78 overflow<nsw> : index
        vector.store %158, %64[%159] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %160 = vector.extract_strided_slice %31 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %161 = arith.addi %150, %78 overflow<nsw> : index
        vector.store %160, %64[%161] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %162 = vector.extract_strided_slice %31 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %163 = arith.addi %154, %78 overflow<nsw> : index
        vector.store %162, %64[%163] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %164 = vector.extract_strided_slice %32 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %165 = arith.addi %142, %87 overflow<nsw> : index
        vector.store %164, %64[%165] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %166 = vector.extract_strided_slice %32 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %167 = arith.addi %146, %87 overflow<nsw> : index
        vector.store %166, %64[%167] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %168 = vector.extract_strided_slice %32 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %169 = arith.addi %150, %87 overflow<nsw> : index
        vector.store %168, %64[%169] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %170 = vector.extract_strided_slice %32 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %171 = arith.addi %154, %87 overflow<nsw> : index
        vector.store %170, %64[%171] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %172 = vector.extract_strided_slice %33 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %173 = arith.addi %142, %96 overflow<nsw> : index
        vector.store %172, %64[%173] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %174 = vector.extract_strided_slice %33 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %175 = arith.addi %146, %96 overflow<nsw> : index
        vector.store %174, %64[%175] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %176 = vector.extract_strided_slice %33 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %177 = arith.addi %150, %96 overflow<nsw> : index
        vector.store %176, %64[%177] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %178 = vector.extract_strided_slice %33 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %179 = arith.addi %154, %96 overflow<nsw> : index
        vector.store %178, %64[%179] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %180 = vector.extract_strided_slice %34 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %181 = arith.addi %142, %105 overflow<nsw> : index
        vector.store %180, %64[%181] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %182 = vector.extract_strided_slice %34 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %183 = arith.addi %146, %105 overflow<nsw> : index
        vector.store %182, %64[%183] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %184 = vector.extract_strided_slice %34 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %185 = arith.addi %150, %105 overflow<nsw> : index
        vector.store %184, %64[%185] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %186 = vector.extract_strided_slice %34 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %187 = arith.addi %154, %105 overflow<nsw> : index
        vector.store %186, %64[%187] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %188 = vector.extract_strided_slice %35 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %189 = arith.addi %142, %114 overflow<nsw> : index
        vector.store %188, %64[%189] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %190 = vector.extract_strided_slice %35 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %191 = arith.addi %146, %114 overflow<nsw> : index
        vector.store %190, %64[%191] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %192 = vector.extract_strided_slice %35 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %193 = arith.addi %150, %114 overflow<nsw> : index
        vector.store %192, %64[%193] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %194 = vector.extract_strided_slice %35 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %195 = arith.addi %154, %114 overflow<nsw> : index
        vector.store %194, %64[%195] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %196 = vector.extract_strided_slice %36 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %197 = arith.addi %142, %123 overflow<nsw> : index
        vector.store %196, %64[%197] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %198 = vector.extract_strided_slice %36 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %199 = arith.addi %146, %123 overflow<nsw> : index
        vector.store %198, %64[%199] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %200 = vector.extract_strided_slice %36 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %201 = arith.addi %150, %123 overflow<nsw> : index
        vector.store %200, %64[%201] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %202 = vector.extract_strided_slice %36 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %203 = arith.addi %154, %123 overflow<nsw> : index
        vector.store %202, %64[%203] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %204 = vector.extract_strided_slice %37 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %205 = arith.addi %142, %132 overflow<nsw> : index
        vector.store %204, %64[%205] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %206 = vector.extract_strided_slice %37 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %207 = arith.addi %146, %132 overflow<nsw> : index
        vector.store %206, %64[%207] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %208 = vector.extract_strided_slice %37 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %209 = arith.addi %150, %132 overflow<nsw> : index
        vector.store %208, %64[%209] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %210 = vector.extract_strided_slice %37 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %211 = arith.addi %154, %132 overflow<nsw> : index
        vector.store %210, %64[%211] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %212 = vector.extract_strided_slice %38 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %213 = affine.apply #map33()[%thread_id_x]
        %214 = arith.muli %213, %c16384 overflow<nsw> : index
        %215 = arith.addi %214, %59 overflow<nsw> : index
        vector.store %212, %64[%215] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %216 = vector.extract_strided_slice %38 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %217 = affine.apply #map34()[%thread_id_x]
        %218 = arith.muli %217, %c16384 overflow<nsw> : index
        %219 = arith.addi %218, %59 overflow<nsw> : index
        vector.store %216, %64[%219] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %220 = vector.extract_strided_slice %38 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %221 = affine.apply #map35()[%thread_id_x]
        %222 = arith.muli %221, %c16384 overflow<nsw> : index
        %223 = arith.addi %222, %59 overflow<nsw> : index
        vector.store %220, %64[%223] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %224 = vector.extract_strided_slice %38 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %225 = affine.apply #map36()[%thread_id_x]
        %226 = arith.muli %225, %c16384 overflow<nsw> : index
        %227 = arith.addi %226, %59 overflow<nsw> : index
        vector.store %224, %64[%227] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %228 = vector.extract_strided_slice %39 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %229 = arith.addi %214, %78 overflow<nsw> : index
        vector.store %228, %64[%229] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %230 = vector.extract_strided_slice %39 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %231 = arith.addi %218, %78 overflow<nsw> : index
        vector.store %230, %64[%231] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %232 = vector.extract_strided_slice %39 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %233 = arith.addi %222, %78 overflow<nsw> : index
        vector.store %232, %64[%233] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %234 = vector.extract_strided_slice %39 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %235 = arith.addi %226, %78 overflow<nsw> : index
        vector.store %234, %64[%235] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %236 = vector.extract_strided_slice %40 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %237 = arith.addi %214, %87 overflow<nsw> : index
        vector.store %236, %64[%237] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %238 = vector.extract_strided_slice %40 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %239 = arith.addi %218, %87 overflow<nsw> : index
        vector.store %238, %64[%239] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %240 = vector.extract_strided_slice %40 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %241 = arith.addi %222, %87 overflow<nsw> : index
        vector.store %240, %64[%241] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %242 = vector.extract_strided_slice %40 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %243 = arith.addi %226, %87 overflow<nsw> : index
        vector.store %242, %64[%243] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %244 = vector.extract_strided_slice %41 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %245 = arith.addi %214, %96 overflow<nsw> : index
        vector.store %244, %64[%245] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %246 = vector.extract_strided_slice %41 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %247 = arith.addi %218, %96 overflow<nsw> : index
        vector.store %246, %64[%247] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %248 = vector.extract_strided_slice %41 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %249 = arith.addi %222, %96 overflow<nsw> : index
        vector.store %248, %64[%249] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %250 = vector.extract_strided_slice %41 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %251 = arith.addi %226, %96 overflow<nsw> : index
        vector.store %250, %64[%251] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %252 = vector.extract_strided_slice %42 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %253 = arith.addi %214, %105 overflow<nsw> : index
        vector.store %252, %64[%253] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %254 = vector.extract_strided_slice %42 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %255 = arith.addi %218, %105 overflow<nsw> : index
        vector.store %254, %64[%255] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %256 = vector.extract_strided_slice %42 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %257 = arith.addi %222, %105 overflow<nsw> : index
        vector.store %256, %64[%257] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %258 = vector.extract_strided_slice %42 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %259 = arith.addi %226, %105 overflow<nsw> : index
        vector.store %258, %64[%259] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %260 = vector.extract_strided_slice %43 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %261 = arith.addi %214, %114 overflow<nsw> : index
        vector.store %260, %64[%261] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %262 = vector.extract_strided_slice %43 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %263 = arith.addi %218, %114 overflow<nsw> : index
        vector.store %262, %64[%263] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %264 = vector.extract_strided_slice %43 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %265 = arith.addi %222, %114 overflow<nsw> : index
        vector.store %264, %64[%265] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %266 = vector.extract_strided_slice %43 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %267 = arith.addi %226, %114 overflow<nsw> : index
        vector.store %266, %64[%267] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %268 = vector.extract_strided_slice %44 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %269 = arith.addi %214, %123 overflow<nsw> : index
        vector.store %268, %64[%269] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %270 = vector.extract_strided_slice %44 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %271 = arith.addi %218, %123 overflow<nsw> : index
        vector.store %270, %64[%271] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %272 = vector.extract_strided_slice %44 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %273 = arith.addi %222, %123 overflow<nsw> : index
        vector.store %272, %64[%273] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %274 = vector.extract_strided_slice %44 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %275 = arith.addi %226, %123 overflow<nsw> : index
        vector.store %274, %64[%275] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %276 = vector.extract_strided_slice %45 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %277 = arith.addi %214, %132 overflow<nsw> : index
        vector.store %276, %64[%277] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %278 = vector.extract_strided_slice %45 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %279 = arith.addi %218, %132 overflow<nsw> : index
        vector.store %278, %64[%279] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %280 = vector.extract_strided_slice %45 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %281 = arith.addi %222, %132 overflow<nsw> : index
        vector.store %280, %64[%281] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %282 = vector.extract_strided_slice %45 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %283 = arith.addi %226, %132 overflow<nsw> : index
        vector.store %282, %64[%283] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %284 = vector.extract_strided_slice %46 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %285 = affine.apply #map37()[%thread_id_x]
        %286 = arith.muli %285, %c16384 overflow<nsw> : index
        %287 = arith.addi %286, %59 overflow<nsw> : index
        vector.store %284, %64[%287] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %288 = vector.extract_strided_slice %46 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %289 = affine.apply #map38()[%thread_id_x]
        %290 = arith.muli %289, %c16384 overflow<nsw> : index
        %291 = arith.addi %290, %59 overflow<nsw> : index
        vector.store %288, %64[%291] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %292 = vector.extract_strided_slice %46 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %293 = affine.apply #map39()[%thread_id_x]
        %294 = arith.muli %293, %c16384 overflow<nsw> : index
        %295 = arith.addi %294, %59 overflow<nsw> : index
        vector.store %292, %64[%295] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %296 = vector.extract_strided_slice %46 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %297 = affine.apply #map40()[%thread_id_x]
        %298 = arith.muli %297, %c16384 overflow<nsw> : index
        %299 = arith.addi %298, %59 overflow<nsw> : index
        vector.store %296, %64[%299] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %300 = vector.extract_strided_slice %47 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %301 = arith.addi %286, %78 overflow<nsw> : index
        vector.store %300, %64[%301] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %302 = vector.extract_strided_slice %47 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %303 = arith.addi %290, %78 overflow<nsw> : index
        vector.store %302, %64[%303] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %304 = vector.extract_strided_slice %47 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %305 = arith.addi %294, %78 overflow<nsw> : index
        vector.store %304, %64[%305] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %306 = vector.extract_strided_slice %47 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %307 = arith.addi %298, %78 overflow<nsw> : index
        vector.store %306, %64[%307] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %308 = vector.extract_strided_slice %48 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %309 = arith.addi %286, %87 overflow<nsw> : index
        vector.store %308, %64[%309] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %310 = vector.extract_strided_slice %48 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %311 = arith.addi %290, %87 overflow<nsw> : index
        vector.store %310, %64[%311] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %312 = vector.extract_strided_slice %48 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %313 = arith.addi %294, %87 overflow<nsw> : index
        vector.store %312, %64[%313] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %314 = vector.extract_strided_slice %48 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %315 = arith.addi %298, %87 overflow<nsw> : index
        vector.store %314, %64[%315] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %316 = vector.extract_strided_slice %49 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %317 = arith.addi %286, %96 overflow<nsw> : index
        vector.store %316, %64[%317] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %318 = vector.extract_strided_slice %49 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %319 = arith.addi %290, %96 overflow<nsw> : index
        vector.store %318, %64[%319] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %320 = vector.extract_strided_slice %49 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %321 = arith.addi %294, %96 overflow<nsw> : index
        vector.store %320, %64[%321] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %322 = vector.extract_strided_slice %49 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %323 = arith.addi %298, %96 overflow<nsw> : index
        vector.store %322, %64[%323] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %324 = vector.extract_strided_slice %50 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %325 = arith.addi %286, %105 overflow<nsw> : index
        vector.store %324, %64[%325] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %326 = vector.extract_strided_slice %50 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %327 = arith.addi %290, %105 overflow<nsw> : index
        vector.store %326, %64[%327] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %328 = vector.extract_strided_slice %50 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %329 = arith.addi %294, %105 overflow<nsw> : index
        vector.store %328, %64[%329] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %330 = vector.extract_strided_slice %50 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %331 = arith.addi %298, %105 overflow<nsw> : index
        vector.store %330, %64[%331] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %332 = vector.extract_strided_slice %51 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %333 = arith.addi %286, %114 overflow<nsw> : index
        vector.store %332, %64[%333] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %334 = vector.extract_strided_slice %51 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %335 = arith.addi %290, %114 overflow<nsw> : index
        vector.store %334, %64[%335] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %336 = vector.extract_strided_slice %51 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %337 = arith.addi %294, %114 overflow<nsw> : index
        vector.store %336, %64[%337] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %338 = vector.extract_strided_slice %51 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %339 = arith.addi %298, %114 overflow<nsw> : index
        vector.store %338, %64[%339] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %340 = vector.extract_strided_slice %52 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %341 = arith.addi %286, %123 overflow<nsw> : index
        vector.store %340, %64[%341] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %342 = vector.extract_strided_slice %52 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %343 = arith.addi %290, %123 overflow<nsw> : index
        vector.store %342, %64[%343] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %344 = vector.extract_strided_slice %52 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %345 = arith.addi %294, %123 overflow<nsw> : index
        vector.store %344, %64[%345] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %346 = vector.extract_strided_slice %52 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %347 = arith.addi %298, %123 overflow<nsw> : index
        vector.store %346, %64[%347] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %348 = vector.extract_strided_slice %53 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %349 = arith.addi %286, %132 overflow<nsw> : index
        vector.store %348, %64[%349] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %350 = vector.extract_strided_slice %53 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %351 = arith.addi %290, %132 overflow<nsw> : index
        vector.store %350, %64[%351] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %352 = vector.extract_strided_slice %53 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %353 = arith.addi %294, %132 overflow<nsw> : index
        vector.store %352, %64[%353] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %354 = vector.extract_strided_slice %53 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %355 = arith.addi %298, %132 overflow<nsw> : index
        vector.store %354, %64[%355] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
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
