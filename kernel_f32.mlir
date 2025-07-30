#map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
#map1 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
#map2 = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 8) * 128)>
#map3 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map4 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map6 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
#map8 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
#map9 = affine_map<()[s0, s1, s2] -> (s1 * 128 + s2 * 256 + s0 floordiv 2 - ((s1 * 128 + s0 floordiv 2) floordiv 256) * 256)>
#map10 = affine_map<()[s0, s1] -> ((s1 * 128 + s0 floordiv 2) mod 256)>
#map11 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 2) * 8)>
#map12 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map13 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map14 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
#map15 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map16 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map17 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map18 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map19 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map20 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map21 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
#map22 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
#map23 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>
#map24 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
#map25 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
#map26 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
#map27 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
#map28 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 - (s1 floordiv 8) * 128)>
#map29 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 - (s1 floordiv 2) * 8)>
#map30 = affine_map<()[s0] -> (s0 * 256)>
#map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
#map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
#map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
#map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
#map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
#map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
#map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
#map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
#map46 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
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
        %c512_i14 = arith.constant 512 : i14
        %c-8192_i14 = arith.constant -8192 : i14
        %c2147483643_i32 = arith.constant 2147483643 : i32
        %c536870910 = arith.constant 536870910 : index
        %c16384 = arith.constant 16384 : index
        %c512 = arith.constant 512 : index
        %c2147483646_i32 = arith.constant 2147483646 : i32
        %c2147483646 = arith.constant 2147483646 : index
        %c8192 = arith.constant 8192 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %block_id_x = gpu.block_id  x upper_bound 64
        %block_id_y = gpu.block_id  y upper_bound 64
        %thread_id_x = gpu.thread_id  x upper_bound 256
        %thread_id_y = gpu.thread_id  y upper_bound 2
        %alloc = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
        %alloc_1 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
        %alloc_2 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<16384x512xi8, strided<[512, 1], offset: ?>>
        %1 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<16384x8192xi8, strided<[8192, 1], offset: ?>>
        %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<16384x512xi8, strided<[512, 1], offset: ?>>
        %3 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x8192xi8, strided<[8192, 1], offset: ?>>
        %4 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
        %5 = arith.muli %4, %c8192 overflow<nsw> : index
        %reinterpret_cast = memref.reinterpret_cast %3 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x8192xi8, strided<[8192, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %6 = amdgpu.fat_raw_buffer_cast %reinterpret_cast validBytes(%c2147483646_i32) cacheSwizzleStride(%c-8192_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %7 = affine.apply #map1()[%thread_id_x, %thread_id_y]
        %8 = affine.apply #map2()[%thread_id_x]
        %9 = affine.apply #map3()[%thread_id_x, %thread_id_y, %block_id_x]
        %10 = arith.muli %9, %c8192 overflow<nsw> : index
        %11 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %12 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
        %13 = arith.muli %12, %c8192 overflow<nsw> : index
        %14 = affine.apply #map6()[%thread_id_x, %thread_id_y]
        %15 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
        %16 = arith.muli %15, %c8192 overflow<nsw> : index
        %17 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        %18 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x]
        %19 = arith.muli %18, %c512 overflow<nsw> : index
        %reinterpret_cast_3 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x512xi8, strided<[512, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %20 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_3 validBytes(%c2147483646_i32) cacheSwizzleStride(%c512_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %21 = affine.apply #map10()[%thread_id_x, %thread_id_y]
        %22 = affine.apply #map11()[%thread_id_x]
        %23 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
        %24 = arith.muli %23, %c8192 overflow<nsw> : index
        %reinterpret_cast_4 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x8192xi8, strided<[8192, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %25 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_4 validBytes(%c2147483646_i32) cacheSwizzleStride(%c-8192_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %26 = affine.apply #map3()[%thread_id_x, %thread_id_y, %block_id_y]
        %27 = arith.muli %26, %c8192 overflow<nsw> : index
        %28 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
        %29 = arith.muli %28, %c8192 overflow<nsw> : index
        %30 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_y]
        %31 = arith.muli %30, %c8192 overflow<nsw> : index
        %32 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_y]
        %33 = arith.muli %32, %c512 overflow<nsw> : index
        %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x512xi8, strided<[512, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %34 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483646_i32) cacheSwizzleStride(%c512_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %35 = affine.apply #map12()[%thread_id_x, %thread_id_y]
        %36 = affine.apply #map13()[%thread_id_x]
        %37 = affine.apply #map14()[%thread_id_x]
        %38 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %39 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %40 = affine.apply #map17()[%thread_id_x, %thread_id_y]
        %41 = affine.apply #map18()[%thread_id_x, %thread_id_y]
        %42 = affine.apply #map19()[%thread_id_x, %thread_id_y]
        %43 = affine.apply #map20()[%thread_id_x, %thread_id_y]
        %44 = affine.apply #map21()[%thread_id_x, %thread_id_y]
        %45 = affine.apply #map22()[%thread_id_x]
        %46 = affine.apply #map23()[%thread_id_x]
        %47 = affine.apply #map24()[%thread_id_x]
        %48 = affine.apply #map25()[%thread_id_x]
        %49 = affine.apply #map26()[%thread_id_x]
        %50 = affine.apply #map27()[%thread_id_x]
        %51:32 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %354 = affine.apply #map28()[%arg5, %thread_id_x]
          %355 = arith.addi %5, %354 overflow<nsw> : index
          %356 = vector.load %6[%355] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          amdgpu.lds_barrier
          vector.store %356, %alloc_2[%7, %8] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %357 = arith.addi %10, %354 overflow<nsw> : index
          %358 = vector.load %6[%357] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %358, %alloc_2[%11, %8] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %359 = arith.addi %13, %354 overflow<nsw> : index
          %360 = vector.load %6[%359] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %360, %alloc_2[%14, %8] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %361 = arith.addi %16, %354 overflow<nsw> : index
          %362 = vector.load %6[%361] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %362, %alloc_2[%17, %8] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %363 = affine.apply #map29()[%arg5, %thread_id_x]
          %364 = arith.addi %19, %363 overflow<nsw> : index
          %365 = vector.load %20[%364] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          vector.store %365, %alloc_1[%21, %22] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<4xi8>
          %366 = arith.addi %24, %354 overflow<nsw> : index
          %367 = vector.load %25[%366] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %367, %alloc_0[%7, %8] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %368 = arith.addi %27, %354 overflow<nsw> : index
          %369 = vector.load %25[%368] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %369, %alloc_0[%11, %8] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %370 = arith.addi %29, %354 overflow<nsw> : index
          %371 = vector.load %25[%370] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %371, %alloc_0[%14, %8] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %372 = arith.addi %31, %354 overflow<nsw> : index
          %373 = vector.load %25[%372] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %373, %alloc_0[%17, %8] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %374 = arith.addi %33, %363 overflow<nsw> : index
          %375 = vector.load %34[%374] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          vector.store %375, %alloc[%21, %22] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<4xi8>
          amdgpu.lds_barrier
          %376 = vector.load %alloc[%35, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %377 = vector.load %alloc[%35, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %378 = vector.load %alloc[%38, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %379 = vector.load %alloc[%38, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %380 = vector.load %alloc[%39, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %381 = vector.load %alloc[%39, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %382 = vector.load %alloc[%40, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %383 = vector.load %alloc[%40, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %384 = vector.load %alloc[%41, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %385 = vector.load %alloc[%41, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %386 = vector.load %alloc[%42, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %387 = vector.load %alloc[%42, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %388 = vector.load %alloc[%43, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %389 = vector.load %alloc[%43, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %390 = vector.load %alloc[%44, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %391 = vector.load %alloc[%44, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %392 = vector.load %alloc_0[%35, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %393 = vector.load %alloc_0[%35, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %394 = vector.load %alloc_0[%38, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %395 = vector.load %alloc_0[%38, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %396 = vector.load %alloc_0[%39, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %397 = vector.load %alloc_0[%39, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %398 = vector.load %alloc_0[%40, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %399 = vector.load %alloc_0[%40, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %400 = vector.load %alloc_0[%41, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %401 = vector.load %alloc_0[%41, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %402 = vector.load %alloc_0[%42, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %403 = vector.load %alloc_0[%42, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %404 = vector.load %alloc_0[%43, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %405 = vector.load %alloc_0[%43, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %406 = vector.load %alloc_0[%44, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %407 = vector.load %alloc_0[%44, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %408 = vector.load %alloc_1[%47, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %409 = vector.load %alloc_1[%47, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %410 = vector.load %alloc_1[%48, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %411 = vector.load %alloc_1[%48, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %412 = vector.load %alloc_1[%49, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %413 = vector.load %alloc_1[%49, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %414 = vector.load %alloc_1[%50, %36] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %415 = vector.load %alloc_1[%50, %37] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %416 = vector.load %alloc_2[%47, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %417 = vector.load %alloc_2[%47, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %418 = vector.load %alloc_2[%48, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %419 = vector.load %alloc_2[%48, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %420 = vector.load %alloc_2[%49, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %421 = vector.load %alloc_2[%49, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %422 = vector.load %alloc_2[%50, %45] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %423 = vector.load %alloc_2[%50, %46] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %424 = vector.bitcast %416 : vector<16xi8> to vector<32xf4E2M1FN>
          %425 = vector.bitcast %417 : vector<16xi8> to vector<32xf4E2M1FN>
          %426 = vector.bitcast %418 : vector<16xi8> to vector<32xf4E2M1FN>
          %427 = vector.bitcast %419 : vector<16xi8> to vector<32xf4E2M1FN>
          %428 = vector.bitcast %420 : vector<16xi8> to vector<32xf4E2M1FN>
          %429 = vector.bitcast %421 : vector<16xi8> to vector<32xf4E2M1FN>
          %430 = vector.bitcast %422 : vector<16xi8> to vector<32xf4E2M1FN>
          %431 = vector.bitcast %423 : vector<16xi8> to vector<32xf4E2M1FN>
          %432 = vector.bitcast %408 : vector<1xi8> to vector<1xf8E8M0FNU>
          %433 = vector.bitcast %409 : vector<1xi8> to vector<1xf8E8M0FNU>
          %434 = vector.bitcast %410 : vector<1xi8> to vector<1xf8E8M0FNU>
          %435 = vector.bitcast %411 : vector<1xi8> to vector<1xf8E8M0FNU>
          %436 = vector.bitcast %412 : vector<1xi8> to vector<1xf8E8M0FNU>
          %437 = vector.bitcast %413 : vector<1xi8> to vector<1xf8E8M0FNU>
          %438 = vector.bitcast %414 : vector<1xi8> to vector<1xf8E8M0FNU>
          %439 = vector.bitcast %415 : vector<1xi8> to vector<1xf8E8M0FNU>
          %440 = vector.bitcast %392 : vector<16xi8> to vector<32xf4E2M1FN>
          %441 = vector.bitcast %393 : vector<16xi8> to vector<32xf4E2M1FN>
          %442 = vector.bitcast %394 : vector<16xi8> to vector<32xf4E2M1FN>
          %443 = vector.bitcast %395 : vector<16xi8> to vector<32xf4E2M1FN>
          %444 = vector.bitcast %396 : vector<16xi8> to vector<32xf4E2M1FN>
          %445 = vector.bitcast %397 : vector<16xi8> to vector<32xf4E2M1FN>
          %446 = vector.bitcast %398 : vector<16xi8> to vector<32xf4E2M1FN>
          %447 = vector.bitcast %399 : vector<16xi8> to vector<32xf4E2M1FN>
          %448 = vector.bitcast %400 : vector<16xi8> to vector<32xf4E2M1FN>
          %449 = vector.bitcast %401 : vector<16xi8> to vector<32xf4E2M1FN>
          %450 = vector.bitcast %402 : vector<16xi8> to vector<32xf4E2M1FN>
          %451 = vector.bitcast %403 : vector<16xi8> to vector<32xf4E2M1FN>
          %452 = vector.bitcast %404 : vector<16xi8> to vector<32xf4E2M1FN>
          %453 = vector.bitcast %405 : vector<16xi8> to vector<32xf4E2M1FN>
          %454 = vector.bitcast %406 : vector<16xi8> to vector<32xf4E2M1FN>
          %455 = vector.bitcast %407 : vector<16xi8> to vector<32xf4E2M1FN>
          %456 = vector.bitcast %376 : vector<1xi8> to vector<1xf8E8M0FNU>
          %457 = vector.bitcast %377 : vector<1xi8> to vector<1xf8E8M0FNU>
          %458 = vector.bitcast %378 : vector<1xi8> to vector<1xf8E8M0FNU>
          %459 = vector.bitcast %379 : vector<1xi8> to vector<1xf8E8M0FNU>
          %460 = vector.bitcast %380 : vector<1xi8> to vector<1xf8E8M0FNU>
          %461 = vector.bitcast %381 : vector<1xi8> to vector<1xf8E8M0FNU>
          %462 = vector.bitcast %382 : vector<1xi8> to vector<1xf8E8M0FNU>
          %463 = vector.bitcast %383 : vector<1xi8> to vector<1xf8E8M0FNU>
          %464 = vector.bitcast %384 : vector<1xi8> to vector<1xf8E8M0FNU>
          %465 = vector.bitcast %385 : vector<1xi8> to vector<1xf8E8M0FNU>
          %466 = vector.bitcast %386 : vector<1xi8> to vector<1xf8E8M0FNU>
          %467 = vector.bitcast %387 : vector<1xi8> to vector<1xf8E8M0FNU>
          %468 = vector.bitcast %388 : vector<1xi8> to vector<1xf8E8M0FNU>
          %469 = vector.bitcast %389 : vector<1xi8> to vector<1xf8E8M0FNU>
          %470 = vector.bitcast %390 : vector<1xi8> to vector<1xf8E8M0FNU>
          %471 = vector.bitcast %391 : vector<1xi8> to vector<1xf8E8M0FNU>
          %472 = vector.extractelement %432[%c0 : index] : vector<1xf8E8M0FNU>
          %473 = vector.extractelement %456[%c0 : index] : vector<1xf8E8M0FNU>
          %474 = amdgpu.scaled_mfma(%472[0] * %424) * (%473[0] * %440) + %arg6 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %475 = vector.extractelement %433[%c0 : index] : vector<1xf8E8M0FNU>
          %476 = vector.extractelement %457[%c0 : index] : vector<1xf8E8M0FNU>
          %477 = amdgpu.scaled_mfma(%475[0] * %425) * (%476[0] * %441) + %474 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %478 = vector.extractelement %458[%c0 : index] : vector<1xf8E8M0FNU>
          %479 = amdgpu.scaled_mfma(%472[0] * %424) * (%478[0] * %442) + %arg7 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %480 = vector.extractelement %459[%c0 : index] : vector<1xf8E8M0FNU>
          %481 = amdgpu.scaled_mfma(%475[0] * %425) * (%480[0] * %443) + %479 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %482 = vector.extractelement %460[%c0 : index] : vector<1xf8E8M0FNU>
          %483 = amdgpu.scaled_mfma(%472[0] * %424) * (%482[0] * %444) + %arg8 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %484 = vector.extractelement %461[%c0 : index] : vector<1xf8E8M0FNU>
          %485 = amdgpu.scaled_mfma(%475[0] * %425) * (%484[0] * %445) + %483 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %486 = vector.extractelement %462[%c0 : index] : vector<1xf8E8M0FNU>
          %487 = amdgpu.scaled_mfma(%472[0] * %424) * (%486[0] * %446) + %arg9 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %488 = vector.extractelement %463[%c0 : index] : vector<1xf8E8M0FNU>
          %489 = amdgpu.scaled_mfma(%475[0] * %425) * (%488[0] * %447) + %487 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %490 = vector.extractelement %464[%c0 : index] : vector<1xf8E8M0FNU>
          %491 = amdgpu.scaled_mfma(%472[0] * %424) * (%490[0] * %448) + %arg10 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %492 = vector.extractelement %465[%c0 : index] : vector<1xf8E8M0FNU>
          %493 = amdgpu.scaled_mfma(%475[0] * %425) * (%492[0] * %449) + %491 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %494 = vector.extractelement %466[%c0 : index] : vector<1xf8E8M0FNU>
          %495 = amdgpu.scaled_mfma(%472[0] * %424) * (%494[0] * %450) + %arg11 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %496 = vector.extractelement %467[%c0 : index] : vector<1xf8E8M0FNU>
          %497 = amdgpu.scaled_mfma(%475[0] * %425) * (%496[0] * %451) + %495 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %498 = vector.extractelement %468[%c0 : index] : vector<1xf8E8M0FNU>
          %499 = amdgpu.scaled_mfma(%472[0] * %424) * (%498[0] * %452) + %arg12 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %500 = vector.extractelement %469[%c0 : index] : vector<1xf8E8M0FNU>
          %501 = amdgpu.scaled_mfma(%475[0] * %425) * (%500[0] * %453) + %499 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %502 = vector.extractelement %470[%c0 : index] : vector<1xf8E8M0FNU>
          %503 = amdgpu.scaled_mfma(%472[0] * %424) * (%502[0] * %454) + %arg13 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %504 = vector.extractelement %471[%c0 : index] : vector<1xf8E8M0FNU>
          %505 = amdgpu.scaled_mfma(%475[0] * %425) * (%504[0] * %455) + %503 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %506 = vector.extractelement %434[%c0 : index] : vector<1xf8E8M0FNU>
          %507 = amdgpu.scaled_mfma(%506[0] * %426) * (%473[0] * %440) + %arg14 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %508 = vector.extractelement %435[%c0 : index] : vector<1xf8E8M0FNU>
          %509 = amdgpu.scaled_mfma(%508[0] * %427) * (%476[0] * %441) + %507 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %510 = amdgpu.scaled_mfma(%506[0] * %426) * (%478[0] * %442) + %arg15 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %511 = amdgpu.scaled_mfma(%508[0] * %427) * (%480[0] * %443) + %510 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %512 = amdgpu.scaled_mfma(%506[0] * %426) * (%482[0] * %444) + %arg16 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %513 = amdgpu.scaled_mfma(%508[0] * %427) * (%484[0] * %445) + %512 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %514 = amdgpu.scaled_mfma(%506[0] * %426) * (%486[0] * %446) + %arg17 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %515 = amdgpu.scaled_mfma(%508[0] * %427) * (%488[0] * %447) + %514 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %516 = amdgpu.scaled_mfma(%506[0] * %426) * (%490[0] * %448) + %arg18 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %517 = amdgpu.scaled_mfma(%508[0] * %427) * (%492[0] * %449) + %516 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %518 = amdgpu.scaled_mfma(%506[0] * %426) * (%494[0] * %450) + %arg19 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %519 = amdgpu.scaled_mfma(%508[0] * %427) * (%496[0] * %451) + %518 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %520 = amdgpu.scaled_mfma(%506[0] * %426) * (%498[0] * %452) + %arg20 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %521 = amdgpu.scaled_mfma(%508[0] * %427) * (%500[0] * %453) + %520 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %522 = amdgpu.scaled_mfma(%506[0] * %426) * (%502[0] * %454) + %arg21 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %523 = amdgpu.scaled_mfma(%508[0] * %427) * (%504[0] * %455) + %522 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %524 = vector.extractelement %436[%c0 : index] : vector<1xf8E8M0FNU>
          %525 = amdgpu.scaled_mfma(%524[0] * %428) * (%473[0] * %440) + %arg22 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %526 = vector.extractelement %437[%c0 : index] : vector<1xf8E8M0FNU>
          %527 = amdgpu.scaled_mfma(%526[0] * %429) * (%476[0] * %441) + %525 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %528 = amdgpu.scaled_mfma(%524[0] * %428) * (%478[0] * %442) + %arg23 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %529 = amdgpu.scaled_mfma(%526[0] * %429) * (%480[0] * %443) + %528 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %530 = amdgpu.scaled_mfma(%524[0] * %428) * (%482[0] * %444) + %arg24 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %531 = amdgpu.scaled_mfma(%526[0] * %429) * (%484[0] * %445) + %530 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %532 = amdgpu.scaled_mfma(%524[0] * %428) * (%486[0] * %446) + %arg25 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %533 = amdgpu.scaled_mfma(%526[0] * %429) * (%488[0] * %447) + %532 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %534 = amdgpu.scaled_mfma(%524[0] * %428) * (%490[0] * %448) + %arg26 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %535 = amdgpu.scaled_mfma(%526[0] * %429) * (%492[0] * %449) + %534 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %536 = amdgpu.scaled_mfma(%524[0] * %428) * (%494[0] * %450) + %arg27 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %537 = amdgpu.scaled_mfma(%526[0] * %429) * (%496[0] * %451) + %536 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %538 = amdgpu.scaled_mfma(%524[0] * %428) * (%498[0] * %452) + %arg28 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %539 = amdgpu.scaled_mfma(%526[0] * %429) * (%500[0] * %453) + %538 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %540 = amdgpu.scaled_mfma(%524[0] * %428) * (%502[0] * %454) + %arg29 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %541 = amdgpu.scaled_mfma(%526[0] * %429) * (%504[0] * %455) + %540 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %542 = vector.extractelement %438[%c0 : index] : vector<1xf8E8M0FNU>
          %543 = amdgpu.scaled_mfma(%542[0] * %430) * (%473[0] * %440) + %arg30 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %544 = vector.extractelement %439[%c0 : index] : vector<1xf8E8M0FNU>
          %545 = amdgpu.scaled_mfma(%544[0] * %431) * (%476[0] * %441) + %543 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %546 = amdgpu.scaled_mfma(%542[0] * %430) * (%478[0] * %442) + %arg31 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %547 = amdgpu.scaled_mfma(%544[0] * %431) * (%480[0] * %443) + %546 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %548 = amdgpu.scaled_mfma(%542[0] * %430) * (%482[0] * %444) + %arg32 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %549 = amdgpu.scaled_mfma(%544[0] * %431) * (%484[0] * %445) + %548 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %550 = amdgpu.scaled_mfma(%542[0] * %430) * (%486[0] * %446) + %arg33 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %551 = amdgpu.scaled_mfma(%544[0] * %431) * (%488[0] * %447) + %550 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %552 = amdgpu.scaled_mfma(%542[0] * %430) * (%490[0] * %448) + %arg34 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %553 = amdgpu.scaled_mfma(%544[0] * %431) * (%492[0] * %449) + %552 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %554 = amdgpu.scaled_mfma(%542[0] * %430) * (%494[0] * %450) + %arg35 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %555 = amdgpu.scaled_mfma(%544[0] * %431) * (%496[0] * %451) + %554 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %556 = amdgpu.scaled_mfma(%542[0] * %430) * (%498[0] * %452) + %arg36 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %557 = amdgpu.scaled_mfma(%544[0] * %431) * (%500[0] * %453) + %556 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %558 = amdgpu.scaled_mfma(%542[0] * %430) * (%502[0] * %454) + %arg37 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %559 = amdgpu.scaled_mfma(%544[0] * %431) * (%504[0] * %455) + %558 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          scf.yield %477, %481, %485, %489, %493, %497, %501, %505, %509, %511, %513, %515, %517, %519, %521, %523, %527, %529, %531, %533, %535, %537, %539, %541, %545, %547, %549, %551, %553, %555, %557, %559 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %52 = vector.extract_strided_slice %51#0 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %53 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<16384x16384xf32, strided<[16384, 1], offset: ?>>
        %54 = affine.apply #map30()[%block_id_x]
        %55 = affine.apply #map30()[%block_id_y]
        %56 = affine.apply #map31()[%thread_id_x]
        %57 = affine.apply #map12()[%thread_id_x, %thread_id_y]
        %58 = arith.muli %54, %c16384 overflow<nsw> : index
        %59 = arith.muli %56, %c16384 overflow<nsw> : index
        %60 = arith.addi %58, %55 overflow<nsw> : index
        %61 = arith.addi %59, %57 overflow<nsw> : index
        %reinterpret_cast_6 = memref.reinterpret_cast %53 to offset: [%60], sizes: [%c536870910], strides: [1] : memref<16384x16384xf32, strided<[16384, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %62 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_6 validBytes(%c2147483643_i32) : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        vector.store %52, %62[%61] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %63 = vector.extract_strided_slice %51#0 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %64 = affine.apply #map32()[%thread_id_x]
        %65 = arith.muli %64, %c16384 overflow<nsw> : index
        %66 = arith.addi %65, %57 overflow<nsw> : index
        vector.store %63, %62[%66] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %67 = vector.extract_strided_slice %51#0 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %68 = affine.apply #map33()[%thread_id_x]
        %69 = arith.muli %68, %c16384 overflow<nsw> : index
        %70 = arith.addi %69, %57 overflow<nsw> : index
        vector.store %67, %62[%70] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %71 = vector.extract_strided_slice %51#0 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %72 = affine.apply #map34()[%thread_id_x]
        %73 = arith.muli %72, %c16384 overflow<nsw> : index
        %74 = arith.addi %73, %57 overflow<nsw> : index
        vector.store %71, %62[%74] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %75 = vector.extract_strided_slice %51#1 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %76 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %77 = arith.addi %59, %76 overflow<nsw> : index
        vector.store %75, %62[%77] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %78 = vector.extract_strided_slice %51#1 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %79 = arith.addi %65, %76 overflow<nsw> : index
        vector.store %78, %62[%79] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %80 = vector.extract_strided_slice %51#1 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %81 = arith.addi %69, %76 overflow<nsw> : index
        vector.store %80, %62[%81] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %82 = vector.extract_strided_slice %51#1 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %83 = arith.addi %73, %76 overflow<nsw> : index
        vector.store %82, %62[%83] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %84 = vector.extract_strided_slice %51#2 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %85 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %86 = arith.addi %59, %85 overflow<nsw> : index
        vector.store %84, %62[%86] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %87 = vector.extract_strided_slice %51#2 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %88 = arith.addi %65, %85 overflow<nsw> : index
        vector.store %87, %62[%88] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %89 = vector.extract_strided_slice %51#2 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %90 = arith.addi %69, %85 overflow<nsw> : index
        vector.store %89, %62[%90] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %91 = vector.extract_strided_slice %51#2 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %92 = arith.addi %73, %85 overflow<nsw> : index
        vector.store %91, %62[%92] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %93 = vector.extract_strided_slice %51#3 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %94 = affine.apply #map17()[%thread_id_x, %thread_id_y]
        %95 = arith.addi %59, %94 overflow<nsw> : index
        vector.store %93, %62[%95] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %96 = vector.extract_strided_slice %51#3 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %97 = arith.addi %65, %94 overflow<nsw> : index
        vector.store %96, %62[%97] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %98 = vector.extract_strided_slice %51#3 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %99 = arith.addi %69, %94 overflow<nsw> : index
        vector.store %98, %62[%99] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %100 = vector.extract_strided_slice %51#3 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %101 = arith.addi %73, %94 overflow<nsw> : index
        vector.store %100, %62[%101] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %102 = vector.extract_strided_slice %51#4 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %103 = affine.apply #map18()[%thread_id_x, %thread_id_y]
        %104 = arith.addi %59, %103 overflow<nsw> : index
        vector.store %102, %62[%104] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %105 = vector.extract_strided_slice %51#4 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %106 = arith.addi %65, %103 overflow<nsw> : index
        vector.store %105, %62[%106] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %107 = vector.extract_strided_slice %51#4 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %108 = arith.addi %69, %103 overflow<nsw> : index
        vector.store %107, %62[%108] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %109 = vector.extract_strided_slice %51#4 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %110 = arith.addi %73, %103 overflow<nsw> : index
        vector.store %109, %62[%110] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %111 = vector.extract_strided_slice %51#5 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %112 = affine.apply #map19()[%thread_id_x, %thread_id_y]
        %113 = arith.addi %59, %112 overflow<nsw> : index
        vector.store %111, %62[%113] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %114 = vector.extract_strided_slice %51#5 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %115 = arith.addi %65, %112 overflow<nsw> : index
        vector.store %114, %62[%115] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %116 = vector.extract_strided_slice %51#5 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %117 = arith.addi %69, %112 overflow<nsw> : index
        vector.store %116, %62[%117] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %118 = vector.extract_strided_slice %51#5 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %119 = arith.addi %73, %112 overflow<nsw> : index
        vector.store %118, %62[%119] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %120 = vector.extract_strided_slice %51#6 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %121 = affine.apply #map20()[%thread_id_x, %thread_id_y]
        %122 = arith.addi %59, %121 overflow<nsw> : index
        vector.store %120, %62[%122] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %123 = vector.extract_strided_slice %51#6 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %124 = arith.addi %65, %121 overflow<nsw> : index
        vector.store %123, %62[%124] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %125 = vector.extract_strided_slice %51#6 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %126 = arith.addi %69, %121 overflow<nsw> : index
        vector.store %125, %62[%126] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %127 = vector.extract_strided_slice %51#6 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %128 = arith.addi %73, %121 overflow<nsw> : index
        vector.store %127, %62[%128] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %129 = vector.extract_strided_slice %51#7 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %130 = affine.apply #map21()[%thread_id_x, %thread_id_y]
        %131 = arith.addi %59, %130 overflow<nsw> : index
        vector.store %129, %62[%131] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %132 = vector.extract_strided_slice %51#7 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %133 = arith.addi %65, %130 overflow<nsw> : index
        vector.store %132, %62[%133] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %134 = vector.extract_strided_slice %51#7 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %135 = arith.addi %69, %130 overflow<nsw> : index
        vector.store %134, %62[%135] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %136 = vector.extract_strided_slice %51#7 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %137 = arith.addi %73, %130 overflow<nsw> : index
        vector.store %136, %62[%137] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %138 = vector.extract_strided_slice %51#8 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %139 = affine.apply #map35()[%thread_id_x]
        %140 = arith.muli %139, %c16384 overflow<nsw> : index
        %141 = arith.addi %140, %57 overflow<nsw> : index
        vector.store %138, %62[%141] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %142 = vector.extract_strided_slice %51#8 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %143 = affine.apply #map36()[%thread_id_x]
        %144 = arith.muli %143, %c16384 overflow<nsw> : index
        %145 = arith.addi %144, %57 overflow<nsw> : index
        vector.store %142, %62[%145] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %146 = vector.extract_strided_slice %51#8 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %147 = affine.apply #map37()[%thread_id_x]
        %148 = arith.muli %147, %c16384 overflow<nsw> : index
        %149 = arith.addi %148, %57 overflow<nsw> : index
        vector.store %146, %62[%149] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %150 = vector.extract_strided_slice %51#8 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %151 = affine.apply #map38()[%thread_id_x]
        %152 = arith.muli %151, %c16384 overflow<nsw> : index
        %153 = arith.addi %152, %57 overflow<nsw> : index
        vector.store %150, %62[%153] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %154 = vector.extract_strided_slice %51#9 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %155 = arith.addi %140, %76 overflow<nsw> : index
        vector.store %154, %62[%155] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %156 = vector.extract_strided_slice %51#9 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %157 = arith.addi %144, %76 overflow<nsw> : index
        vector.store %156, %62[%157] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %158 = vector.extract_strided_slice %51#9 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %159 = arith.addi %148, %76 overflow<nsw> : index
        vector.store %158, %62[%159] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %160 = vector.extract_strided_slice %51#9 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %161 = arith.addi %152, %76 overflow<nsw> : index
        vector.store %160, %62[%161] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %162 = vector.extract_strided_slice %51#10 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %163 = arith.addi %140, %85 overflow<nsw> : index
        vector.store %162, %62[%163] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %164 = vector.extract_strided_slice %51#10 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %165 = arith.addi %144, %85 overflow<nsw> : index
        vector.store %164, %62[%165] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %166 = vector.extract_strided_slice %51#10 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %167 = arith.addi %148, %85 overflow<nsw> : index
        vector.store %166, %62[%167] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %168 = vector.extract_strided_slice %51#10 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %169 = arith.addi %152, %85 overflow<nsw> : index
        vector.store %168, %62[%169] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %170 = vector.extract_strided_slice %51#11 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %171 = arith.addi %140, %94 overflow<nsw> : index
        vector.store %170, %62[%171] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %172 = vector.extract_strided_slice %51#11 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %173 = arith.addi %144, %94 overflow<nsw> : index
        vector.store %172, %62[%173] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %174 = vector.extract_strided_slice %51#11 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %175 = arith.addi %148, %94 overflow<nsw> : index
        vector.store %174, %62[%175] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %176 = vector.extract_strided_slice %51#11 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %177 = arith.addi %152, %94 overflow<nsw> : index
        vector.store %176, %62[%177] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %178 = vector.extract_strided_slice %51#12 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %179 = arith.addi %140, %103 overflow<nsw> : index
        vector.store %178, %62[%179] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %180 = vector.extract_strided_slice %51#12 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %181 = arith.addi %144, %103 overflow<nsw> : index
        vector.store %180, %62[%181] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %182 = vector.extract_strided_slice %51#12 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %183 = arith.addi %148, %103 overflow<nsw> : index
        vector.store %182, %62[%183] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %184 = vector.extract_strided_slice %51#12 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %185 = arith.addi %152, %103 overflow<nsw> : index
        vector.store %184, %62[%185] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %186 = vector.extract_strided_slice %51#13 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %187 = arith.addi %140, %112 overflow<nsw> : index
        vector.store %186, %62[%187] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %188 = vector.extract_strided_slice %51#13 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %189 = arith.addi %144, %112 overflow<nsw> : index
        vector.store %188, %62[%189] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %190 = vector.extract_strided_slice %51#13 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %191 = arith.addi %148, %112 overflow<nsw> : index
        vector.store %190, %62[%191] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %192 = vector.extract_strided_slice %51#13 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %193 = arith.addi %152, %112 overflow<nsw> : index
        vector.store %192, %62[%193] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %194 = vector.extract_strided_slice %51#14 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %195 = arith.addi %140, %121 overflow<nsw> : index
        vector.store %194, %62[%195] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %196 = vector.extract_strided_slice %51#14 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %197 = arith.addi %144, %121 overflow<nsw> : index
        vector.store %196, %62[%197] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %198 = vector.extract_strided_slice %51#14 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %199 = arith.addi %148, %121 overflow<nsw> : index
        vector.store %198, %62[%199] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %200 = vector.extract_strided_slice %51#14 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %201 = arith.addi %152, %121 overflow<nsw> : index
        vector.store %200, %62[%201] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %202 = vector.extract_strided_slice %51#15 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %203 = arith.addi %140, %130 overflow<nsw> : index
        vector.store %202, %62[%203] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %204 = vector.extract_strided_slice %51#15 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %205 = arith.addi %144, %130 overflow<nsw> : index
        vector.store %204, %62[%205] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %206 = vector.extract_strided_slice %51#15 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %207 = arith.addi %148, %130 overflow<nsw> : index
        vector.store %206, %62[%207] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %208 = vector.extract_strided_slice %51#15 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %209 = arith.addi %152, %130 overflow<nsw> : index
        vector.store %208, %62[%209] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %210 = vector.extract_strided_slice %51#16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %211 = affine.apply #map39()[%thread_id_x]
        %212 = arith.muli %211, %c16384 overflow<nsw> : index
        %213 = arith.addi %212, %57 overflow<nsw> : index
        vector.store %210, %62[%213] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %214 = vector.extract_strided_slice %51#16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %215 = affine.apply #map40()[%thread_id_x]
        %216 = arith.muli %215, %c16384 overflow<nsw> : index
        %217 = arith.addi %216, %57 overflow<nsw> : index
        vector.store %214, %62[%217] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %218 = vector.extract_strided_slice %51#16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %219 = affine.apply #map41()[%thread_id_x]
        %220 = arith.muli %219, %c16384 overflow<nsw> : index
        %221 = arith.addi %220, %57 overflow<nsw> : index
        vector.store %218, %62[%221] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %222 = vector.extract_strided_slice %51#16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %223 = affine.apply #map42()[%thread_id_x]
        %224 = arith.muli %223, %c16384 overflow<nsw> : index
        %225 = arith.addi %224, %57 overflow<nsw> : index
        vector.store %222, %62[%225] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %226 = vector.extract_strided_slice %51#17 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %227 = arith.addi %212, %76 overflow<nsw> : index
        vector.store %226, %62[%227] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %228 = vector.extract_strided_slice %51#17 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %229 = arith.addi %216, %76 overflow<nsw> : index
        vector.store %228, %62[%229] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %230 = vector.extract_strided_slice %51#17 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %231 = arith.addi %220, %76 overflow<nsw> : index
        vector.store %230, %62[%231] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %232 = vector.extract_strided_slice %51#17 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %233 = arith.addi %224, %76 overflow<nsw> : index
        vector.store %232, %62[%233] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %234 = vector.extract_strided_slice %51#18 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %235 = arith.addi %212, %85 overflow<nsw> : index
        vector.store %234, %62[%235] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %236 = vector.extract_strided_slice %51#18 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %237 = arith.addi %216, %85 overflow<nsw> : index
        vector.store %236, %62[%237] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %238 = vector.extract_strided_slice %51#18 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %239 = arith.addi %220, %85 overflow<nsw> : index
        vector.store %238, %62[%239] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %240 = vector.extract_strided_slice %51#18 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %241 = arith.addi %224, %85 overflow<nsw> : index
        vector.store %240, %62[%241] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %242 = vector.extract_strided_slice %51#19 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %243 = arith.addi %212, %94 overflow<nsw> : index
        vector.store %242, %62[%243] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %244 = vector.extract_strided_slice %51#19 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %245 = arith.addi %216, %94 overflow<nsw> : index
        vector.store %244, %62[%245] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %246 = vector.extract_strided_slice %51#19 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %247 = arith.addi %220, %94 overflow<nsw> : index
        vector.store %246, %62[%247] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %248 = vector.extract_strided_slice %51#19 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %249 = arith.addi %224, %94 overflow<nsw> : index
        vector.store %248, %62[%249] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %250 = vector.extract_strided_slice %51#20 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %251 = arith.addi %212, %103 overflow<nsw> : index
        vector.store %250, %62[%251] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %252 = vector.extract_strided_slice %51#20 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %253 = arith.addi %216, %103 overflow<nsw> : index
        vector.store %252, %62[%253] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %254 = vector.extract_strided_slice %51#20 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %255 = arith.addi %220, %103 overflow<nsw> : index
        vector.store %254, %62[%255] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %256 = vector.extract_strided_slice %51#20 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %257 = arith.addi %224, %103 overflow<nsw> : index
        vector.store %256, %62[%257] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %258 = vector.extract_strided_slice %51#21 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %259 = arith.addi %212, %112 overflow<nsw> : index
        vector.store %258, %62[%259] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %260 = vector.extract_strided_slice %51#21 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %261 = arith.addi %216, %112 overflow<nsw> : index
        vector.store %260, %62[%261] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %262 = vector.extract_strided_slice %51#21 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %263 = arith.addi %220, %112 overflow<nsw> : index
        vector.store %262, %62[%263] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %264 = vector.extract_strided_slice %51#21 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %265 = arith.addi %224, %112 overflow<nsw> : index
        vector.store %264, %62[%265] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %266 = vector.extract_strided_slice %51#22 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %267 = arith.addi %212, %121 overflow<nsw> : index
        vector.store %266, %62[%267] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %268 = vector.extract_strided_slice %51#22 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %269 = arith.addi %216, %121 overflow<nsw> : index
        vector.store %268, %62[%269] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %270 = vector.extract_strided_slice %51#22 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %271 = arith.addi %220, %121 overflow<nsw> : index
        vector.store %270, %62[%271] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %272 = vector.extract_strided_slice %51#22 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %273 = arith.addi %224, %121 overflow<nsw> : index
        vector.store %272, %62[%273] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %274 = vector.extract_strided_slice %51#23 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %275 = arith.addi %212, %130 overflow<nsw> : index
        vector.store %274, %62[%275] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %276 = vector.extract_strided_slice %51#23 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %277 = arith.addi %216, %130 overflow<nsw> : index
        vector.store %276, %62[%277] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %278 = vector.extract_strided_slice %51#23 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %279 = arith.addi %220, %130 overflow<nsw> : index
        vector.store %278, %62[%279] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %280 = vector.extract_strided_slice %51#23 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %281 = arith.addi %224, %130 overflow<nsw> : index
        vector.store %280, %62[%281] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %282 = vector.extract_strided_slice %51#24 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %283 = affine.apply #map43()[%thread_id_x]
        %284 = arith.muli %283, %c16384 overflow<nsw> : index
        %285 = arith.addi %284, %57 overflow<nsw> : index
        vector.store %282, %62[%285] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %286 = vector.extract_strided_slice %51#24 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %287 = affine.apply #map44()[%thread_id_x]
        %288 = arith.muli %287, %c16384 overflow<nsw> : index
        %289 = arith.addi %288, %57 overflow<nsw> : index
        vector.store %286, %62[%289] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %290 = vector.extract_strided_slice %51#24 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %291 = affine.apply #map45()[%thread_id_x]
        %292 = arith.muli %291, %c16384 overflow<nsw> : index
        %293 = arith.addi %292, %57 overflow<nsw> : index
        vector.store %290, %62[%293] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %294 = vector.extract_strided_slice %51#24 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %295 = affine.apply #map46()[%thread_id_x]
        %296 = arith.muli %295, %c16384 overflow<nsw> : index
        %297 = arith.addi %296, %57 overflow<nsw> : index
        vector.store %294, %62[%297] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %298 = vector.extract_strided_slice %51#25 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %299 = arith.addi %284, %76 overflow<nsw> : index
        vector.store %298, %62[%299] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %300 = vector.extract_strided_slice %51#25 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %301 = arith.addi %288, %76 overflow<nsw> : index
        vector.store %300, %62[%301] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %302 = vector.extract_strided_slice %51#25 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %303 = arith.addi %292, %76 overflow<nsw> : index
        vector.store %302, %62[%303] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %304 = vector.extract_strided_slice %51#25 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %305 = arith.addi %296, %76 overflow<nsw> : index
        vector.store %304, %62[%305] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %306 = vector.extract_strided_slice %51#26 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %307 = arith.addi %284, %85 overflow<nsw> : index
        vector.store %306, %62[%307] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %308 = vector.extract_strided_slice %51#26 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %309 = arith.addi %288, %85 overflow<nsw> : index
        vector.store %308, %62[%309] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %310 = vector.extract_strided_slice %51#26 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %311 = arith.addi %292, %85 overflow<nsw> : index
        vector.store %310, %62[%311] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %312 = vector.extract_strided_slice %51#26 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %313 = arith.addi %296, %85 overflow<nsw> : index
        vector.store %312, %62[%313] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %314 = vector.extract_strided_slice %51#27 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %315 = arith.addi %284, %94 overflow<nsw> : index
        vector.store %314, %62[%315] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %316 = vector.extract_strided_slice %51#27 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %317 = arith.addi %288, %94 overflow<nsw> : index
        vector.store %316, %62[%317] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %318 = vector.extract_strided_slice %51#27 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %319 = arith.addi %292, %94 overflow<nsw> : index
        vector.store %318, %62[%319] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %320 = vector.extract_strided_slice %51#27 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %321 = arith.addi %296, %94 overflow<nsw> : index
        vector.store %320, %62[%321] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %322 = vector.extract_strided_slice %51#28 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %323 = arith.addi %284, %103 overflow<nsw> : index
        vector.store %322, %62[%323] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %324 = vector.extract_strided_slice %51#28 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %325 = arith.addi %288, %103 overflow<nsw> : index
        vector.store %324, %62[%325] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %326 = vector.extract_strided_slice %51#28 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %327 = arith.addi %292, %103 overflow<nsw> : index
        vector.store %326, %62[%327] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %328 = vector.extract_strided_slice %51#28 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %329 = arith.addi %296, %103 overflow<nsw> : index
        vector.store %328, %62[%329] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %330 = vector.extract_strided_slice %51#29 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %331 = arith.addi %284, %112 overflow<nsw> : index
        vector.store %330, %62[%331] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %332 = vector.extract_strided_slice %51#29 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %333 = arith.addi %288, %112 overflow<nsw> : index
        vector.store %332, %62[%333] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %334 = vector.extract_strided_slice %51#29 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %335 = arith.addi %292, %112 overflow<nsw> : index
        vector.store %334, %62[%335] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %336 = vector.extract_strided_slice %51#29 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %337 = arith.addi %296, %112 overflow<nsw> : index
        vector.store %336, %62[%337] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %338 = vector.extract_strided_slice %51#30 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %339 = arith.addi %284, %121 overflow<nsw> : index
        vector.store %338, %62[%339] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %340 = vector.extract_strided_slice %51#30 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %341 = arith.addi %288, %121 overflow<nsw> : index
        vector.store %340, %62[%341] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %342 = vector.extract_strided_slice %51#30 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %343 = arith.addi %292, %121 overflow<nsw> : index
        vector.store %342, %62[%343] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %344 = vector.extract_strided_slice %51#30 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %345 = arith.addi %296, %121 overflow<nsw> : index
        vector.store %344, %62[%345] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %346 = vector.extract_strided_slice %51#31 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %347 = arith.addi %284, %130 overflow<nsw> : index
        vector.store %346, %62[%347] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %348 = vector.extract_strided_slice %51#31 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %349 = arith.addi %288, %130 overflow<nsw> : index
        vector.store %348, %62[%349] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %350 = vector.extract_strided_slice %51#31 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %351 = arith.addi %292, %130 overflow<nsw> : index
        vector.store %350, %62[%351] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %352 = vector.extract_strided_slice %51#31 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %353 = arith.addi %296, %130 overflow<nsw> : index
        vector.store %352, %62[%353] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<16384x8192xi8>, %arg1: tensor<16384x512xi8>, %arg2: tensor<16384x8192xi8>, %arg3: tensor<16384x512xi8>, %arg4: tensor<16384x16384xf32>) -> tensor<16384x16384xf32> {
    %0 = flow.dispatch @gemm_afp4_wfp4_wave::@gemm_afp4_wfp4_wave(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x16384xf32>) -> %arg4
    return %0 : tensor<16384x16384xf32>
  }
}
