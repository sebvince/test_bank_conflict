#map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>

#map2 = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 8) * 128)>
#map3 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>

#map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>

#map1 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
#map4 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map6 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map8 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>



#map10 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
#map11 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>

#map9 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map12 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map13 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map14 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map15 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map16 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map17 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map18 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>

#map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
#map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
#map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
#map22 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>

#map23 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 - (s1 floordiv 8) * 128)>
#map24 = affine_map<()[s0] -> (s0 * 256)>
#map25 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
#map26 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map27 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map28 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
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
        %cst = arith.constant 5.877470e-39 : f8E8M0FNU
        %c-8192_i14 = arith.constant -8192 : i14
        %c2147483643_i32 = arith.constant 2147483643 : i32
        %c536870910 = arith.constant 536870910 : index
        %c16384 = arith.constant 16384 : index
        %c2147483646_i32 = arith.constant 2147483646 : i32
        %c2147483646 = arith.constant 2147483646 : index
        %c8192 = arith.constant 8192 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
        %block_id_x = gpu.block_id  x upper_bound 64
        %block_id_y = gpu.block_id  y upper_bound 64
        %thread_id_x = gpu.thread_id  x upper_bound 256
        %thread_id_y = gpu.thread_id  y upper_bound 2
        %alloc = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
        %alloc_1 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<16384x8192xi8, strided<[8192, 1], offset: ?>>
        %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x8192xi8, strided<[8192, 1], offset: ?>>
        %2 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
        %3 = arith.muli %2, %c8192 overflow<nsw> : index
        %reinterpret_cast = memref.reinterpret_cast %1 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x8192xi8, strided<[8192, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %4 = amdgpu.fat_raw_buffer_cast %reinterpret_cast validBytes(%c2147483646_i32) cacheSwizzleStride(%c-8192_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %5 = affine.apply #map1()[%thread_id_x, %thread_id_y]
        %6 = affine.apply #map2()[%thread_id_x]
        %7 = affine.apply #map3()[%thread_id_x, %thread_id_y, %block_id_x]
        %8 = arith.muli %7, %c8192 overflow<nsw> : index
        %9 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %10 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
        %11 = arith.muli %10, %c8192 overflow<nsw> : index
        %12 = affine.apply #map6()[%thread_id_x, %thread_id_y]
        %13 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
        %14 = arith.muli %13, %c8192 overflow<nsw> : index
        %15 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        %16 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
        %17 = arith.muli %16, %c8192 overflow<nsw> : index
        %reinterpret_cast_2 = memref.reinterpret_cast %0 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x8192xi8, strided<[8192, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %18 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_2 validBytes(%c2147483646_i32) cacheSwizzleStride(%c-8192_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %19 = affine.apply #map3()[%thread_id_x, %thread_id_y, %block_id_y]
        %20 = arith.muli %19, %c8192 overflow<nsw> : index
        %21 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
        %22 = arith.muli %21, %c8192 overflow<nsw> : index
        %23 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_y]
        %24 = arith.muli %23, %c8192 overflow<nsw> : index
        %25 = affine.apply #map9()[%thread_id_x, %thread_id_y]
        %26 = affine.apply #map10()[%thread_id_x]
        %27 = affine.apply #map11()[%thread_id_x]
        %28 = affine.apply #map12()[%thread_id_x, %thread_id_y]
        %29 = affine.apply #map13()[%thread_id_x, %thread_id_y]
        %30 = affine.apply #map14()[%thread_id_x, %thread_id_y]
        %31 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %32 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %33 = affine.apply #map17()[%thread_id_x, %thread_id_y]
        %34 = affine.apply #map18()[%thread_id_x, %thread_id_y]
        %35 = affine.apply #map19()[%thread_id_x]
        %36 = affine.apply #map20()[%thread_id_x]
        %37 = affine.apply #map21()[%thread_id_x]
        %38 = affine.apply #map22()[%thread_id_x]
        %39:32 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %cst_0, %arg7 = %cst_0, %arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_0, %arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0, %arg17 = %cst_0, %arg18 = %cst_0, %arg19 = %cst_0, %arg20 = %cst_0, %arg21 = %cst_0, %arg22 = %cst_0, %arg23 = %cst_0, %arg24 = %cst_0, %arg25 = %cst_0, %arg26 = %cst_0, %arg27 = %cst_0, %arg28 = %cst_0, %arg29 = %cst_0, %arg30 = %cst_0, %arg31 = %cst_0, %arg32 = %cst_0, %arg33 = %cst_0, %arg34 = %cst_0, %arg35 = %cst_0, %arg36 = %cst_0, %arg37 = %cst_0) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %342 = affine.apply #map23()[%arg5, %thread_id_x]
          %343 = arith.addi %3, %342 overflow<nsw> : index
          %344 = vector.load %4[%343] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          amdgpu.lds_barrier

          //%6
          // #map2 = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 8) * 128)>
          %store_col = affine.apply affine_map<()[s0] -> (s0 - (s0 floordiv 8) * 8)>()[%thread_id_x]

          //Store B
          // #map1 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
          // #map4 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
          // #map6 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
          // #map8 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>

          %store_b_row_0 = affine.apply affine_map<()[s0,s1] -> (((s1 * 32 + s0 floordiv 8) mod 256) mod 8)>()[%thread_id_x, %thread_id_y]
          %store_b_row_1 = affine.apply affine_map<()[s0,s1] -> (((s1 * 32 + s0 floordiv 8) mod 256 + 64) mod 8)>()[%thread_id_x, %thread_id_y]
          %store_b_row_2 = affine.apply affine_map<()[s0,s1] -> (((s1 * 32 + s0 floordiv 8) mod 256 + 128) mod 8)>()[%thread_id_x, %thread_id_y]
          %store_b_row_3 = affine.apply affine_map<()[s0,s1] -> (((s1 * 32 + s0 floordiv 8) mod 256 + 192) mod 8)>()[%thread_id_x, %thread_id_y]

          %store_b_col_0_swizzle = arith.xori %store_b_row_0, %store_col : index 
          %store_b_col_1_swizzle = arith.xori %store_b_row_1, %store_col : index 
          %store_b_col_2_swizzle = arith.xori %store_b_row_2, %store_col : index 
          %store_b_col_3_swizzle = arith.xori %store_b_row_3, %store_col : index 

          %store_b_col_0_swizzle_b = arith.muli %store_b_col_0_swizzle,%c16 : index 
          %store_b_col_1_swizzle_b = arith.muli %store_b_col_1_swizzle,%c16 : index 
          %store_b_col_2_swizzle_b = arith.muli %store_b_col_2_swizzle,%c16 : index 
          %store_b_col_3_swizzle_b = arith.muli %store_b_col_3_swizzle,%c16 : index 


          vector.store %344, %alloc_1[%5, %store_b_col_0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %345 = arith.addi %8, %342 overflow<nsw> : index
          %346 = vector.load %4[%345] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %346, %alloc_1[%9, %store_b_col_1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %347 = arith.addi %11, %342 overflow<nsw> : index
          %348 = vector.load %4[%347] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %348, %alloc_1[%12, %store_b_col_2_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %349 = arith.addi %14, %342 overflow<nsw> : index
          %350 = vector.load %4[%349] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %350, %alloc_1[%15, %store_b_col_3_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>


          %351 = arith.addi %17, %342 overflow<nsw> : index
          %352 = vector.load %18[%351] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          
          %store_a_row_0 = affine.apply affine_map<()[s0,s1] -> ((s1 * 32 + s0 floordiv 8) mod 8)>()[%thread_id_x, %thread_id_y]
          %store_a_row_1 = affine.apply affine_map<()[s0,s1] -> ((s1 * 32 + s0 floordiv 8 + 64) mod 8)>()[%thread_id_x, %thread_id_y]
          %store_a_row_2 = affine.apply affine_map<()[s0,s1] -> ((s1 * 32 + s0 floordiv 8 + 128) mod 8)>()[%thread_id_x, %thread_id_y]
          %store_a_row_3 = affine.apply affine_map<()[s0,s1] -> ((s1 * 32 + s0 floordiv 8 + 192) mod 8)>()[%thread_id_x, %thread_id_y]

          %store_a_col_0_swizzle = arith.xori %store_a_row_0, %store_col : index 
          %store_a_col_1_swizzle = arith.xori %store_a_row_1, %store_col : index 
          %store_a_col_2_swizzle = arith.xori %store_a_row_2, %store_col : index 
          %store_a_col_3_swizzle = arith.xori %store_a_row_3, %store_col : index 
          // #map1 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
          // #map4 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
          // #map6 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
          // #map8 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>

          
          %store_a_col_0_swizzle_b = arith.muli %store_a_col_0_swizzle,%c16 : index 
          %store_a_col_1_swizzle_b = arith.muli %store_a_col_1_swizzle,%c16 : index 
          %store_a_col_2_swizzle_b = arith.muli %store_a_col_2_swizzle,%c16 : index 
          %store_a_col_3_swizzle_b = arith.muli %store_a_col_3_swizzle,%c16 : index 

          vector.store %352, %alloc[%5, %store_a_col_0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %353 = arith.addi %20, %342 overflow<nsw> : index
          %354 = vector.load %18[%353] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %354, %alloc[%9, %store_a_col_1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %355 = arith.addi %22, %342 overflow<nsw> : index
          %356 = vector.load %18[%355] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %356, %alloc[%12, %store_a_col_2_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %357 = arith.addi %24, %342 overflow<nsw> : index
          %358 = vector.load %18[%357] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          vector.store %358, %alloc[%15, %store_a_col_3_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>

          amdgpu.lds_barrier

          // #map10 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
          // #map11 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>

          // Get col in element (not byte offet)
          %col_0 = affine.apply affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>()[%thread_id_x]
          %col_1 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 16)+4)>()[%thread_id_x]


          // #map9 = affine_map<()[s0, s1] ->  (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
          // #map12 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
          // #map13 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
          // #map14 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
          // #map15 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
          // #map16 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
          // #map17 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
          // #map18 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>

          //A. 8 rows
          // Use maxPhase of 8
          %row_0 = affine.apply affine_map<()[s0,s1] -> ((s0 + s1 * 128 - (s0 floordiv 16) * 16)       mod 8)>()[%thread_id_x, %thread_id_y]
          %row_1 = affine.apply affine_map<()[s0,s1] -> ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)  mod 8)>()[%thread_id_x, %thread_id_y]
          %row_2 = affine.apply affine_map<()[s0,s1] -> ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)  mod 8)>()[%thread_id_x, %thread_id_y]
          %row_3 = affine.apply affine_map<()[s0,s1] -> ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)  mod 8)>()[%thread_id_x, %thread_id_y]
          %row_4 = affine.apply affine_map<()[s0,s1] -> ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)  mod 8)>()[%thread_id_x, %thread_id_y]
          %row_5 = affine.apply affine_map<()[s0,s1] -> ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)  mod 8)>()[%thread_id_x, %thread_id_y]
          %row_6 = affine.apply affine_map<()[s0,s1] -> ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)  mod 8)>()[%thread_id_x, %thread_id_y]
          %row_7 = affine.apply affine_map<()[s0,s1] -> ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112) mod 8)>()[%thread_id_x, %thread_id_y]


          //XOR swizzling
          %r0_c0_swizzle = arith.xori %row_0,%col_0 : index 
          %r0_c1_swizzle = arith.xori %row_0,%col_1 : index 
          %r1_c0_swizzle = arith.xori %row_1,%col_0 : index 
          %r1_c1_swizzle = arith.xori %row_1,%col_1 : index 
          %r2_c0_swizzle = arith.xori %row_2,%col_0 : index 
          %r2_c1_swizzle = arith.xori %row_2,%col_1 : index 
          %r3_c0_swizzle = arith.xori %row_3,%col_0 : index 
          %r3_c1_swizzle = arith.xori %row_3,%col_1 : index 
          %r4_c0_swizzle = arith.xori %row_4,%col_0 : index 
          %r4_c1_swizzle = arith.xori %row_4,%col_1 : index 
          %r5_c0_swizzle = arith.xori %row_5,%col_0 : index 
          %r5_c1_swizzle = arith.xori %row_5,%col_1 : index 
          %r6_c0_swizzle = arith.xori %row_6,%col_0 : index 
          %r6_c1_swizzle = arith.xori %row_6,%col_1 : index 
          %r7_c0_swizzle = arith.xori %row_7,%col_0 : index 
          %r7_c1_swizzle = arith.xori %row_7,%col_1 : index 

          //Back to byte offset
          
          %r0_c0_swizzle_b = arith.muli %r0_c0_swizzle,%c16 : index 
          %r0_c1_swizzle_b = arith.muli %r0_c1_swizzle,%c16 : index 
          %r1_c0_swizzle_b = arith.muli %r1_c0_swizzle,%c16 : index 
          %r1_c1_swizzle_b = arith.muli %r1_c1_swizzle,%c16 : index 
          %r2_c0_swizzle_b = arith.muli %r2_c0_swizzle,%c16 : index 
          %r2_c1_swizzle_b = arith.muli %r2_c1_swizzle,%c16 : index 
          %r3_c0_swizzle_b = arith.muli %r3_c0_swizzle,%c16 : index 
          %r3_c1_swizzle_b = arith.muli %r3_c1_swizzle,%c16 : index 
          %r4_c0_swizzle_b = arith.muli %r4_c0_swizzle,%c16 : index 
          %r4_c1_swizzle_b = arith.muli %r4_c1_swizzle,%c16 : index 
          %r5_c0_swizzle_b = arith.muli %r5_c0_swizzle,%c16 : index 
          %r5_c1_swizzle_b = arith.muli %r5_c1_swizzle,%c16 : index 
          %r6_c0_swizzle_b = arith.muli %r6_c0_swizzle,%c16 : index 
          %r6_c1_swizzle_b = arith.muli %r6_c1_swizzle,%c16 : index 
          %r7_c0_swizzle_b = arith.muli %r7_c0_swizzle,%c16 : index 
          %r7_c1_swizzle_b = arith.muli %r7_c1_swizzle,%c16 : index 

          %359 = vector.load %alloc[%25, %r0_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %360 = vector.load %alloc[%25, %r0_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %361 = vector.load %alloc[%28, %r1_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %362 = vector.load %alloc[%28, %r1_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %363 = vector.load %alloc[%29, %r2_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %364 = vector.load %alloc[%29, %r2_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %365 = vector.load %alloc[%30, %r3_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %366 = vector.load %alloc[%30, %r3_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %367 = vector.load %alloc[%31, %r4_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %368 = vector.load %alloc[%31, %r4_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %369 = vector.load %alloc[%32, %r5_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %370 = vector.load %alloc[%32, %r5_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %371 = vector.load %alloc[%33, %r6_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %372 = vector.load %alloc[%33, %r6_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %373 = vector.load %alloc[%34, %r7_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %374 = vector.load %alloc[%34, %r7_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>



          // %359 = vector.load %alloc[%25, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %360 = vector.load %alloc[%25, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %361 = vector.load %alloc[%28, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %362 = vector.load %alloc[%28, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %363 = vector.load %alloc[%29, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %364 = vector.load %alloc[%29, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %365 = vector.load %alloc[%30, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %366 = vector.load %alloc[%30, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %367 = vector.load %alloc[%31, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %368 = vector.load %alloc[%31, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %369 = vector.load %alloc[%32, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %370 = vector.load %alloc[%32, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %371 = vector.load %alloc[%33, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %372 = vector.load %alloc[%33, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %373 = vector.load %alloc[%34, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %374 = vector.load %alloc[%34, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>

          //Load B
          // #map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64     )>
          // #map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
          // #map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
          // #map22 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
          // Use maxPhase of 8
          %row_b_0 = affine.apply affine_map<()[s0] -> ((s0 mod 16 + (s0 floordiv 64) * 64     )  mod 8)>()[%thread_id_x]
          %row_b_1 = affine.apply affine_map<()[s0] -> ((s0 mod 16 + (s0 floordiv 64) * 64 + 16)  mod 8)>()[%thread_id_x]
          %row_b_2 = affine.apply affine_map<()[s0] -> ((s0 mod 16 + (s0 floordiv 64) * 64 + 32)  mod 8)>()[%thread_id_x]
          %row_b_3 = affine.apply affine_map<()[s0] -> ((s0 mod 16 + (s0 floordiv 64) * 64 + 48)  mod 8)>()[%thread_id_x]
          

          //XOR swizzling
          %rb_0_c0_swizzle = arith.xori %row_b_0,%col_0 : index 
          %rb_0_c1_swizzle = arith.xori %row_b_0,%col_1 : index 
          %rb_1_c0_swizzle = arith.xori %row_b_1,%col_0 : index 
          %rb_1_c1_swizzle = arith.xori %row_b_1,%col_1 : index 
          %rb_2_c0_swizzle = arith.xori %row_b_2,%col_0 : index 
          %rb_2_c1_swizzle = arith.xori %row_b_2,%col_1 : index 
          %rb_3_c0_swizzle = arith.xori %row_b_3,%col_0 : index 
          %rb_3_c1_swizzle = arith.xori %row_b_3,%col_1 : index 


          %rb_0_c0_swizzle_b = arith.muli %rb_0_c0_swizzle, %c16 : index 
          %rb_0_c1_swizzle_b = arith.muli %rb_0_c1_swizzle, %c16 : index 
          %rb_1_c0_swizzle_b = arith.muli %rb_1_c0_swizzle, %c16 : index 
          %rb_1_c1_swizzle_b = arith.muli %rb_1_c1_swizzle, %c16 : index 
          %rb_2_c0_swizzle_b = arith.muli %rb_2_c0_swizzle, %c16 : index 
          %rb_2_c1_swizzle_b = arith.muli %rb_2_c1_swizzle, %c16 : index 
          %rb_3_c0_swizzle_b = arith.muli %rb_3_c0_swizzle, %c16 : index 
          %rb_3_c1_swizzle_b = arith.muli %rb_3_c1_swizzle, %c16 : index 

          %375 = vector.load %alloc_1[%35, %rb_0_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %376 = vector.load %alloc_1[%35, %rb_0_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %377 = vector.load %alloc_1[%36, %rb_1_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %378 = vector.load %alloc_1[%36, %rb_1_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %379 = vector.load %alloc_1[%37, %rb_2_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %380 = vector.load %alloc_1[%37, %rb_2_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %381 = vector.load %alloc_1[%38, %rb_3_c0_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %382 = vector.load %alloc_1[%38, %rb_3_c1_swizzle_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>

          // %375 = vector.load %alloc_1[%35, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %376 = vector.load %alloc_1[%35, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %377 = vector.load %alloc_1[%36, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %378 = vector.load %alloc_1[%36, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %379 = vector.load %alloc_1[%37, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %380 = vector.load %alloc_1[%37, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %381 = vector.load %alloc_1[%38, %26] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          // %382 = vector.load %alloc_1[%38, %27] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>


          %383 = vector.bitcast %375 : vector<16xi8> to vector<32xf4E2M1FN>
          %384 = vector.bitcast %376 : vector<16xi8> to vector<32xf4E2M1FN>
          %385 = vector.bitcast %377 : vector<16xi8> to vector<32xf4E2M1FN>
          %386 = vector.bitcast %378 : vector<16xi8> to vector<32xf4E2M1FN>
          %387 = vector.bitcast %379 : vector<16xi8> to vector<32xf4E2M1FN>
          %388 = vector.bitcast %380 : vector<16xi8> to vector<32xf4E2M1FN>
          %389 = vector.bitcast %381 : vector<16xi8> to vector<32xf4E2M1FN>
          %390 = vector.bitcast %382 : vector<16xi8> to vector<32xf4E2M1FN>
          %391 = vector.bitcast %359 : vector<16xi8> to vector<32xf4E2M1FN>
          %392 = vector.bitcast %360 : vector<16xi8> to vector<32xf4E2M1FN>
          %393 = vector.bitcast %361 : vector<16xi8> to vector<32xf4E2M1FN>
          %394 = vector.bitcast %362 : vector<16xi8> to vector<32xf4E2M1FN>
          %395 = vector.bitcast %363 : vector<16xi8> to vector<32xf4E2M1FN>
          %396 = vector.bitcast %364 : vector<16xi8> to vector<32xf4E2M1FN>
          %397 = vector.bitcast %365 : vector<16xi8> to vector<32xf4E2M1FN>
          %398 = vector.bitcast %366 : vector<16xi8> to vector<32xf4E2M1FN>
          %399 = vector.bitcast %367 : vector<16xi8> to vector<32xf4E2M1FN>
          %400 = vector.bitcast %368 : vector<16xi8> to vector<32xf4E2M1FN>
          %401 = vector.bitcast %369 : vector<16xi8> to vector<32xf4E2M1FN>
          %402 = vector.bitcast %370 : vector<16xi8> to vector<32xf4E2M1FN>
          %403 = vector.bitcast %371 : vector<16xi8> to vector<32xf4E2M1FN>
          %404 = vector.bitcast %372 : vector<16xi8> to vector<32xf4E2M1FN>
          %405 = vector.bitcast %373 : vector<16xi8> to vector<32xf4E2M1FN>
          %406 = vector.bitcast %374 : vector<16xi8> to vector<32xf4E2M1FN>
          %407 = amdgpu.scaled_mfma(%cst[0] * %383) * (%cst[0] * %391) + %arg6 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %408 = amdgpu.scaled_mfma(%cst[0] * %384) * (%cst[0] * %392) + %407 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %409 = amdgpu.scaled_mfma(%cst[0] * %383) * (%cst[0] * %393) + %arg7 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %410 = amdgpu.scaled_mfma(%cst[0] * %384) * (%cst[0] * %394) + %409 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %411 = amdgpu.scaled_mfma(%cst[0] * %383) * (%cst[0] * %395) + %arg8 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %412 = amdgpu.scaled_mfma(%cst[0] * %384) * (%cst[0] * %396) + %411 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %413 = amdgpu.scaled_mfma(%cst[0] * %383) * (%cst[0] * %397) + %arg9 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %414 = amdgpu.scaled_mfma(%cst[0] * %384) * (%cst[0] * %398) + %413 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %415 = amdgpu.scaled_mfma(%cst[0] * %383) * (%cst[0] * %399) + %arg10 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %416 = amdgpu.scaled_mfma(%cst[0] * %384) * (%cst[0] * %400) + %415 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %417 = amdgpu.scaled_mfma(%cst[0] * %383) * (%cst[0] * %401) + %arg11 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %418 = amdgpu.scaled_mfma(%cst[0] * %384) * (%cst[0] * %402) + %417 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %419 = amdgpu.scaled_mfma(%cst[0] * %383) * (%cst[0] * %403) + %arg12 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %420 = amdgpu.scaled_mfma(%cst[0] * %384) * (%cst[0] * %404) + %419 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %421 = amdgpu.scaled_mfma(%cst[0] * %383) * (%cst[0] * %405) + %arg13 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %422 = amdgpu.scaled_mfma(%cst[0] * %384) * (%cst[0] * %406) + %421 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %423 = amdgpu.scaled_mfma(%cst[0] * %385) * (%cst[0] * %391) + %arg14 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %424 = amdgpu.scaled_mfma(%cst[0] * %386) * (%cst[0] * %392) + %423 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %425 = amdgpu.scaled_mfma(%cst[0] * %385) * (%cst[0] * %393) + %arg15 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %426 = amdgpu.scaled_mfma(%cst[0] * %386) * (%cst[0] * %394) + %425 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %427 = amdgpu.scaled_mfma(%cst[0] * %385) * (%cst[0] * %395) + %arg16 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %428 = amdgpu.scaled_mfma(%cst[0] * %386) * (%cst[0] * %396) + %427 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %429 = amdgpu.scaled_mfma(%cst[0] * %385) * (%cst[0] * %397) + %arg17 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %430 = amdgpu.scaled_mfma(%cst[0] * %386) * (%cst[0] * %398) + %429 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %431 = amdgpu.scaled_mfma(%cst[0] * %385) * (%cst[0] * %399) + %arg18 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %432 = amdgpu.scaled_mfma(%cst[0] * %386) * (%cst[0] * %400) + %431 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %433 = amdgpu.scaled_mfma(%cst[0] * %385) * (%cst[0] * %401) + %arg19 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %434 = amdgpu.scaled_mfma(%cst[0] * %386) * (%cst[0] * %402) + %433 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %435 = amdgpu.scaled_mfma(%cst[0] * %385) * (%cst[0] * %403) + %arg20 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %436 = amdgpu.scaled_mfma(%cst[0] * %386) * (%cst[0] * %404) + %435 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %437 = amdgpu.scaled_mfma(%cst[0] * %385) * (%cst[0] * %405) + %arg21 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %438 = amdgpu.scaled_mfma(%cst[0] * %386) * (%cst[0] * %406) + %437 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %439 = amdgpu.scaled_mfma(%cst[0] * %387) * (%cst[0] * %391) + %arg22 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %440 = amdgpu.scaled_mfma(%cst[0] * %388) * (%cst[0] * %392) + %439 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %441 = amdgpu.scaled_mfma(%cst[0] * %387) * (%cst[0] * %393) + %arg23 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %442 = amdgpu.scaled_mfma(%cst[0] * %388) * (%cst[0] * %394) + %441 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %443 = amdgpu.scaled_mfma(%cst[0] * %387) * (%cst[0] * %395) + %arg24 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %444 = amdgpu.scaled_mfma(%cst[0] * %388) * (%cst[0] * %396) + %443 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %445 = amdgpu.scaled_mfma(%cst[0] * %387) * (%cst[0] * %397) + %arg25 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %446 = amdgpu.scaled_mfma(%cst[0] * %388) * (%cst[0] * %398) + %445 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %447 = amdgpu.scaled_mfma(%cst[0] * %387) * (%cst[0] * %399) + %arg26 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %448 = amdgpu.scaled_mfma(%cst[0] * %388) * (%cst[0] * %400) + %447 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %449 = amdgpu.scaled_mfma(%cst[0] * %387) * (%cst[0] * %401) + %arg27 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %450 = amdgpu.scaled_mfma(%cst[0] * %388) * (%cst[0] * %402) + %449 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %451 = amdgpu.scaled_mfma(%cst[0] * %387) * (%cst[0] * %403) + %arg28 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %452 = amdgpu.scaled_mfma(%cst[0] * %388) * (%cst[0] * %404) + %451 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %453 = amdgpu.scaled_mfma(%cst[0] * %387) * (%cst[0] * %405) + %arg29 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %454 = amdgpu.scaled_mfma(%cst[0] * %388) * (%cst[0] * %406) + %453 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %455 = amdgpu.scaled_mfma(%cst[0] * %389) * (%cst[0] * %391) + %arg30 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %456 = amdgpu.scaled_mfma(%cst[0] * %390) * (%cst[0] * %392) + %455 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %457 = amdgpu.scaled_mfma(%cst[0] * %389) * (%cst[0] * %393) + %arg31 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %458 = amdgpu.scaled_mfma(%cst[0] * %390) * (%cst[0] * %394) + %457 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %459 = amdgpu.scaled_mfma(%cst[0] * %389) * (%cst[0] * %395) + %arg32 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %460 = amdgpu.scaled_mfma(%cst[0] * %390) * (%cst[0] * %396) + %459 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %461 = amdgpu.scaled_mfma(%cst[0] * %389) * (%cst[0] * %397) + %arg33 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %462 = amdgpu.scaled_mfma(%cst[0] * %390) * (%cst[0] * %398) + %461 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %463 = amdgpu.scaled_mfma(%cst[0] * %389) * (%cst[0] * %399) + %arg34 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %464 = amdgpu.scaled_mfma(%cst[0] * %390) * (%cst[0] * %400) + %463 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %465 = amdgpu.scaled_mfma(%cst[0] * %389) * (%cst[0] * %401) + %arg35 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %466 = amdgpu.scaled_mfma(%cst[0] * %390) * (%cst[0] * %402) + %465 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %467 = amdgpu.scaled_mfma(%cst[0] * %389) * (%cst[0] * %403) + %arg36 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %468 = amdgpu.scaled_mfma(%cst[0] * %390) * (%cst[0] * %404) + %467 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %469 = amdgpu.scaled_mfma(%cst[0] * %389) * (%cst[0] * %405) + %arg37 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %470 = amdgpu.scaled_mfma(%cst[0] * %390) * (%cst[0] * %406) + %469 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          scf.yield %408, %410, %412, %414, %416, %418, %420, %422, %424, %426, %428, %430, %432, %434, %436, %438, %440, %442, %444, %446, %448, %450, %452, %454, %456, %458, %460, %462, %464, %466, %468, %470 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %40 = vector.extract_strided_slice %39#0 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %41 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<16384x16384xf32, strided<[16384, 1], offset: ?>>
        %42 = affine.apply #map24()[%block_id_x]
        %43 = affine.apply #map24()[%block_id_y]
        %44 = affine.apply #map25()[%thread_id_x]
        %45 = affine.apply #map9()[%thread_id_x, %thread_id_y]
        %46 = arith.muli %42, %c16384 overflow<nsw> : index
        %47 = arith.muli %44, %c16384 overflow<nsw> : index
        %48 = arith.addi %46, %43 overflow<nsw> : index
        %49 = arith.addi %47, %45 overflow<nsw> : index
        %reinterpret_cast_3 = memref.reinterpret_cast %41 to offset: [%48], sizes: [%c536870910], strides: [1] : memref<16384x16384xf32, strided<[16384, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %50 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_3 validBytes(%c2147483643_i32) : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        vector.store %40, %50[%49] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %51 = vector.extract_strided_slice %39#0 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %52 = affine.apply #map26()[%thread_id_x]
        %53 = arith.muli %52, %c16384 overflow<nsw> : index
        %54 = arith.addi %53, %45 overflow<nsw> : index
        vector.store %51, %50[%54] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %55 = vector.extract_strided_slice %39#0 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %56 = affine.apply #map27()[%thread_id_x]
        %57 = arith.muli %56, %c16384 overflow<nsw> : index
        %58 = arith.addi %57, %45 overflow<nsw> : index
        vector.store %55, %50[%58] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %59 = vector.extract_strided_slice %39#0 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %60 = affine.apply #map28()[%thread_id_x]
        %61 = arith.muli %60, %c16384 overflow<nsw> : index
        %62 = arith.addi %61, %45 overflow<nsw> : index
        vector.store %59, %50[%62] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %63 = vector.extract_strided_slice %39#1 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %64 = affine.apply #map12()[%thread_id_x, %thread_id_y]
        %65 = arith.addi %47, %64 overflow<nsw> : index
        vector.store %63, %50[%65] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %66 = vector.extract_strided_slice %39#1 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %67 = arith.addi %53, %64 overflow<nsw> : index
        vector.store %66, %50[%67] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %68 = vector.extract_strided_slice %39#1 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %69 = arith.addi %57, %64 overflow<nsw> : index
        vector.store %68, %50[%69] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %70 = vector.extract_strided_slice %39#1 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %71 = arith.addi %61, %64 overflow<nsw> : index
        vector.store %70, %50[%71] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %72 = vector.extract_strided_slice %39#2 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %73 = affine.apply #map13()[%thread_id_x, %thread_id_y]
        %74 = arith.addi %47, %73 overflow<nsw> : index
        vector.store %72, %50[%74] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %75 = vector.extract_strided_slice %39#2 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %76 = arith.addi %53, %73 overflow<nsw> : index
        vector.store %75, %50[%76] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %77 = vector.extract_strided_slice %39#2 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %78 = arith.addi %57, %73 overflow<nsw> : index
        vector.store %77, %50[%78] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %79 = vector.extract_strided_slice %39#2 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %80 = arith.addi %61, %73 overflow<nsw> : index
        vector.store %79, %50[%80] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %81 = vector.extract_strided_slice %39#3 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %82 = affine.apply #map14()[%thread_id_x, %thread_id_y]
        %83 = arith.addi %47, %82 overflow<nsw> : index
        vector.store %81, %50[%83] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %84 = vector.extract_strided_slice %39#3 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %85 = arith.addi %53, %82 overflow<nsw> : index
        vector.store %84, %50[%85] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %86 = vector.extract_strided_slice %39#3 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %87 = arith.addi %57, %82 overflow<nsw> : index
        vector.store %86, %50[%87] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %88 = vector.extract_strided_slice %39#3 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %89 = arith.addi %61, %82 overflow<nsw> : index
        vector.store %88, %50[%89] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %90 = vector.extract_strided_slice %39#4 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %91 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %92 = arith.addi %47, %91 overflow<nsw> : index
        vector.store %90, %50[%92] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %93 = vector.extract_strided_slice %39#4 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %94 = arith.addi %53, %91 overflow<nsw> : index
        vector.store %93, %50[%94] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %95 = vector.extract_strided_slice %39#4 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %96 = arith.addi %57, %91 overflow<nsw> : index
        vector.store %95, %50[%96] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %97 = vector.extract_strided_slice %39#4 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %98 = arith.addi %61, %91 overflow<nsw> : index
        vector.store %97, %50[%98] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %99 = vector.extract_strided_slice %39#5 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %100 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %101 = arith.addi %47, %100 overflow<nsw> : index
        vector.store %99, %50[%101] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %102 = vector.extract_strided_slice %39#5 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %103 = arith.addi %53, %100 overflow<nsw> : index
        vector.store %102, %50[%103] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %104 = vector.extract_strided_slice %39#5 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %105 = arith.addi %57, %100 overflow<nsw> : index
        vector.store %104, %50[%105] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %106 = vector.extract_strided_slice %39#5 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %107 = arith.addi %61, %100 overflow<nsw> : index
        vector.store %106, %50[%107] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %108 = vector.extract_strided_slice %39#6 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %109 = affine.apply #map17()[%thread_id_x, %thread_id_y]
        %110 = arith.addi %47, %109 overflow<nsw> : index
        vector.store %108, %50[%110] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %111 = vector.extract_strided_slice %39#6 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %112 = arith.addi %53, %109 overflow<nsw> : index
        vector.store %111, %50[%112] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %113 = vector.extract_strided_slice %39#6 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %114 = arith.addi %57, %109 overflow<nsw> : index
        vector.store %113, %50[%114] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %115 = vector.extract_strided_slice %39#6 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %116 = arith.addi %61, %109 overflow<nsw> : index
        vector.store %115, %50[%116] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %117 = vector.extract_strided_slice %39#7 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %118 = affine.apply #map18()[%thread_id_x, %thread_id_y]
        %119 = arith.addi %47, %118 overflow<nsw> : index
        vector.store %117, %50[%119] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %120 = vector.extract_strided_slice %39#7 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %121 = arith.addi %53, %118 overflow<nsw> : index
        vector.store %120, %50[%121] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %122 = vector.extract_strided_slice %39#7 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %123 = arith.addi %57, %118 overflow<nsw> : index
        vector.store %122, %50[%123] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %124 = vector.extract_strided_slice %39#7 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %125 = arith.addi %61, %118 overflow<nsw> : index
        vector.store %124, %50[%125] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %126 = vector.extract_strided_slice %39#8 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %127 = affine.apply #map29()[%thread_id_x]
        %128 = arith.muli %127, %c16384 overflow<nsw> : index
        %129 = arith.addi %128, %45 overflow<nsw> : index
        vector.store %126, %50[%129] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %130 = vector.extract_strided_slice %39#8 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %131 = affine.apply #map30()[%thread_id_x]
        %132 = arith.muli %131, %c16384 overflow<nsw> : index
        %133 = arith.addi %132, %45 overflow<nsw> : index
        vector.store %130, %50[%133] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %134 = vector.extract_strided_slice %39#8 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %135 = affine.apply #map31()[%thread_id_x]
        %136 = arith.muli %135, %c16384 overflow<nsw> : index
        %137 = arith.addi %136, %45 overflow<nsw> : index
        vector.store %134, %50[%137] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %138 = vector.extract_strided_slice %39#8 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %139 = affine.apply #map32()[%thread_id_x]
        %140 = arith.muli %139, %c16384 overflow<nsw> : index
        %141 = arith.addi %140, %45 overflow<nsw> : index
        vector.store %138, %50[%141] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %142 = vector.extract_strided_slice %39#9 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %143 = arith.addi %128, %64 overflow<nsw> : index
        vector.store %142, %50[%143] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %144 = vector.extract_strided_slice %39#9 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %145 = arith.addi %132, %64 overflow<nsw> : index
        vector.store %144, %50[%145] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %146 = vector.extract_strided_slice %39#9 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %147 = arith.addi %136, %64 overflow<nsw> : index
        vector.store %146, %50[%147] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %148 = vector.extract_strided_slice %39#9 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %149 = arith.addi %140, %64 overflow<nsw> : index
        vector.store %148, %50[%149] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %150 = vector.extract_strided_slice %39#10 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %151 = arith.addi %128, %73 overflow<nsw> : index
        vector.store %150, %50[%151] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %152 = vector.extract_strided_slice %39#10 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %153 = arith.addi %132, %73 overflow<nsw> : index
        vector.store %152, %50[%153] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %154 = vector.extract_strided_slice %39#10 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %155 = arith.addi %136, %73 overflow<nsw> : index
        vector.store %154, %50[%155] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %156 = vector.extract_strided_slice %39#10 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %157 = arith.addi %140, %73 overflow<nsw> : index
        vector.store %156, %50[%157] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %158 = vector.extract_strided_slice %39#11 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %159 = arith.addi %128, %82 overflow<nsw> : index
        vector.store %158, %50[%159] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %160 = vector.extract_strided_slice %39#11 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %161 = arith.addi %132, %82 overflow<nsw> : index
        vector.store %160, %50[%161] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %162 = vector.extract_strided_slice %39#11 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %163 = arith.addi %136, %82 overflow<nsw> : index
        vector.store %162, %50[%163] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %164 = vector.extract_strided_slice %39#11 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %165 = arith.addi %140, %82 overflow<nsw> : index
        vector.store %164, %50[%165] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %166 = vector.extract_strided_slice %39#12 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %167 = arith.addi %128, %91 overflow<nsw> : index
        vector.store %166, %50[%167] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %168 = vector.extract_strided_slice %39#12 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %169 = arith.addi %132, %91 overflow<nsw> : index
        vector.store %168, %50[%169] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %170 = vector.extract_strided_slice %39#12 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %171 = arith.addi %136, %91 overflow<nsw> : index
        vector.store %170, %50[%171] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %172 = vector.extract_strided_slice %39#12 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %173 = arith.addi %140, %91 overflow<nsw> : index
        vector.store %172, %50[%173] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %174 = vector.extract_strided_slice %39#13 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %175 = arith.addi %128, %100 overflow<nsw> : index
        vector.store %174, %50[%175] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %176 = vector.extract_strided_slice %39#13 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %177 = arith.addi %132, %100 overflow<nsw> : index
        vector.store %176, %50[%177] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %178 = vector.extract_strided_slice %39#13 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %179 = arith.addi %136, %100 overflow<nsw> : index
        vector.store %178, %50[%179] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %180 = vector.extract_strided_slice %39#13 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %181 = arith.addi %140, %100 overflow<nsw> : index
        vector.store %180, %50[%181] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %182 = vector.extract_strided_slice %39#14 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %183 = arith.addi %128, %109 overflow<nsw> : index
        vector.store %182, %50[%183] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %184 = vector.extract_strided_slice %39#14 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %185 = arith.addi %132, %109 overflow<nsw> : index
        vector.store %184, %50[%185] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %186 = vector.extract_strided_slice %39#14 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %187 = arith.addi %136, %109 overflow<nsw> : index
        vector.store %186, %50[%187] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %188 = vector.extract_strided_slice %39#14 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %189 = arith.addi %140, %109 overflow<nsw> : index
        vector.store %188, %50[%189] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %190 = vector.extract_strided_slice %39#15 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %191 = arith.addi %128, %118 overflow<nsw> : index
        vector.store %190, %50[%191] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %192 = vector.extract_strided_slice %39#15 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %193 = arith.addi %132, %118 overflow<nsw> : index
        vector.store %192, %50[%193] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %194 = vector.extract_strided_slice %39#15 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %195 = arith.addi %136, %118 overflow<nsw> : index
        vector.store %194, %50[%195] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %196 = vector.extract_strided_slice %39#15 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %197 = arith.addi %140, %118 overflow<nsw> : index
        vector.store %196, %50[%197] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %198 = vector.extract_strided_slice %39#16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %199 = affine.apply #map33()[%thread_id_x]
        %200 = arith.muli %199, %c16384 overflow<nsw> : index
        %201 = arith.addi %200, %45 overflow<nsw> : index
        vector.store %198, %50[%201] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %202 = vector.extract_strided_slice %39#16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %203 = affine.apply #map34()[%thread_id_x]
        %204 = arith.muli %203, %c16384 overflow<nsw> : index
        %205 = arith.addi %204, %45 overflow<nsw> : index
        vector.store %202, %50[%205] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %206 = vector.extract_strided_slice %39#16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %207 = affine.apply #map35()[%thread_id_x]
        %208 = arith.muli %207, %c16384 overflow<nsw> : index
        %209 = arith.addi %208, %45 overflow<nsw> : index
        vector.store %206, %50[%209] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %210 = vector.extract_strided_slice %39#16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %211 = affine.apply #map36()[%thread_id_x]
        %212 = arith.muli %211, %c16384 overflow<nsw> : index
        %213 = arith.addi %212, %45 overflow<nsw> : index
        vector.store %210, %50[%213] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %214 = vector.extract_strided_slice %39#17 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %215 = arith.addi %200, %64 overflow<nsw> : index
        vector.store %214, %50[%215] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %216 = vector.extract_strided_slice %39#17 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %217 = arith.addi %204, %64 overflow<nsw> : index
        vector.store %216, %50[%217] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %218 = vector.extract_strided_slice %39#17 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %219 = arith.addi %208, %64 overflow<nsw> : index
        vector.store %218, %50[%219] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %220 = vector.extract_strided_slice %39#17 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %221 = arith.addi %212, %64 overflow<nsw> : index
        vector.store %220, %50[%221] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %222 = vector.extract_strided_slice %39#18 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %223 = arith.addi %200, %73 overflow<nsw> : index
        vector.store %222, %50[%223] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %224 = vector.extract_strided_slice %39#18 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %225 = arith.addi %204, %73 overflow<nsw> : index
        vector.store %224, %50[%225] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %226 = vector.extract_strided_slice %39#18 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %227 = arith.addi %208, %73 overflow<nsw> : index
        vector.store %226, %50[%227] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %228 = vector.extract_strided_slice %39#18 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %229 = arith.addi %212, %73 overflow<nsw> : index
        vector.store %228, %50[%229] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %230 = vector.extract_strided_slice %39#19 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %231 = arith.addi %200, %82 overflow<nsw> : index
        vector.store %230, %50[%231] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %232 = vector.extract_strided_slice %39#19 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %233 = arith.addi %204, %82 overflow<nsw> : index
        vector.store %232, %50[%233] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %234 = vector.extract_strided_slice %39#19 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %235 = arith.addi %208, %82 overflow<nsw> : index
        vector.store %234, %50[%235] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %236 = vector.extract_strided_slice %39#19 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %237 = arith.addi %212, %82 overflow<nsw> : index
        vector.store %236, %50[%237] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %238 = vector.extract_strided_slice %39#20 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %239 = arith.addi %200, %91 overflow<nsw> : index
        vector.store %238, %50[%239] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %240 = vector.extract_strided_slice %39#20 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %241 = arith.addi %204, %91 overflow<nsw> : index
        vector.store %240, %50[%241] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %242 = vector.extract_strided_slice %39#20 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %243 = arith.addi %208, %91 overflow<nsw> : index
        vector.store %242, %50[%243] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %244 = vector.extract_strided_slice %39#20 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %245 = arith.addi %212, %91 overflow<nsw> : index
        vector.store %244, %50[%245] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %246 = vector.extract_strided_slice %39#21 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %247 = arith.addi %200, %100 overflow<nsw> : index
        vector.store %246, %50[%247] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %248 = vector.extract_strided_slice %39#21 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %249 = arith.addi %204, %100 overflow<nsw> : index
        vector.store %248, %50[%249] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %250 = vector.extract_strided_slice %39#21 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %251 = arith.addi %208, %100 overflow<nsw> : index
        vector.store %250, %50[%251] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %252 = vector.extract_strided_slice %39#21 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %253 = arith.addi %212, %100 overflow<nsw> : index
        vector.store %252, %50[%253] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %254 = vector.extract_strided_slice %39#22 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %255 = arith.addi %200, %109 overflow<nsw> : index
        vector.store %254, %50[%255] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %256 = vector.extract_strided_slice %39#22 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %257 = arith.addi %204, %109 overflow<nsw> : index
        vector.store %256, %50[%257] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %258 = vector.extract_strided_slice %39#22 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %259 = arith.addi %208, %109 overflow<nsw> : index
        vector.store %258, %50[%259] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %260 = vector.extract_strided_slice %39#22 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %261 = arith.addi %212, %109 overflow<nsw> : index
        vector.store %260, %50[%261] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %262 = vector.extract_strided_slice %39#23 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %263 = arith.addi %200, %118 overflow<nsw> : index
        vector.store %262, %50[%263] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %264 = vector.extract_strided_slice %39#23 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %265 = arith.addi %204, %118 overflow<nsw> : index
        vector.store %264, %50[%265] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %266 = vector.extract_strided_slice %39#23 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %267 = arith.addi %208, %118 overflow<nsw> : index
        vector.store %266, %50[%267] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %268 = vector.extract_strided_slice %39#23 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %269 = arith.addi %212, %118 overflow<nsw> : index
        vector.store %268, %50[%269] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %270 = vector.extract_strided_slice %39#24 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %271 = affine.apply #map37()[%thread_id_x]
        %272 = arith.muli %271, %c16384 overflow<nsw> : index
        %273 = arith.addi %272, %45 overflow<nsw> : index
        vector.store %270, %50[%273] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %274 = vector.extract_strided_slice %39#24 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %275 = affine.apply #map38()[%thread_id_x]
        %276 = arith.muli %275, %c16384 overflow<nsw> : index
        %277 = arith.addi %276, %45 overflow<nsw> : index
        vector.store %274, %50[%277] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %278 = vector.extract_strided_slice %39#24 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %279 = affine.apply #map39()[%thread_id_x]
        %280 = arith.muli %279, %c16384 overflow<nsw> : index
        %281 = arith.addi %280, %45 overflow<nsw> : index
        vector.store %278, %50[%281] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %282 = vector.extract_strided_slice %39#24 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %283 = affine.apply #map40()[%thread_id_x]
        %284 = arith.muli %283, %c16384 overflow<nsw> : index
        %285 = arith.addi %284, %45 overflow<nsw> : index
        vector.store %282, %50[%285] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %286 = vector.extract_strided_slice %39#25 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %287 = arith.addi %272, %64 overflow<nsw> : index
        vector.store %286, %50[%287] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %288 = vector.extract_strided_slice %39#25 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %289 = arith.addi %276, %64 overflow<nsw> : index
        vector.store %288, %50[%289] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %290 = vector.extract_strided_slice %39#25 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %291 = arith.addi %280, %64 overflow<nsw> : index
        vector.store %290, %50[%291] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %292 = vector.extract_strided_slice %39#25 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %293 = arith.addi %284, %64 overflow<nsw> : index
        vector.store %292, %50[%293] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %294 = vector.extract_strided_slice %39#26 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %295 = arith.addi %272, %73 overflow<nsw> : index
        vector.store %294, %50[%295] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %296 = vector.extract_strided_slice %39#26 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %297 = arith.addi %276, %73 overflow<nsw> : index
        vector.store %296, %50[%297] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %298 = vector.extract_strided_slice %39#26 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %299 = arith.addi %280, %73 overflow<nsw> : index
        vector.store %298, %50[%299] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %300 = vector.extract_strided_slice %39#26 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %301 = arith.addi %284, %73 overflow<nsw> : index
        vector.store %300, %50[%301] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %302 = vector.extract_strided_slice %39#27 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %303 = arith.addi %272, %82 overflow<nsw> : index
        vector.store %302, %50[%303] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %304 = vector.extract_strided_slice %39#27 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %305 = arith.addi %276, %82 overflow<nsw> : index
        vector.store %304, %50[%305] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %306 = vector.extract_strided_slice %39#27 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %307 = arith.addi %280, %82 overflow<nsw> : index
        vector.store %306, %50[%307] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %308 = vector.extract_strided_slice %39#27 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %309 = arith.addi %284, %82 overflow<nsw> : index
        vector.store %308, %50[%309] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %310 = vector.extract_strided_slice %39#28 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %311 = arith.addi %272, %91 overflow<nsw> : index
        vector.store %310, %50[%311] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %312 = vector.extract_strided_slice %39#28 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %313 = arith.addi %276, %91 overflow<nsw> : index
        vector.store %312, %50[%313] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %314 = vector.extract_strided_slice %39#28 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %315 = arith.addi %280, %91 overflow<nsw> : index
        vector.store %314, %50[%315] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %316 = vector.extract_strided_slice %39#28 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %317 = arith.addi %284, %91 overflow<nsw> : index
        vector.store %316, %50[%317] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %318 = vector.extract_strided_slice %39#29 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %319 = arith.addi %272, %100 overflow<nsw> : index
        vector.store %318, %50[%319] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %320 = vector.extract_strided_slice %39#29 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %321 = arith.addi %276, %100 overflow<nsw> : index
        vector.store %320, %50[%321] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %322 = vector.extract_strided_slice %39#29 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %323 = arith.addi %280, %100 overflow<nsw> : index
        vector.store %322, %50[%323] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %324 = vector.extract_strided_slice %39#29 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %325 = arith.addi %284, %100 overflow<nsw> : index
        vector.store %324, %50[%325] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %326 = vector.extract_strided_slice %39#30 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %327 = arith.addi %272, %109 overflow<nsw> : index
        vector.store %326, %50[%327] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %328 = vector.extract_strided_slice %39#30 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %329 = arith.addi %276, %109 overflow<nsw> : index
        vector.store %328, %50[%329] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %330 = vector.extract_strided_slice %39#30 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %331 = arith.addi %280, %109 overflow<nsw> : index
        vector.store %330, %50[%331] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %332 = vector.extract_strided_slice %39#30 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %333 = arith.addi %284, %109 overflow<nsw> : index
        vector.store %332, %50[%333] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %334 = vector.extract_strided_slice %39#31 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %335 = arith.addi %272, %118 overflow<nsw> : index
        vector.store %334, %50[%335] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %336 = vector.extract_strided_slice %39#31 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %337 = arith.addi %276, %118 overflow<nsw> : index
        vector.store %336, %50[%337] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %338 = vector.extract_strided_slice %39#31 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %339 = arith.addi %280, %118 overflow<nsw> : index
        vector.store %338, %50[%339] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %340 = vector.extract_strided_slice %39#31 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %341 = arith.addi %284, %118 overflow<nsw> : index
        vector.store %340, %50[%341] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<16384x8192xi8>, %arg1: tensor<16384x512xi8>, %arg2: tensor<16384x8192xi8>, %arg3: tensor<16384x512xi8>, %arg4: tensor<16384x16384xf32>) -> tensor<16384x16384xf32> {
    %0 = flow.dispatch @gemm_afp4_wfp4_wave::@gemm_afp4_wfp4_wave(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x16384xf32>) -> %arg4
    return %0 : tensor<16384x16384xf32>
  }
}
