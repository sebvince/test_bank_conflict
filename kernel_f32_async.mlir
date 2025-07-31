#map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
#map1 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
#map2 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map3 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
#map4 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map5 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
#map6 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
#map7 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
#map8 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map9 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
#map10 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>
#map11 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map12 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map13 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map14 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map15 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map16 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map17 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
#map18 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
#map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
#map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
#map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
#map22 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 - (s1 floordiv 8) * 128)>
#map23 = affine_map<()[s0] -> (s0 * 256)>
#map24 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
#map25 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map26 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map27 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map28 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
#map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
#map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
#map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
#map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
#map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
#map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
#map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
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
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
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
        %3 = affine.apply #map1()[%thread_id_x, %thread_id_y]
        %4 = arith.muli %2, %c8192 overflow<nsw> : index
        %reinterpret_cast = memref.reinterpret_cast %1 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x8192xi8, strided<[8192, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %5 = amdgpu.fat_raw_buffer_cast %reinterpret_cast validBytes(%c2147483646_i32) cacheSwizzleStride(%c-8192_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %6 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_x]
        %7 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %8 = arith.muli %6, %c8192 overflow<nsw> : index
        %9 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_x]
        %10 = affine.apply #map5()[%thread_id_x, %thread_id_y]
        %11 = arith.muli %9, %c8192 overflow<nsw> : index
        %12 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_x]
        %13 = affine.apply #map7()[%thread_id_x, %thread_id_y]
        %14 = arith.muli %12, %c8192 overflow<nsw> : index
        %15 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
        %16 = arith.muli %15, %c8192 overflow<nsw> : index
        %reinterpret_cast_2 = memref.reinterpret_cast %0 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x8192xi8, strided<[8192, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %17 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_2 validBytes(%c2147483646_i32) cacheSwizzleStride(%c-8192_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %18 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y]
        %19 = arith.muli %18, %c8192 overflow<nsw> : index
        %20 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_y]
        %21 = arith.muli %20, %c8192 overflow<nsw> : index
        %22 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_y]
        %23 = arith.muli %22, %c8192 overflow<nsw> : index
        %24 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        %25 = affine.apply #map9()[%thread_id_x]
        %26 = affine.apply #map10()[%thread_id_x]
        %27 = affine.apply #map11()[%thread_id_x, %thread_id_y]
        %28 = affine.apply #map12()[%thread_id_x, %thread_id_y]
        %29 = affine.apply #map13()[%thread_id_x, %thread_id_y]
        %30 = affine.apply #map14()[%thread_id_x, %thread_id_y]
        %31 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %32 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %33 = affine.apply #map17()[%thread_id_x, %thread_id_y]
        %34 = affine.apply #map18()[%thread_id_x]
        %35 = affine.apply #map19()[%thread_id_x]
        %36 = affine.apply #map20()[%thread_id_x]
        %37 = affine.apply #map21()[%thread_id_x]
        %38:32 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %cst_0, %arg7 = %cst_0, %arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_0, %arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0, %arg17 = %cst_0, %arg18 = %cst_0, %arg19 = %cst_0, %arg20 = %cst_0, %arg21 = %cst_0, %arg22 = %cst_0, %arg23 = %cst_0, %arg24 = %cst_0, %arg25 = %cst_0, %arg26 = %cst_0, %arg27 = %cst_0, %arg28 = %cst_0, %arg29 = %cst_0, %arg30 = %cst_0, %arg31 = %cst_0, %arg32 = %cst_0, %arg33 = %cst_0, %arg34 = %cst_0, %arg35 = %cst_0, %arg36 = %cst_0, %arg37 = %cst_0) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          amdgpu.lds_barrier

          //((s1 * 4 + s0 floordiv 64) mod 32) * 8 : warpId * 8 as 8 rows per warp
          //%3,%7,%10,%13 are row index in LDS


          //Apply inverse swizzling
          %read_col = affine.apply affine_map<()[s0,s1] -> ((s1 mod 8) + s0 *8)>()[%arg5, %thread_id_x]
          %read_row_a_0_mod_8 = arith.remui %2, %c8 : index
          %read_row_a_1_mod_8 = arith.remui %6, %c8 : index
          %read_row_a_2_mod_8 = arith.remui %9, %c8 : index
          %read_row_a_3_mod_8 = arith.remui %12, %c8 : index
          %read_row_b_0_mod_8 = arith.remui %15, %c8 : index
          %read_row_b_1_mod_8 = arith.remui %18, %c8 : index
          %read_row_b_2_mod_8 = arith.remui %20, %c8 : index
          %read_row_b_3_mod_8 = arith.remui %22, %c8 : index

          %col_a_swizzle_0 = arith.xori %read_row_a_0_mod_8, %read_col : index
          %col_a_swizzle_1 = arith.xori %read_row_a_1_mod_8, %read_col : index
          %col_a_swizzle_2 = arith.xori %read_row_a_2_mod_8, %read_col : index
          %col_a_swizzle_3 = arith.xori %read_row_a_3_mod_8, %read_col : index
          %col_b_swizzle_0 = arith.xori %read_row_b_0_mod_8, %read_col : index
          %col_b_swizzle_1 = arith.xori %read_row_b_1_mod_8, %read_col : index
          %col_b_swizzle_2 = arith.xori %read_row_b_2_mod_8, %read_col : index
          %col_b_swizzle_3 = arith.xori %read_row_b_3_mod_8, %read_col : index

          %col_a_swizzle_0_b = arith.muli %col_a_swizzle_0, %c16 : index
          %col_a_swizzle_1_b = arith.muli %col_a_swizzle_1, %c16 : index 
          %col_a_swizzle_2_b = arith.muli %col_a_swizzle_2, %c16 : index
          %col_a_swizzle_3_b = arith.muli %col_a_swizzle_3, %c16 : index
          %col_b_swizzle_0_b = arith.muli %col_b_swizzle_0, %c16 : index
          %col_b_swizzle_1_b = arith.muli %col_b_swizzle_1, %c16 : index
          %col_b_swizzle_2_b = arith.muli %col_b_swizzle_2, %c16 : index
          %col_b_swizzle_3_b = arith.muli %col_b_swizzle_3, %c16 : index


          %342 = arith.addi %4,  %col_a_swizzle_0_b overflow<nsw> : index
          %343 = arith.addi %8,  %col_a_swizzle_1_b overflow<nsw> : index
          %344 = arith.addi %11, %col_a_swizzle_2_b overflow<nsw> : index
          %345 = arith.addi %14, %col_a_swizzle_3_b overflow<nsw> : index
          %346 = arith.addi %16, %col_b_swizzle_0_b overflow<nsw> : index
          %347 = arith.addi %19, %col_b_swizzle_1_b overflow<nsw> : index
          %348 = arith.addi %21, %col_b_swizzle_2_b overflow<nsw> : index
          %349 = arith.addi %23, %col_b_swizzle_3_b overflow<nsw> : index

          amdgpu.gather_to_lds %5[%342], %alloc_1[%3] : vector<16xi8>, memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          amdgpu.gather_to_lds %5[%343], %alloc_1[%7] : vector<16xi8>, memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          amdgpu.gather_to_lds %5[%344], %alloc_1[%10] : vector<16xi8>, memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          amdgpu.gather_to_lds %5[%345], %alloc_1[%13] : vector<16xi8>, memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          amdgpu.gather_to_lds %17[%346], %alloc[%3] : vector<16xi8>, memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          amdgpu.gather_to_lds %17[%347], %alloc[%7] : vector<16xi8>, memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          amdgpu.gather_to_lds %17[%348], %alloc[%10] : vector<16xi8>, memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
          amdgpu.gather_to_lds %17[%349], %alloc[%13] : vector<16xi8>, memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>

          rocdl.s.waitcnt 16368

          //Apply Swizzling on LDS reads . Row %8 ^ col

          //col back to elements (not bytes)
          %c4 = arith.constant 4 : index
          %col_0 = arith.shrui %25, %c4 : index
          %col_1 = arith.shrui %26, %c4 : index

          //mod8 on rows
          %row_a_8_0 = arith.remui %24, %c8 : index
          %row_a_8_1 = arith.remui %27, %c8 : index
          %row_a_8_2 = arith.remui %28, %c8 : index
          %row_a_8_3 = arith.remui %29, %c8 : index
          %row_a_8_4 = arith.remui %30, %c8 : index
          %row_a_8_5 = arith.remui %31, %c8 : index
          %row_a_8_6 = arith.remui %32, %c8 : index
          %row_a_8_7 = arith.remui %33, %c8 : index
          
          //XOR
          %col_a_0_0 = arith.xori %row_a_8_0, %col_0 : index
          %col_a_0_1 = arith.xori %row_a_8_0, %col_1 : index
          %col_a_1_0 = arith.xori %row_a_8_1, %col_0 : index
          %col_a_1_1 = arith.xori %row_a_8_1, %col_1 : index
          %col_a_2_0 = arith.xori %row_a_8_2, %col_0 : index
          %col_a_2_1 = arith.xori %row_a_8_2, %col_1 : index
          %col_a_3_0 = arith.xori %row_a_8_3, %col_0 : index
          %col_a_3_1 = arith.xori %row_a_8_3, %col_1 : index
          %col_a_4_0 = arith.xori %row_a_8_4, %col_0 : index
          %col_a_4_1 = arith.xori %row_a_8_4, %col_1 : index
          %col_a_5_0 = arith.xori %row_a_8_5, %col_0 : index
          %col_a_5_1 = arith.xori %row_a_8_5, %col_1 : index
          %col_a_6_0 = arith.xori %row_a_8_6, %col_0 : index
          %col_a_6_1 = arith.xori %row_a_8_6, %col_1 : index
          %col_a_7_0 = arith.xori %row_a_8_7, %col_0 : index
          %col_a_7_1 = arith.xori %row_a_8_7, %col_1 : index

          //offset in bytes
          %col_a_0_0_b = arith.muli %col_a_0_0, %c16 : index
          %col_a_0_1_b = arith.muli %col_a_0_1, %c16 : index
          %col_a_1_0_b = arith.muli %col_a_1_0, %c16 : index
          %col_a_1_1_b = arith.muli %col_a_1_1, %c16 : index
          %col_a_2_0_b = arith.muli %col_a_2_0, %c16 : index
          %col_a_2_1_b = arith.muli %col_a_2_1, %c16 : index
          %col_a_3_0_b = arith.muli %col_a_3_0, %c16 : index
          %col_a_3_1_b = arith.muli %col_a_3_1, %c16 : index
          %col_a_4_0_b = arith.muli %col_a_4_0, %c16 : index
          %col_a_4_1_b = arith.muli %col_a_4_1, %c16 : index
          %col_a_5_0_b = arith.muli %col_a_5_0, %c16 : index
          %col_a_5_1_b = arith.muli %col_a_5_1, %c16 : index
          %col_a_6_0_b = arith.muli %col_a_6_0, %c16 : index
          %col_a_6_1_b = arith.muli %col_a_6_1, %c16 : index
          %col_a_7_0_b = arith.muli %col_a_7_0, %c16 : index
          %col_a_7_1_b = arith.muli %col_a_7_1, %c16 : index

          amdgpu.lds_barrier
          %350 = vector.load %alloc[%24, %col_a_0_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %351 = vector.load %alloc[%24, %col_a_0_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %352 = vector.load %alloc[%27, %col_a_1_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %353 = vector.load %alloc[%27, %col_a_1_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %354 = vector.load %alloc[%28, %col_a_2_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %355 = vector.load %alloc[%28, %col_a_2_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %356 = vector.load %alloc[%29, %col_a_3_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %357 = vector.load %alloc[%29, %col_a_3_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %358 = vector.load %alloc[%30, %col_a_4_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %359 = vector.load %alloc[%30, %col_a_4_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %360 = vector.load %alloc[%31, %col_a_5_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %361 = vector.load %alloc[%31, %col_a_5_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %362 = vector.load %alloc[%32, %col_a_6_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %363 = vector.load %alloc[%32, %col_a_6_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %364 = vector.load %alloc[%33, %col_a_7_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %365 = vector.load %alloc[%33, %col_a_7_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>

 //mod8 on rows
          %row_b_8_0 = arith.remui %34, %c8 : index
          %row_b_8_1 = arith.remui %35, %c8 : index
          %row_b_8_2 = arith.remui %36, %c8 : index
          %row_b_8_3 = arith.remui %37, %c8 : index
          
          //XOR
          %col_b_0_0 = arith.xori %row_b_8_0, %col_0 : index
          %col_b_0_1 = arith.xori %row_b_8_0, %col_1 : index
          %col_b_1_0 = arith.xori %row_b_8_1, %col_0 : index
          %col_b_1_1 = arith.xori %row_b_8_1, %col_1 : index
          %col_b_2_0 = arith.xori %row_b_8_2, %col_0 : index
          %col_b_2_1 = arith.xori %row_b_8_2, %col_1 : index
          %col_b_3_0 = arith.xori %row_b_8_3, %col_0 : index
          %col_b_3_1 = arith.xori %row_b_8_3, %col_1 : index
          

          //offset in bytes
          %col_b_0_0_b = arith.muli %col_b_0_0, %c16 : index
          %col_b_0_1_b = arith.muli %col_b_0_1, %c16 : index
          %col_b_1_0_b = arith.muli %col_b_1_0, %c16 : index
          %col_b_1_1_b = arith.muli %col_b_1_1, %c16 : index
          %col_b_2_0_b = arith.muli %col_b_2_0, %c16 : index
          %col_b_2_1_b = arith.muli %col_b_2_1, %c16 : index
          %col_b_3_0_b = arith.muli %col_b_3_0, %c16 : index
          %col_b_3_1_b = arith.muli %col_b_3_1, %c16 : index


          %366 = vector.load %alloc_1[%34, %col_b_0_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %367 = vector.load %alloc_1[%34, %col_b_0_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %368 = vector.load %alloc_1[%35, %col_b_1_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %369 = vector.load %alloc_1[%35, %col_b_1_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %370 = vector.load %alloc_1[%36, %col_b_2_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %371 = vector.load %alloc_1[%36, %col_b_2_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %372 = vector.load %alloc_1[%37, %col_b_3_0_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %373 = vector.load %alloc_1[%37, %col_b_3_1_b] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>

          %374 = vector.bitcast %366 : vector<16xi8> to vector<32xf4E2M1FN>
          %375 = vector.bitcast %367 : vector<16xi8> to vector<32xf4E2M1FN>
          %376 = vector.bitcast %368 : vector<16xi8> to vector<32xf4E2M1FN>
          %377 = vector.bitcast %369 : vector<16xi8> to vector<32xf4E2M1FN>
          %378 = vector.bitcast %370 : vector<16xi8> to vector<32xf4E2M1FN>
          %379 = vector.bitcast %371 : vector<16xi8> to vector<32xf4E2M1FN>
          %380 = vector.bitcast %372 : vector<16xi8> to vector<32xf4E2M1FN>
          %381 = vector.bitcast %373 : vector<16xi8> to vector<32xf4E2M1FN>
          %382 = vector.bitcast %350 : vector<16xi8> to vector<32xf4E2M1FN>
          %383 = vector.bitcast %351 : vector<16xi8> to vector<32xf4E2M1FN>
          %384 = vector.bitcast %352 : vector<16xi8> to vector<32xf4E2M1FN>
          %385 = vector.bitcast %353 : vector<16xi8> to vector<32xf4E2M1FN>
          %386 = vector.bitcast %354 : vector<16xi8> to vector<32xf4E2M1FN>
          %387 = vector.bitcast %355 : vector<16xi8> to vector<32xf4E2M1FN>
          %388 = vector.bitcast %356 : vector<16xi8> to vector<32xf4E2M1FN>
          %389 = vector.bitcast %357 : vector<16xi8> to vector<32xf4E2M1FN>
          %390 = vector.bitcast %358 : vector<16xi8> to vector<32xf4E2M1FN>
          %391 = vector.bitcast %359 : vector<16xi8> to vector<32xf4E2M1FN>
          %392 = vector.bitcast %360 : vector<16xi8> to vector<32xf4E2M1FN>
          %393 = vector.bitcast %361 : vector<16xi8> to vector<32xf4E2M1FN>
          %394 = vector.bitcast %362 : vector<16xi8> to vector<32xf4E2M1FN>
          %395 = vector.bitcast %363 : vector<16xi8> to vector<32xf4E2M1FN>
          %396 = vector.bitcast %364 : vector<16xi8> to vector<32xf4E2M1FN>
          %397 = vector.bitcast %365 : vector<16xi8> to vector<32xf4E2M1FN>
          %398 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %382) + %arg6 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %399 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %383) + %398 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %400 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %384) + %arg7 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %401 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %385) + %400 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %402 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %386) + %arg8 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %403 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %387) + %402 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %404 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %388) + %arg9 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %405 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %389) + %404 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %406 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %390) + %arg10 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %407 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %391) + %406 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %408 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %392) + %arg11 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %409 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %393) + %408 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %410 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %394) + %arg12 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %411 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %395) + %410 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %412 = amdgpu.scaled_mfma(%cst[0] * %374) * (%cst[0] * %396) + %arg13 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %413 = amdgpu.scaled_mfma(%cst[0] * %375) * (%cst[0] * %397) + %412 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %414 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %382) + %arg14 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %415 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %383) + %414 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %416 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %384) + %arg15 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %417 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %385) + %416 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %418 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %386) + %arg16 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %419 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %387) + %418 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %420 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %388) + %arg17 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %421 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %389) + %420 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %422 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %390) + %arg18 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %423 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %391) + %422 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %424 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %392) + %arg19 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %425 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %393) + %424 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %426 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %394) + %arg20 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %427 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %395) + %426 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %428 = amdgpu.scaled_mfma(%cst[0] * %376) * (%cst[0] * %396) + %arg21 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %429 = amdgpu.scaled_mfma(%cst[0] * %377) * (%cst[0] * %397) + %428 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %430 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %382) + %arg22 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %431 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %383) + %430 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %432 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %384) + %arg23 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %433 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %385) + %432 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %434 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %386) + %arg24 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %435 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %387) + %434 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %436 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %388) + %arg25 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %437 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %389) + %436 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %438 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %390) + %arg26 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %439 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %391) + %438 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %440 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %392) + %arg27 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %441 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %393) + %440 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %442 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %394) + %arg28 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %443 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %395) + %442 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %444 = amdgpu.scaled_mfma(%cst[0] * %378) * (%cst[0] * %396) + %arg29 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %445 = amdgpu.scaled_mfma(%cst[0] * %379) * (%cst[0] * %397) + %444 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %446 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %382) + %arg30 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %447 = amdgpu.scaled_mfma(%cst[0] * %381) * (%cst[0] * %383) + %446 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %448 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %384) + %arg31 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %449 = amdgpu.scaled_mfma(%cst[0] * %381) * (%cst[0] * %385) + %448 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %450 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %386) + %arg32 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %451 = amdgpu.scaled_mfma(%cst[0] * %381) * (%cst[0] * %387) + %450 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %452 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %388) + %arg33 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %453 = amdgpu.scaled_mfma(%cst[0] * %381) * (%cst[0] * %389) + %452 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %454 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %390) + %arg34 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %455 = amdgpu.scaled_mfma(%cst[0] * %381) * (%cst[0] * %391) + %454 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %456 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %392) + %arg35 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %457 = amdgpu.scaled_mfma(%cst[0] * %381) * (%cst[0] * %393) + %456 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %458 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %394) + %arg36 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %459 = amdgpu.scaled_mfma(%cst[0] * %381) * (%cst[0] * %395) + %458 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %460 = amdgpu.scaled_mfma(%cst[0] * %380) * (%cst[0] * %396) + %arg37 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %461 = amdgpu.scaled_mfma(%cst[0] * %381) * (%cst[0] * %397) + %460 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          scf.yield %399, %401, %403, %405, %407, %409, %411, %413, %415, %417, %419, %421, %423, %425, %427, %429, %431, %433, %435, %437, %439, %441, %443, %445, %447, %449, %451, %453, %455, %457, %459, %461 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %39 = vector.extract_strided_slice %38#0 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %40 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<16384x16384xf32, strided<[16384, 1], offset: ?>>
        %41 = affine.apply #map23()[%block_id_x]
        %42 = affine.apply #map23()[%block_id_y]
        %43 = affine.apply #map24()[%thread_id_x]
        %44 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        %45 = arith.muli %41, %c16384 overflow<nsw> : index
        %46 = arith.muli %43, %c16384 overflow<nsw> : index
        %47 = arith.addi %45, %42 overflow<nsw> : index
        %48 = arith.addi %46, %44 overflow<nsw> : index
        %reinterpret_cast_3 = memref.reinterpret_cast %40 to offset: [%47], sizes: [%c536870910], strides: [1] : memref<16384x16384xf32, strided<[16384, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %49 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_3 validBytes(%c2147483643_i32) : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        vector.store %39, %49[%48] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %50 = vector.extract_strided_slice %38#0 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %51 = affine.apply #map25()[%thread_id_x]
        %52 = arith.muli %51, %c16384 overflow<nsw> : index
        %53 = arith.addi %52, %44 overflow<nsw> : index
        vector.store %50, %49[%53] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %54 = vector.extract_strided_slice %38#0 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %55 = affine.apply #map26()[%thread_id_x]
        %56 = arith.muli %55, %c16384 overflow<nsw> : index
        %57 = arith.addi %56, %44 overflow<nsw> : index
        vector.store %54, %49[%57] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %58 = vector.extract_strided_slice %38#0 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %59 = affine.apply #map27()[%thread_id_x]
        %60 = arith.muli %59, %c16384 overflow<nsw> : index
        %61 = arith.addi %60, %44 overflow<nsw> : index
        vector.store %58, %49[%61] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %62 = vector.extract_strided_slice %38#1 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %63 = affine.apply #map11()[%thread_id_x, %thread_id_y]
        %64 = arith.addi %46, %63 overflow<nsw> : index
        vector.store %62, %49[%64] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %65 = vector.extract_strided_slice %38#1 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %66 = arith.addi %52, %63 overflow<nsw> : index
        vector.store %65, %49[%66] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %67 = vector.extract_strided_slice %38#1 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %68 = arith.addi %56, %63 overflow<nsw> : index
        vector.store %67, %49[%68] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %69 = vector.extract_strided_slice %38#1 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %70 = arith.addi %60, %63 overflow<nsw> : index
        vector.store %69, %49[%70] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %71 = vector.extract_strided_slice %38#2 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %72 = affine.apply #map12()[%thread_id_x, %thread_id_y]
        %73 = arith.addi %46, %72 overflow<nsw> : index
        vector.store %71, %49[%73] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %74 = vector.extract_strided_slice %38#2 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %75 = arith.addi %52, %72 overflow<nsw> : index
        vector.store %74, %49[%75] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %76 = vector.extract_strided_slice %38#2 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %77 = arith.addi %56, %72 overflow<nsw> : index
        vector.store %76, %49[%77] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %78 = vector.extract_strided_slice %38#2 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %79 = arith.addi %60, %72 overflow<nsw> : index
        vector.store %78, %49[%79] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %80 = vector.extract_strided_slice %38#3 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %81 = affine.apply #map13()[%thread_id_x, %thread_id_y]
        %82 = arith.addi %46, %81 overflow<nsw> : index
        vector.store %80, %49[%82] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %83 = vector.extract_strided_slice %38#3 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %84 = arith.addi %52, %81 overflow<nsw> : index
        vector.store %83, %49[%84] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %85 = vector.extract_strided_slice %38#3 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %86 = arith.addi %56, %81 overflow<nsw> : index
        vector.store %85, %49[%86] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %87 = vector.extract_strided_slice %38#3 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %88 = arith.addi %60, %81 overflow<nsw> : index
        vector.store %87, %49[%88] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %89 = vector.extract_strided_slice %38#4 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %90 = affine.apply #map14()[%thread_id_x, %thread_id_y]
        %91 = arith.addi %46, %90 overflow<nsw> : index
        vector.store %89, %49[%91] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %92 = vector.extract_strided_slice %38#4 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %93 = arith.addi %52, %90 overflow<nsw> : index
        vector.store %92, %49[%93] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %94 = vector.extract_strided_slice %38#4 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %95 = arith.addi %56, %90 overflow<nsw> : index
        vector.store %94, %49[%95] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %96 = vector.extract_strided_slice %38#4 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %97 = arith.addi %60, %90 overflow<nsw> : index
        vector.store %96, %49[%97] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %98 = vector.extract_strided_slice %38#5 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %99 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %100 = arith.addi %46, %99 overflow<nsw> : index
        vector.store %98, %49[%100] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %101 = vector.extract_strided_slice %38#5 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %102 = arith.addi %52, %99 overflow<nsw> : index
        vector.store %101, %49[%102] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %103 = vector.extract_strided_slice %38#5 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %104 = arith.addi %56, %99 overflow<nsw> : index
        vector.store %103, %49[%104] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %105 = vector.extract_strided_slice %38#5 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %106 = arith.addi %60, %99 overflow<nsw> : index
        vector.store %105, %49[%106] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %107 = vector.extract_strided_slice %38#6 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %108 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %109 = arith.addi %46, %108 overflow<nsw> : index
        vector.store %107, %49[%109] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %110 = vector.extract_strided_slice %38#6 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %111 = arith.addi %52, %108 overflow<nsw> : index
        vector.store %110, %49[%111] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %112 = vector.extract_strided_slice %38#6 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %113 = arith.addi %56, %108 overflow<nsw> : index
        vector.store %112, %49[%113] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %114 = vector.extract_strided_slice %38#6 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %115 = arith.addi %60, %108 overflow<nsw> : index
        vector.store %114, %49[%115] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %116 = vector.extract_strided_slice %38#7 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %117 = affine.apply #map17()[%thread_id_x, %thread_id_y]
        %118 = arith.addi %46, %117 overflow<nsw> : index
        vector.store %116, %49[%118] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %119 = vector.extract_strided_slice %38#7 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %120 = arith.addi %52, %117 overflow<nsw> : index
        vector.store %119, %49[%120] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %121 = vector.extract_strided_slice %38#7 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %122 = arith.addi %56, %117 overflow<nsw> : index
        vector.store %121, %49[%122] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %123 = vector.extract_strided_slice %38#7 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %124 = arith.addi %60, %117 overflow<nsw> : index
        vector.store %123, %49[%124] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %125 = vector.extract_strided_slice %38#8 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %126 = affine.apply #map28()[%thread_id_x]
        %127 = arith.muli %126, %c16384 overflow<nsw> : index
        %128 = arith.addi %127, %44 overflow<nsw> : index
        vector.store %125, %49[%128] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %129 = vector.extract_strided_slice %38#8 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %130 = affine.apply #map29()[%thread_id_x]
        %131 = arith.muli %130, %c16384 overflow<nsw> : index
        %132 = arith.addi %131, %44 overflow<nsw> : index
        vector.store %129, %49[%132] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %133 = vector.extract_strided_slice %38#8 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %134 = affine.apply #map30()[%thread_id_x]
        %135 = arith.muli %134, %c16384 overflow<nsw> : index
        %136 = arith.addi %135, %44 overflow<nsw> : index
        vector.store %133, %49[%136] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %137 = vector.extract_strided_slice %38#8 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %138 = affine.apply #map31()[%thread_id_x]
        %139 = arith.muli %138, %c16384 overflow<nsw> : index
        %140 = arith.addi %139, %44 overflow<nsw> : index
        vector.store %137, %49[%140] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %141 = vector.extract_strided_slice %38#9 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %142 = arith.addi %127, %63 overflow<nsw> : index
        vector.store %141, %49[%142] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %143 = vector.extract_strided_slice %38#9 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %144 = arith.addi %131, %63 overflow<nsw> : index
        vector.store %143, %49[%144] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %145 = vector.extract_strided_slice %38#9 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %146 = arith.addi %135, %63 overflow<nsw> : index
        vector.store %145, %49[%146] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %147 = vector.extract_strided_slice %38#9 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %148 = arith.addi %139, %63 overflow<nsw> : index
        vector.store %147, %49[%148] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %149 = vector.extract_strided_slice %38#10 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %150 = arith.addi %127, %72 overflow<nsw> : index
        vector.store %149, %49[%150] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %151 = vector.extract_strided_slice %38#10 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %152 = arith.addi %131, %72 overflow<nsw> : index
        vector.store %151, %49[%152] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %153 = vector.extract_strided_slice %38#10 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %154 = arith.addi %135, %72 overflow<nsw> : index
        vector.store %153, %49[%154] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %155 = vector.extract_strided_slice %38#10 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %156 = arith.addi %139, %72 overflow<nsw> : index
        vector.store %155, %49[%156] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %157 = vector.extract_strided_slice %38#11 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %158 = arith.addi %127, %81 overflow<nsw> : index
        vector.store %157, %49[%158] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %159 = vector.extract_strided_slice %38#11 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %160 = arith.addi %131, %81 overflow<nsw> : index
        vector.store %159, %49[%160] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %161 = vector.extract_strided_slice %38#11 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %162 = arith.addi %135, %81 overflow<nsw> : index
        vector.store %161, %49[%162] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %163 = vector.extract_strided_slice %38#11 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %164 = arith.addi %139, %81 overflow<nsw> : index
        vector.store %163, %49[%164] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %165 = vector.extract_strided_slice %38#12 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %166 = arith.addi %127, %90 overflow<nsw> : index
        vector.store %165, %49[%166] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %167 = vector.extract_strided_slice %38#12 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %168 = arith.addi %131, %90 overflow<nsw> : index
        vector.store %167, %49[%168] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %169 = vector.extract_strided_slice %38#12 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %170 = arith.addi %135, %90 overflow<nsw> : index
        vector.store %169, %49[%170] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %171 = vector.extract_strided_slice %38#12 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %172 = arith.addi %139, %90 overflow<nsw> : index
        vector.store %171, %49[%172] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %173 = vector.extract_strided_slice %38#13 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %174 = arith.addi %127, %99 overflow<nsw> : index
        vector.store %173, %49[%174] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %175 = vector.extract_strided_slice %38#13 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %176 = arith.addi %131, %99 overflow<nsw> : index
        vector.store %175, %49[%176] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %177 = vector.extract_strided_slice %38#13 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %178 = arith.addi %135, %99 overflow<nsw> : index
        vector.store %177, %49[%178] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %179 = vector.extract_strided_slice %38#13 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %180 = arith.addi %139, %99 overflow<nsw> : index
        vector.store %179, %49[%180] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %181 = vector.extract_strided_slice %38#14 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %182 = arith.addi %127, %108 overflow<nsw> : index
        vector.store %181, %49[%182] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %183 = vector.extract_strided_slice %38#14 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %184 = arith.addi %131, %108 overflow<nsw> : index
        vector.store %183, %49[%184] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %185 = vector.extract_strided_slice %38#14 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %186 = arith.addi %135, %108 overflow<nsw> : index
        vector.store %185, %49[%186] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %187 = vector.extract_strided_slice %38#14 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %188 = arith.addi %139, %108 overflow<nsw> : index
        vector.store %187, %49[%188] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %189 = vector.extract_strided_slice %38#15 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %190 = arith.addi %127, %117 overflow<nsw> : index
        vector.store %189, %49[%190] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %191 = vector.extract_strided_slice %38#15 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %192 = arith.addi %131, %117 overflow<nsw> : index
        vector.store %191, %49[%192] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %193 = vector.extract_strided_slice %38#15 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %194 = arith.addi %135, %117 overflow<nsw> : index
        vector.store %193, %49[%194] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %195 = vector.extract_strided_slice %38#15 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %196 = arith.addi %139, %117 overflow<nsw> : index
        vector.store %195, %49[%196] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %197 = vector.extract_strided_slice %38#16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %198 = affine.apply #map32()[%thread_id_x]
        %199 = arith.muli %198, %c16384 overflow<nsw> : index
        %200 = arith.addi %199, %44 overflow<nsw> : index
        vector.store %197, %49[%200] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %201 = vector.extract_strided_slice %38#16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %202 = affine.apply #map33()[%thread_id_x]
        %203 = arith.muli %202, %c16384 overflow<nsw> : index
        %204 = arith.addi %203, %44 overflow<nsw> : index
        vector.store %201, %49[%204] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %205 = vector.extract_strided_slice %38#16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %206 = affine.apply #map34()[%thread_id_x]
        %207 = arith.muli %206, %c16384 overflow<nsw> : index
        %208 = arith.addi %207, %44 overflow<nsw> : index
        vector.store %205, %49[%208] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %209 = vector.extract_strided_slice %38#16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %210 = affine.apply #map35()[%thread_id_x]
        %211 = arith.muli %210, %c16384 overflow<nsw> : index
        %212 = arith.addi %211, %44 overflow<nsw> : index
        vector.store %209, %49[%212] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %213 = vector.extract_strided_slice %38#17 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %214 = arith.addi %199, %63 overflow<nsw> : index
        vector.store %213, %49[%214] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %215 = vector.extract_strided_slice %38#17 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %216 = arith.addi %203, %63 overflow<nsw> : index
        vector.store %215, %49[%216] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %217 = vector.extract_strided_slice %38#17 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %218 = arith.addi %207, %63 overflow<nsw> : index
        vector.store %217, %49[%218] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %219 = vector.extract_strided_slice %38#17 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %220 = arith.addi %211, %63 overflow<nsw> : index
        vector.store %219, %49[%220] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %221 = vector.extract_strided_slice %38#18 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %222 = arith.addi %199, %72 overflow<nsw> : index
        vector.store %221, %49[%222] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %223 = vector.extract_strided_slice %38#18 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %224 = arith.addi %203, %72 overflow<nsw> : index
        vector.store %223, %49[%224] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %225 = vector.extract_strided_slice %38#18 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %226 = arith.addi %207, %72 overflow<nsw> : index
        vector.store %225, %49[%226] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %227 = vector.extract_strided_slice %38#18 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %228 = arith.addi %211, %72 overflow<nsw> : index
        vector.store %227, %49[%228] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %229 = vector.extract_strided_slice %38#19 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %230 = arith.addi %199, %81 overflow<nsw> : index
        vector.store %229, %49[%230] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %231 = vector.extract_strided_slice %38#19 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %232 = arith.addi %203, %81 overflow<nsw> : index
        vector.store %231, %49[%232] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %233 = vector.extract_strided_slice %38#19 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %234 = arith.addi %207, %81 overflow<nsw> : index
        vector.store %233, %49[%234] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %235 = vector.extract_strided_slice %38#19 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %236 = arith.addi %211, %81 overflow<nsw> : index
        vector.store %235, %49[%236] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %237 = vector.extract_strided_slice %38#20 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %238 = arith.addi %199, %90 overflow<nsw> : index
        vector.store %237, %49[%238] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %239 = vector.extract_strided_slice %38#20 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %240 = arith.addi %203, %90 overflow<nsw> : index
        vector.store %239, %49[%240] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %241 = vector.extract_strided_slice %38#20 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %242 = arith.addi %207, %90 overflow<nsw> : index
        vector.store %241, %49[%242] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %243 = vector.extract_strided_slice %38#20 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %244 = arith.addi %211, %90 overflow<nsw> : index
        vector.store %243, %49[%244] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %245 = vector.extract_strided_slice %38#21 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %246 = arith.addi %199, %99 overflow<nsw> : index
        vector.store %245, %49[%246] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %247 = vector.extract_strided_slice %38#21 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %248 = arith.addi %203, %99 overflow<nsw> : index
        vector.store %247, %49[%248] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %249 = vector.extract_strided_slice %38#21 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %250 = arith.addi %207, %99 overflow<nsw> : index
        vector.store %249, %49[%250] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %251 = vector.extract_strided_slice %38#21 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %252 = arith.addi %211, %99 overflow<nsw> : index
        vector.store %251, %49[%252] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %253 = vector.extract_strided_slice %38#22 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %254 = arith.addi %199, %108 overflow<nsw> : index
        vector.store %253, %49[%254] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %255 = vector.extract_strided_slice %38#22 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %256 = arith.addi %203, %108 overflow<nsw> : index
        vector.store %255, %49[%256] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %257 = vector.extract_strided_slice %38#22 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %258 = arith.addi %207, %108 overflow<nsw> : index
        vector.store %257, %49[%258] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %259 = vector.extract_strided_slice %38#22 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %260 = arith.addi %211, %108 overflow<nsw> : index
        vector.store %259, %49[%260] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %261 = vector.extract_strided_slice %38#23 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %262 = arith.addi %199, %117 overflow<nsw> : index
        vector.store %261, %49[%262] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %263 = vector.extract_strided_slice %38#23 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %264 = arith.addi %203, %117 overflow<nsw> : index
        vector.store %263, %49[%264] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %265 = vector.extract_strided_slice %38#23 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %266 = arith.addi %207, %117 overflow<nsw> : index
        vector.store %265, %49[%266] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %267 = vector.extract_strided_slice %38#23 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %268 = arith.addi %211, %117 overflow<nsw> : index
        vector.store %267, %49[%268] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %269 = vector.extract_strided_slice %38#24 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %270 = affine.apply #map36()[%thread_id_x]
        %271 = arith.muli %270, %c16384 overflow<nsw> : index
        %272 = arith.addi %271, %44 overflow<nsw> : index
        vector.store %269, %49[%272] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %273 = vector.extract_strided_slice %38#24 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %274 = affine.apply #map37()[%thread_id_x]
        %275 = arith.muli %274, %c16384 overflow<nsw> : index
        %276 = arith.addi %275, %44 overflow<nsw> : index
        vector.store %273, %49[%276] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %277 = vector.extract_strided_slice %38#24 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %278 = affine.apply #map38()[%thread_id_x]
        %279 = arith.muli %278, %c16384 overflow<nsw> : index
        %280 = arith.addi %279, %44 overflow<nsw> : index
        vector.store %277, %49[%280] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %281 = vector.extract_strided_slice %38#24 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %282 = affine.apply #map39()[%thread_id_x]
        %283 = arith.muli %282, %c16384 overflow<nsw> : index
        %284 = arith.addi %283, %44 overflow<nsw> : index
        vector.store %281, %49[%284] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %285 = vector.extract_strided_slice %38#25 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %286 = arith.addi %271, %63 overflow<nsw> : index
        vector.store %285, %49[%286] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %287 = vector.extract_strided_slice %38#25 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %288 = arith.addi %275, %63 overflow<nsw> : index
        vector.store %287, %49[%288] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %289 = vector.extract_strided_slice %38#25 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %290 = arith.addi %279, %63 overflow<nsw> : index
        vector.store %289, %49[%290] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %291 = vector.extract_strided_slice %38#25 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %292 = arith.addi %283, %63 overflow<nsw> : index
        vector.store %291, %49[%292] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %293 = vector.extract_strided_slice %38#26 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %294 = arith.addi %271, %72 overflow<nsw> : index
        vector.store %293, %49[%294] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %295 = vector.extract_strided_slice %38#26 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %296 = arith.addi %275, %72 overflow<nsw> : index
        vector.store %295, %49[%296] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %297 = vector.extract_strided_slice %38#26 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %298 = arith.addi %279, %72 overflow<nsw> : index
        vector.store %297, %49[%298] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %299 = vector.extract_strided_slice %38#26 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %300 = arith.addi %283, %72 overflow<nsw> : index
        vector.store %299, %49[%300] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %301 = vector.extract_strided_slice %38#27 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %302 = arith.addi %271, %81 overflow<nsw> : index
        vector.store %301, %49[%302] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %303 = vector.extract_strided_slice %38#27 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %304 = arith.addi %275, %81 overflow<nsw> : index
        vector.store %303, %49[%304] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %305 = vector.extract_strided_slice %38#27 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %306 = arith.addi %279, %81 overflow<nsw> : index
        vector.store %305, %49[%306] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %307 = vector.extract_strided_slice %38#27 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %308 = arith.addi %283, %81 overflow<nsw> : index
        vector.store %307, %49[%308] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %309 = vector.extract_strided_slice %38#28 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %310 = arith.addi %271, %90 overflow<nsw> : index
        vector.store %309, %49[%310] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %311 = vector.extract_strided_slice %38#28 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %312 = arith.addi %275, %90 overflow<nsw> : index
        vector.store %311, %49[%312] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %313 = vector.extract_strided_slice %38#28 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %314 = arith.addi %279, %90 overflow<nsw> : index
        vector.store %313, %49[%314] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %315 = vector.extract_strided_slice %38#28 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %316 = arith.addi %283, %90 overflow<nsw> : index
        vector.store %315, %49[%316] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %317 = vector.extract_strided_slice %38#29 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %318 = arith.addi %271, %99 overflow<nsw> : index
        vector.store %317, %49[%318] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %319 = vector.extract_strided_slice %38#29 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %320 = arith.addi %275, %99 overflow<nsw> : index
        vector.store %319, %49[%320] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %321 = vector.extract_strided_slice %38#29 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %322 = arith.addi %279, %99 overflow<nsw> : index
        vector.store %321, %49[%322] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %323 = vector.extract_strided_slice %38#29 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %324 = arith.addi %283, %99 overflow<nsw> : index
        vector.store %323, %49[%324] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %325 = vector.extract_strided_slice %38#30 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %326 = arith.addi %271, %108 overflow<nsw> : index
        vector.store %325, %49[%326] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %327 = vector.extract_strided_slice %38#30 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %328 = arith.addi %275, %108 overflow<nsw> : index
        vector.store %327, %49[%328] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %329 = vector.extract_strided_slice %38#30 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %330 = arith.addi %279, %108 overflow<nsw> : index
        vector.store %329, %49[%330] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %331 = vector.extract_strided_slice %38#30 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %332 = arith.addi %283, %108 overflow<nsw> : index
        vector.store %331, %49[%332] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %333 = vector.extract_strided_slice %38#31 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %334 = arith.addi %271, %117 overflow<nsw> : index
        vector.store %333, %49[%334] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %335 = vector.extract_strided_slice %38#31 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %336 = arith.addi %275, %117 overflow<nsw> : index
        vector.store %335, %49[%336] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %337 = vector.extract_strided_slice %38#31 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %338 = arith.addi %279, %117 overflow<nsw> : index
        vector.store %337, %49[%338] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        %339 = vector.extract_strided_slice %38#31 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %340 = arith.addi %283, %117 overflow<nsw> : index
        vector.store %339, %49[%340] : memref<?xf32, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<16384x8192xi8>, %arg1: tensor<16384x512xi8>, %arg2: tensor<16384x8192xi8>, %arg3: tensor<16384x512xi8>, %arg4: tensor<16384x16384xf32>) -> tensor<16384x16384xf32> {
    %0 = flow.dispatch @gemm_afp4_wfp4_wave::@gemm_afp4_wfp4_wave(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x16384xf32>) -> %arg4
    return %0 : tensor<16384x16384xf32>
  }
}
