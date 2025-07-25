#map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
#map1 = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 8) * 128)>
#map2 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map3 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map4 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
#map5 = affine_map<()[s0, s1, s2] -> (s1 * 128 + s2 * 256 + s0 floordiv 2 - ((s1 * 128 + s0 floordiv 2) floordiv 256) * 256)>
#map6 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 2) * 8)>
#map7 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
#map8 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map9 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map10 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
#map11 = affine_map<()[s0, s1] -> ((s1 * 128 + s0 floordiv 2) mod 256)>
#map12 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map13 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map14 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map15 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map16 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map17 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map18 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map19 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map20 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
#map21 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
#map22 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
#map23 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
#map24 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
#map25 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
#map26 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
#map27 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>
#map28 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 - (s1 floordiv 2) * 8 + 8)>
#map29 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 - (s1 floordiv 8) * 128 + 128)>
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
        %c2147483645_i32 = arith.constant 2147483645 : i32
        %c1073741822 = arith.constant 1073741822 : index
        %c16384 = arith.constant 16384 : index
        %c63 = arith.constant 63 : index
        %c512 = arith.constant 512 : index
        %c2147483646_i32 = arith.constant 2147483646 : i32
        %c2147483646 = arith.constant 2147483646 : index
        %c8192 = arith.constant 8192 : index
        %c1 = arith.constant 1 : index
        %c34816 = arith.constant 34816 : index
        %c73728 = arith.constant 73728 : index
        %c0 = arith.constant 0 : index
        %c69632 = arith.constant 69632 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %block_id_x = gpu.block_id  x upper_bound 64
        %block_id_y = gpu.block_id  y upper_bound 64
        %thread_id_x = gpu.thread_id  x upper_bound 256
        %thread_id_y = gpu.thread_id  y upper_bound 2
        %alloc = memref.alloc() : memref<81920xi8, #gpu.address_space<workgroup>>
        %view = memref.view %alloc[%c69632][] : memref<81920xi8, #gpu.address_space<workgroup>> to memref<256x16xi8, #gpu.address_space<workgroup>>
        %view_0 = memref.view %alloc[%c0][] : memref<81920xi8, #gpu.address_space<workgroup>> to memref<256x144xi8, #gpu.address_space<workgroup>>
        %view_1 = memref.view %alloc[%c73728][] : memref<81920xi8, #gpu.address_space<workgroup>> to memref<256x16xi8, #gpu.address_space<workgroup>>
        %view_2 = memref.view %alloc[%c34816][] : memref<81920xi8, #gpu.address_space<workgroup>> to memref<256x144xi8, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x8192xi8, strided<[8192, 1], offset: ?>>
        %1 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
        %2 = affine.apply #map1()[%thread_id_x]
        %3 = arith.muli %1, %c8192 overflow<nsw> : index
        %4 = arith.addi %3, %2 overflow<nsw> : index
        %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x8192xi8, strided<[8192, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %5 = amdgpu.fat_raw_buffer_cast %reinterpret_cast validBytes(%c2147483646_i32) cacheSwizzleStride(%c-8192_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %6 = vector.load %5[%4] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %7 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_x]
        %8 = arith.muli %7, %c8192 overflow<nsw> : index
        %9 = arith.addi %8, %2 overflow<nsw> : index
        %10 = vector.load %5[%9] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %11 = affine.apply #map3()[%thread_id_x, %thread_id_y, %block_id_x]
        %12 = arith.muli %11, %c8192 overflow<nsw> : index
        %13 = arith.addi %12, %2 overflow<nsw> : index
        %14 = vector.load %5[%13] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %15 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_x]
        %16 = arith.muli %15, %c8192 overflow<nsw> : index
        %17 = arith.addi %16, %2 overflow<nsw> : index
        %18 = vector.load %5[%17] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %19 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<16384x512xi8, strided<[512, 1], offset: ?>>
        %20 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
        %21 = affine.apply #map6()[%thread_id_x]
        %22 = arith.muli %20, %c512 overflow<nsw> : index
        %23 = arith.addi %22, %21 overflow<nsw> : index
        %reinterpret_cast_3 = memref.reinterpret_cast %19 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x512xi8, strided<[512, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %24 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_3 validBytes(%c2147483646_i32) cacheSwizzleStride(%c512_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %25 = vector.load %24[%23] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %26 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<16384x8192xi8, strided<[8192, 1], offset: ?>>
        %27 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
        %28 = arith.muli %27, %c8192 overflow<nsw> : index
        %29 = arith.addi %28, %2 overflow<nsw> : index
        %reinterpret_cast_4 = memref.reinterpret_cast %26 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x8192xi8, strided<[8192, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %30 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_4 validBytes(%c2147483646_i32) cacheSwizzleStride(%c-8192_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %31 = vector.load %30[%29] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %32 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y]
        %33 = arith.muli %32, %c8192 overflow<nsw> : index
        %34 = arith.addi %33, %2 overflow<nsw> : index
        %35 = vector.load %30[%34] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %36 = affine.apply #map3()[%thread_id_x, %thread_id_y, %block_id_y]
        %37 = arith.muli %36, %c8192 overflow<nsw> : index
        %38 = arith.addi %37, %2 overflow<nsw> : index
        %39 = vector.load %30[%38] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %40 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_y]
        %41 = arith.muli %40, %c8192 overflow<nsw> : index
        %42 = arith.addi %41, %2 overflow<nsw> : index
        %43 = vector.load %30[%42] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
        %44 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<16384x512xi8, strided<[512, 1], offset: ?>>
        %45 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
        %46 = arith.muli %45, %c512 overflow<nsw> : index
        %47 = arith.addi %46, %21 overflow<nsw> : index
        %reinterpret_cast_5 = memref.reinterpret_cast %44 to offset: [%c0], sizes: [%c2147483646], strides: [1] : memref<16384x512xi8, strided<[512, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
        %48 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483646_i32) cacheSwizzleStride(%c512_i14) : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %49 = vector.load %48[%47] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
        %50 = affine.apply #map7()[%thread_id_x, %thread_id_y]
        vector.store %6, %view_2[%50, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %51 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        vector.store %10, %view_2[%51, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %52 = affine.apply #map9()[%thread_id_x, %thread_id_y]
        vector.store %14, %view_2[%52, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %53 = affine.apply #map10()[%thread_id_x, %thread_id_y]
        vector.store %18, %view_2[%53, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %54 = affine.apply #map11()[%thread_id_x, %thread_id_y]
        // vector.store %25, %view_1[%54, %21] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<4xi8>
        vector.store %31, %view_0[%50, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %35, %view_0[%51, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %39, %view_0[%52, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %43, %view_0[%53, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        // vector.store %49, %view[%54, %21] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<4xi8>
        %55 = affine.apply #map12()[%thread_id_x, %thread_id_y]
        %56 = affine.apply #map13()[%thread_id_x]
        %57 = affine.apply #map14()[%thread_id_x, %thread_id_y]
        %58 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %59 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %60 = affine.apply #map17()[%thread_id_x, %thread_id_y]
        %61 = affine.apply #map18()[%thread_id_x, %thread_id_y]
        %62 = affine.apply #map19()[%thread_id_x, %thread_id_y]
        %63 = affine.apply #map20()[%thread_id_x, %thread_id_y]
        %64 = affine.apply #map21()[%thread_id_x]
        %65 = affine.apply #map22()[%thread_id_x]
        %66 = affine.apply #map23()[%thread_id_x]
        %67 = affine.apply #map24()[%thread_id_x]
        %68 = affine.apply #map25()[%thread_id_x]
        %69 = affine.apply #map26()[%thread_id_x]
        %70 = affine.apply #map27()[%thread_id_x]
        %71:32 = scf.for %arg5 = %c0 to %c63 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %598 = affine.apply #map28()[%arg5, %thread_id_x]
          %599 = arith.addi %46, %598 overflow<nsw> : index
          %600 = vector.load %48[%599] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
          amdgpu.lds_barrier

          // // Scale A
          // %601 = vector.load %view[%55, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %602 = vector.load %view[%57, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %603 = vector.bitcast %601 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %604 = vector.load %view[%58, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %605 = vector.bitcast %602 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %606 = vector.load %view[%59, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %607 = vector.bitcast %604 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %608 = vector.load %view[%60, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %609 = vector.bitcast %606 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %610 = vector.load %view[%61, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %611 = vector.bitcast %608 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %612 = vector.load %view[%62, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %613 = vector.bitcast %610 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %614 = vector.load %view[%63, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %615 = vector.bitcast %612 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %616 = vector.load %view[%55, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %617 = vector.bitcast %614 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %618 = vector.load %view[%57, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %619 = vector.bitcast %616 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %620 = vector.load %view[%58, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %621 = vector.bitcast %618 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %622 = vector.load %view[%59, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %623 = vector.bitcast %620 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %624 = vector.load %view[%60, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %625 = vector.bitcast %622 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %626 = vector.load %view[%61, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %627 = vector.bitcast %624 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %628 = vector.load %view[%62, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %629 = vector.bitcast %626 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %630 = vector.load %view[%63, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          // %631 = vector.bitcast %628 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %634 = vector.bitcast %630 : vector<1xi8> to vector<1xf8E8M0FNU>

          // Scale A to 1
          %603 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %605 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %607 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %609 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %611 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %613 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %615 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %617 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %619 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %621 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %623 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %625 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %627 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %629 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %631 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>
          %634 = arith.constant dense<1.0> :vector<1xf8E8M0FNU>

          %632 = arith.addi %22, %598 overflow<nsw> : index
          %633 = vector.load %24[%632] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>


          %635 = vector.load %view_1[%65, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %636 = vector.load %view_1[%66, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %638 = vector.load %view_1[%67, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %640 = vector.load %view_1[%68, %56] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %642 = vector.load %view_1[%65, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %644 = vector.load %view_1[%66, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %646 = vector.load %view_1[%67, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          %648 = vector.load %view_1[%68, %64] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
          
          // %637 = vector.bitcast %635 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %639 = vector.bitcast %636 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %641 = vector.bitcast %638 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %643 = vector.bitcast %640 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %645 = vector.bitcast %642 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %647 = vector.bitcast %644 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %649 = vector.bitcast %646 : vector<1xi8> to vector<1xf8E8M0FNU>
          // %653 = vector.bitcast %648 : vector<1xi8> to vector<1xf8E8M0FNU>

          //Scale B
          %637 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
          %639 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
          %641 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
          %643 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
          %645 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
          %647 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
          %649 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
          %653 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>

          %650 = affine.apply #map29()[%arg5, %thread_id_x]
          %651 = arith.addi %33, %650 overflow<nsw> : index
          %652 = vector.load %30[%651] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          
          %654 = arith.addi %37, %650 overflow<nsw> : index
          %655 = vector.load %30[%654] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %656 = arith.addi %28, %650 overflow<nsw> : index
          %657 = vector.load %30[%656] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %658 = arith.addi %41, %650 overflow<nsw> : index
          %659 = vector.load %30[%658] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %660 = vector.load %view_0[%55, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %661 = vector.load %view_0[%57, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %662 = vector.bitcast %660 : vector<16xi8> to vector<32xf4E2M1FN>
          %663 = vector.load %view_0[%58, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %664 = vector.bitcast %661 : vector<16xi8> to vector<32xf4E2M1FN>
          %665 = vector.load %view_0[%59, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %666 = vector.bitcast %663 : vector<16xi8> to vector<32xf4E2M1FN>
          %667 = vector.load %view_0[%60, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %668 = vector.bitcast %665 : vector<16xi8> to vector<32xf4E2M1FN>
          %669 = vector.load %view_0[%61, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %670 = vector.bitcast %667 : vector<16xi8> to vector<32xf4E2M1FN>
          %671 = vector.load %view_0[%62, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %672 = vector.bitcast %669 : vector<16xi8> to vector<32xf4E2M1FN>
          %673 = vector.load %view_0[%63, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %674 = vector.bitcast %671 : vector<16xi8> to vector<32xf4E2M1FN>
          %675 = vector.load %view_0[%55, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %676 = vector.bitcast %673 : vector<16xi8> to vector<32xf4E2M1FN>
          %677 = vector.load %view_0[%57, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %678 = vector.bitcast %675 : vector<16xi8> to vector<32xf4E2M1FN>
          %679 = vector.load %view_0[%58, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %680 = vector.bitcast %677 : vector<16xi8> to vector<32xf4E2M1FN>
          %681 = vector.load %view_0[%59, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %682 = vector.bitcast %679 : vector<16xi8> to vector<32xf4E2M1FN>
          %683 = vector.load %view_0[%60, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %684 = vector.bitcast %681 : vector<16xi8> to vector<32xf4E2M1FN>
          %685 = vector.load %view_0[%61, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %686 = vector.bitcast %683 : vector<16xi8> to vector<32xf4E2M1FN>
          %687 = vector.load %view_0[%62, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %688 = vector.bitcast %685 : vector<16xi8> to vector<32xf4E2M1FN>
          %689 = vector.load %view_0[%63, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %690 = vector.bitcast %687 : vector<16xi8> to vector<32xf4E2M1FN>
          %691 = arith.addi %12, %650 overflow<nsw> : index
          %692 = vector.load %5[%691] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %693 = vector.bitcast %689 : vector<16xi8> to vector<32xf4E2M1FN>
          %694 = arith.addi %16, %650 overflow<nsw> : index
          %695 = vector.load %5[%694] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %696 = arith.addi %3, %650 overflow<nsw> : index
          %697 = vector.load %5[%696] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %698 = arith.addi %8, %650 overflow<nsw> : index
          %699 = vector.load %5[%698] : memref<?xi8, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
          %700 = vector.load %view_2[%65, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %701 = vector.load %view_2[%66, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %702 = vector.bitcast %700 : vector<16xi8> to vector<32xf4E2M1FN>
          %703 = vector.load %view_2[%67, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %704 = vector.bitcast %701 : vector<16xi8> to vector<32xf4E2M1FN>
          %705 = vector.load %view_2[%68, %69] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %706 = vector.bitcast %703 : vector<16xi8> to vector<32xf4E2M1FN>
          %707 = vector.load %view_2[%65, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %708 = vector.bitcast %705 : vector<16xi8> to vector<32xf4E2M1FN>
          %709 = vector.load %view_2[%66, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %710 = vector.bitcast %707 : vector<16xi8> to vector<32xf4E2M1FN>
          %711 = vector.load %view_2[%67, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %712 = vector.bitcast %709 : vector<16xi8> to vector<32xf4E2M1FN>
          %713 = vector.load %view_2[%68, %70] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %714 = vector.bitcast %711 : vector<16xi8> to vector<32xf4E2M1FN>
          %715 = vector.bitcast %713 : vector<16xi8> to vector<32xf4E2M1FN>
          %716 = vector.extractelement %637[%c0 : index] : vector<1xf8E8M0FNU>
          %717 = vector.extractelement %603[%c0 : index] : vector<1xf8E8M0FNU>
          %718 = amdgpu.scaled_mfma(%716[0] * %702) * (%717[0] * %662) + %arg6 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %719 = vector.extractelement %605[%c0 : index] : vector<1xf8E8M0FNU>
          %720 = amdgpu.scaled_mfma(%716[0] * %702) * (%719[0] * %664) + %arg7 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %721 = vector.extractelement %607[%c0 : index] : vector<1xf8E8M0FNU>
          %722 = amdgpu.scaled_mfma(%716[0] * %702) * (%721[0] * %666) + %arg8 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %723 = vector.extractelement %609[%c0 : index] : vector<1xf8E8M0FNU>
          %724 = amdgpu.scaled_mfma(%716[0] * %702) * (%723[0] * %668) + %arg9 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %725 = vector.extractelement %611[%c0 : index] : vector<1xf8E8M0FNU>
          %726 = amdgpu.scaled_mfma(%716[0] * %702) * (%725[0] * %670) + %arg10 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %727 = vector.extractelement %613[%c0 : index] : vector<1xf8E8M0FNU>
          %728 = amdgpu.scaled_mfma(%716[0] * %702) * (%727[0] * %672) + %arg11 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %729 = vector.extractelement %615[%c0 : index] : vector<1xf8E8M0FNU>
          %730 = amdgpu.scaled_mfma(%716[0] * %702) * (%729[0] * %674) + %arg12 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %731 = vector.extractelement %617[%c0 : index] : vector<1xf8E8M0FNU>
          %732 = amdgpu.scaled_mfma(%716[0] * %702) * (%731[0] * %676) + %arg13 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %733 = vector.extractelement %639[%c0 : index] : vector<1xf8E8M0FNU>
          %734 = amdgpu.scaled_mfma(%733[0] * %704) * (%717[0] * %662) + %arg14 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %735 = amdgpu.scaled_mfma(%733[0] * %704) * (%719[0] * %664) + %arg15 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %736 = amdgpu.scaled_mfma(%733[0] * %704) * (%721[0] * %666) + %arg16 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %737 = amdgpu.scaled_mfma(%733[0] * %704) * (%723[0] * %668) + %arg17 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %738 = amdgpu.scaled_mfma(%733[0] * %704) * (%725[0] * %670) + %arg18 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %739 = amdgpu.scaled_mfma(%733[0] * %704) * (%727[0] * %672) + %arg19 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %740 = amdgpu.scaled_mfma(%733[0] * %704) * (%729[0] * %674) + %arg20 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %741 = amdgpu.scaled_mfma(%733[0] * %704) * (%731[0] * %676) + %arg21 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %742 = vector.extractelement %641[%c0 : index] : vector<1xf8E8M0FNU>
          %743 = amdgpu.scaled_mfma(%742[0] * %706) * (%717[0] * %662) + %arg22 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %744 = amdgpu.scaled_mfma(%742[0] * %706) * (%719[0] * %664) + %arg23 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %745 = amdgpu.scaled_mfma(%742[0] * %706) * (%721[0] * %666) + %arg24 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %746 = amdgpu.scaled_mfma(%742[0] * %706) * (%723[0] * %668) + %arg25 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %747 = amdgpu.scaled_mfma(%742[0] * %706) * (%725[0] * %670) + %arg26 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %748 = amdgpu.scaled_mfma(%742[0] * %706) * (%727[0] * %672) + %arg27 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %749 = amdgpu.scaled_mfma(%742[0] * %706) * (%729[0] * %674) + %arg28 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %750 = amdgpu.scaled_mfma(%742[0] * %706) * (%731[0] * %676) + %arg29 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %751 = vector.extractelement %643[%c0 : index] : vector<1xf8E8M0FNU>
          %752 = amdgpu.scaled_mfma(%751[0] * %708) * (%717[0] * %662) + %arg30 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %753 = amdgpu.scaled_mfma(%751[0] * %708) * (%719[0] * %664) + %arg31 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %754 = amdgpu.scaled_mfma(%751[0] * %708) * (%721[0] * %666) + %arg32 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %755 = amdgpu.scaled_mfma(%751[0] * %708) * (%723[0] * %668) + %arg33 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %756 = amdgpu.scaled_mfma(%751[0] * %708) * (%725[0] * %670) + %arg34 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %757 = amdgpu.scaled_mfma(%751[0] * %708) * (%727[0] * %672) + %arg35 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %758 = amdgpu.scaled_mfma(%751[0] * %708) * (%729[0] * %674) + %arg36 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %759 = amdgpu.scaled_mfma(%751[0] * %708) * (%731[0] * %676) + %arg37 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %760 = vector.extractelement %645[%c0 : index] : vector<1xf8E8M0FNU>
          %761 = vector.extractelement %619[%c0 : index] : vector<1xf8E8M0FNU>
          %762 = amdgpu.scaled_mfma(%760[0] * %710) * (%761[0] * %678) + %718 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %763 = vector.extractelement %621[%c0 : index] : vector<1xf8E8M0FNU>
          %764 = amdgpu.scaled_mfma(%760[0] * %710) * (%763[0] * %680) + %720 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %765 = vector.extractelement %623[%c0 : index] : vector<1xf8E8M0FNU>
          %766 = amdgpu.scaled_mfma(%760[0] * %710) * (%765[0] * %682) + %722 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %767 = vector.extractelement %625[%c0 : index] : vector<1xf8E8M0FNU>
          %768 = amdgpu.scaled_mfma(%760[0] * %710) * (%767[0] * %684) + %724 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %769 = vector.extractelement %627[%c0 : index] : vector<1xf8E8M0FNU>
          %770 = amdgpu.scaled_mfma(%760[0] * %710) * (%769[0] * %686) + %726 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %771 = vector.extractelement %629[%c0 : index] : vector<1xf8E8M0FNU>
          %772 = amdgpu.scaled_mfma(%760[0] * %710) * (%771[0] * %688) + %728 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %773 = vector.extractelement %631[%c0 : index] : vector<1xf8E8M0FNU>
          %774 = amdgpu.scaled_mfma(%760[0] * %710) * (%773[0] * %690) + %730 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %775 = vector.extractelement %634[%c0 : index] : vector<1xf8E8M0FNU>
          %776 = amdgpu.scaled_mfma(%760[0] * %710) * (%775[0] * %693) + %732 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %777 = vector.extractelement %647[%c0 : index] : vector<1xf8E8M0FNU>
          %778 = amdgpu.scaled_mfma(%777[0] * %712) * (%761[0] * %678) + %734 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %779 = amdgpu.scaled_mfma(%777[0] * %712) * (%763[0] * %680) + %735 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %780 = amdgpu.scaled_mfma(%777[0] * %712) * (%765[0] * %682) + %736 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %781 = amdgpu.scaled_mfma(%777[0] * %712) * (%767[0] * %684) + %737 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %782 = amdgpu.scaled_mfma(%777[0] * %712) * (%769[0] * %686) + %738 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %783 = amdgpu.scaled_mfma(%777[0] * %712) * (%771[0] * %688) + %739 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %784 = amdgpu.scaled_mfma(%777[0] * %712) * (%773[0] * %690) + %740 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %785 = amdgpu.scaled_mfma(%777[0] * %712) * (%775[0] * %693) + %741 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %786 = vector.extractelement %649[%c0 : index] : vector<1xf8E8M0FNU>
          %787 = amdgpu.scaled_mfma(%786[0] * %714) * (%761[0] * %678) + %743 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %788 = amdgpu.scaled_mfma(%786[0] * %714) * (%763[0] * %680) + %744 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %789 = amdgpu.scaled_mfma(%786[0] * %714) * (%765[0] * %682) + %745 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %790 = amdgpu.scaled_mfma(%786[0] * %714) * (%767[0] * %684) + %746 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %791 = amdgpu.scaled_mfma(%786[0] * %714) * (%769[0] * %686) + %747 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %792 = amdgpu.scaled_mfma(%786[0] * %714) * (%771[0] * %688) + %748 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %793 = amdgpu.scaled_mfma(%786[0] * %714) * (%773[0] * %690) + %749 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %794 = amdgpu.scaled_mfma(%786[0] * %714) * (%775[0] * %693) + %750 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %795 = vector.extractelement %653[%c0 : index] : vector<1xf8E8M0FNU>
          %796 = amdgpu.scaled_mfma(%795[0] * %715) * (%761[0] * %678) + %752 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %797 = amdgpu.scaled_mfma(%795[0] * %715) * (%763[0] * %680) + %753 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %798 = amdgpu.scaled_mfma(%795[0] * %715) * (%765[0] * %682) + %754 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %799 = amdgpu.scaled_mfma(%795[0] * %715) * (%767[0] * %684) + %755 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %800 = amdgpu.scaled_mfma(%795[0] * %715) * (%769[0] * %686) + %756 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %801 = amdgpu.scaled_mfma(%795[0] * %715) * (%771[0] * %688) + %757 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %802 = amdgpu.scaled_mfma(%795[0] * %715) * (%773[0] * %690) + %758 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          %803 = amdgpu.scaled_mfma(%795[0] * %715) * (%775[0] * %693) + %759 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
          amdgpu.lds_barrier
          // vector.store %633, %view_1[%54, %21] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<4xi8>
          // vector.store %600, %view[%54, %21] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<4xi8>
          vector.store %699, %view_2[%51, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %697, %view_2[%50, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %695, %view_2[%53, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %692, %view_2[%52, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %652, %view_0[%51, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %655, %view_0[%52, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %657, %view_0[%50, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %659, %view_0[%53, %2] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          scf.yield %762, %764, %766, %768, %770, %772, %774, %776, %778, %779, %780, %781, %782, %783, %784, %785, %787, %788, %789, %790, %791, %792, %793, %794, %796, %797, %798, %799, %800, %801, %802, %803 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        amdgpu.lds_barrier
        %72 = affine.apply #map12()[%thread_id_x, %thread_id_y]
        %73 = affine.apply #map13()[%thread_id_x]

        %74 = vector.load %view[%72, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %75 = affine.apply #map21()[%thread_id_x]
        %76 = vector.load %view[%72, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %77 = affine.apply #map14()[%thread_id_x, %thread_id_y]
        %78 = vector.load %view[%77, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %79 = vector.load %view[%77, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %80 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %81 = vector.load %view[%80, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %82 = vector.load %view[%80, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %83 = affine.apply #map16()[%thread_id_x, %thread_id_y]
        %84 = vector.load %view[%83, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %85 = vector.load %view[%83, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %86 = affine.apply #map17()[%thread_id_x, %thread_id_y]
        %87 = vector.load %view[%86, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %88 = vector.load %view[%86, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %89 = affine.apply #map18()[%thread_id_x, %thread_id_y]
        %90 = vector.load %view[%89, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %91 = vector.load %view[%89, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %92 = affine.apply #map19()[%thread_id_x, %thread_id_y]
        %93 = vector.load %view[%92, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %94 = vector.load %view[%92, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %95 = affine.apply #map20()[%thread_id_x, %thread_id_y]
        %96 = vector.load %view[%95, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %97 = vector.load %view[%95, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %98 = affine.apply #map26()[%thread_id_x]
        %99 = vector.load %view_0[%72, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %100 = affine.apply #map27()[%thread_id_x]
        %101 = vector.load %view_0[%72, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %102 = vector.load %view_0[%77, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %103 = vector.load %view_0[%77, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %104 = vector.load %view_0[%80, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %105 = vector.load %view_0[%80, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %106 = vector.load %view_0[%83, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %107 = vector.load %view_0[%83, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %108 = vector.load %view_0[%86, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %109 = vector.load %view_0[%86, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %110 = vector.load %view_0[%89, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %111 = vector.load %view_0[%89, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %112 = vector.load %view_0[%92, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %113 = vector.load %view_0[%92, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %114 = vector.load %view_0[%95, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %115 = vector.load %view_0[%95, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %116 = affine.apply #map22()[%thread_id_x]
        %117 = vector.load %view_1[%116, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %118 = vector.load %view_1[%116, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %119 = affine.apply #map23()[%thread_id_x]
        %120 = vector.load %view_1[%119, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %121 = vector.load %view_1[%119, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %122 = affine.apply #map24()[%thread_id_x]
        %123 = vector.load %view_1[%122, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %124 = vector.load %view_1[%122, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %125 = affine.apply #map25()[%thread_id_x]
        %126 = vector.load %view_1[%125, %73] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %127 = vector.load %view_1[%125, %75] : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
        %128 = vector.load %view_2[%116, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %129 = vector.load %view_2[%116, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %130 = vector.load %view_2[%119, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %131 = vector.load %view_2[%119, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %132 = vector.load %view_2[%122, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %133 = vector.load %view_2[%122, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %134 = vector.load %view_2[%125, %98] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %135 = vector.load %view_2[%125, %100] : memref<256x144xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %136 = vector.bitcast %128 : vector<16xi8> to vector<32xf4E2M1FN>
        %137 = vector.bitcast %129 : vector<16xi8> to vector<32xf4E2M1FN>
        %138 = vector.bitcast %130 : vector<16xi8> to vector<32xf4E2M1FN>
        %139 = vector.bitcast %131 : vector<16xi8> to vector<32xf4E2M1FN>
        %140 = vector.bitcast %132 : vector<16xi8> to vector<32xf4E2M1FN>
        %141 = vector.bitcast %133 : vector<16xi8> to vector<32xf4E2M1FN>
        %142 = vector.bitcast %134 : vector<16xi8> to vector<32xf4E2M1FN>
        %143 = vector.bitcast %135 : vector<16xi8> to vector<32xf4E2M1FN>

        //Scale B
        // %144 = vector.bitcast %117 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %145 = vector.bitcast %118 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %146 = vector.bitcast %120 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %147 = vector.bitcast %121 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %148 = vector.bitcast %123 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %149 = vector.bitcast %124 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %150 = vector.bitcast %126 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %151 = vector.bitcast %127 : vector<1xi8> to vector<1xf8E8M0FNU>

        %144 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %145 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %146 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %147 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %148 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %149 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %150 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %151 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
       
        %152 = vector.bitcast %99 : vector<16xi8> to vector<32xf4E2M1FN>
        %153 = vector.bitcast %101 : vector<16xi8> to vector<32xf4E2M1FN>
        %154 = vector.bitcast %102 : vector<16xi8> to vector<32xf4E2M1FN>
        %155 = vector.bitcast %103 : vector<16xi8> to vector<32xf4E2M1FN>
        %156 = vector.bitcast %104 : vector<16xi8> to vector<32xf4E2M1FN>
        %157 = vector.bitcast %105 : vector<16xi8> to vector<32xf4E2M1FN>
        %158 = vector.bitcast %106 : vector<16xi8> to vector<32xf4E2M1FN>
        %159 = vector.bitcast %107 : vector<16xi8> to vector<32xf4E2M1FN>
        %160 = vector.bitcast %108 : vector<16xi8> to vector<32xf4E2M1FN>
        %161 = vector.bitcast %109 : vector<16xi8> to vector<32xf4E2M1FN>
        %162 = vector.bitcast %110 : vector<16xi8> to vector<32xf4E2M1FN>
        %163 = vector.bitcast %111 : vector<16xi8> to vector<32xf4E2M1FN>
        %164 = vector.bitcast %112 : vector<16xi8> to vector<32xf4E2M1FN>
        %165 = vector.bitcast %113 : vector<16xi8> to vector<32xf4E2M1FN>
        %166 = vector.bitcast %114 : vector<16xi8> to vector<32xf4E2M1FN>
        %167 = vector.bitcast %115 : vector<16xi8> to vector<32xf4E2M1FN>

        //scale A - ref
        // %168 = vector.bitcast %74 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %169 = vector.bitcast %76 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %170 = vector.bitcast %78 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %171 = vector.bitcast %79 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %172 = vector.bitcast %81 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %173 = vector.bitcast %82 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %174 = vector.bitcast %84 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %175 = vector.bitcast %85 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %176 = vector.bitcast %87 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %177 = vector.bitcast %88 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %178 = vector.bitcast %90 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %179 = vector.bitcast %91 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %180 = vector.bitcast %93 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %181 = vector.bitcast %94 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %182 = vector.bitcast %96 : vector<1xi8> to vector<1xf8E8M0FNU>
        // %183 = vector.bitcast %97 : vector<1xi8> to vector<1xf8E8M0FNU>
        //scale A - CST
        %168 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %169 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %170 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %171 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %172 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %173 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %174 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %175 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %176 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %177 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %178 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %179 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %180 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %181 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %182 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>
        %183 = arith.constant dense<1.0> : vector<1xf8E8M0FNU>


        %184 = vector.extractelement %144[%c0 : index] : vector<1xf8E8M0FNU>
        %185 = vector.extractelement %168[%c0 : index] : vector<1xf8E8M0FNU>
        %186 = amdgpu.scaled_mfma(%184[0] * %136) * (%185[0] * %152) + %71#0 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %187 = vector.extractelement %145[%c0 : index] : vector<1xf8E8M0FNU>
        %188 = vector.extractelement %169[%c0 : index] : vector<1xf8E8M0FNU>
        %189 = amdgpu.scaled_mfma(%187[0] * %137) * (%188[0] * %153) + %186 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %190 = vector.extractelement %170[%c0 : index] : vector<1xf8E8M0FNU>
        %191 = amdgpu.scaled_mfma(%184[0] * %136) * (%190[0] * %154) + %71#1 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %192 = vector.extractelement %171[%c0 : index] : vector<1xf8E8M0FNU>
        %193 = amdgpu.scaled_mfma(%187[0] * %137) * (%192[0] * %155) + %191 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %194 = vector.extractelement %172[%c0 : index] : vector<1xf8E8M0FNU>
        %195 = amdgpu.scaled_mfma(%184[0] * %136) * (%194[0] * %156) + %71#2 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %196 = vector.extractelement %173[%c0 : index] : vector<1xf8E8M0FNU>
        %197 = amdgpu.scaled_mfma(%187[0] * %137) * (%196[0] * %157) + %195 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %198 = vector.extractelement %174[%c0 : index] : vector<1xf8E8M0FNU>
        %199 = amdgpu.scaled_mfma(%184[0] * %136) * (%198[0] * %158) + %71#3 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %200 = vector.extractelement %175[%c0 : index] : vector<1xf8E8M0FNU>
        %201 = amdgpu.scaled_mfma(%187[0] * %137) * (%200[0] * %159) + %199 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %202 = vector.extractelement %176[%c0 : index] : vector<1xf8E8M0FNU>
        %203 = amdgpu.scaled_mfma(%184[0] * %136) * (%202[0] * %160) + %71#4 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %204 = vector.extractelement %177[%c0 : index] : vector<1xf8E8M0FNU>
        %205 = amdgpu.scaled_mfma(%187[0] * %137) * (%204[0] * %161) + %203 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %206 = vector.extractelement %178[%c0 : index] : vector<1xf8E8M0FNU>
        %207 = amdgpu.scaled_mfma(%184[0] * %136) * (%206[0] * %162) + %71#5 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %208 = vector.extractelement %179[%c0 : index] : vector<1xf8E8M0FNU>
        %209 = amdgpu.scaled_mfma(%187[0] * %137) * (%208[0] * %163) + %207 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %210 = vector.extractelement %180[%c0 : index] : vector<1xf8E8M0FNU>
        %211 = amdgpu.scaled_mfma(%184[0] * %136) * (%210[0] * %164) + %71#6 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %212 = vector.extractelement %181[%c0 : index] : vector<1xf8E8M0FNU>
        %213 = amdgpu.scaled_mfma(%187[0] * %137) * (%212[0] * %165) + %211 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %214 = vector.extractelement %182[%c0 : index] : vector<1xf8E8M0FNU>
        %215 = amdgpu.scaled_mfma(%184[0] * %136) * (%214[0] * %166) + %71#7 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %216 = vector.extractelement %183[%c0 : index] : vector<1xf8E8M0FNU>
        %217 = amdgpu.scaled_mfma(%187[0] * %137) * (%216[0] * %167) + %215 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %218 = vector.extractelement %146[%c0 : index] : vector<1xf8E8M0FNU>
        %219 = amdgpu.scaled_mfma(%218[0] * %138) * (%185[0] * %152) + %71#8 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %220 = vector.extractelement %147[%c0 : index] : vector<1xf8E8M0FNU>
        %221 = amdgpu.scaled_mfma(%220[0] * %139) * (%188[0] * %153) + %219 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %222 = amdgpu.scaled_mfma(%218[0] * %138) * (%190[0] * %154) + %71#9 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %223 = amdgpu.scaled_mfma(%220[0] * %139) * (%192[0] * %155) + %222 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %224 = amdgpu.scaled_mfma(%218[0] * %138) * (%194[0] * %156) + %71#10 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %225 = amdgpu.scaled_mfma(%220[0] * %139) * (%196[0] * %157) + %224 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %226 = amdgpu.scaled_mfma(%218[0] * %138) * (%198[0] * %158) + %71#11 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %227 = amdgpu.scaled_mfma(%220[0] * %139) * (%200[0] * %159) + %226 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %228 = amdgpu.scaled_mfma(%218[0] * %138) * (%202[0] * %160) + %71#12 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %229 = amdgpu.scaled_mfma(%220[0] * %139) * (%204[0] * %161) + %228 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %230 = amdgpu.scaled_mfma(%218[0] * %138) * (%206[0] * %162) + %71#13 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %231 = amdgpu.scaled_mfma(%220[0] * %139) * (%208[0] * %163) + %230 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %232 = amdgpu.scaled_mfma(%218[0] * %138) * (%210[0] * %164) + %71#14 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %233 = amdgpu.scaled_mfma(%220[0] * %139) * (%212[0] * %165) + %232 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %234 = amdgpu.scaled_mfma(%218[0] * %138) * (%214[0] * %166) + %71#15 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %235 = amdgpu.scaled_mfma(%220[0] * %139) * (%216[0] * %167) + %234 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %236 = vector.extractelement %148[%c0 : index] : vector<1xf8E8M0FNU>
        %237 = amdgpu.scaled_mfma(%236[0] * %140) * (%185[0] * %152) + %71#16 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %238 = vector.extractelement %149[%c0 : index] : vector<1xf8E8M0FNU>
        %239 = amdgpu.scaled_mfma(%238[0] * %141) * (%188[0] * %153) + %237 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %240 = amdgpu.scaled_mfma(%236[0] * %140) * (%190[0] * %154) + %71#17 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %241 = amdgpu.scaled_mfma(%238[0] * %141) * (%192[0] * %155) + %240 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %242 = amdgpu.scaled_mfma(%236[0] * %140) * (%194[0] * %156) + %71#18 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %243 = amdgpu.scaled_mfma(%238[0] * %141) * (%196[0] * %157) + %242 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %244 = amdgpu.scaled_mfma(%236[0] * %140) * (%198[0] * %158) + %71#19 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %245 = amdgpu.scaled_mfma(%238[0] * %141) * (%200[0] * %159) + %244 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %246 = amdgpu.scaled_mfma(%236[0] * %140) * (%202[0] * %160) + %71#20 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %247 = amdgpu.scaled_mfma(%238[0] * %141) * (%204[0] * %161) + %246 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %248 = amdgpu.scaled_mfma(%236[0] * %140) * (%206[0] * %162) + %71#21 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %249 = amdgpu.scaled_mfma(%238[0] * %141) * (%208[0] * %163) + %248 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %250 = amdgpu.scaled_mfma(%236[0] * %140) * (%210[0] * %164) + %71#22 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %251 = amdgpu.scaled_mfma(%238[0] * %141) * (%212[0] * %165) + %250 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %252 = amdgpu.scaled_mfma(%236[0] * %140) * (%214[0] * %166) + %71#23 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %253 = amdgpu.scaled_mfma(%238[0] * %141) * (%216[0] * %167) + %252 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %254 = vector.extractelement %150[%c0 : index] : vector<1xf8E8M0FNU>
        %255 = amdgpu.scaled_mfma(%254[0] * %142) * (%185[0] * %152) + %71#24 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %256 = vector.extractelement %151[%c0 : index] : vector<1xf8E8M0FNU>
        %257 = amdgpu.scaled_mfma(%256[0] * %143) * (%188[0] * %153) + %255 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %258 = amdgpu.scaled_mfma(%254[0] * %142) * (%190[0] * %154) + %71#25 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %259 = amdgpu.scaled_mfma(%256[0] * %143) * (%192[0] * %155) + %258 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %260 = amdgpu.scaled_mfma(%254[0] * %142) * (%194[0] * %156) + %71#26 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %261 = amdgpu.scaled_mfma(%256[0] * %143) * (%196[0] * %157) + %260 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %262 = amdgpu.scaled_mfma(%254[0] * %142) * (%198[0] * %158) + %71#27 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %263 = amdgpu.scaled_mfma(%256[0] * %143) * (%200[0] * %159) + %262 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %264 = amdgpu.scaled_mfma(%254[0] * %142) * (%202[0] * %160) + %71#28 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %265 = amdgpu.scaled_mfma(%256[0] * %143) * (%204[0] * %161) + %264 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %266 = amdgpu.scaled_mfma(%254[0] * %142) * (%206[0] * %162) + %71#29 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %267 = amdgpu.scaled_mfma(%256[0] * %143) * (%208[0] * %163) + %266 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %268 = amdgpu.scaled_mfma(%254[0] * %142) * (%210[0] * %164) + %71#30 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %269 = amdgpu.scaled_mfma(%256[0] * %143) * (%212[0] * %165) + %268 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %270 = amdgpu.scaled_mfma(%254[0] * %142) * (%214[0] * %166) + %71#31 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %271 = amdgpu.scaled_mfma(%256[0] * %143) * (%216[0] * %167) + %270 {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
        %272 = arith.truncf %189 : vector<4xf32> to vector<4xbf16>
        %273 = arith.truncf %193 : vector<4xf32> to vector<4xbf16>
        %274 = arith.truncf %197 : vector<4xf32> to vector<4xbf16>
        %275 = arith.truncf %201 : vector<4xf32> to vector<4xbf16>
        %276 = arith.truncf %205 : vector<4xf32> to vector<4xbf16>
        %277 = arith.truncf %209 : vector<4xf32> to vector<4xbf16>
        %278 = arith.truncf %213 : vector<4xf32> to vector<4xbf16>
        %279 = arith.truncf %217 : vector<4xf32> to vector<4xbf16>
        %280 = arith.truncf %221 : vector<4xf32> to vector<4xbf16>
        %281 = arith.truncf %223 : vector<4xf32> to vector<4xbf16>
        %282 = arith.truncf %225 : vector<4xf32> to vector<4xbf16>
        %283 = arith.truncf %227 : vector<4xf32> to vector<4xbf16>
        %284 = arith.truncf %229 : vector<4xf32> to vector<4xbf16>
        %285 = arith.truncf %231 : vector<4xf32> to vector<4xbf16>
        %286 = arith.truncf %233 : vector<4xf32> to vector<4xbf16>
        %287 = arith.truncf %235 : vector<4xf32> to vector<4xbf16>
        %288 = arith.truncf %239 : vector<4xf32> to vector<4xbf16>
        %289 = arith.truncf %241 : vector<4xf32> to vector<4xbf16>
        %290 = arith.truncf %243 : vector<4xf32> to vector<4xbf16>
        %291 = arith.truncf %245 : vector<4xf32> to vector<4xbf16>
        %292 = arith.truncf %247 : vector<4xf32> to vector<4xbf16>
        %293 = arith.truncf %249 : vector<4xf32> to vector<4xbf16>
        %294 = arith.truncf %251 : vector<4xf32> to vector<4xbf16>
        %295 = arith.truncf %253 : vector<4xf32> to vector<4xbf16>
        %296 = arith.truncf %257 : vector<4xf32> to vector<4xbf16>
        %297 = arith.truncf %259 : vector<4xf32> to vector<4xbf16>
        %298 = arith.truncf %261 : vector<4xf32> to vector<4xbf16>
        %299 = arith.truncf %263 : vector<4xf32> to vector<4xbf16>
        %300 = arith.truncf %265 : vector<4xf32> to vector<4xbf16>
        %301 = arith.truncf %267 : vector<4xf32> to vector<4xbf16>
        %302 = arith.truncf %269 : vector<4xf32> to vector<4xbf16>
        %303 = arith.truncf %271 : vector<4xf32> to vector<4xbf16>
        %304 = vector.extract_strided_slice %272 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %305 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<16384x16384xbf16, strided<[16384, 1], offset: ?>>
        %306 = affine.apply #map30()[%block_id_x]
        %307 = affine.apply #map30()[%block_id_y]
        %308 = affine.apply #map31()[%thread_id_x]
        %309 = arith.muli %306, %c16384 overflow<nsw> : index
        %310 = arith.muli %308, %c16384 overflow<nsw> : index
        %311 = arith.addi %309, %307 overflow<nsw> : index
        %312 = arith.addi %310, %72 overflow<nsw> : index
        %reinterpret_cast_6 = memref.reinterpret_cast %305 to offset: [%311], sizes: [%c1073741822], strides: [1] : memref<16384x16384xbf16, strided<[16384, 1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
        %313 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_6 validBytes(%c2147483645_i32) : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        vector.store %304, %313[%312] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %314 = vector.extract_strided_slice %272 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %315 = affine.apply #map32()[%thread_id_x]
        %316 = arith.muli %315, %c16384 overflow<nsw> : index
        %317 = arith.addi %316, %72 overflow<nsw> : index
        vector.store %314, %313[%317] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %318 = vector.extract_strided_slice %272 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %319 = affine.apply #map33()[%thread_id_x]
        %320 = arith.muli %319, %c16384 overflow<nsw> : index
        %321 = arith.addi %320, %72 overflow<nsw> : index
        vector.store %318, %313[%321] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %322 = vector.extract_strided_slice %272 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %323 = affine.apply #map34()[%thread_id_x]
        %324 = arith.muli %323, %c16384 overflow<nsw> : index
        %325 = arith.addi %324, %72 overflow<nsw> : index
        vector.store %322, %313[%325] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %326 = vector.extract_strided_slice %273 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %327 = arith.addi %310, %77 overflow<nsw> : index
        vector.store %326, %313[%327] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %328 = vector.extract_strided_slice %273 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %329 = arith.addi %316, %77 overflow<nsw> : index
        vector.store %328, %313[%329] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %330 = vector.extract_strided_slice %273 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %331 = arith.addi %320, %77 overflow<nsw> : index
        vector.store %330, %313[%331] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %332 = vector.extract_strided_slice %273 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %333 = arith.addi %324, %77 overflow<nsw> : index
        vector.store %332, %313[%333] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %334 = vector.extract_strided_slice %274 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %335 = arith.addi %310, %80 overflow<nsw> : index
        vector.store %334, %313[%335] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %336 = vector.extract_strided_slice %274 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %337 = arith.addi %316, %80 overflow<nsw> : index
        vector.store %336, %313[%337] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %338 = vector.extract_strided_slice %274 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %339 = arith.addi %320, %80 overflow<nsw> : index
        vector.store %338, %313[%339] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %340 = vector.extract_strided_slice %274 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %341 = arith.addi %324, %80 overflow<nsw> : index
        vector.store %340, %313[%341] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %342 = vector.extract_strided_slice %275 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %343 = arith.addi %310, %83 overflow<nsw> : index
        vector.store %342, %313[%343] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %344 = vector.extract_strided_slice %275 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %345 = arith.addi %316, %83 overflow<nsw> : index
        vector.store %344, %313[%345] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %346 = vector.extract_strided_slice %275 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %347 = arith.addi %320, %83 overflow<nsw> : index
        vector.store %346, %313[%347] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %348 = vector.extract_strided_slice %275 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %349 = arith.addi %324, %83 overflow<nsw> : index
        vector.store %348, %313[%349] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %350 = vector.extract_strided_slice %276 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %351 = arith.addi %310, %86 overflow<nsw> : index
        vector.store %350, %313[%351] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %352 = vector.extract_strided_slice %276 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %353 = arith.addi %316, %86 overflow<nsw> : index
        vector.store %352, %313[%353] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %354 = vector.extract_strided_slice %276 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %355 = arith.addi %320, %86 overflow<nsw> : index
        vector.store %354, %313[%355] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %356 = vector.extract_strided_slice %276 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %357 = arith.addi %324, %86 overflow<nsw> : index
        vector.store %356, %313[%357] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %358 = vector.extract_strided_slice %277 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %359 = arith.addi %310, %89 overflow<nsw> : index
        vector.store %358, %313[%359] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %360 = vector.extract_strided_slice %277 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %361 = arith.addi %316, %89 overflow<nsw> : index
        vector.store %360, %313[%361] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %362 = vector.extract_strided_slice %277 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %363 = arith.addi %320, %89 overflow<nsw> : index
        vector.store %362, %313[%363] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %364 = vector.extract_strided_slice %277 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %365 = arith.addi %324, %89 overflow<nsw> : index
        vector.store %364, %313[%365] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %366 = vector.extract_strided_slice %278 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %367 = arith.addi %310, %92 overflow<nsw> : index
        vector.store %366, %313[%367] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %368 = vector.extract_strided_slice %278 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %369 = arith.addi %316, %92 overflow<nsw> : index
        vector.store %368, %313[%369] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %370 = vector.extract_strided_slice %278 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %371 = arith.addi %320, %92 overflow<nsw> : index
        vector.store %370, %313[%371] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %372 = vector.extract_strided_slice %278 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %373 = arith.addi %324, %92 overflow<nsw> : index
        vector.store %372, %313[%373] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %374 = vector.extract_strided_slice %279 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %375 = arith.addi %310, %95 overflow<nsw> : index
        vector.store %374, %313[%375] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %376 = vector.extract_strided_slice %279 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %377 = arith.addi %316, %95 overflow<nsw> : index
        vector.store %376, %313[%377] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %378 = vector.extract_strided_slice %279 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %379 = arith.addi %320, %95 overflow<nsw> : index
        vector.store %378, %313[%379] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %380 = vector.extract_strided_slice %279 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %381 = arith.addi %324, %95 overflow<nsw> : index
        vector.store %380, %313[%381] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %382 = vector.extract_strided_slice %280 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %383 = affine.apply #map35()[%thread_id_x]
        %384 = arith.muli %383, %c16384 overflow<nsw> : index
        %385 = arith.addi %384, %72 overflow<nsw> : index
        vector.store %382, %313[%385] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %386 = vector.extract_strided_slice %280 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %387 = affine.apply #map36()[%thread_id_x]
        %388 = arith.muli %387, %c16384 overflow<nsw> : index
        %389 = arith.addi %388, %72 overflow<nsw> : index
        vector.store %386, %313[%389] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %390 = vector.extract_strided_slice %280 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %391 = affine.apply #map37()[%thread_id_x]
        %392 = arith.muli %391, %c16384 overflow<nsw> : index
        %393 = arith.addi %392, %72 overflow<nsw> : index
        vector.store %390, %313[%393] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %394 = vector.extract_strided_slice %280 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %395 = affine.apply #map38()[%thread_id_x]
        %396 = arith.muli %395, %c16384 overflow<nsw> : index
        %397 = arith.addi %396, %72 overflow<nsw> : index
        vector.store %394, %313[%397] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %398 = vector.extract_strided_slice %281 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %399 = arith.addi %384, %77 overflow<nsw> : index
        vector.store %398, %313[%399] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %400 = vector.extract_strided_slice %281 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %401 = arith.addi %388, %77 overflow<nsw> : index
        vector.store %400, %313[%401] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %402 = vector.extract_strided_slice %281 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %403 = arith.addi %392, %77 overflow<nsw> : index
        vector.store %402, %313[%403] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %404 = vector.extract_strided_slice %281 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %405 = arith.addi %396, %77 overflow<nsw> : index
        vector.store %404, %313[%405] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %406 = vector.extract_strided_slice %282 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %407 = arith.addi %384, %80 overflow<nsw> : index
        vector.store %406, %313[%407] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %408 = vector.extract_strided_slice %282 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %409 = arith.addi %388, %80 overflow<nsw> : index
        vector.store %408, %313[%409] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %410 = vector.extract_strided_slice %282 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %411 = arith.addi %392, %80 overflow<nsw> : index
        vector.store %410, %313[%411] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %412 = vector.extract_strided_slice %282 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %413 = arith.addi %396, %80 overflow<nsw> : index
        vector.store %412, %313[%413] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %414 = vector.extract_strided_slice %283 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %415 = arith.addi %384, %83 overflow<nsw> : index
        vector.store %414, %313[%415] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %416 = vector.extract_strided_slice %283 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %417 = arith.addi %388, %83 overflow<nsw> : index
        vector.store %416, %313[%417] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %418 = vector.extract_strided_slice %283 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %419 = arith.addi %392, %83 overflow<nsw> : index
        vector.store %418, %313[%419] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %420 = vector.extract_strided_slice %283 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %421 = arith.addi %396, %83 overflow<nsw> : index
        vector.store %420, %313[%421] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %422 = vector.extract_strided_slice %284 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %423 = arith.addi %384, %86 overflow<nsw> : index
        vector.store %422, %313[%423] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %424 = vector.extract_strided_slice %284 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %425 = arith.addi %388, %86 overflow<nsw> : index
        vector.store %424, %313[%425] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %426 = vector.extract_strided_slice %284 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %427 = arith.addi %392, %86 overflow<nsw> : index
        vector.store %426, %313[%427] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %428 = vector.extract_strided_slice %284 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %429 = arith.addi %396, %86 overflow<nsw> : index
        vector.store %428, %313[%429] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %430 = vector.extract_strided_slice %285 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %431 = arith.addi %384, %89 overflow<nsw> : index
        vector.store %430, %313[%431] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %432 = vector.extract_strided_slice %285 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %433 = arith.addi %388, %89 overflow<nsw> : index
        vector.store %432, %313[%433] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %434 = vector.extract_strided_slice %285 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %435 = arith.addi %392, %89 overflow<nsw> : index
        vector.store %434, %313[%435] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %436 = vector.extract_strided_slice %285 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %437 = arith.addi %396, %89 overflow<nsw> : index
        vector.store %436, %313[%437] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %438 = vector.extract_strided_slice %286 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %439 = arith.addi %384, %92 overflow<nsw> : index
        vector.store %438, %313[%439] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %440 = vector.extract_strided_slice %286 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %441 = arith.addi %388, %92 overflow<nsw> : index
        vector.store %440, %313[%441] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %442 = vector.extract_strided_slice %286 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %443 = arith.addi %392, %92 overflow<nsw> : index
        vector.store %442, %313[%443] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %444 = vector.extract_strided_slice %286 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %445 = arith.addi %396, %92 overflow<nsw> : index
        vector.store %444, %313[%445] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %446 = vector.extract_strided_slice %287 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %447 = arith.addi %384, %95 overflow<nsw> : index
        vector.store %446, %313[%447] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %448 = vector.extract_strided_slice %287 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %449 = arith.addi %388, %95 overflow<nsw> : index
        vector.store %448, %313[%449] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %450 = vector.extract_strided_slice %287 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %451 = arith.addi %392, %95 overflow<nsw> : index
        vector.store %450, %313[%451] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %452 = vector.extract_strided_slice %287 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %453 = arith.addi %396, %95 overflow<nsw> : index
        vector.store %452, %313[%453] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %454 = vector.extract_strided_slice %288 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %455 = affine.apply #map39()[%thread_id_x]
        %456 = arith.muli %455, %c16384 overflow<nsw> : index
        %457 = arith.addi %456, %72 overflow<nsw> : index
        vector.store %454, %313[%457] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %458 = vector.extract_strided_slice %288 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %459 = affine.apply #map40()[%thread_id_x]
        %460 = arith.muli %459, %c16384 overflow<nsw> : index
        %461 = arith.addi %460, %72 overflow<nsw> : index
        vector.store %458, %313[%461] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %462 = vector.extract_strided_slice %288 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %463 = affine.apply #map41()[%thread_id_x]
        %464 = arith.muli %463, %c16384 overflow<nsw> : index
        %465 = arith.addi %464, %72 overflow<nsw> : index
        vector.store %462, %313[%465] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %466 = vector.extract_strided_slice %288 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %467 = affine.apply #map42()[%thread_id_x]
        %468 = arith.muli %467, %c16384 overflow<nsw> : index
        %469 = arith.addi %468, %72 overflow<nsw> : index
        vector.store %466, %313[%469] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %470 = vector.extract_strided_slice %289 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %471 = arith.addi %456, %77 overflow<nsw> : index
        vector.store %470, %313[%471] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %472 = vector.extract_strided_slice %289 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %473 = arith.addi %460, %77 overflow<nsw> : index
        vector.store %472, %313[%473] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %474 = vector.extract_strided_slice %289 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %475 = arith.addi %464, %77 overflow<nsw> : index
        vector.store %474, %313[%475] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %476 = vector.extract_strided_slice %289 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %477 = arith.addi %468, %77 overflow<nsw> : index
        vector.store %476, %313[%477] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %478 = vector.extract_strided_slice %290 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %479 = arith.addi %456, %80 overflow<nsw> : index
        vector.store %478, %313[%479] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %480 = vector.extract_strided_slice %290 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %481 = arith.addi %460, %80 overflow<nsw> : index
        vector.store %480, %313[%481] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %482 = vector.extract_strided_slice %290 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %483 = arith.addi %464, %80 overflow<nsw> : index
        vector.store %482, %313[%483] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %484 = vector.extract_strided_slice %290 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %485 = arith.addi %468, %80 overflow<nsw> : index
        vector.store %484, %313[%485] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %486 = vector.extract_strided_slice %291 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %487 = arith.addi %456, %83 overflow<nsw> : index
        vector.store %486, %313[%487] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %488 = vector.extract_strided_slice %291 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %489 = arith.addi %460, %83 overflow<nsw> : index
        vector.store %488, %313[%489] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %490 = vector.extract_strided_slice %291 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %491 = arith.addi %464, %83 overflow<nsw> : index
        vector.store %490, %313[%491] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %492 = vector.extract_strided_slice %291 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %493 = arith.addi %468, %83 overflow<nsw> : index
        vector.store %492, %313[%493] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %494 = vector.extract_strided_slice %292 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %495 = arith.addi %456, %86 overflow<nsw> : index
        vector.store %494, %313[%495] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %496 = vector.extract_strided_slice %292 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %497 = arith.addi %460, %86 overflow<nsw> : index
        vector.store %496, %313[%497] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %498 = vector.extract_strided_slice %292 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %499 = arith.addi %464, %86 overflow<nsw> : index
        vector.store %498, %313[%499] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %500 = vector.extract_strided_slice %292 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %501 = arith.addi %468, %86 overflow<nsw> : index
        vector.store %500, %313[%501] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %502 = vector.extract_strided_slice %293 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %503 = arith.addi %456, %89 overflow<nsw> : index
        vector.store %502, %313[%503] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %504 = vector.extract_strided_slice %293 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %505 = arith.addi %460, %89 overflow<nsw> : index
        vector.store %504, %313[%505] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %506 = vector.extract_strided_slice %293 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %507 = arith.addi %464, %89 overflow<nsw> : index
        vector.store %506, %313[%507] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %508 = vector.extract_strided_slice %293 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %509 = arith.addi %468, %89 overflow<nsw> : index
        vector.store %508, %313[%509] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %510 = vector.extract_strided_slice %294 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %511 = arith.addi %456, %92 overflow<nsw> : index
        vector.store %510, %313[%511] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %512 = vector.extract_strided_slice %294 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %513 = arith.addi %460, %92 overflow<nsw> : index
        vector.store %512, %313[%513] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %514 = vector.extract_strided_slice %294 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %515 = arith.addi %464, %92 overflow<nsw> : index
        vector.store %514, %313[%515] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %516 = vector.extract_strided_slice %294 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %517 = arith.addi %468, %92 overflow<nsw> : index
        vector.store %516, %313[%517] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %518 = vector.extract_strided_slice %295 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %519 = arith.addi %456, %95 overflow<nsw> : index
        vector.store %518, %313[%519] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %520 = vector.extract_strided_slice %295 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %521 = arith.addi %460, %95 overflow<nsw> : index
        vector.store %520, %313[%521] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %522 = vector.extract_strided_slice %295 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %523 = arith.addi %464, %95 overflow<nsw> : index
        vector.store %522, %313[%523] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %524 = vector.extract_strided_slice %295 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %525 = arith.addi %468, %95 overflow<nsw> : index
        vector.store %524, %313[%525] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %526 = vector.extract_strided_slice %296 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %527 = affine.apply #map43()[%thread_id_x]
        %528 = arith.muli %527, %c16384 overflow<nsw> : index
        %529 = arith.addi %528, %72 overflow<nsw> : index
        vector.store %526, %313[%529] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %530 = vector.extract_strided_slice %296 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %531 = affine.apply #map44()[%thread_id_x]
        %532 = arith.muli %531, %c16384 overflow<nsw> : index
        %533 = arith.addi %532, %72 overflow<nsw> : index
        vector.store %530, %313[%533] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %534 = vector.extract_strided_slice %296 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %535 = affine.apply #map45()[%thread_id_x]
        %536 = arith.muli %535, %c16384 overflow<nsw> : index
        %537 = arith.addi %536, %72 overflow<nsw> : index
        vector.store %534, %313[%537] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %538 = vector.extract_strided_slice %296 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %539 = affine.apply #map46()[%thread_id_x]
        %540 = arith.muli %539, %c16384 overflow<nsw> : index
        %541 = arith.addi %540, %72 overflow<nsw> : index
        vector.store %538, %313[%541] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %542 = vector.extract_strided_slice %297 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %543 = arith.addi %528, %77 overflow<nsw> : index
        vector.store %542, %313[%543] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %544 = vector.extract_strided_slice %297 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %545 = arith.addi %532, %77 overflow<nsw> : index
        vector.store %544, %313[%545] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %546 = vector.extract_strided_slice %297 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %547 = arith.addi %536, %77 overflow<nsw> : index
        vector.store %546, %313[%547] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %548 = vector.extract_strided_slice %297 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %549 = arith.addi %540, %77 overflow<nsw> : index
        vector.store %548, %313[%549] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %550 = vector.extract_strided_slice %298 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %551 = arith.addi %528, %80 overflow<nsw> : index
        vector.store %550, %313[%551] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %552 = vector.extract_strided_slice %298 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %553 = arith.addi %532, %80 overflow<nsw> : index
        vector.store %552, %313[%553] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %554 = vector.extract_strided_slice %298 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %555 = arith.addi %536, %80 overflow<nsw> : index
        vector.store %554, %313[%555] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %556 = vector.extract_strided_slice %298 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %557 = arith.addi %540, %80 overflow<nsw> : index
        vector.store %556, %313[%557] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %558 = vector.extract_strided_slice %299 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %559 = arith.addi %528, %83 overflow<nsw> : index
        vector.store %558, %313[%559] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %560 = vector.extract_strided_slice %299 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %561 = arith.addi %532, %83 overflow<nsw> : index
        vector.store %560, %313[%561] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %562 = vector.extract_strided_slice %299 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %563 = arith.addi %536, %83 overflow<nsw> : index
        vector.store %562, %313[%563] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %564 = vector.extract_strided_slice %299 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %565 = arith.addi %540, %83 overflow<nsw> : index
        vector.store %564, %313[%565] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %566 = vector.extract_strided_slice %300 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %567 = arith.addi %528, %86 overflow<nsw> : index
        vector.store %566, %313[%567] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %568 = vector.extract_strided_slice %300 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %569 = arith.addi %532, %86 overflow<nsw> : index
        vector.store %568, %313[%569] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %570 = vector.extract_strided_slice %300 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %571 = arith.addi %536, %86 overflow<nsw> : index
        vector.store %570, %313[%571] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %572 = vector.extract_strided_slice %300 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %573 = arith.addi %540, %86 overflow<nsw> : index
        vector.store %572, %313[%573] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %574 = vector.extract_strided_slice %301 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %575 = arith.addi %528, %89 overflow<nsw> : index
        vector.store %574, %313[%575] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %576 = vector.extract_strided_slice %301 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %577 = arith.addi %532, %89 overflow<nsw> : index
        vector.store %576, %313[%577] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %578 = vector.extract_strided_slice %301 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %579 = arith.addi %536, %89 overflow<nsw> : index
        vector.store %578, %313[%579] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %580 = vector.extract_strided_slice %301 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %581 = arith.addi %540, %89 overflow<nsw> : index
        vector.store %580, %313[%581] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %582 = vector.extract_strided_slice %302 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %583 = arith.addi %528, %92 overflow<nsw> : index
        vector.store %582, %313[%583] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %584 = vector.extract_strided_slice %302 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %585 = arith.addi %532, %92 overflow<nsw> : index
        vector.store %584, %313[%585] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %586 = vector.extract_strided_slice %302 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %587 = arith.addi %536, %92 overflow<nsw> : index
        vector.store %586, %313[%587] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %588 = vector.extract_strided_slice %302 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %589 = arith.addi %540, %92 overflow<nsw> : index
        vector.store %588, %313[%589] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %590 = vector.extract_strided_slice %303 {offsets = [0], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %591 = arith.addi %528, %95 overflow<nsw> : index
        vector.store %590, %313[%591] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %592 = vector.extract_strided_slice %303 {offsets = [1], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %593 = arith.addi %532, %95 overflow<nsw> : index
        vector.store %592, %313[%593] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %594 = vector.extract_strided_slice %303 {offsets = [2], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %595 = arith.addi %536, %95 overflow<nsw> : index
        vector.store %594, %313[%595] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
        %596 = vector.extract_strided_slice %303 {offsets = [3], sizes = [1], strides = [1]} : vector<4xbf16> to vector<1xbf16>
        %597 = arith.addi %540, %95 overflow<nsw> : index
        vector.store %596, %313[%597] : memref<?xbf16, strided<[1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xbf16>
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
