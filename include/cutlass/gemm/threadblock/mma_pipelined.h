/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<> default_mma_core_sm70.Shape(blkShape)
  typename Shape_,
  /// Iterates over tiles of A operand in global memory 
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorA_,
  /// Iterates over tiles of A operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorA_,
  /// Iterates over tiles of B operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorB_,
  /// Iterates over tiles of B operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorB_,
  /// Data type of accumulator matrix
  typename ElementC_,
  /// Data type of accumulator matrix
  typename LayoutC_,
  /// Policy describing tuning details (concept: MmaPolicy) default_mma_core_sm70.MmaPolicy
  typename Policy_,
  /// Transformation applied to A operand
  typename TransformA_ = NumericArrayConverter<
    typename SmemIteratorA_::Element, 
    typename IteratorA_::Element, 
    IteratorA_::Fragment::kElements>,
  ///
  /// Transformation applied to B operand
  typename TransformB_ = NumericArrayConverter<
    typename SmemIteratorB_::Element, 
    typename IteratorB_::Element, 
    IteratorB_::Fragment::kElements>,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaPipelined : public MmaBase<Shape_, Policy_, 2> {
public:

  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 2>;

  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using ElementC = ElementC_;       ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;         ///< Layout of accumulator matrix
  using Policy = Policy_;           ///< Policy describing tuning details

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  //
  // Dependent types
  //

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Obtain the arch tag from the warp-level operator
  using ArchTag = typename Policy::Operator::ArchTag;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  // staticaly assert kStages for MmaPipelined is two (Double-buffered pipeline)
  static_assert((Base::kStages==2), "MmaPipelined requires kStages set to value 2");

private:

  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;

protected:

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaPipelined(
    typename Base::SharedStorage &shared_storage,       ///< Shared storage needed for internal use by threadblock-scoped GEMM
    int thread_idx,                                     ///< ID within the threadblock
    int warp_idx,                                       ///< ID of warp
    int lane_idx                                        ///< ID of each thread within a warp
  ):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),  //BTBT 这里shared_storage.operand_A_ref()是一块smem内存,在上一行MmaBase构造时也把同一块smem传入warp_tile_iterator_A_.
    smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx) {    //因此,smem_iterator_A_负责从glb写入数据到smem,而warp_tile_iterator_A_负责从smem读相应数据给到wrp的mma运算使用

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC &accum,                                 ///< destination accumulator tile
    IteratorA iterator_A,                             ///< iterator over A operand in global memory
    IteratorB iterator_B,                             ///< iterator over B operand in global memory
    FragmentC const &src_accum,                       ///< source accumulator tile
    TransformA transform_A = TransformA(),            ///< transformation applied to A fragment
    TransformB transform_B = TransformB()) {          ///< transformation applied to B fragment

    //
    // Prologue
    //

    // Perform accumulation in the 'd' output operand
    accum = src_accum;
    //BTBT IteratorA:Fragment=Array<half_t, 4*8>//kCount=1*4,kElementsPerAccess=8//predicated_tile_iterator.h#643<pitch_linear_thread_map.h#284
    FragmentA tb_frag_A;//BTBT A:[shpThrdBlk.K*(128/bitOf(half))/4/1_]=1 * [(shpThrdBlk.M/8) / ((shpThrdBlk/shpWrp).MNK*wrpSz/(4*8))_]=4 ->Array<half_t, 4*8>(kElementsPerAccess=8)
    FragmentB tb_frag_B;//BTBT B:[shpThrdBlk.N*(128/bitOf(half))/8/1_]=2 * [(shpThrdBlk.K/4) / ((shpThrdBlk/shpWrp).MNK*wrpSz/(8*4))_]=2 ->Array<half_t, 4*8>(kElementsPerAccess=8)

    tb_frag_A.clear();//BTBT IteratorA predicated_tile_iterator.h#343#311用到了predicated_tile_access_iterator.h#315计算内存位置,然后用memory.h中的cutlass::arch::global_load的汇编把数据加载进来
    tb_frag_B.clear();//BTBT iterator_A&B在kernel/gemm.h的operator()()中初始化. 这里是从glb到reg.

    // The last kblock is loaded in the prolog
    iterator_A.load(tb_frag_A);
    iterator_B.load(tb_frag_B);

    ++iterator_A;
    ++iterator_B;
    //从GLB拿到SMEM，让mma从SMEM中通过warp_tile_iterator_A_取数据计算
    this->smem_iterator_A_.store(transform_A(tb_frag_A));//SmemIteratorA<-'regular_tile_iterator_tensor_op_sm70.h#1352并使用了tensor_op_multiplicand_sm70.h#943'<-default_mma_core_sm70.h#434#454
    this->smem_iterator_B_.store(transform_B(tb_frag_B));//BTBT ??? TransformA和B不知道template是调了numeric_conversion.h#666还是#697,如果都是half的话,前者做了多余的类型转换

    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA warp_frag_A[2];//BTBT =Array<half,32*16/32*2><-mma_tensor_op_tile_iterator_sm70.h#1571#2050<-mma_tensor_op_sm70.h#136<-default_mma_core_sm70.h#499<-default_gemm.h
    WarpFragmentB warp_frag_B[2];//BTBT =Array<half,32*16/32*2><-mma_tensor_op_tile_iterator_sm70.h#483#905...
    //BTBT bias_relu sm70 warp_tile_iterator_A_和B都是MmaPolicy.MmaVoltaTensorOp::MmaVoltaTensorOpMultiplicandTileIterator
    this->warp_tile_iterator_A_.set_kgroup_index(0);//BTBT warp_tile_iterator_A_在父类mma_base.h中初始化,包含了指向smem的指针.来自mma_tensor_op_tile_iterator_sm70.h#1476#2050
    this->warp_tile_iterator_B_.set_kgroup_index(0);//BTBT warp_tile_iterator_B_在父类mma_base.h中初始化,包含了指向smem的指针.来自mma_tensor_op_tile_iterator_sm70.h#394#905

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    Operator warp_mma;

    int smem_write_stage_idx = 1;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing 
    // shared memory loads (which have the tighest latency requirement).

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2. 
    //BTBT 这个循环称为wrp循环,共gemm_k_iterations次迭代,每次迭代wrp的每个thrd把数据从SMEM拿到REG并调mma.sync做计算
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {//BlkTil窗口沿PblmSz移动gemm_k_iterations次,每次取BlkTil个元素。这是blk循环，相当于blkTil这个窗口沿着k移动，每次blk循环都要做多次下面的wrp内循环
      //
      // Loop over GEMM K dimension
      //
      //WrpTil窗口沿BlkTil移动kWarpGemmIterations次,每次取WrpTil个元素。这是wrp内循环，会涉及整个wrpTil的元素，多个wrpTil会凑成一个blkTil. wrp内循环会基于SMEM中的数据去做，做完了就相当于blkTil向前一步，再做下一步blkTil的wrp内循环
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {//BTBT kWarpGemmIterations=wrpTil.K/arch::Mma::Sharp.K=32/4=8.这里的arch::Mma::Sharp不是<8,8,4>而是<16,16,4>,前者指8thrd做mma.m8n8k4的shape,后者是一个wrp内32thrd可以同时做4个mma.m8n8k4的shape,而一般都是一个wrp内4组8个thrd同时做的,只是排布可以变化而已.

        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          //在该次wrp迭代结束前，把在做上一轮wrp迭代的同时从GLB拿到REG的数据放入SMEM
          // Write fragments to shared memory
          this->smem_iterator_A_.store(transform_A(tb_frag_A));

          this->smem_iterator_B_.store(transform_B(tb_frag_B));

          __syncthreads();
          
          ++this->smem_iterator_A_;
          ++this->smem_iterator_B_;
          // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
          //SMEM分连续的甲乙两区,所有wrp循环开始前会写一次甲区以初始化数据,然后每次wrp循环结束的前一次迭代会写上一次写时没有写过的另一个区.
            //当stage_idx=1时,表示上一轮写SMEM的甲区,现在乙区是刚写的数据,smem_iterator_A_的写指针指向了乙区末端,因为刚写了一轮数据,所以这里要把写指针重置到甲区首端,以便下轮从甲区开始写;而warp_tile_iterator_A_的读指针指向甲区末端,因为刚做了一轮wrp循环,所以读指针不用重置就可以继续使用刚写入SMEM乙区的内容.
            //当stage_idx=0时,表示上一轮写SMEM的乙区,现在甲区是刚写的数据,smem_iterator_A_的写指针指向了甲区末端,因为刚写了一轮数据,所以不必重置写指针下轮也可以从乙区开始写;而warp_tile_iterator_A_的读指针指向乙区末端,因为刚做了一轮wrp循环,所以读指针要重置才可以使用刚写入SMEM甲区的内容.
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
          }
          else {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
        }
        //warp_tile_iterator_A_在这一轮wrp迭代内会从SMEM中获取下一轮wrp迭代中mma计算要用的数据,这就使得smem_iterator_A_能够在该次blk循环结束前把下一次blkTil的数据放入SMEM ??? 这是指REG的双buf甲乙区么,又好像不是,更像是REG有kWarpGemmIterations个区,这次迭代从SMEM加载数据到下次迭代要用的区(注意,在整个循环开始前已经加载过一次到REG以初始化数据)
        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        
        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);//BTBT 写入双buff的warp_frag_A另一个区(写区) ??? 这是指REG的双buf甲乙区么,不大像
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {//BTBT ??? TOREFACTOR iterator_A之类的在gemm_k_iterations==1时是没必要load的,这里是不是要加个判断，否则即使mask置0了，但还是要空load一次，还是因为无论怎样都要移动iterator_A?
          //在blk循环开始前已经从GLB->SMEM了一次，然后每次blk循环开始获取下一次blk循环要放到SMEM中的blkTil数据(属于该thrd的blkTil的数据)，在从GLB取数的同时，下面的warp_mma会在每轮wrp内循环中同时用warp_tile_iterator_A_获取上次blk循环放入SMEM的blkTil数据进行计算，做指令级别并行//??? 从GLB加载到REG?如果是A100就不用先加载到REG再到SMEM了?但也未必,因为GLB,SMEM的排布不一样
          iterator_A.load(tb_frag_A);
          iterator_B.load(tb_frag_B);

          ++iterator_A;
          ++iterator_B;

          // Avoid reading out of bounds if this was the last loop iteration
          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
        }
        //BTBT mma_tensor_op_sm70.h<default_mma_core_sm70.h#505
        warp_mma(accum, warp_frag_A[warp_mma_k % 2],//BTBT 读取双buff的warp_frag_A另一个区(读区)
                 warp_frag_B[warp_mma_k % 2], accum);
      }
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
