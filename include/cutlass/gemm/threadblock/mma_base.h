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

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Policy object describing MmaTensorOp
template <
    /// Warp-level GEMM operator (concept: gemm::warp::Mma)
    typename Operator_,
    /// Padding used for A operand in shared memory (concept: MatrixShape)
    typename SmemPaddingA_,
    /// Padding used for B operand in shared memory (concept: MatrixShape)
    typename SmemPaddingB_,
    /// Number of partitions of K dimension of GEMM
    int PartitionsK = 1>
struct MmaPolicy {
  /// Warp-level GEMM operator (concept: gemm::warp::MmaTensorOp or gemm::warp::MmaSimt)
  using Operator = Operator_;

  /// Padding used for A operand in shared memory
  using SmemPaddingA = SmemPaddingA_;

  /// Padding used for B operand in shared memory
  using SmemPaddingB = SmemPaddingB_;

  /// Number of partitions of K dimension
  static int const kPartitionsK = PartitionsK;
};

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>   default_mma_core_sm70.Shape(blkShape)
    typename Shape_,
    /// Policy describing tuning details (concept: MmaPolicy)  default_mma_core_sm70.h.MmaPolicy which defined in this file
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class MmaBase {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<> //BTBT bias_relu BlkTil<128,128,32>
  using Shape = Shape_;

  ///< Policy describing tuning details
  using Policy = Policy_;

  //
  // Dependent types
  //

  /// Warp-level Mma BTBT default_mma_core_sm70.MmaVoltaTensorOp
  using Operator = typename Policy::Operator;

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Policy::Operator::Shape;//BTBT default_mma_core_sm70.MmaVoltaTensorOp.Shape which is WarpShape

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<Shape::kM / WarpGemm::kM,  //blkTilM/wrpTilM
                              Shape::kN / WarpGemm::kN,
                              Shape::kK / WarpGemm::kK>;

  /// Number of warp-level GEMM oeprations 分母在default_mma_core_sm70中,为MmaTensorOpPolicy::Operator::Shape<16,16,4>其实也是同一页要调用的指令的对应的arch::Mma::Shape,
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::MmaShape::kK);//这里是wrpK/mmaK,即一个wrp的数据要处理完需要迭代做多少次mma

  /// Number of stages //BTBT bias_relu 在mma_pipelined中设为2
  static int const kStages = Stages;

  /// Tensor reference to the A operand
  using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;//default_mma_core_sm70.SmemLayoutA  RowMajorVoltaTensorOpMultiplicandCrosswise@tensor_op_multiplicand_sm70.h:948

  /// Tensor reference to the B operand
  using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;//default_mma_core_sm70.SmemLayoutB   RowMajorVoltaTensorOpMultiplicandBCongruous

  //
  // Nested structs
  //

  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the A matrix operand in shared memory //BTBT Policy::SmemPaddingA 在default_mma_core_sm70#517中为<0,0>
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                               Shape::kK * kStages +
                                   Policy::SmemPaddingA::kColumn>;//bias_relu sm70 <128, 32*2>

    /// Shape of the B matrix operand in shared memory
    using ShapeB =
        MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                    Shape::kN + Policy::SmemPaddingB::kColumn>;

   public:
    //
    // Data members
    //

    /// Buffer for A operand //BTBT bias_relu sm70 <half,128*64=8192>
    AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

    /// Buffer for B operand
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

   public:

    //
    // Methods
    //

    /// Returns a layout object for the A matrix //BTBT RowMajorVoltaTensorOpMultiplicandCrosswise@tensor_op_multiplicand_sm70.h:948 <128,64>
    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {//return RowMajorVoltaTensorOpMultiplicandCrosswise.strid=ShapeA.row=128=blkTilM without pad
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_ref() {
      return TensorRefA{operand_A.data(), LayoutA()};
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }
  };

 protected:

  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A operand from shared memory
  typename Operator::IteratorA warp_tile_iterator_A_;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  typename Operator::IteratorB warp_tile_iterator_B_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
      warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx) {

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
