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
    \brief Templates implementing warp-level matrix multiply-accumulate operations targeting
      Tensor Cores.

    This is a work in progress.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/mma.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaVoltaTensorOp {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape) //BTBT WrpTil
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Architecture tag
  using ArchTag = arch::Sm70;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Indicates math operator 
  using MathOperator = typename ArchMmaOperator::Operator;
  
  /// Underlying instruction shape
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

  /// interleaved 32x32 tiles
  using InterleavedTileShape = GemmShape<32, 32, 4>;

  static_assert(!(Shape::kM % InterleavedTileShape::kM) &&
                !(Shape::kN % InterleavedTileShape::kN),
                "Shape must be a multiple of InterleavedTileShape.");
public:

  /// Iterates over the A operand in memory //BTBT mma_tensor_op_tile_iterator_sm70#2041
  using IteratorA = MmaVoltaTensorOpMultiplicandTileIterator<
    MatrixShape<Shape::kM, Shape::kK>,//BTBT <WrpTil.M,WrpTil.K>
    Operand::kA,
    ElementA,
    LayoutA,
    MatrixShape<   //BTBT InstructionShape_<16,16,4>
      ArchMmaOperator::Shape::kM,
      ArchMmaOperator::Shape::kK
    >,
    Policy::OpDelta::kRow,
    kThreadCount
  >;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Iterates over the B operand in memory //BTBT mma_tensor_op_tile_iterator_sm70#910
  using IteratorB = MmaVoltaTensorOpMultiplicandTileIterator<
    MatrixShape<Shape::kK, Shape::kN>,//BTBT <WrpTil.K,WrpTil.M>
    Operand::kB,
    ElementB,
    LayoutB,
    MatrixShape<
      ArchMmaOperator::Shape::kK,
      ArchMmaOperator::Shape::kN
    >,
    Policy::OpDelta::kRow,
    kThreadCount
  >;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Iterates over the C operand in memory//BTBT mma_tensor_op_tile_iterator_sm70.h#1149
  using IteratorC = MmaVoltaTensorOpAccumulatorTileIterator<
    MatrixShape<Shape::kM, Shape::kN>,
    ElementC,
    LayoutC,
    typename ArchMmaOperator::Shape,
    typename Policy::OpDelta
  >;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;//BTBT Array<half,WrpTilM*WrpTilN/32=128> mma_tensor_op_tile_iterator_sm70.h#1227
private:

  static_assert(
    !(Shape::kM % ArchMmaOperator::Shape::kM) && 
    !(Shape::kN % ArchMmaOperator::Shape::kN),
    "Shape of warp-level Mma must be divisible by operator shape.");

  /// Number of mma operations performed
  using MmaIterations = MatrixShape<
    InterleavedTileShape::kM / ArchMmaOperator::Shape::kM, //BTBT 32/ArchMmaShp.M=2
    InterleavedTileShape::kN / ArchMmaOperator::Shape::kN  //BTBT 32/ArchMmaShp.N=2
  >;
  using TileIterations = MatrixShape<
    Shape::kM / InterleavedTileShape::kM, //BTBT row:WrpTilM/32=2
    Shape::kN / InterleavedTileShape::kN  //BTBT col:WrpTilN/32=2
  >;

  // Whether matrix B is reordered
  bool reorder_B_;

public:

  /// Underlying matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;//BTBT mma_sm70#637->#257 or #559

public:

  //
  // Methods
  //
  
  /// Ctor
  CUTLASS_DEVICE
  MmaVoltaTensorOp() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D,       //Array<half,128> 128=WrpTil.M(64)*WrpTil.N(64)/32thrds
    FragmentA const &A, //Array<half,16> col.row的情况下，16=WrpTil.Contiguous(M=64) * arch::Mma::Shape.Strid(K=4) / 32thrds * 2 //BTBT 这里最后为何乘2? 见s9593v2#17#18线程T0一次迭代中要做4次MMA，在MMA0矩阵的quad 0和MMA2矩阵的quad 0用的是一次LDS.128取数的低64b，而T0在MMA1矩阵的quad 0和MMA3矩阵的quad 0用的是同一次LDS.128的高64b，即一次LDS.128包含了共2*4个也就是2*Mma::Shape.K个half可用于4个MMA操作,  WrpTil.M * arch::Mma::Shape.K / 32thrds得到的是只做一次MMA操作的每个thrd每个矩阵要处理的half数量，所以要在最后乘2，对于B矩阵也一样
    FragmentB const &B,  //Array<half,16>
    FragmentC const &C)  {

    using MmaOperandA = typename ArchMmaOperator::FragmentA;//mma_sm70#257 Array<half,4>
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;//mma_sm70#257 Array<half,8>

    D = C;

    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    CUTLASS_PRAGMA_UNROLL
    for (int outer_col = 0; outer_col < TileIterations::kColumn; ++outer_col) {//BTBT WrpTil.N/InterleavedTileShape.N(32) 见s9593v2#18,一个tc操作涉及整个wrp共32thrd,一个wrp计算m32n32k4
      CUTLASS_PRAGMA_UNROLL
      for (int inner_col = 0; inner_col < MmaIterations::kColumn; ++inner_col) {//BTBT InterleavedTileShape.N(32)/ArchMmaShp.N(16),在sm70场景相当于硬编码2， 见s9593v2#18,一个tc操作将wrp分4份每份16thrd交错参与,每份负责C中一个MMA矩阵(间隔分布)的计算(图中C的四种不同颜色),一个MMA矩阵由m16n16k4计算得到
        CUTLASS_PRAGMA_UNROLL
        for (int outer_row = 0; outer_row < TileIterations::kRow; ++outer_row) {//BTBT WrpTil.M/InterleavedTileShape.M=2
          CUTLASS_PRAGMA_UNROLL

          for (int inner_row = 0; inner_row < MmaIterations::kRow; ++inner_row) {//BTBT InterleavedTileShape.M(32)/ArchMmaShp.M(16),在sm70场景相当于硬编码2
      
            int op_col = inner_col + MmaIterations::kColumn * outer_col;

            // Column-major serpentine sequence to maximize reuse of A operand.
            // Serpentine visitation order maximizing reuse of Rb  //BTBT ??? 参考gemm/warp/mma_tensor_op.h在(__CUDA_ARCH__ < 800)时有这样的优化,但不知为何对A有好处?且Rb是啥?
            // The visitation order is like
            //      _   
            //   | | | |
            //   | | | |
            //   |_| |_|
            //
            // Down Up Down Up
            int inner_row_serp = inner_row;
            int outer_row_serp = outer_row;
            if (op_col & 1) {
              inner_row_serp = MmaIterations::kRow - inner_row - 1;
              outer_row_serp = TileIterations::kRow - outer_row - 1;
            }
            int op_row = inner_row_serp + MmaIterations::kRow * outer_row_serp;
            int op_idx = inner_row_serp + MmaIterations::kRow * 
                         (inner_col + MmaIterations::kColumn * 
                          (outer_row_serp + TileIterations::kRow * outer_col));
            mma(
              ptr_D[op_idx],//BTBT s9593v2#15#16#17#18
              ptr_A[op_row],//续上,每个MMA矩阵由16thrd参与,然后又分成4个quad矩阵,每个quad由该MMA中的16thrd中的8thrd交错参与#17,组成MMA的quad也是间隔分布的#15,由m8n8k4计算得到,这就是该quad所涉及的8个thrd都要调用的mma.sync.aligned.m8n8k4的作用#17,所以A,B的类型为<half,4>得到的C,D的类型为<half,8>
              ptr_B[op_col],//类型为Array<half,4>
              ptr_D[op_idx]);//类型为Array<half,8> //BTBT 无论A,B是什么major,D出来都是row major的

          }
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass
