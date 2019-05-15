#pragma once

#include "../../data/FluidTensor.hpp"
#include <Eigen/Core>
#include <Eigen/Eigen>

/**
 Utility functions for converting between FluidTensorView and Eigen wrappers
 around raw poiniters
 **/

namespace fluid {
namespace algorithm {
namespace _impl {

using Eigen::Map;
using Eigen::PlainObjectBase;
using Eigen::Stride;

using Eigen::AlignmentType;
using Eigen::ColMajor;
using Eigen::Dynamic;
using Eigen::RowMajor;

/**
 Convert an Eigen Matrix or Array to a FluidTensorView
 **/
template <typename Derived>
auto asFluid(PlainObjectBase<Derived> &a)
    -> FluidTensorView<typename PlainObjectBase<Derived>::Scalar,
                       (PlainObjectBase<Derived>::IsVectorAtCompileTime ? 1 : 2)> {
  constexpr size_t N = PlainObjectBase<Derived>::IsVectorAtCompileTime ? 1 : 2;

  if (N == 2) {
    if (a.Options == ColMajor) {
      // Respect the colmajorness of an eigen type
      auto slice = FluidTensorSlice<N>(
          0, {static_cast<size_t>(a.rows()), static_cast<size_t>(a.cols())},
          {1, static_cast<size_t>(a.rows())});
      return {slice, a.data()};
    }
    return {a.data(), 0, a.rows(), a.cols()};
  } else
    return {a.data(), 0, a.rows()};
}

/**
 Convert an lvalue FluidTensorView to an Eigen Matrix or Array map (you need to
say which as a template parmeter, e.g. makeWrapper<Matrix>(myView)
**/
template <template <typename, int, int, int, int, int> class EigenType,
          typename T, size_t N, template <typename, size_t> class F>
auto asFluid(F<T, N> &a)
    -> Map<EigenType<T, Dynamic, Dynamic, RowMajor, Dynamic, Dynamic>,
           Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>> {
  static_assert(N < 3,
                "Can't convert to Eigen types with more than two dimensions");

  if (N == 2) {
    return {a.data(), static_cast<Eigen::Index>(a.rows()),
            static_cast<Eigen::Index>(a.cols()),
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0],
                                     a.descriptor().strides[1])};
  } else {
    return {a.data(), static_cast<Eigen::Index>(a.rows()), 1,
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0], 1)};
  }
}

/**
 Convert an const lvalue FluidTensorView to a const Eigen Matrix or Array map
 (you need to say which as a template parmeter, e.g.
 makeWrapper<Matrix>(myConstView)
 **/
template <template <typename, int, int, int, int, int> class EigenType,
          typename T, size_t N, template <typename, size_t> class F>
auto asEigen(const F<T, N> &a)
    -> Map<const EigenType<T, Dynamic, Dynamic, RowMajor, Dynamic, Dynamic>,
           Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>> {
  static_assert(N < 3,
                "Can't convert to Eigen types with more than two dimensions");

  if (N == 2) {
    return {a.data(), static_cast<Eigen::Index>(a.rows()),
            static_cast<Eigen::Index>(a.cols()),
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0],
                                     a.descriptor().strides[1])};
  } else {
    return {a.data(), static_cast<Eigen::Index>(a.rows()), 1,
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0], 1)};
  }
}

/**
 Convert an rvalue FluidTensorView to a Eigen Matrix or Array map (you need to
 say which as a template parmeter, e.g. makeWrapper<Matrix>(myView.row(0))
 **/
template <template <typename, int, int, int, int, int> class EigenType,
          typename T, size_t N>
auto asEigen(
    FluidTensorView<T, N> &&a) // restrict this to FluidTensorView because
                               // passing in an rvalue FluidTensor would be silly
    -> Map<EigenType<T, Dynamic, Dynamic, RowMajor, Dynamic, Dynamic>,
           Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>> {
  return asEigen<EigenType>(a);
}

/**
 Convert a const rvalue FluidTensorView to a const Eigen Matrix or Array map
 (you need to say which as a template parmeter, e.g.
 makeWrapper<Matrix>(myCOnstView.row(0))
 **/
template <template <typename, int, int, int, int, int> class EigenType,
          typename T, size_t N>
auto asEigen(const FluidTensorView<T, N>
                 &&a) // restrict this to FluidTensorView because passing in an
                      // rvalue FluidTensor would be silly
    -> Map<const EigenType<T, Dynamic, Dynamic, RowMajor, Dynamic, Dynamic>,
           Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>> {
  return asEigen<EigenType>(a);
}
} // namespace _impl
} // namespace algorithm
} // namespace fluid
