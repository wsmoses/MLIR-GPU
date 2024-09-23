//===- PolygeistDialect.cpp - Polygeist dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "polygeist/Ops.h"

using namespace mlir;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// PolygeistDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with polygeist
/// operations.
struct PolygeistInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// Returns true if the given region 'src' can be inlined into the region
  /// 'dest' that is attached to an operation registered to the current dialect.
  /// 'wouldBeCloned' is set if the region is cloned into its new location
  /// rather than moved, indicating there may be other users.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  /// Returns true if the given operation 'op', that is registered to this
  /// dialect, can be inlined into the given region, false otherwise.
  bool isLegalToInline(Operation *op, Region *region, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  /// Polygeist regions should be analyzed recursively.
  bool shouldAnalyzeRecursively(Operation *op) const final { return true; }
};
} // namespace
//===----------------------------------------------------------------------===//
// Polygeist dialect.
//===----------------------------------------------------------------------===//

void PolygeistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "polygeist/PolygeistOps.cpp.inc"
      >();
  addInterfaces<PolygeistInlinerInterface>();
}

#include "polygeist/PolygeistOpsDialect.cpp.inc"
