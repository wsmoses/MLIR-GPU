// RUN: cgeist %s --no-inline --function=* --struct-abi=0 -memref-abi=0 -S | FileCheck %s

void run0(void*);
void run1(void*);
void run2(void*);

class M {
    public:
 M() { run0(this); }
};


struct _Alloc_hider : M
      {
	_Alloc_hider() { run1(this); }

      };
  
    class basic_ostringstream 
    {
    public:
      _Alloc_hider _M_stringbuf;
      basic_ostringstream() { run2(this); }
    };

void a() {
    ::basic_ostringstream a;
}

// CHECK-LABEL:   func.func @_Z1av() 
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(struct<(i8)>)> : (i64) -> !llvm.ptr
// CHECK:           call @_ZN19basic_ostringstreamC1Ev(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN19basic_ostringstreamC1Ev(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !llvm.ptr) 
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i8)>)>
// CHECK:           call @_ZN12_Alloc_hiderC1Ev(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK:           %[[VAL_2:.*]] = "polygeist.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           call @_Z4run2Pv(%[[VAL_2]]) : (memref<?xi8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN12_Alloc_hiderC1Ev(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !llvm.ptr) 
// CHECK:           call @_ZN1MC1Ev(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK:           %[[VAL_1:.*]] = "polygeist.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           call @_Z4run1Pv(%[[VAL_1]]) : (memref<?xi8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN1MC1Ev(
// CHECK-SAME:                         %[[VAL_0:.*]]: !llvm.ptr) 
// CHECK:           %[[VAL_1:.*]] = "polygeist.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           call @_Z4run0Pv(%[[VAL_1]]) : (memref<?xi8>) -> ()
// CHECK:           return
// CHECK:         }

