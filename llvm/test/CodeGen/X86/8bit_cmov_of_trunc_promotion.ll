; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=i386-unknown-unknown -mattr=-cmov | FileCheck %s --check-prefix=I386-NOCMOV
; RUN: llc < %s -mtriple=i386-unknown-unknown -mattr=+cmov | FileCheck %s --check-prefix=I386-CMOV
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=-cmov | FileCheck %s --check-prefix=I686-NOCMOV
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+cmov | FileCheck %s --check-prefix=I686-CMOV
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=-cmov | FileCheck %s --check-prefix=X86_64
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+cmov | FileCheck %s --check-prefix=X86_64

; Values don't come from regs. All good.

define i8 @t0(i32 %a1_wide_orig, i32 %a2_wide_orig, i32 %inc) nounwind {
; I386-NOCMOV-LABEL: t0:
; I386-NOCMOV:       # %bb.0:
; I386-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I386-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I386-NOCMOV-NEXT:    addl %ecx, %eax
; I386-NOCMOV-NEXT:    addl {{[0-9]+}}(%esp), %ecx
; I386-NOCMOV-NEXT:    cmpb %cl, %al
; I386-NOCMOV-NEXT:    jg .LBB0_2
; I386-NOCMOV-NEXT:  # %bb.1:
; I386-NOCMOV-NEXT:    movl %ecx, %eax
; I386-NOCMOV-NEXT:  .LBB0_2:
; I386-NOCMOV-NEXT:    # kill: def $al killed $al killed $eax
; I386-NOCMOV-NEXT:    retl
;
; I386-CMOV-LABEL: t0:
; I386-CMOV:       # %bb.0:
; I386-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I386-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I386-CMOV-NEXT:    addl %eax, %ecx
; I386-CMOV-NEXT:    addl {{[0-9]+}}(%esp), %eax
; I386-CMOV-NEXT:    cmpb %al, %cl
; I386-CMOV-NEXT:    cmovgl %ecx, %eax
; I386-CMOV-NEXT:    # kill: def $al killed $al killed $eax
; I386-CMOV-NEXT:    retl
;
; I686-NOCMOV-LABEL: t0:
; I686-NOCMOV:       # %bb.0:
; I686-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I686-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I686-NOCMOV-NEXT:    addl %ecx, %eax
; I686-NOCMOV-NEXT:    addl {{[0-9]+}}(%esp), %ecx
; I686-NOCMOV-NEXT:    cmpb %cl, %al
; I686-NOCMOV-NEXT:    jg .LBB0_2
; I686-NOCMOV-NEXT:  # %bb.1:
; I686-NOCMOV-NEXT:    movl %ecx, %eax
; I686-NOCMOV-NEXT:  .LBB0_2:
; I686-NOCMOV-NEXT:    # kill: def $al killed $al killed $eax
; I686-NOCMOV-NEXT:    retl
;
; I686-CMOV-LABEL: t0:
; I686-CMOV:       # %bb.0:
; I686-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I686-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I686-CMOV-NEXT:    addl %eax, %ecx
; I686-CMOV-NEXT:    addl {{[0-9]+}}(%esp), %eax
; I686-CMOV-NEXT:    cmpb %al, %cl
; I686-CMOV-NEXT:    cmovgl %ecx, %eax
; I686-CMOV-NEXT:    # kill: def $al killed $al killed $eax
; I686-CMOV-NEXT:    retl
;
; X86_64-LABEL: t0:
; X86_64:       # %bb.0:
; X86_64-NEXT:    movl %esi, %eax
; X86_64-NEXT:    addl %edx, %edi
; X86_64-NEXT:    addl %edx, %eax
; X86_64-NEXT:    cmpb %al, %dil
; X86_64-NEXT:    cmovgl %edi, %eax
; X86_64-NEXT:    # kill: def $al killed $al killed $eax
; X86_64-NEXT:    retq
  %a1_wide = add i32 %a1_wide_orig, %inc
  %a2_wide = add i32 %a2_wide_orig, %inc
  %a1 = trunc i32 %a1_wide to i8
  %a2 = trunc i32 %a2_wide to i8
  %t1 = icmp sgt i8 %a1, %a2
  %t2 = select i1 %t1, i8 %a1, i8 %a2
  ret i8 %t2
}

; Values don't come from regs, but there is only one truncation.

define i8 @neg_only_one_truncation(i32 %a1_wide_orig, i8 %a2_orig, i32 %inc) nounwind {
; I386-NOCMOV-LABEL: neg_only_one_truncation:
; I386-NOCMOV:       # %bb.0:
; I386-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I386-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I386-NOCMOV-NEXT:    addl %ecx, %eax
; I386-NOCMOV-NEXT:    addb {{[0-9]+}}(%esp), %cl
; I386-NOCMOV-NEXT:    cmpb %cl, %al
; I386-NOCMOV-NEXT:    jg .LBB1_2
; I386-NOCMOV-NEXT:  # %bb.1:
; I386-NOCMOV-NEXT:    movl %ecx, %eax
; I386-NOCMOV-NEXT:  .LBB1_2:
; I386-NOCMOV-NEXT:    # kill: def $al killed $al killed $eax
; I386-NOCMOV-NEXT:    retl
;
; I386-CMOV-LABEL: neg_only_one_truncation:
; I386-CMOV:       # %bb.0:
; I386-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I386-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I386-CMOV-NEXT:    addl %eax, %ecx
; I386-CMOV-NEXT:    addb {{[0-9]+}}(%esp), %al
; I386-CMOV-NEXT:    cmpb %al, %cl
; I386-CMOV-NEXT:    movzbl %al, %eax
; I386-CMOV-NEXT:    cmovgl %ecx, %eax
; I386-CMOV-NEXT:    # kill: def $al killed $al killed $eax
; I386-CMOV-NEXT:    retl
;
; I686-NOCMOV-LABEL: neg_only_one_truncation:
; I686-NOCMOV:       # %bb.0:
; I686-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I686-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I686-NOCMOV-NEXT:    addl %ecx, %eax
; I686-NOCMOV-NEXT:    addb {{[0-9]+}}(%esp), %cl
; I686-NOCMOV-NEXT:    cmpb %cl, %al
; I686-NOCMOV-NEXT:    jg .LBB1_2
; I686-NOCMOV-NEXT:  # %bb.1:
; I686-NOCMOV-NEXT:    movl %ecx, %eax
; I686-NOCMOV-NEXT:  .LBB1_2:
; I686-NOCMOV-NEXT:    # kill: def $al killed $al killed $eax
; I686-NOCMOV-NEXT:    retl
;
; I686-CMOV-LABEL: neg_only_one_truncation:
; I686-CMOV:       # %bb.0:
; I686-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I686-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I686-CMOV-NEXT:    addl %eax, %ecx
; I686-CMOV-NEXT:    addb {{[0-9]+}}(%esp), %al
; I686-CMOV-NEXT:    cmpb %al, %cl
; I686-CMOV-NEXT:    movzbl %al, %eax
; I686-CMOV-NEXT:    cmovgl %ecx, %eax
; I686-CMOV-NEXT:    # kill: def $al killed $al killed $eax
; I686-CMOV-NEXT:    retl
;
; X86_64-LABEL: neg_only_one_truncation:
; X86_64:       # %bb.0:
; X86_64-NEXT:    addl %edx, %edi
; X86_64-NEXT:    addb %sil, %dl
; X86_64-NEXT:    cmpb %dl, %dil
; X86_64-NEXT:    movzbl %dl, %eax
; X86_64-NEXT:    cmovgl %edi, %eax
; X86_64-NEXT:    # kill: def $al killed $al killed $eax
; X86_64-NEXT:    retq
  %a1_wide = add i32 %a1_wide_orig, %inc
  %inc_short = trunc i32 %inc to i8
  %a2 = add i8 %a2_orig, %inc_short
  %a1 = trunc i32 %a1_wide to i8
  %t1 = icmp sgt i8 %a1, %a2
  %t2 = select i1 %t1, i8 %a1, i8 %a2
  ret i8 %t2
}

; Values don't come from regs, but truncation from different types.

define i8 @neg_type_mismatch(i32 %a1_wide_orig, i16 %a2_wide_orig, i32 %inc) nounwind {
; I386-NOCMOV-LABEL: neg_type_mismatch:
; I386-NOCMOV:       # %bb.0:
; I386-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I386-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I386-NOCMOV-NEXT:    addl %ecx, %eax
; I386-NOCMOV-NEXT:    addw {{[0-9]+}}(%esp), %cx
; I386-NOCMOV-NEXT:    cmpb %cl, %al
; I386-NOCMOV-NEXT:    jg .LBB2_2
; I386-NOCMOV-NEXT:  # %bb.1:
; I386-NOCMOV-NEXT:    movl %ecx, %eax
; I386-NOCMOV-NEXT:  .LBB2_2:
; I386-NOCMOV-NEXT:    # kill: def $al killed $al killed $eax
; I386-NOCMOV-NEXT:    retl
;
; I386-CMOV-LABEL: neg_type_mismatch:
; I386-CMOV:       # %bb.0:
; I386-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I386-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I386-CMOV-NEXT:    addl %eax, %ecx
; I386-CMOV-NEXT:    addw {{[0-9]+}}(%esp), %ax
; I386-CMOV-NEXT:    cmpb %al, %cl
; I386-CMOV-NEXT:    cmovgl %ecx, %eax
; I386-CMOV-NEXT:    # kill: def $al killed $al killed $eax
; I386-CMOV-NEXT:    retl
;
; I686-NOCMOV-LABEL: neg_type_mismatch:
; I686-NOCMOV:       # %bb.0:
; I686-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I686-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I686-NOCMOV-NEXT:    addl %ecx, %eax
; I686-NOCMOV-NEXT:    addw {{[0-9]+}}(%esp), %cx
; I686-NOCMOV-NEXT:    cmpb %cl, %al
; I686-NOCMOV-NEXT:    jg .LBB2_2
; I686-NOCMOV-NEXT:  # %bb.1:
; I686-NOCMOV-NEXT:    movl %ecx, %eax
; I686-NOCMOV-NEXT:  .LBB2_2:
; I686-NOCMOV-NEXT:    # kill: def $al killed $al killed $eax
; I686-NOCMOV-NEXT:    retl
;
; I686-CMOV-LABEL: neg_type_mismatch:
; I686-CMOV:       # %bb.0:
; I686-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I686-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I686-CMOV-NEXT:    addl %eax, %ecx
; I686-CMOV-NEXT:    addw {{[0-9]+}}(%esp), %ax
; I686-CMOV-NEXT:    cmpb %al, %cl
; I686-CMOV-NEXT:    cmovgl %ecx, %eax
; I686-CMOV-NEXT:    # kill: def $al killed $al killed $eax
; I686-CMOV-NEXT:    retl
;
; X86_64-LABEL: neg_type_mismatch:
; X86_64:       # %bb.0:
; X86_64-NEXT:    movl %esi, %eax
; X86_64-NEXT:    addl %edx, %edi
; X86_64-NEXT:    addl %edx, %eax
; X86_64-NEXT:    cmpb %al, %dil
; X86_64-NEXT:    cmovgl %edi, %eax
; X86_64-NEXT:    # kill: def $al killed $al killed $eax
; X86_64-NEXT:    retq
  %a1_wide = add i32 %a1_wide_orig, %inc
  %inc_short = trunc i32 %inc to i16
  %a2_wide = add i16 %a2_wide_orig, %inc_short
  %a1 = trunc i32 %a1_wide to i8
  %a2 = trunc i16 %a2_wide to i8
  %t1 = icmp sgt i8 %a1, %a2
  %t2 = select i1 %t1, i8 %a1, i8 %a2
  ret i8 %t2
}

; One value come from regs

define i8 @negative_CopyFromReg(i32 %a1_wide, i32 %a2_wide_orig, i32 %inc) nounwind {
; I386-NOCMOV-LABEL: negative_CopyFromReg:
; I386-NOCMOV:       # %bb.0:
; I386-NOCMOV-NEXT:    movb {{[0-9]+}}(%esp), %al
; I386-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I386-NOCMOV-NEXT:    addl {{[0-9]+}}(%esp), %ecx
; I386-NOCMOV-NEXT:    cmpb %cl, %al
; I386-NOCMOV-NEXT:    jg .LBB3_2
; I386-NOCMOV-NEXT:  # %bb.1:
; I386-NOCMOV-NEXT:    movl %ecx, %eax
; I386-NOCMOV-NEXT:  .LBB3_2:
; I386-NOCMOV-NEXT:    retl
;
; I386-CMOV-LABEL: negative_CopyFromReg:
; I386-CMOV:       # %bb.0:
; I386-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I386-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I386-CMOV-NEXT:    addl {{[0-9]+}}(%esp), %eax
; I386-CMOV-NEXT:    cmpb %al, %cl
; I386-CMOV-NEXT:    cmovgl %ecx, %eax
; I386-CMOV-NEXT:    # kill: def $al killed $al killed $eax
; I386-CMOV-NEXT:    retl
;
; I686-NOCMOV-LABEL: negative_CopyFromReg:
; I686-NOCMOV:       # %bb.0:
; I686-NOCMOV-NEXT:    movb {{[0-9]+}}(%esp), %al
; I686-NOCMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I686-NOCMOV-NEXT:    addl {{[0-9]+}}(%esp), %ecx
; I686-NOCMOV-NEXT:    cmpb %cl, %al
; I686-NOCMOV-NEXT:    jg .LBB3_2
; I686-NOCMOV-NEXT:  # %bb.1:
; I686-NOCMOV-NEXT:    movl %ecx, %eax
; I686-NOCMOV-NEXT:  .LBB3_2:
; I686-NOCMOV-NEXT:    retl
;
; I686-CMOV-LABEL: negative_CopyFromReg:
; I686-CMOV:       # %bb.0:
; I686-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I686-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I686-CMOV-NEXT:    addl {{[0-9]+}}(%esp), %eax
; I686-CMOV-NEXT:    cmpb %al, %cl
; I686-CMOV-NEXT:    cmovgl %ecx, %eax
; I686-CMOV-NEXT:    # kill: def $al killed $al killed $eax
; I686-CMOV-NEXT:    retl
;
; X86_64-LABEL: negative_CopyFromReg:
; X86_64:       # %bb.0:
; X86_64-NEXT:    movl %esi, %eax
; X86_64-NEXT:    addl %edx, %eax
; X86_64-NEXT:    cmpb %al, %dil
; X86_64-NEXT:    cmovgl %edi, %eax
; X86_64-NEXT:    # kill: def $al killed $al killed $eax
; X86_64-NEXT:    retq
  %a2_wide = add i32 %a2_wide_orig, %inc
  %a1 = trunc i32 %a1_wide to i8
  %a2 = trunc i32 %a2_wide to i8
  %t1 = icmp sgt i8 %a1, %a2
  %t2 = select i1 %t1, i8 %a1, i8 %a2
  ret i8 %t2
}

; Both values come from regs

define i8 @negative_CopyFromRegs(i32 %a1_wide, i32 %a2_wide) nounwind {
; I386-NOCMOV-LABEL: negative_CopyFromRegs:
; I386-NOCMOV:       # %bb.0:
; I386-NOCMOV-NEXT:    movb {{[0-9]+}}(%esp), %cl
; I386-NOCMOV-NEXT:    movb {{[0-9]+}}(%esp), %al
; I386-NOCMOV-NEXT:    cmpb %cl, %al
; I386-NOCMOV-NEXT:    jg .LBB4_2
; I386-NOCMOV-NEXT:  # %bb.1:
; I386-NOCMOV-NEXT:    movl %ecx, %eax
; I386-NOCMOV-NEXT:  .LBB4_2:
; I386-NOCMOV-NEXT:    retl
;
; I386-CMOV-LABEL: negative_CopyFromRegs:
; I386-CMOV:       # %bb.0:
; I386-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I386-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I386-CMOV-NEXT:    cmpb %al, %cl
; I386-CMOV-NEXT:    cmovgl %ecx, %eax
; I386-CMOV-NEXT:    # kill: def $al killed $al killed $eax
; I386-CMOV-NEXT:    retl
;
; I686-NOCMOV-LABEL: negative_CopyFromRegs:
; I686-NOCMOV:       # %bb.0:
; I686-NOCMOV-NEXT:    movb {{[0-9]+}}(%esp), %cl
; I686-NOCMOV-NEXT:    movb {{[0-9]+}}(%esp), %al
; I686-NOCMOV-NEXT:    cmpb %cl, %al
; I686-NOCMOV-NEXT:    jg .LBB4_2
; I686-NOCMOV-NEXT:  # %bb.1:
; I686-NOCMOV-NEXT:    movl %ecx, %eax
; I686-NOCMOV-NEXT:  .LBB4_2:
; I686-NOCMOV-NEXT:    retl
;
; I686-CMOV-LABEL: negative_CopyFromRegs:
; I686-CMOV:       # %bb.0:
; I686-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %eax
; I686-CMOV-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; I686-CMOV-NEXT:    cmpb %al, %cl
; I686-CMOV-NEXT:    cmovgl %ecx, %eax
; I686-CMOV-NEXT:    # kill: def $al killed $al killed $eax
; I686-CMOV-NEXT:    retl
;
; X86_64-LABEL: negative_CopyFromRegs:
; X86_64:       # %bb.0:
; X86_64-NEXT:    movl %esi, %eax
; X86_64-NEXT:    cmpb %al, %dil
; X86_64-NEXT:    cmovgl %edi, %eax
; X86_64-NEXT:    # kill: def $al killed $al killed $eax
; X86_64-NEXT:    retq
  %a1 = trunc i32 %a1_wide to i8
  %a2 = trunc i32 %a2_wide to i8
  %t1 = icmp sgt i8 %a1, %a2
  %t2 = select i1 %t1, i8 %a1, i8 %a2
  ret i8 %t2
}
