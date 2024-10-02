// RUN: cgeist %s %stdinclude --function=* --print-debug-info -S | FileCheck %s

int main() { return 0; }
//CHECK: loc
