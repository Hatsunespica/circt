// RUN: circt-opt --lower-pipeline-to-hw="outline-stages" %s | FileCheck %s

// CHECK-LABEL:   hw.module @testBasic_p0(
// CHECK-SAME:                            %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i1, valid: i1) {
// CHECK:           hw.output %[[VAL_0]], %[[VAL_1]] : i1, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testBasic(
// CHECK-SAME:                         %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) -> (out: i1) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = hw.instance "testBasic_p0" @testBasic_p0(in0: %[[VAL_0]]: i1, enable: %[[VAL_0]]: i1, clk: %[[VAL_1]]: i1, rst: %[[VAL_2]]: i1) -> (out0: i1, valid: i1)
// CHECK:           hw.output %[[VAL_3]] : i1
// CHECK:         }
hw.module @testBasic(%arg0: i1, %clk: i1, %rst: i1) -> (out: i1) {
  %0:2 = pipeline.scheduled(%arg0) clock %clk reset %rst go %arg0 : (i1) -> (i1) {
  ^bb0(%a0: i1, %s0_valid: i1):
    pipeline.return %a0 : i1
  }
  hw.output %0#0 : i1
}

// CHECK-LABEL:   hw.module @testLatency1_p0(
// CHECK-SAME:             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = hw.instance "testLatency1_p0_s0" @testLatency1_p0_s0(in0: %[[VAL_0]]: i32, enable: %[[VAL_1]]: i1, clk: %[[VAL_2]]: i1, rst: %[[VAL_3]]: i1) -> (out0: i32, valid: i1)
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "testLatency1_p0_s1" @testLatency1_p0_s1(in0: %[[VAL_4]]: i32, enable: %[[VAL_5]]: i1, clk: %[[VAL_2]]: i1, rst: %[[VAL_3]]: i1) -> (out0: i32, valid: i1)
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = hw.instance "testLatency1_p0_s2" @testLatency1_p0_s2(in0: %[[VAL_6]]: i32, enable: %[[VAL_7]]: i1, clk: %[[VAL_2]]: i1, rst: %[[VAL_3]]: i1) -> (out0: i32, valid: i1)
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = hw.instance "testLatency1_p0_s3" @testLatency1_p0_s3(in0: %[[VAL_8]]: i32, enable: %[[VAL_9]]: i1, clk: %[[VAL_2]]: i1, rst: %[[VAL_3]]: i1) -> (out0: i32, valid: i1)
// CHECK:           hw.output %[[VAL_10]], %[[VAL_11]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testLatency1_p0_s0(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1,  %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_4:.*]] = comb.add %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_5:.*]] = hw.constant false
// CHECK:           %[[VAL_6:.*]] = seq.compreg %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_5]]  : i1
// CHECK:           hw.output %[[VAL_4]], %[[VAL_6]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testLatency1_p0_s1(
// CHECK-SAME:               %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_4:.*]] = hw.constant false
// CHECK:           %[[VAL_5:.*]] = seq.compreg %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]]  : i1
// CHECK:           hw.output %[[VAL_0]], %[[VAL_5]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testLatency1_p0_s2(
// CHECK-SAME:                %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_4:.*]] = comb.mux %[[VAL_1]], %[[VAL_0]], %[[VAL_5:.*]] : i32
// CHECK:           %[[VAL_5]] = seq.compreg %[[VAL_4]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_6:.*]] = hw.constant false
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_6]]  : i1
// CHECK:           hw.output %[[VAL_5]], %[[VAL_7]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testLatency1_p0_s3(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_4:.*]] = comb.mux %[[VAL_1]], %[[VAL_0]], %[[VAL_5:.*]] : i32
// CHECK:           %[[VAL_5]] = seq.compreg %[[VAL_4]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_6:.*]] = hw.constant false
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_6]]  : i1
// CHECK:           hw.output %[[VAL_5]], %[[VAL_7]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testLatency1(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32, done: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "testLatency1_p0" @testLatency1_p0(in0: %[[VAL_0]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, valid: i1)
// CHECK:           hw.output %[[VAL_5]], %[[VAL_6]] : i32, i1
// CHECK:         }
hw.module @testLatency1(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32, done: i1) {
  %out, %done = pipeline.scheduled(%arg0) clock %clk reset %rst go %go : (i32) -> i32 {
  ^bb0(%arg0_0: i32, %s0_valid: i1):
    %1 = pipeline.latency 2 -> (i32) {
      %6 = comb.add %arg0_0, %arg0_0 : i32
      pipeline.latency.return %6 : i32
    }
    pipeline.stage ^bb1 pass(%1 : i32)
  ^bb1(%2: i32, %s1_valid: i1):  // pred: ^bb0
    pipeline.stage ^bb2 pass(%2 : i32)
  ^bb2(%3: i32, %s2_valid: i1):  // pred: ^bb1
    pipeline.stage ^bb3 regs(%3 : i32)
  ^bb3(%4: i32, %s3_valid: i1):  // pred: ^bb2
    pipeline.stage ^bb4 regs(%4 : i32)
  ^bb4(%5: i32, %s4_valid: i1):  // pred: ^bb3
    pipeline.return %5 : i32
  }
  hw.output %out, %done : i32, i1
}

// CHECK-LABEL:   hw.module @testSingle_p0(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "testSingle_p0_s0" @testSingle_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, valid: i1)
// CHECK:           %[[VAL_8:.*]] = comb.add %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:           hw.output %[[VAL_8]], %[[VAL_7]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingle_p0_s0(
// CHECK-SAME:                                %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = comb.mux %[[VAL_2]], %[[VAL_5]], %[[VAL_7:.*]] : i32
// CHECK:           %[[VAL_7]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = comb.mux %[[VAL_2]], %[[VAL_0]], %[[VAL_9:.*]] : i32
// CHECK:           %[[VAL_9]] = seq.compreg %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]]  : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_9]], %[[VAL_11]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingle(
// CHECK-SAME:                          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "testSingle_p0" @testSingle_p0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, valid: i1)
// CHECK:           hw.output %[[VAL_5]], %[[VAL_6]] : i32, i1
// CHECK:         }
hw.module @testSingle(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst go %go : (i32, i32) -> (i32) {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %s0_valid : i1):
    %1 = comb.sub %arg0_0, %arg1_1 : i32
    pipeline.stage ^bb1 regs(%1, %arg0_0 : i32, i32)
  ^bb1(%6: i32, %7: i32, %s1_valid : i1):  // pred: ^bb1
    %8 = comb.add %6, %7 : i32
    pipeline.return %8 : i32
  }
  hw.output %0#0, %0#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testMultiple_p0(
// CHECK-SAME:                               %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "testMultiple_p0_s0" @testMultiple_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, valid: i1)
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]] = hw.instance "testMultiple_p0_s1" @testMultiple_p0_s1(in0: %[[VAL_5]]: i32, in1: %[[VAL_6]]: i32, enable: %[[VAL_7]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, valid: i1)
// CHECK:           %[[VAL_11:.*]] = comb.mul %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           hw.output %[[VAL_11]], %[[VAL_10]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p0_s0(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = comb.mux %[[VAL_2]], %[[VAL_5]], %[[VAL_7:.*]] : i32
// CHECK:           %[[VAL_7]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = comb.mux %[[VAL_2]], %[[VAL_0]], %[[VAL_9:.*]] : i32
// CHECK:           %[[VAL_9]] = seq.compreg %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]]  : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_9]], %[[VAL_11]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p0_s1(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = comb.mux %[[VAL_2]], %[[VAL_5]], %[[VAL_7:.*]] : i32
// CHECK:           %[[VAL_7]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = comb.mux %[[VAL_2]], %[[VAL_0]], %[[VAL_9:.*]] : i32
// CHECK:           %[[VAL_9]] = seq.compreg %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]]  : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_9]], %[[VAL_11]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p1(
// CHECK-SAME:                               %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "testMultiple_p1_s0" @testMultiple_p1_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, valid: i1)
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]] = hw.instance "testMultiple_p1_s1" @testMultiple_p1_s1(in0: %[[VAL_5]]: i32, in1: %[[VAL_6]]: i32, enable: %[[VAL_7]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, valid: i1)
// CHECK:           %[[VAL_11:.*]] = comb.mul %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           hw.output %[[VAL_11]], %[[VAL_10]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p1_s0(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = comb.mux %[[VAL_2]], %[[VAL_5]], %[[VAL_7:.*]] : i32
// CHECK:           %[[VAL_7]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = comb.mux %[[VAL_2]], %[[VAL_0]], %[[VAL_9:.*]] : i32
// CHECK:           %[[VAL_9]] = seq.compreg %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]]  : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_9]], %[[VAL_11]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p1_s1(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = comb.mux %[[VAL_2]], %[[VAL_5]], %[[VAL_7:.*]] : i32
// CHECK:           %[[VAL_7]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = comb.mux %[[VAL_2]], %[[VAL_0]], %[[VAL_9:.*]] : i32
// CHECK:           %[[VAL_9]] = seq.compreg %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]]  : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_9]], %[[VAL_11]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple(
// CHECK-SAME:                            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "testMultiple_p0" @testMultiple_p0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, valid: i1)
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = hw.instance "testMultiple_p1" @testMultiple_p1(in0: %[[VAL_5]]: i32, in1: %[[VAL_1]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, valid: i1)
// CHECK:           hw.output %[[VAL_5]], %[[VAL_6]] : i32, i1
// CHECK:         }
hw.module @testMultiple(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst go %go : (i32, i32) -> (i32) {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %s0_valid: i1):
    %1 = comb.sub %arg0_0, %arg1_1 : i32
    pipeline.stage ^bb1 regs(%1, %arg0_0 : i32, i32)
  ^bb1(%2: i32, %3: i32, %s1_valid: i1):  // pred: ^bb0
    %5 = comb.add %2, %3 : i32
    pipeline.stage ^bb2 regs(%5, %2 : i32, i32)
  ^bb2(%6: i32, %7: i32, %s2_valid: i1):  // pred: ^bb1
    %8 = comb.mul %6, %7 : i32
    pipeline.return %8 : i32
  }

  %1:2 = pipeline.scheduled(%0#0, %arg1) clock %clk reset %rst go %go : (i32, i32) -> (i32) {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %s0_valid: i1):
    %1 = comb.sub %arg0_0, %arg1_1 : i32
    pipeline.stage ^bb1 regs(%1, %arg0_0 : i32, i32)
  ^bb1(%2: i32, %3: i32, %s1_valid: i1):  // pred: ^bb0
    %5 = comb.add %2, %3 : i32
    pipeline.stage ^bb2 regs(%5, %2 : i32, i32)
  ^bb2(%6: i32, %7: i32, %s2_valid: i1):  // pred: ^bb1
    %8 = comb.mul %6, %7 : i32
    pipeline.return %8 : i32
  }

  hw.output %0#0, %0#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testSingleWithExt_p0(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1, %[[VAL_5:.*]]: i1) -> (out0: i32, out1: i32, valid: i1) {
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "testSingleWithExt_p0_s0" @testSingleWithExt_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, enable: %[[VAL_3]]: i1, clk: %[[VAL_4]]: i1, rst: %[[VAL_5]]: i1) -> (out0: i32, valid: i1)
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = hw.instance "testSingleWithExt_p0_s1" @testSingleWithExt_p0_s1(in0: %[[VAL_6]]: i32, extIn0: %[[VAL_2]]: i32, enable: %[[VAL_7]]: i1, clk: %[[VAL_4]]: i1, rst: %[[VAL_5]]: i1) -> (out0: i32, valid: i1)
// CHECK:           hw.output %[[VAL_8]], %[[VAL_2]], %[[VAL_9]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingleWithExt_p0_s0(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = hw.constant true
// CHECK:           %[[VAL_6:.*]] = comb.sub %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_7:.*]] = comb.mux %[[VAL_2]], %[[VAL_6]], %[[VAL_8:.*]] : i32
// CHECK:           %[[VAL_8]] = seq.compreg %[[VAL_7]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_9:.*]] = hw.constant false
// CHECK:           %[[VAL_10:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_9]]  : i1
// CHECK:           hw.output %[[VAL_8]], %[[VAL_10]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingleWithExt_p0_s1(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = comb.mux %[[VAL_2]], %[[VAL_5]], %[[VAL_7:.*]] : i32
// CHECK:           %[[VAL_7]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = hw.constant false
// CHECK:           %[[VAL_9:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_8]]  : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_9]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingleWithExt(
// CHECK-SAME:                                 %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "testSingleWithExt_p0" @testSingleWithExt_p0(in0: %[[VAL_0]]: i32, in1: %[[VAL_0]]: i32, extIn0: %[[VAL_1]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, valid: i1)
// CHECK:           hw.output %[[VAL_5]], %[[VAL_6]] : i32, i32
// CHECK:         }
hw.module @testSingleWithExt(%arg0: i32, %ext1: i32, %go : i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i32) {
  %0:3 = pipeline.scheduled(%arg0, %arg0) ext (%ext1 : i32) clock %clk reset %rst go %go : (i32, i32) -> (i32, i32) {
  ^bb0(%a0: i32, %a1 : i32, %ext0: i32, %s0_valid: i1):
    %true = hw.constant true
    %1 = comb.sub %a0, %a0 : i32
    pipeline.stage ^bb1 regs(%1 : i32)

  ^bb1(%6: i32, %s1_valid: i1):
    // Use the external value inside a stage
    %8 = comb.add %6, %ext0 : i32
    pipeline.stage ^bb2 regs(%8 : i32)
  
  ^bb2(%9 : i32, %s2_valid: i1):
  // Use the external value in the exit stage.
    pipeline.return %9, %ext0  : i32, i32
  }
  hw.output %0#0, %0#1 : i32, i32
}

// CHECK-LABEL:   hw.module @testControlUsage_p0(
// CHECK-SAME:                                   %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1, %[[VAL_5:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "testControlUsage_p0_s0" @testControlUsage_p0_s0(in0: %[[VAL_0]]: i32, extIn0: %[[VAL_1]]: i1, extIn1: %[[VAL_2]]: i1, enable: %[[VAL_3]]: i1, clk: %[[VAL_4]]: i1, rst: %[[VAL_5]]: i1) -> (out0: i32, valid: i1)
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = hw.instance "testControlUsage_p0_s1" @testControlUsage_p0_s1(in0: %[[VAL_6]]: i32, extIn0: %[[VAL_1]]: i1, extIn1: %[[VAL_2]]: i1, enable: %[[VAL_7]]: i1, clk: %[[VAL_4]]: i1, rst: %[[VAL_5]]: i1) -> (out0: i32, valid: i1)
// CHECK:           %[[VAL_10:.*]] = hw.constant 0 : i32
// CHECK:           %[[VAL_11:.*]] = sv.wire : !hw.inout<i32>
// CHECK:           %[[VAL_12:.*]] = sv.read_inout %[[VAL_11]] : !hw.inout<i32>
// CHECK:           %[[VAL_13:.*]] = comb.add %[[VAL_12]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_14:.*]] = seq.compreg.ce %[[VAL_13]], %[[VAL_1]], %[[VAL_9]], %[[VAL_2]], %[[VAL_10]]  : i32
// CHECK:           sv.assign %[[VAL_11]], %[[VAL_14]] : i32
// CHECK:           hw.output %[[VAL_14]], %[[VAL_9]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testControlUsage_p0_s0(
// CHECK-SAME:                                      %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1, %[[VAL_5:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_6:.*]] = hw.constant 0 : i32
// CHECK:           %[[VAL_7:.*]] = sv.wire : !hw.inout<i32>
// CHECK:           %[[VAL_8:.*]] = sv.read_inout %[[VAL_7]] : !hw.inout<i32>
// CHECK:           %[[VAL_9:.*]] = comb.add %[[VAL_8]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_10:.*]] = seq.compreg.ce %[[VAL_9]], %[[VAL_1]], %[[VAL_3]], %[[VAL_2]], %[[VAL_6]]  : i32
// CHECK:           sv.assign %[[VAL_7]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_11:.*]] = comb.mux %[[VAL_3]], %[[VAL_10]], %[[VAL_12:.*]] : i32
// CHECK:           %[[VAL_12]] = seq.compreg %[[VAL_11]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_13:.*]] = hw.constant false
// CHECK:           %[[VAL_14:.*]] = seq.compreg %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_13]]  : i1
// CHECK:           hw.output %[[VAL_12]], %[[VAL_14]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testControlUsage_p0_s1(
// CHECK-SAME:                                      %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1, %[[VAL_5:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_6:.*]] = hw.constant 0 : i32
// CHECK:           %[[VAL_7:.*]] = sv.wire : !hw.inout<i32>
// CHECK:           %[[VAL_8:.*]] = sv.read_inout %[[VAL_7]] : !hw.inout<i32>
// CHECK:           %[[VAL_9:.*]] = comb.add %[[VAL_8]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_10:.*]] = seq.compreg.ce %[[VAL_9]], %[[VAL_1]], %[[VAL_3]], %[[VAL_2]], %[[VAL_6]]  : i32
// CHECK:           sv.assign %[[VAL_7]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_11:.*]] = comb.mux %[[VAL_3]], %[[VAL_10]], %[[VAL_12:.*]] : i32
// CHECK:           %[[VAL_12]] = seq.compreg %[[VAL_11]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_13:.*]] = hw.constant false
// CHECK:           %[[VAL_14:.*]] = seq.compreg %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_13]]  : i1
// CHECK:           hw.output %[[VAL_12]], %[[VAL_14]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testControlUsage(
// CHECK-SAME:                                %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32) {
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = hw.instance "testControlUsage_p0" @testControlUsage_p0(in0: %[[VAL_0]]: i32, extIn0: %[[VAL_2]]: i1, extIn1: %[[VAL_3]]: i1, enable: %[[VAL_1]]: i1, clk: %[[VAL_2]]: i1, rst: %[[VAL_3]]: i1) -> (out0: i32, valid: i1)
// CHECK:           hw.output %[[VAL_4]] : i32
// CHECK:         }
hw.module @testControlUsage(%arg0: i32, %go : i1, %clk: i1, %rst: i1) -> (out0: i32) {
  %0:2 = pipeline.scheduled(%arg0) ext (%clk, %rst : i1, i1) clock %clk reset %rst go %go : (i32) -> (i32) {
  ^bb0(%a0: i32, %ext_clk: i1, %ext_rst : i1, %s0_valid: i1):
    %zero = hw.constant 0 : i32
    %reg_out_wire = sv.wire : !hw.inout<i32>
    %reg_out = sv.read_inout %reg_out_wire : !hw.inout<i32>
    %add0 = comb.add %reg_out, %a0 : i32
    %out = seq.compreg.ce %add0, %ext_clk, %s0_valid, %ext_rst, %zero : i32
    sv.assign %reg_out_wire, %out : i32
    pipeline.stage ^bb1 regs(%out : i32)

  ^bb1(%6: i32, %s1_valid: i1):
    %reg1_out_wire = sv.wire : !hw.inout<i32>
    %reg1_out = sv.read_inout %reg1_out_wire : !hw.inout<i32>
    %add1 = comb.add %reg1_out, %6 : i32
    %out1 = seq.compreg.ce %add1, %ext_clk, %s1_valid, %ext_rst, %zero : i32
    sv.assign %reg1_out_wire, %out1 : i32

    pipeline.stage ^bb2 regs(%out1 : i32)
  
  ^bb2(%9 : i32, %s2_valid: i1):
    %reg2_out_wire = sv.wire : !hw.inout<i32>
    %reg2_out = sv.read_inout %reg2_out_wire : !hw.inout<i32>
    %add2 = comb.add %reg2_out, %9 : i32
    %out2 = seq.compreg.ce %add2, %ext_clk, %s2_valid, %ext_rst, %zero : i32
    sv.assign %reg2_out_wire, %out2 : i32
    pipeline.return %out2  : i32
  }
  hw.output %0#0 : i32
}

// -----

// CHECK-LABEL:   hw.module @testWithStall_p0(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "testWithStall_p0_s0" @testWithStall_p0_s0(in0: %[[VAL_0]]: i32, enable: %[[VAL_1]]: i1, stall: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, valid: i1)
// CHECK:           hw.output %[[VAL_5]], %[[VAL_6]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testWithStall_p0_s0(
// CHECK-SAME:             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = hw.constant true
// CHECK:           %[[VAL_6:.*]] = comb.xor %[[VAL_2]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_7:.*]] = comb.and %[[VAL_1]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_8:.*]] = comb.mux %[[VAL_7]], %[[VAL_0]], %[[VAL_9:.*]] : i32
// CHECK:           %[[VAL_9]] = seq.compreg %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = comb.mux %[[VAL_6]], %[[VAL_1]], %[[VAL_12:.*]] : i1
// CHECK:           %[[VAL_12]] = seq.compreg %[[VAL_11]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]]  : i1
// CHECK:           hw.output %[[VAL_9]], %[[VAL_12]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testWithStall(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "testWithStall_p0" @testWithStall_p0(in0: %[[VAL_0]]: i32, enable: %[[VAL_1]]: i1, stall: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, valid: i1)
// CHECK:           hw.output %[[VAL_5]], %[[VAL_6]] : i32, i1
// CHECK:         }
hw.module @testWithStall(%arg0: i32, %go: i1, %stall : i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%arg0) stall %stall clock %clk reset %rst go %go : (i32) -> (i32) {
  ^bb0(%arg0_0: i32, %s0_valid : i1):
    pipeline.stage ^bb1 regs(%arg0_0 : i32)
  ^bb1(%1: i32, %s1_valid : i1):  // pred: ^bb1
    pipeline.return %1 : i32
  }
  hw.output %0#0, %0#1 : i32, i1
}