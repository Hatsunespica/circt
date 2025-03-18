static int getConstraint(std::vector<APInt> arg0){
  APInt arg0_0=arg0[0];
  int result=arg0_0.eq(arg0_0);
  return result;
}
static int getInstanceConstraint(std::vector<APInt> arg0,APInt inst){
  APInt arg0_0=arg0[0];
  int result=arg0_0.eq(arg0_0);
  return result;
}
static int isValidKnownBit(std::vector<APInt> arg0){
  APInt arg0_0=arg0[0];
  APInt arg0_1=arg0[1];
  APInt andi=arg0_0&arg0_1;
  APInt const0(arg0_0.getBitWidth(),0);
  int result=andi.eq(const0);
  return result;
}
static int inKnownBits(std::vector<APInt> arg0,APInt inst){
  APInt arg0_0=arg0[0];
  APInt arg0_1=arg0[1];
  APInt neg_inst=~inst;
  APInt or1=neg_inst|arg0_0;
  APInt or2=inst|arg0_1;
  int cmp1=or1.eq(neg_inst);
  int cmp2=or2.eq(inst);
  int result=cmp1&cmp2;
  return result;
}
static int inSameEq(std::vector<APInt> arg0,APInt inst,APInt inst1){
  APInt arg0_0=arg0[0];
  APInt eqinst=arg0_0&inst;
  APInt eqinst1=arg0_0&inst1;
  int eq=eqinst.eq(eqinst1);
  return eq;
}
static std::vector<APInt> XORImpl1(std::vector<APInt> arg0){
  return arg0;
}
static std::vector<APInt> XORImpl0(std::vector<APInt> arg0){
  return arg0;
}
static std::vector<APInt> AndImpl0(std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1){
  APInt arg0_0=arg0[0];
  APInt op0_0=op0[0];
  APInt op0_1=op0[1];
  APInt op1_0=op1[0];
  APInt op1_1=op1[1];
  APInt neg_op1_0=~op1_0;
  APInt result_1=neg_op1_0&arg0_0;
  std::vector<APInt> result=std::vector<APInt>{result_1};
  return result;
}
static std::vector<APInt> AndImpl1(std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1){
  APInt arg0_0=arg0[0];
  APInt op0_0=op0[0];
  APInt op0_1=op0[1];
  APInt op1_0=op1[0];
  APInt op1_1=op1[1];
  APInt neg_op1_0=~op1_0;
  APInt and_neg=neg_op1_0&op0_0;
  APInt neg_and=~and_neg;
  APInt result_1=neg_and&arg0_0;
  std::vector<APInt> result=std::vector<APInt>{result_1};
  return result;
}
static std::vector<APInt> OrImpl0(std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1){
  APInt arg0_0=arg0[0];
  APInt op0_0=op0[0];
  APInt op0_1=op0[1];
  APInt op1_0=op1[0];
  APInt op1_1=op1[1];
  APInt neg_op1_1=~op1_1;
  APInt result_1=neg_op1_1&arg0_0;
  std::vector<APInt> result=std::vector<APInt>{result_1};
  return result;
}
static std::vector<APInt> OrImpl1(std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1){
  APInt arg0_0=arg0[0];
  APInt op0_0=op0[0];
  APInt op0_1=op0[1];
  APInt op1_0=op1[0];
  APInt op1_1=op1[1];
  APInt neg_op1_1=~op1_1;
  APInt and_neg=neg_op1_1&op0_1;
  APInt neg_and=~and_neg;
  APInt result_1=neg_and&arg0_0;
  std::vector<APInt> result=std::vector<APInt>{result_1};
  return result;
}
static std::vector<APInt> determineLiveOperandBitsAddCarry(int operationNo,std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1,APInt carryZero,APInt carryOne){
  APInt arg0_0=arg0[0];
  APInt op0_0=op0[0];
  APInt op0_1=op0[1];
  APInt op1_0=op1[0];
  APInt op1_1=op1[1];
  APInt and_0_0=op0_0&op1_0;
  APInt and_1_1=op0_1&op1_1;
  APInt bound=and_0_0|and_1_1;
  APInt rbound=bound.reverseBits();
  APInt rarg0_0=arg0_0.reverseBits();
  APInt neg_rbound=~rbound;
  APInt or_rarg0_0_neg=rarg0_0|neg_rbound;
  APInt rprop=rarg0_0+or_rarg0_0_neg;
  APInt racarry=rprop^neg_rbound;
  APInt acarry=racarry.reverseBits();
  APInt neg_op0_0=~op0_0;
  APInt neg_op0_1=~op0_1;
  APInt neg_op1_0=~op1_0;
  APInt neg_op1_1=~op1_1;
  APInt or_0_0_neg_1=op0_0|neg_op1_0;
  APInt or_0_neg_0_1=neg_op0_0|op1_0;
  APInt or_1_0_neg_1=op0_1|neg_op1_1;
  APInt or_1_neg_0_1=neg_op0_1|op1_1;
  APInt neededToMaintainCarryZero=operationNo ? or_0_0_neg_1 : or_1_0_neg_1 ;
  APInt neededToMaintainCarryOne=operationNo ? or_0_neg_0_1 : or_1_neg_0_1 ;
  APInt one(arg0_0.getBitWidth(),1);
  APInt negCarryZero=one-carryZero;
  APInt possibleSumZeroTmp=neg_op0_0+neg_op1_0;
  APInt possibleSumZero=possibleSumZeroTmp+negCarryZero;
  APInt neg_possibleSumZero=~possibleSumZero;
  APInt possibleSumOneTmp=neg_op0_1+neg_op1_1;
  APInt possibleSumOne=possibleSumOneTmp+carryOne;
  APInt neededToMaintainCarry_0=neg_possibleSumZero|neededToMaintainCarryZero;
  APInt neededToMaintainCarry_1=possibleSumOne|neededToMaintainCarryOne;
  APInt neededToMaintainCarry=neededToMaintainCarry_0&neededToMaintainCarry_1;
  APInt carryAnd=acarry&neededToMaintainCarry;
  APInt result_1=arg0_0|carryAnd;
  std::vector<APInt> result=std::vector<APInt>{result_1};
  return result;
}
static std::vector<APInt> AddImpl0(std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1){
  int const0 = 0;
  int const1 = 1;
  APInt arg0_0=arg0[0];
  APInt transfer_const0(arg0_0.getBitWidth(),0);
  APInt transfer_const1(arg0_0.getBitWidth(),1);
  std::vector<APInt> result=determineLiveOperandBitsAddCarry(const0,arg0,op0,op1,transfer_const1,transfer_const0);
  return result;
}
static std::vector<APInt> AddImpl1(std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1){
  int const0 = 0;
  int const1 = 1;
  APInt arg0_0=arg0[0];
  APInt transfer_const0(arg0_0.getBitWidth(),0);
  APInt transfer_const1(arg0_0.getBitWidth(),1);
  std::vector<APInt> result=determineLiveOperandBitsAddCarry(const1,arg0,op0,op1,transfer_const1,transfer_const0);
  return result;
}
static std::vector<APInt> SubImpl0(std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1){
  int const0 = 0;
  int const1 = 1;
  APInt arg0_0=arg0[0];
  APInt op1_0=op1[0];
  APInt op1_1=op1[1];
  std::vector<APInt> newOp=std::vector<APInt>{op1_1,op1_0};
  APInt transfer_const0(arg0_0.getBitWidth(),0);
  APInt transfer_const1(arg0_0.getBitWidth(),1);
  std::vector<APInt> result=determineLiveOperandBitsAddCarry(const0,arg0,op0,newOp,transfer_const0,transfer_const1);
  return result;
}
static std::vector<APInt> SubImpl1(std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1){
  int const0 = 0;
  int const1 = 1;
  APInt arg0_0=arg0[0];
  APInt op1_0=op1[0];
  APInt op1_1=op1[1];
  std::vector<APInt> newOp=std::vector<APInt>{op1_1,op1_0};
  APInt transfer_const0(arg0_0.getBitWidth(),0);
  APInt transfer_const1(arg0_0.getBitWidth(),1);
  std::vector<APInt> result=determineLiveOperandBitsAddCarry(const1,arg0,op0,newOp,transfer_const0,transfer_const1);
  return result;
}
static int isConstant_i1(std::vector<int> arg0){
  int arg0_0=arg0[0];
  int arg0_1=arg0[1];
  int arg0_0_arith = arg0_0;
  int arg0_1_arith = arg0_1;
  int add_res=arg0_0_arith+arg0_1_arith;
  int all_ones = 1;
  int cmp_res=(add_res==all_ones);
  return cmp_res;
}
static int getConstant_i1(std::vector<int> arg0){
  int arg0_1=arg0[1];
  int arg0_1_arith = arg0_1;
  return arg0_1_arith;
}
static std::vector<APInt> MUXImplHelper(std::vector<APInt> arg0,std::vector<int> cond,int branchNo){
  APInt arg0_0=arg0[0];
  APInt const0(arg0_0.getBitWidth(),0);
  int cond_const=isConstant_i1(cond);
  int cond_val=getConstant_i1(cond);
  int cond_eq_branch=(cond_val==branchNo);
  APInt cond_res=cond_eq_branch ? arg0_0 : const0 ;
  std::vector<APInt> result=std::vector<APInt>{cond_res};
  return result;
}
static std::vector<APInt> MUXImpl0(std::vector<APInt> arg0,std::vector<int> cond){
  int const0 = 0;
  std::vector<APInt> result=MUXImplHelper(arg0,cond,const0);
  return result;
}
static std::vector<APInt> MUXImpl1(std::vector<APInt> arg0,std::vector<int> cond){
  int const1 = 1;
  std::vector<APInt> result=MUXImplHelper(arg0,cond,const1);
  return result;
}
static std::vector<APInt> EXTRACTImpl(std::vector<APInt> arg0,std::vector<APInt> op0,APInt len,APInt low_bit){
  APInt arg0_0=arg0[0];
  APInt op0_0=op0[0];
  APInt op0_1=op0[0];
  APInt const0(op0_0.getBitWidth(),0);
  APInt concat_res=const0.concat(arg0_0);
  APInt shl_res=concat_res.shl(low_bit.getZExtValue());
  unsigned bitwidth_autocast=op0_0.getBitWidth();
  APInt bitwidth(op0_0.getBitWidth(),bitwidth_autocast);
  APInt result_0=shl_res.extractBits(bitwidth.getZExtValue(),const0.getZExtValue());
  std::vector<APInt> result=std::vector<APInt>{result_0};
  return result;
}
static std::vector<APInt> ConcatImpl0(std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1){
  APInt arg0_0=arg0[0];
  APInt op0_0=op0[0];
  APInt op0_1=op0[1];
  APInt op1_0=op1[0];
  APInt op1_1=op1[1];
  APInt const0(op0_0.getBitWidth(),0);
  unsigned bitwidth0_autocast=op0_0.getBitWidth();
  APInt bitwidth0(op0_0.getBitWidth(),bitwidth0_autocast);
  unsigned bitwidth1_autocast=op1_0.getBitWidth();
  APInt bitwidth1(op1_0.getBitWidth(),bitwidth1_autocast);
  APInt result_0=arg0_0.extractBits(bitwidth0.getZExtValue(),bitwidth1.getZExtValue());
  std::vector<APInt> result=std::vector<APInt>{result_0};
  return result;
}
static std::vector<APInt> ConcatImpl1(std::vector<APInt> arg0,std::vector<APInt> op0,std::vector<APInt> op1){
  APInt arg0_0=arg0[0];
  APInt op0_0=op0[0];
  APInt op0_1=op0[1];
  APInt op1_0=op1[0];
  APInt op1_1=op1[1];
  APInt const0(op0_0.getBitWidth(),0);
  unsigned bitwidth0_autocast=op0_0.getBitWidth();
  APInt bitwidth0(op0_0.getBitWidth(),bitwidth0_autocast);
  unsigned bitwidth1_autocast=op1_0.getBitWidth();
  APInt bitwidth1(op1_0.getBitWidth(),bitwidth1_autocast);
  APInt result_0=arg0_0.extractBits(bitwidth1.getZExtValue(),const0.getZExtValue());
  std::vector<APInt> result=std::vector<APInt>{result_0};
  return result;
}
static std::optional<std::vector<APInt>> naiveDispatcher(Operation* op, std::vector<std::vector<llvm::APInt>> operands, unsigned operationNo){
  if(auto castedOp=dyn_cast<circt::comb::XorOp>(op);castedOp&&operationNo == 0){
    return XORImpl1(operands[0]);
  }
  if(auto castedOp=dyn_cast<circt::comb::XorOp>(op);castedOp&&operationNo == 1){
    return XORImpl0(operands[0]);
  }
  if(auto castedOp=dyn_cast<circt::comb::AndOp>(op);castedOp&&operationNo == 0){
    return AndImpl0(operands[0], operands[1], operands[2]);
  }
  if(auto castedOp=dyn_cast<circt::comb::AndOp>(op);castedOp&&operationNo == 1){
    return AndImpl1(operands[0], operands[1], operands[2]);
  }
  if(auto castedOp=dyn_cast<circt::comb::OrOp>(op);castedOp&&operationNo == 0){
    return OrImpl0(operands[0], operands[1], operands[2]);
  }
  if(auto castedOp=dyn_cast<circt::comb::OrOp>(op);castedOp&&operationNo == 1){
    return OrImpl1(operands[0], operands[1], operands[2]);
  }
  if(auto castedOp=dyn_cast<circt::comb::AddOp>(op);castedOp&&operationNo == 0){
    return AddImpl0(operands[0], operands[1], operands[2]);
  }
  if(auto castedOp=dyn_cast<circt::comb::AddOp>(op);castedOp&&operationNo == 0){
    return AddImpl1(operands[0], operands[1], operands[2]);
  }
  if(auto castedOp=dyn_cast<circt::comb::SubOp>(op);castedOp&&operationNo == 0){
    return SubImpl0(operands[0], operands[1], operands[2]);
  }
  if(auto castedOp=dyn_cast<circt::comb::SubOp>(op);castedOp&&operationNo == 1){
    return SubImpl1(operands[0], operands[1], operands[2]);
  }

  if(auto castedOp=dyn_cast<circt::comb::ConcatOp>(op);castedOp&&operationNo == 0){
    return ConcatImpl0(operands[0], operands[1], operands[2]);
  }
  if(auto castedOp=dyn_cast<circt::comb::ConcatOp>(op);castedOp&&operationNo == 1){
    return ConcatImpl1(operands[0], operands[1], operands[2]);
  }
  return {};
}

