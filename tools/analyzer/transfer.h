int getConstraint(std::tuple<APInt,APInt> arg0){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt andi=arg0_0&arg0_1;
  APInt const0(arg0_0.getBitWidth(),0);
  int result=andi.eq(const0);
  return result;
}
int getInstanceConstraint(std::tuple<APInt,APInt> arg0,APInt inst){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt neg_inst=~inst;
  APInt or1=neg_inst|arg0_0;
  APInt or2=inst|arg0_1;
  int cmp1=or1.eq(neg_inst);
  int cmp2=or2.eq(inst);
  int result=cmp1&cmp2;
  return result;
}
APInt OR(APInt arg0,APInt arg1){
  APInt autogen0=arg0|arg1;
  return autogen0;
}
std::tuple<APInt,APInt> ORImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt arg1_0=std::get<0>(arg1);
  APInt arg1_1=std::get<1>(arg1);
  APInt result_0=arg0_0&arg1_0;
  APInt result_1=arg0_1|arg1_1;
  std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
  return result;
}
std::tuple<APInt,APInt> intersection(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt arg1_0=std::get<0>(arg1);
  APInt arg1_1=std::get<1>(arg1);
  APInt result_0=arg0_0&arg1_0;
  APInt result_1=arg0_1&arg1_1;
  std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
  return result;
}
int isConstant(std::tuple<APInt,APInt> arg0){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt add_res=arg0_0|arg0_1;
  APInt all_ones = APInt::getAllOnes(arg0_1.getBitWidth());
  int cmp_res=add_res.eq(all_ones);
  return cmp_res;
}
APInt getConstant(std::tuple<APInt,APInt> arg0){
  APInt arg0_1=std::get<1>(arg0);
  return arg0_1;
}
APInt AND(APInt arg0,APInt arg1){
  APInt autogen1=arg0&arg1;
  return autogen1;
}
std::tuple<APInt,APInt> ANDImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt arg1_0=std::get<0>(arg1);
  APInt arg1_1=std::get<1>(arg1);
  APInt result_0=arg0_0|arg1_0;
  APInt result_1=arg0_1&arg1_1;
  std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
  return result;
}
APInt XOR(APInt arg0,APInt arg1){
  APInt autogen2=arg0^arg1;
  return autogen2;
}
std::tuple<APInt,APInt> XORImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt arg1_0=std::get<0>(arg1);
  APInt arg1_1=std::get<1>(arg1);
  APInt and_00=arg0_0&arg1_0;
  APInt and_11=arg0_1&arg1_1;
  APInt and_01=arg0_0&arg1_1;
  APInt and_10=arg0_1&arg1_0;
  APInt result_0=and_00|and_11;
  APInt result_1=and_01|and_10;
  std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
  return result;
}
APInt getMaxValue(std::tuple<APInt,APInt> arg0){
  APInt arg0_0=std::get<0>(arg0);
  APInt result=~arg0_0;
  return result;
}
APInt getMinValue(std::tuple<APInt,APInt> arg0){
  APInt arg0_1=std::get<1>(arg0);
  return arg0_1;
}
APInt countMinTrailingZeros(std::tuple<APInt,APInt> arg0){
  APInt arg0_0=std::get<0>(arg0);
  unsigned result_autocast=arg0_0.countr_one();
  APInt result(arg0_0.getBitWidth(),result_autocast);
  return result;
}
APInt countMinTrailingOnes(std::tuple<APInt,APInt> arg0){
  APInt arg0_1=std::get<1>(arg0);
  unsigned result_autocast=arg0_1.countr_one();
  APInt result(arg0_1.getBitWidth(),result_autocast);
  return result;
}
std::tuple<APInt,APInt> computeForAddCarry(std::tuple<APInt,APInt> lhs,std::tuple<APInt,APInt> rhs,APInt carryZero,APInt carryOne){
  APInt lhs0=std::get<0>(lhs);
  APInt lhs1=std::get<1>(lhs);
  APInt rhs0=std::get<0>(rhs);
  APInt rhs1=std::get<1>(rhs);
  APInt one(lhs0.getBitWidth(),1);
  APInt negCarryZero=one-carryZero;
  APInt lhsMax=getMaxValue(lhs);
  APInt lhsMin=getMinValue(lhs);
  APInt rhsMax=getMaxValue(rhs);
  APInt rhsMin=getMinValue(rhs);
  APInt possibleSumZeroTmp=lhsMax+rhsMax;
  APInt possibleSumZero=possibleSumZeroTmp+negCarryZero;
  APInt possibleSumOneTmp=lhsMin+rhsMin;
  APInt possibleSumOne=possibleSumOneTmp+carryOne;
  APInt carryKnownZeroTmp0=possibleSumZero^lhs0;
  APInt carryKnownZeroTmp1=carryKnownZeroTmp0^rhs0;
  APInt carryKnownZero=~carryKnownZeroTmp1;
  APInt carryKnownOneTmp=possibleSumOne^lhs1;
  APInt carryKnownOne=carryKnownOneTmp^rhs1;
  APInt lhsKnownUnion=lhs0|lhs1;
  APInt rhsKnownUnion=rhs0|rhs1;
  APInt carryKnownUnion=carryKnownZero|carryKnownOne;
  APInt knownTmp=lhsKnownUnion&rhsKnownUnion;
  APInt known=knownTmp&carryKnownUnion;
  APInt knownZeroTmp=~possibleSumZero;
  APInt knownZero=knownZeroTmp&known;
  APInt knownOne=possibleSumOne&known;
  std::tuple<APInt,APInt> result=std::make_tuple(knownZero,knownOne);
  return result;
}
APInt ADD(APInt arg0,APInt arg1){
  APInt autogen3=arg0+arg1;
  return autogen3;
}
std::tuple<APInt,APInt> ADDImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg1_0=std::get<0>(arg1);
  APInt one(arg1_0.getBitWidth(),1);
  APInt zero(arg1_0.getBitWidth(),0);
  std::tuple<APInt,APInt> result=computeForAddCarry(arg0,arg1,one,zero);
  return result;
}
APInt SUB(APInt arg0,APInt arg1){
  APInt autogen4=arg0-arg1;
  return autogen4;
}
std::tuple<APInt,APInt> SUBImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg1_0=std::get<0>(arg1);
  APInt arg1_1=std::get<1>(arg1);
  std::tuple<APInt,APInt> newRhs=std::make_tuple(arg1_1,arg1_0);
  APInt one(arg1_0.getBitWidth(),1);
  APInt zero(arg1_1.getBitWidth(),0);
  std::tuple<APInt,APInt> result=computeForAddCarry(arg0,newRhs,zero,one);
  return result;
}
APInt MUL(APInt arg0,APInt arg1){
  APInt autogen5=arg0*arg1;
  return autogen5;
}
std::tuple<APInt,APInt> MULImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0Max=getMaxValue(arg0);
  APInt arg1Max=getMaxValue(arg1);
  APInt umaxResult=arg0Max*arg1Max;
  bool umaxResultOverflow;
  arg0Max.umul_ov(arg1Max,umaxResultOverflow);
  APInt zero(arg0Max.getBitWidth(),0);
  unsigned umaxResult_cnt_l_zero_autocast=umaxResult.countl_zero();
  APInt umaxResult_cnt_l_zero(umaxResult.getBitWidth(),umaxResult_cnt_l_zero_autocast);
  APInt leadZ=umaxResultOverflow ? zero : umaxResult_cnt_l_zero ;
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt arg1_0=std::get<0>(arg1);
  APInt arg1_1=std::get<1>(arg1);
  APInt lhs_union=arg0_0|arg0_1;
  APInt rhs_union=arg1_0|arg1_1;
  unsigned trailBitsKnown0_autocast=lhs_union.countr_one();
  APInt trailBitsKnown0(lhs_union.getBitWidth(),trailBitsKnown0_autocast);
  unsigned trailBitsKnown1_autocast=rhs_union.countr_one();
  APInt trailBitsKnown1(rhs_union.getBitWidth(),trailBitsKnown1_autocast);
  unsigned trailZero0_autocast=arg0_0.countr_one();
  APInt trailZero0(arg0_0.getBitWidth(),trailZero0_autocast);
  unsigned trailZero1_autocast=arg1_0.countr_one();
  APInt trailZero1(arg1_0.getBitWidth(),trailZero1_autocast);
  APInt trailZ=trailZero0+trailZero1;
  APInt smallestOperand_arg0=trailBitsKnown0-trailZero0;
  APInt smallestOperand_arg1=trailBitsKnown1-trailZero1;
  APInt smallestOperand=smallestOperand_arg0.ule(smallestOperand_arg1)?smallestOperand_arg0:smallestOperand_arg1;
  APInt resultBitsKnown_arg0=smallestOperand+trailZ;
  unsigned bitwidth_autocast=arg0_0.getBitWidth();
  APInt bitwidth(arg0_0.getBitWidth(),bitwidth_autocast);
  APInt resultBitsKnown=resultBitsKnown_arg0.ule(bitwidth)?resultBitsKnown_arg0:bitwidth;
  APInt bottomKnown_arg0=arg0_1.getLoBits(trailBitsKnown0.getZExtValue());
  APInt bottomKnown_arg1=arg1_1.getLoBits(trailBitsKnown1.getZExtValue());
  APInt bottomKnown=bottomKnown_arg0*bottomKnown_arg1;
  APInt bottomKnown_neg=~bottomKnown;
  APInt resZerotmp2=bottomKnown_neg.getLoBits(resultBitsKnown.getZExtValue());
  APInt resZerotmp=zero;
  resZerotmp.setHighBits(leadZ.getZExtValue());
  APInt resZero=resZerotmp|resZerotmp2;
  APInt resOne=bottomKnown.getLoBits(resultBitsKnown.getZExtValue());
  std::tuple<APInt,APInt> result=std::make_tuple(resZero,resOne);
  return result;
}
APInt CONCAT(APInt arg0,APInt arg1){
  APInt autogen6=arg0.concat(arg1);
  return autogen6;
}
std::tuple<APInt,APInt> CONCATImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt arg1_0=std::get<0>(arg1);
  APInt arg1_1=std::get<1>(arg1);
  APInt result_0=arg0_0.concat(arg1_0);
  APInt result_1=arg0_1.concat(arg1_1);
  std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
  return result;
}
APInt EXTRACT(APInt arg0,APInt arg1,APInt arg2){
  APInt autogen7=arg0.extractBits(arg1.getZExtValue(),arg2.getZExtValue());
  return autogen7;
}
std::tuple<APInt,APInt> EXTRACTImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1,std::tuple<APInt,APInt> arg2){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt arg1_val=getConstant(arg1);
  APInt arg2_val=getConstant(arg2);
  APInt result_0=arg0_0.extractBits(arg1_val.getZExtValue(),arg2_val.getZExtValue());
  APInt result_1=arg0_1.extractBits(arg1_val.getZExtValue(),arg2_val.getZExtValue());
  std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
  return result;
}
APInt SHL(APInt arg0,APInt arg1){
  APInt autogen8=arg0.shl(arg1.getZExtValue());
  return autogen8;
}
std::tuple<APInt,APInt> SHLImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt const0(arg0_0.getBitWidth(),0);
  std::tuple<APInt,APInt> result=std::make_tuple(const0,const0);
  return result;
}
APInt ASHR(APInt arg0,APInt arg1){
  APInt autogen9=arg0.ashr(arg1.getZExtValue());
  return autogen9;
}
std::tuple<APInt,APInt> SHRSImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt const0(arg0_0.getBitWidth(),0);
  std::tuple<APInt,APInt> result=std::make_tuple(const0,const0);
  return result;
}
APInt LSHR(APInt arg0,APInt arg1){
  APInt autogen10=arg0.lshr(arg1.getZExtValue());
  return autogen10;
}
std::tuple<APInt,APInt> SHRUImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt const0(arg0_0.getBitWidth(),0);
  std::tuple<APInt,APInt> result=std::make_tuple(const0,const0);
  return result;
}
int eq(APInt arg0,APInt arg1){
  int autogen11=arg0.eq(arg1);
  return autogen11;
}
std::tuple<APInt,APInt> EQImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  int arg0_const=isConstant(arg0);
  int arg1_const=isConstant(arg1);
  int constCheck=arg0_const&arg1_const;
  APInt arg0_val=getConstant(arg0);
  APInt arg1_val=getConstant(arg1);
  int const0 = 0;
  int const1 = 1;
  int res1_1=arg0_val.eq(arg1_val);
  int res1_0=res1_1^const1;
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt arg1_0=std::get<0>(arg1);
  APInt arg1_1=std::get<1>(arg1);
  int cond1=arg0_1.intersects(arg1_0);
  int cond2=arg0_0.intersects(arg1_1);
  int cond=cond1|cond2;
  int result1_0=cond ? const1 : const0 ;
  int result_0_i1=constCheck ? res1_0 : result1_0 ;
  int result_1_i1=constCheck ? res1_1 : const0 ;
  APInt result_0(1, result_0_i1);
  APInt result_1(1, result_1_i1);
  std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
  return result;
}
int ne(APInt arg0,APInt arg1){
  int autogen12=arg0.ne(arg1);
  return autogen12;
}
std::tuple<APInt,APInt> NEImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  std::tuple<APInt,APInt> eqRes=EQImpl(arg0,arg1);
  int const0_i1 = 0;
  APInt const0(1, const0_i1);
  APInt eqRes_0=std::get<0>(eqRes);
  APInt eqRes_1=std::get<1>(eqRes);
  int eqConst=isConstant(eqRes);
  APInt res_0=eqConst ? eqRes_1 : const0 ;
  APInt res_1=eqConst ? eqRes_0 : const0 ;
  std::tuple<APInt,APInt> result=std::make_tuple(res_0,res_1);
  return result;
}
int slt(APInt arg0,APInt arg1){
  int autogen13=arg0.slt(arg1);
  return autogen13;
}
std::tuple<int,int> SLTImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  int const0 = 0;
  std::tuple<int,int> result=std::make_tuple(const0,const0);
  return result;
}
int sle(APInt arg0,APInt arg1){
  int autogen14=arg0.sle(arg1);
  return autogen14;
}
std::tuple<int,int> SLEImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  int const0 = 0;
  std::tuple<int,int> result=std::make_tuple(const0,const0);
  return result;
}
int sgt(APInt arg0,APInt arg1){
  int autogen15=arg0.sgt(arg1);
  return autogen15;
}
std::tuple<int,int> SGTImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  int const0 = 0;
  std::tuple<int,int> result=std::make_tuple(const0,const0);
  return result;
}
int sge(APInt arg0,APInt arg1){
  int autogen16=arg0.sge(arg1);
  return autogen16;
}
std::tuple<int,int> SGEImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  int const0 = 0;
  std::tuple<int,int> result=std::make_tuple(const0,const0);
  return result;
}
int ult(APInt arg0,APInt arg1){
  int autogen17=arg0.ult(arg1);
  return autogen17;
}
std::tuple<int,int> ULTImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  int const0 = 0;
  std::tuple<int,int> result=std::make_tuple(const0,const0);
  return result;
}
int ule(APInt arg0,APInt arg1){
  int autogen18=arg0.ule(arg1);
  return autogen18;
}
std::tuple<int,int> ULEImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  int const0 = 0;
  std::tuple<int,int> result=std::make_tuple(const0,const0);
  return result;
}
int ugt(APInt arg0,APInt arg1){
  int autogen19=arg0.ugt(arg1);
  return autogen19;
}
std::tuple<int,int> UGTImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  int const0 = 0;
  std::tuple<int,int> result=std::make_tuple(const0,const0);
  return result;
}
int uge(APInt arg0,APInt arg1){
  int autogen20=arg0.uge(arg1);
  return autogen20;
}
std::tuple<int,int> UGEImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  int const0 = 0;
  std::tuple<int,int> result=std::make_tuple(const0,const0);
  return result;
}
APInt MUX(int cond,APInt arg0,APInt arg1){
  APInt autogen21=cond ? arg0 : arg1 ;
  return autogen21;
}
std::tuple<APInt,APInt> MUXImpl(std::tuple<APInt,APInt> cond,std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt arg1_0=std::get<0>(arg1);
  APInt arg1_1=std::get<1>(arg1);
  APInt cond_0=std::get<0>(cond);
  int cond_const=isConstant(cond);
  APInt cond_val=getConstant(cond);
  APInt const1(cond_0.getBitWidth(),1);
  int cond_eq_1=cond_val.eq(const1);
  APInt cond_res_0=cond_eq_1 ? arg0_0 : arg1_0 ;
  APInt cond_res_1=cond_eq_1 ? arg0_1 : arg1_1 ;
  std::tuple<APInt,APInt> intersection_res=intersection(arg0,arg1);
  APInt intersection_0=std::get<0>(intersection_res);
  APInt intersection_1=std::get<1>(intersection_res);
  APInt result_0=cond_const ? cond_res_0 : intersection_0 ;
  APInt result_1=cond_const ? cond_res_1 : intersection_1 ;
  std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
  return result;
}
APInt REPEAT(APInt arg0,APInt arg1){
  APInt autogen22 = arg0;
  for(APInt i(arg1.getBitWidth(),1);i.ult(arg1);++i){
    autogen22 = autogen22.concat(arg0);
  }
  return autogen22;
}
std::tuple<APInt,APInt> REPEATImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
  APInt arg0_0=std::get<0>(arg0);
  APInt arg0_1=std::get<1>(arg0);
  APInt arg1_val=getConstant(arg1);
  APInt result_0 = arg0_0;
  for(APInt i(arg1_val.getBitWidth(),1);i.ult(arg1_val);++i){
    result_0 = result_0.concat(arg0_0);
  }
  APInt result_1 = arg0_1;
  for(APInt i(arg1_val.getBitWidth(),1);i.ult(arg1_val);++i){
    result_1 = result_1.concat(arg0_1);
  }
  std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
  return result;
}

std::tuple<APInt,APInt> CONCATImpl(ArrayRef<std::tuple<APInt,APInt>> operands){
  std::tuple<APInt,APInt> result=CONCATImpl(operands[0], operands[1]);
  for(int i=2;i<operands.size();++i){
    result=CONCATImpl(result, operands[i]);
  }
  return result;
}

std::optional<std::tuple<APInt,APInt>> naiveDispatcher(Operation* op, ArrayRef<std::tuple<APInt,APInt>> operands){
  if(auto castedOp=dyn_cast<circt::comb::OrOp>(op);castedOp){
    return ORImpl(operands[0], operands[1]);
  }
  if(auto castedOp=dyn_cast<circt::comb::AndOp>(op);castedOp){
    return ANDImpl(operands[0], operands[1]);
  }
  if(auto castedOp=dyn_cast<circt::comb::XorOp>(op);castedOp){
    return XORImpl(operands[0], operands[1]);
  }
  if(auto castedOp=dyn_cast<circt::comb::AddOp>(op);castedOp){
    return ADDImpl(operands[0], operands[1]);
  }
  if(auto castedOp=dyn_cast<circt::comb::SubOp>(op);castedOp){
    return SUBImpl(operands[0], operands[1]);
  }
  if(auto castedOp=dyn_cast<circt::comb::MulOp>(op);castedOp){
    return MULImpl(operands[0], operands[1]);
  }
  if(auto castedOp=dyn_cast<circt::comb::ConcatOp>(op);castedOp){
    return CONCATImpl(operands);
  }
  if(auto castedOp=dyn_cast<circt::comb::ExtractOp>(op);castedOp){
    return EXTRACTImpl(operands[0], operands[1], operands[2]);
  }
  if(auto castedOp=dyn_cast<circt::comb::ShlOp>(op);castedOp){
    return SHLImpl(operands[0], operands[1]);
  }
  if(auto castedOp=dyn_cast<circt::comb::ShrSOp>(op);castedOp){
    return SHRSImpl(operands[0], operands[1]);
  }
  if(auto castedOp=dyn_cast<circt::comb::ShrUOp>(op);castedOp){
    return SHRUImpl(operands[0], operands[1]);
  }
  if(auto castedOp=dyn_cast<circt::comb::MuxOp>(op);castedOp){
    return MUXImpl(operands[0], operands[1], operands[2]);
  }
  if(auto castedOp=dyn_cast<circt::comb::ReplicateOp>(op);castedOp){
    return REPEATImpl(operands[0], operands[1]);
  }
  return {};
}

