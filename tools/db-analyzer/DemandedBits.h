//
// Created by spica on 9/12/24.
//

#ifndef DEMANDEDBITS_H
#define DEMANDEDBITS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/InitAllDialects.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/ADT/StringSet.h"
#include <string>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/SourceMgr.h>
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;


template<typename FunctionOp, typename ReturnOp>
class MLIRDemandedBits {
public:
    MLIRDemandedBits(FunctionOp f):f(f), analyzed(false){}
    FunctionOp getFunction(){return f;}
    llvm::APInt getDemandedBits(mlir::Operation *I);
    void performAnalysis();
    void determineLiveOperandBits(mlir::Operation *UserI,
                                  mlir::Value Val, unsigned OperandNo,
                                  const llvm::APInt &AOut, llvm::APInt &AB,
                                  llvm::KnownBits &Known, llvm::KnownBits &Known2, bool &KnownBitsComputed);
    llvm::DenseMap<mlir::Operation*, llvm::APInt> getAliveBits(){return aliveBits;};
    void print();
private:
    FunctionOp f;

    bool analyzed;
    llvm::DenseMap<mlir::Operation*, llvm::APInt>aliveBits;
    llvm::SmallPtrSet<mlir::Operation*, 32> visited;
    llvm::SmallPtrSet<mlir::Value*, 16> deadUses;
};

using namespace std;
using namespace mlir;
using namespace llvm;

#include "transfer.h"

static std::string toString(const llvm::KnownBits& kb){
  std::string res;
  res.resize(kb.getBitWidth());
  for(size_t i=0;i<res.size();++i){
    unsigned N = res.size() - i - 1;
    if(kb.Zero[N]&&kb.One[N]){
      res[i]='!';
    }else if(kb.Zero[N]){
      res[i]='0';
    }else if(kb.One[N]){
      res[i]='1';
    }else{
      res[i]='?';
    }
  }
  return res;
}

static std::string toString(const llvm::APInt apInt){
  std::string res;
  res.resize(apInt.getBitWidth());
  for(size_t i=0;i<res.size();++i){
    unsigned N = res.size() - i - 1;
    if(apInt[N]){
      res[i]='1';
    }else{
      res[i]='0';
    }
  }
  return res;
}


static llvm::KnownBits fromString(std::string& str){
  llvm::KnownBits result(str.size());
  std::reverse(str.begin(),str.end());
  for(int i=0;i<str.size();++i){
    if(str[i]=='1'){
      result.One.setBit(i);
    }else if(str[i]=='0'){
      result.Zero.setBit(i);
    }
  }
  return result;
}

template<typename FunctionOp, typename ReturnOp>
llvm::APInt MLIRDemandedBits<FunctionOp, ReturnOp>::getDemandedBits(mlir::Operation *I){
  performAnalysis();
  auto found=aliveBits.find(I);
  if(found!=aliveBits.end()){
    return found->second;
  }
  return llvm::APInt::getAllOnes(I->getResult(0).getType().getIntOrFloatBitWidth());
}

template<typename FunctionOp, typename ReturnOp>
void MLIRDemandedBits<FunctionOp, ReturnOp>::print(){
  f.walk([&](mlir::Operation* op){
    if(auto it=aliveBits.find(op);it!=aliveBits.end()){
      it->first->dump();
      llvm::errs()<<"DB: "<<toString(it->second)<<"\n";
    }
  });
}

template<typename FunctionOp, typename ReturnOp>
void MLIRDemandedBits<FunctionOp, ReturnOp>::performAnalysis(){
  if(analyzed){
    return;
  }

  analyzed=true;
  visited.clear();
  aliveBits.clear();
  llvm::SmallSetVector<mlir::Operation*, 16> worklist;
  f.walk([&](ReturnOp op){
    mlir::Operation* oper=op.getOperation();
    for(size_t i=0;i<oper->getNumOperands();++i){
      mlir::Type retTy=oper->getOperand(i).getType();
      aliveBits.try_emplace(oper, llvm::APInt(retTy.getIntOrFloatBitWidth(), 0));
      worklist.insert(oper);
    }

  });
  while(!worklist.empty()){
    mlir::Operation* op = worklist.pop_back_val();
    llvm::KnownBits known, known2;
    llvm::APInt AOut = aliveBits[op];
    unsigned operNo=0;
    bool knownBitsComputed=false;
    if(op->getNumOperands()>=1){
      auto V0=op->getOperand(0);
      if(llvm::isa<mlir::OpResult>(V0)){
        assert(V0.getDefiningOp()->hasAttr("kb"));
        mlir::Attribute attr=V0.getDefiningOp()->getAttr("kb");
        std::string str=dyn_cast<mlir::StringAttr>(attr).getValue().str();
        known=fromString(str);
      }else{
        known=llvm::KnownBits(V0.getType().getIntOrFloatBitWidth());
      }
    }

    if(op->getNumOperands()>=2){
      auto V1=op->getOperand(1);
      if(llvm::isa<mlir::OpResult>(V1)){
        assert(V1.getDefiningOp()->hasAttr("kb"));
        mlir::Attribute attr=V1.getDefiningOp()->getAttr("kb");
        std::string str=dyn_cast<mlir::StringAttr>(attr).getValue().str();
        known2=fromString(str);
      }else{
        known2=llvm::KnownBits(V1.getType().getIntOrFloatBitWidth());
      }
    }

    for(auto operand: op->getOperands()){
      if(llvm::isa<mlir::OpResult>(operand)){
        assert(operand.getType().isIntOrIndex());
        unsigned width=operand.getType().getIntOrFloatBitWidth();
        llvm::APInt AB=llvm::APInt::getAllOnes(width);

        determineLiveOperandBits(op, operand, operNo, AOut, AB, known, known2, knownBitsComputed);

        mlir::Operation* defOp=operand.getDefiningOp();
        auto res=aliveBits.try_emplace(defOp);
        if(res.second || (AB |= res.first->second) != res.first->second){
          res.first->second=AB;
          worklist.insert(defOp);
        }
      }
      operNo+=1;
    }
  }
}

template<typename FunctionOp, typename ReturnOp>
void MLIRDemandedBits<FunctionOp, ReturnOp>::determineLiveOperandBits(mlir::Operation *UserI,
                                                                      mlir::Value val, unsigned OperandNo,
                                                                      const llvm::APInt &AOut, llvm::APInt &AB,
                                                                      llvm::KnownBits &Known, llvm::KnownBits &Known2, bool &KnownBitsComputed){
  std::vector<std::vector<APInt>> args;
  args.push_back({AOut});
  args.push_back({Known.Zero, Known.One});
  args.push_back({Known2.Zero, Known2.One});
  std::optional<std::vector<APInt>> res;
  if(auto castedOp=dyn_cast<circt::comb::MuxOp>(UserI);castedOp&&OperandNo == 1){
    assert( Known.Zero.getBitWidth()==1);
    res= MUXImpl0(args[0], {Known.Zero.getLimitedValue(), Known.One.getLimitedValue()});
  }
  if(auto castedOp=dyn_cast<circt::comb::MuxOp>(UserI);castedOp&&OperandNo == 2){
    assert( Known.Zero.getBitWidth()==1);
    res= MUXImpl1(args[0], {Known.Zero.getLimitedValue(), Known.One.getLimitedValue()});
  }
  if(auto castedOp=dyn_cast<circt::comb::ExtractOp>(UserI);castedOp&&OperandNo == 0){
    int lowBit=castedOp.getLowBit();
    auto resType=castedOp.getResult().getType().dyn_cast_or_null<mlir::IntegerType>();
    int resLen=resType.getWidth();
    res= EXTRACTImpl(args[0], args[1], llvm::APInt(32, resLen), llvm::APInt(32, lowBit));
  }else{
    res = naiveDispatcher(UserI, args, OperandNo);
  }


  if(res){
    AB=(*res)[0];
  }else{
    unsigned width=val.getType().getIntOrFloatBitWidth();
    AB=llvm::APInt::getAllOnes(width);
  }
}


#endif //DEMANDEDBITS_H
