//===- LowerComb.cpp - Lower some ops in comb -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "DemandedBits.h"

using namespace circt;
using namespace circt::comb;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_DBTEST
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

namespace {
/// Lower truth tables to mux trees.
} // namespace

namespace {
class DBTestPass : public circt::comb::impl::DBTestBase<DBTestPass> {
public:
  using DBTestBase::DBTestBase;

  void runOnOperation() override;
};
} // namespace

extern std::pair<long long,long long> analyzeModule(ModuleOp m,bool debug=false);

static mlir::TypedAttr getIntAttr(const llvm::APInt &value, mlir::MLIRContext *context) {
  return mlir::IntegerAttr::get(mlir::IntegerType::get(context, value.getBitWidth()),
                          value);
}

static mlir::OpFoldResult comb_subSimplify(mlir::Operation& op, mlir::MLIRContext* context){
  if(llvm::isa<circt::comb::SubOp>(op)){
    if(op.getOperand(0) == op.getOperand(1)){
      unsigned width = op.getOperand(0).getType().getIntOrFloatBitWidth();
      llvm::APInt result(width, 0);

      return getIntAttr(result, context);
    }
    llvm::APInt const0;
    if(matchPattern(op.getOperand(1), m_ConstantInt(&const0))){
      unsigned width = op.getOperand(0).getType().getIntOrFloatBitWidth();
      llvm::APInt result(width, 0);
      if(const0 == result){
        return op.getOperand(1);
      }
    }
  }
  return {};
}

static mlir::OpFoldResult comb_xorSimplify(mlir::Operation& op, mlir::MLIRContext* context){
  if(llvm::isa<circt::comb::XorOp>(op)){
    if(op.getOperand(0) == op.getOperand(1)){
      unsigned width = op.getOperand(0).getType().getIntOrFloatBitWidth();
      llvm::APInt result(width, 0);

      return getIntAttr(result, context);
    }
    llvm::APInt const0;
    if(matchPattern(op.getOperand(0), m_ConstantInt(&const0))){
      unsigned width = op.getOperand(0).getType().getIntOrFloatBitWidth();
      llvm::APInt result(width, 0);
      if(const0 == result){
        return op.getOperand(0);
      }
    }
  }
  return {};
}

static mlir::OpFoldResult comb_orSimplify(mlir::Operation& op, mlir::MLIRContext* context){
  if(llvm::isa<circt::comb::OrOp>(op)){
    if(op.getOperand(0) == op.getOperand(1)){
      return op.getOperand(0);
    }
    llvm::APInt const0;
    if(matchPattern(op.getOperand(0), m_ConstantInt(&const0))){
      unsigned width = op.getOperand(0).getType().getIntOrFloatBitWidth();
      llvm::APInt result(width, 255);
      if(const0 == result){
        return getIntAttr(result, context);
      }
    }
    llvm::APInt const1;
    if(matchPattern(op.getOperand(0), m_ConstantInt(&const1))){
      unsigned width = op.getOperand(0).getType().getIntOrFloatBitWidth();
      llvm::APInt result(width, 0);
      if(const1 == result){
        return op.getOperand(0);
      }
    }
  }
  return {};
}


static bool isComb(mlir::Operation *op) {
  if (llvm::isa<circt::comb::AddOp>(op) || llvm::isa<circt::comb::AndOp>(op) ||
      llvm::isa<circt::comb::ConcatOp>(op) ||
      llvm::isa<circt::comb::DivSOp>(op) ||
      llvm::isa<circt::comb::DivUOp>(op) ||
      llvm::isa<circt::comb::ExtractOp>(op) ||
      llvm::isa<circt::comb::ICmpOp>(op) ||
      llvm::isa<circt::comb::ModSOp>(op) ||
      llvm::isa<circt::comb::ModUOp>(op) || llvm::isa<circt::comb::MulOp>(op) ||
      llvm::isa<circt::comb::MuxOp>(op) || llvm::isa<circt::comb::OrOp>(op) ||
      llvm::isa<circt::comb::ParityOp>(op) ||
      llvm::isa<circt::comb::ReplicateOp>(op) ||
      llvm::isa<circt::comb::ShlOp>(op) || llvm::isa<circt::comb::ShrUOp>(op) ||
      llvm::isa<circt::comb::ShrSOp>(op) || llvm::isa<circt::comb::XorOp>(op) ||
      llvm::isa<circt::comb::SubOp>(op)) {
    return true;
  }
  return false;
}

static void opSimplify(mlir::Operation* op, mlir::OpBuilder& opBuilder) {
  if(isComb(op)){
    mlir::MLIRContext* context=op->getContext();
    mlir::OpFoldResult opRes=comb_orSimplify(*op, context);
    if(opRes){
      if(opRes.is<mlir::Attribute>()){
        opBuilder.setInsertionPoint(op);
        auto int_val=opRes.get<mlir::Attribute>().cast<mlir::IntegerAttr>().getValue();
        auto constantOp=opBuilder.create<hw::ConstantOp>(op->getLoc(),int_val);
        op->getResult(0).replaceAllUsesWith(constantOp.getResult());
      }else{
        mlir::Value newVal=opRes.get<mlir::Value>();
        op->getResult(0).replaceAllUsesWith(newVal);
      }
    }
    opRes=comb_xorSimplify(*op,context);
    if(opRes){
      if(opRes.is<mlir::Attribute>()){
        opBuilder.setInsertionPoint(op);
        auto int_val=opRes.get<mlir::Attribute>().cast<mlir::IntegerAttr>().getValue();
        auto constantOp=opBuilder.create<hw::ConstantOp>(op->getLoc(),int_val);
        op->getResult(0).replaceAllUsesWith(constantOp.getResult());
      }else{
        mlir::Value newVal=opRes.get<mlir::Value>();
        op->getResult(0).replaceAllUsesWith(newVal);
      }
    }
    opRes=comb_subSimplify(*op,context);
    if(opRes){
      if(opRes.is<mlir::Attribute>()){
        opBuilder.setInsertionPoint(op);
        auto int_val=opRes.get<mlir::Attribute>().cast<mlir::IntegerAttr>().getValue();
        auto constantOp=opBuilder.create<hw::ConstantOp>(op->getLoc(),int_val);
        op->getResult(0).replaceAllUsesWith(constantOp.getResult());
      }else{
        mlir::Value newVal=opRes.get<mlir::Value>();
        op->getResult(0).replaceAllUsesWith(newVal);
      }
    }
  }
}

static bool isBinaryOperator(mlir::Operation* op){
  if(llvm::isa<circt::comb::AndOp>(op)||llvm::isa<circt::comb::OrOp>(op)||llvm::isa<circt::comb::XorOp>(op)){
    return true;
  }
  return false;
}

static llvm::KnownBits getKnownBits(mlir::Value val, unsigned bitwidth){
  if(mlir::Operation* op=val.getDefiningOp();op!=nullptr){
    if(op->hasAttr("kb")){
      std::string kbStr=op->getAttr("kb").cast<mlir::StringAttr>().str();
      return fromString(kbStr);
    }
  }
  return llvm::KnownBits(bitwidth);
}

static void instCombineSimplifyDB(mlir::Operation* op, mlir::OpBuilder& opBuilder, MLIRDemandedBits<mlir::func::FuncOp,mlir::func::ReturnOp>& db){
  if(false&&op->getResult(0).hasOneUse()){


  }else if(isComb(op)){
    auto dbVal=db.getDemandedBits(op);
    auto bitwidth=dbVal.getBitWidth();
    llvm::KnownBits LHSKnown(bitwidth),RHSKnown(bitwidth);
    llvm::KnownBits known(bitwidth);
    mlir::Type retTy=op->getResult(0).getType();

    if(op->hasAttr("kb")){
      std::string kbStr=op->getAttr("kb").cast<mlir::StringAttr>().str();
      known=fromString(kbStr);
    }
    if(llvm::isa<circt::comb::AndOp>(op)){
      LHSKnown= getKnownBits(op->getOperand(0), bitwidth);
      RHSKnown= getKnownBits(op->getOperand(1), bitwidth);
      /*
       * if (DemandedMask.isSubsetOf(Known.Zero | Known.One))
  return Constant::getIntegerValue(ITy, Known.One);

      // If all of the demanded bits are known 1 on one side, return the other.
      // These bits cannot contribute to the result of the 'and' in this context.
      if (DemandedMask.isSubsetOf(LHSKnown.Zero | RHSKnown.One))
        return I->getOperand(0);
      if (DemandedMask.isSubsetOf(RHSKnown.Zero | LHSKnown.One))
        return I->getOperand(1);
      */
      if(dbVal.isSubsetOf(known.Zero|known.One)){
        opBuilder.setInsertionPoint(op);
        auto constVal=opBuilder.create<hw::ConstantOp>(op->getLoc(),known.One);
        op->replaceUsesOfWith(op->getResult(0), constVal.getResult());
        llvm::errs()<<"And1 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(LHSKnown.Zero|RHSKnown.One)){
        op->replaceUsesOfWith(op->getResult(0), op->getOperand(0));
        llvm::errs()<<"And2 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(RHSKnown.Zero| LHSKnown.One)){
        op->replaceUsesOfWith(op->getResult(0), op->getOperand(1));
        llvm::errs()<<"And3 triggered\n";
        return;
      }


    }else if(llvm::isa<circt::comb::OrOp>(op)){
      LHSKnown= getKnownBits(op->getOperand(0), bitwidth);
      RHSKnown= getKnownBits(op->getOperand(1), bitwidth);
      /*
       * // constant.
if (DemandedMask.isSubsetOf(Known.Zero | Known.One))
  return Constant::getIntegerValue(ITy, Known.One);

// We can simplify (X|Y) -> X or Y in the user's context if we know that
// only bits from X or Y are demanded.
// If all of the demanded bits are known zero on one side, return the other.
// These bits cannot contribute to the result of the 'or' in this context.
if (DemandedMask.isSubsetOf(LHSKnown.One | RHSKnown.Zero))
  return I->getOperand(0);
if (DemandedMask.isSubsetOf(RHSKnown.One | LHSKnown.Zero))
  return I->getOperand(1);

       */
      if(dbVal.isSubsetOf(known.Zero|known.One)){
        opBuilder.setInsertionPoint(op);
        auto constVal=opBuilder.create<hw::ConstantOp>(op->getLoc(),known.One);
        op->replaceUsesOfWith(op->getResult(0), constVal.getResult());
        llvm::errs()<<"Or1 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(LHSKnown.One|RHSKnown.Zero)){
        op->replaceUsesOfWith(op->getResult(0), op->getOperand(0));
        llvm::errs()<<"Or2 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(RHSKnown.One| LHSKnown.Zero)){
        //op->replaceUsesOfWith(op->getResult(0), op->getOperand(1));
        op->dump();
        op->getResult(0).replaceAllUsesWith(op->getOperand(1));
        op->getParentOp()->dump();
        llvm::errs()<<"Or3 triggered\n";
        return;
      }

    }else if(llvm::isa<circt::comb::XorOp>(op)){
      LHSKnown= getKnownBits(op->getOperand(0), bitwidth);
      RHSKnown= getKnownBits(op->getOperand(1), bitwidth);
/*
 * // If the client is only demanding bits that we know, return the known
      // constant.
      if (DemandedMask.isSubsetOf(Known.Zero | Known.One))
        return Constant::getIntegerValue(ITy, Known.One);

      // We can simplify (X^Y) -> X or Y in the user's context if we know that
      // only bits from X or Y are demanded.
      // If all of the demanded bits are known zero on one side, return the other.
      if (DemandedMask.isSubsetOf(RHSKnown.Zero))
        return I->getOperand(0);
      if (DemandedMask.isSubsetOf(LHSKnown.Zero))
        return I->getOperand(1);
 */
      if(dbVal.isSubsetOf(known.Zero|known.One)){
        opBuilder.setInsertionPoint(op);
        auto constVal=opBuilder.create<hw::ConstantOp>(op->getLoc(),known.One);
        op->replaceUsesOfWith(op->getResult(0), constVal.getResult());
        llvm::errs()<<"Xor1 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(RHSKnown.Zero)){
        op->replaceUsesOfWith(op->getResult(0), op->getOperand(0));
        llvm::errs()<<"Xor2 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(LHSKnown.Zero)){
        op->replaceUsesOfWith(op->getResult(0), op->getOperand(1));
        llvm::errs()<<"Xor3 triggered\n";
        return;
      }
    }
  }
}


void DBTestPass::runOnOperation() {
  ModuleOp module = getOperation();

  auto res= analyzeModule(module);
  std::vector<MLIRDemandedBits<mlir::func::FuncOp,mlir::func::ReturnOp>> v;
  module.walk([&v](mlir::func::FuncOp op){
    v.push_back(op);
  });
  v[0].performAnalysis();
  MLIRContext& context=*module.getOperation()->getContext();
  v[0].getFunction()->walk([&v, &context](mlir::Operation* op){
    if(isComb(op)){
      std::string tmpStr;
      llvm::APInt db=v[0].getDemandedBits(op);
      tmpStr.resize(db.getBitWidth());
      for(size_t i=0;i<tmpStr.size();++i){
        unsigned N = tmpStr.size() - i - 1;
        if(db[N]){
          tmpStr[i]='1';
        }else{
          tmpStr[i]='0';
        }
      }
      Twine tmpTwine(tmpStr);
      mlir::StringAttr dbAttr=mlir::StringAttr::get(&context, tmpTwine);
      op->setAttr("db", dbAttr);
    }
  });
  //BDCE
  mlir::OpBuilder opBuilder(&context);
  v[0].getFunction()->walk([&module, &context, &v, &opBuilder](mlir::Operation* op){
    instCombineSimplifyDB(op, opBuilder, v[0]);
    if(isBinaryOperator(op)){
      mlir::Value op0 = op->getOperand(0), op1=op->getOperand(1);
      auto db=v[0].getDemandedBits(op);
      if(!db.isAllOnes()){
        bool canBeSimplified=false;
        if(mlir::Operation* op1op=op1.getDefiningOp(); op1op!=nullptr&&op1op->hasAttr("kb")){
          mlir::StringAttr attr=op1op->getAttr("kb").cast<mlir::StringAttr>();
          std::string knownBitsStr= attr.data();
          auto kb = fromString(knownBitsStr);
          if(kb.isConstant()){
            auto kbVal=kb.getConstant();
            if(llvm::isa<circt::comb::AndOp>(op)){
              canBeSimplified=db.isSubsetOf(kbVal);
            }else if(llvm::isa<circt::comb::OrOp>(op)||llvm::isa<circt::comb::XorOp>(op)){
              canBeSimplified=!db.intersects(kbVal);
            }
          }
        }
        if(canBeSimplified) {
          llvm::errs() << "DB triggered\n";
          op->replaceUsesOfWith(op->getResult(0), op0);
          return;
        }
      }
    }
    for(int i=0;i<op->getNumOperands();++i){
      if(mlir::Operation* opop=op->getOperand(i).getDefiningOp();opop!=nullptr){
        if(auto opdb=v[0].getDemandedBits(opop);opdb.isZero()&&opop->getResult(0).getType().isIntOrIndex()){
          mlir::OperationState opState(opop->getLoc(), "constant");
          //auto const0 = circt::hw::ConstantOp::build(opBuilder, opState, opop->getResult(0).getType(),0);
          opBuilder.setInsertionPoint(op);
          auto const0 = opBuilder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(&context), opop->getResult(0).getType(), 0);
          op->setOperand(i, const0);
          llvm::errs()<<"Replace 0 triggered\n";
        }
      }
    }
  });
}
