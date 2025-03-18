#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "DemandedBits.h"

using namespace std;
using namespace mlir;

llvm::cl::OptionCategory MlirTvCategory("mlir-tv options", "");
llvm::cl::OptionCategory MLIR_MUTATE_CAT("mlir-mutate-tv options", "");

llvm::cl::opt<string> filename_src(llvm::cl::Positional,
                                   llvm::cl::desc("first-mlir-file"),
                                   llvm::cl::Required,
                                   llvm::cl::value_desc("filename"),
                                   llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<string> output_folder("o",
                                    llvm::cl::desc("Specify output folder"),
                                    llvm::cl::value_desc("folder name"),
                                    llvm::cl::Optional);
llvm::cl::opt<bool>
    arg_verbose("verbose", llvm::cl::desc("Be verbose about what's going on"),
                llvm::cl::Hidden, llvm::cl::init(false),
                llvm::cl::cat(MLIR_MUTATE_CAT));

// Defined in the test directory, no public header.
namespace circt {
namespace test {
void registerAnalysisTestPasses();
} // namespace test
} // namespace circt

filesystem::path inputPath, outputPath;
bool isValidInputPath(), isComb(mlir::Operation *op);
void visit(mlir::Operation *op, std::vector<mlir::Operation *> &tmp,
           std::unordered_set<mlir::Operation *> &visited);

mlir::BlockArgument addParameter(mlir::func::FuncOp &func, mlir::Type ty) {
  func.insertArgument(func.getNumArguments(), ty, {}, func->getLoc());
  return func.getArgument(func.getNumArguments() - 1);
}

void addResult(mlir::func::FuncOp &func, mlir::Value val) {
  mlir::Operation &retOp = func.getFunctionBody().getBlocks().front().back();
  retOp.insertOperands(retOp.getNumOperands(), val);
}

mlir::func::FuncOp moveToFunc(MLIRContext &context,
                              std::vector<mlir::Operation *> ops,
                              mlir::Location loc, unordered_map<mlir::Operation*, mlir::Operation*>& copyToOrigin) {

  mlir::OpBuilder builder(&context);

  mlir::FunctionType funcTy = mlir::FunctionType::get(&context, {}, {});
  auto func = builder.create<mlir::func::FuncOp>(loc, "tmp", funcTy);
  mlir::Block *blk = func.addEntryBlock();

  auto retOp = builder.create<mlir::func::ReturnOp>(func->getLoc());
  blk->push_back(retOp.getOperation());

  unordered_set<mlir::Operation *> needReturn;
  unordered_map<mlir::Operation *, mlir::Operation *> um;
  // arg_num -> current arg_num;
  unordered_map<int, mlir::BlockArgument> arg_um;
  std::vector<mlir::Operation *> stk;

  for (auto op : ops) {
    mlir::Operation *cur = op->clone();
    copyToOrigin.emplace(cur, op);
    stk.push_back(cur);
    um.insert({op, cur});
    needReturn.insert(cur);

    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      mlir::Value arg = op->getOperand(i);
      mlir::Type arg_ty = arg.getType();
      if (mlir::Operation *definingOp = arg.getDefiningOp(); definingOp) {
        if (auto it = um.find(definingOp); it != um.end()) {
          /*
           * Calc the result index in definingOp
           * Assume there are multiple returns
           */
          size_t idx = 0;
          needReturn.erase(it->second);
          for (; idx < definingOp->getNumResults(); ++idx) {
            if (definingOp->getResult(idx) == arg) {
              cur->setOperand(i, it->second->getResult(idx));
            }
          }
        } else {
          mlir::BlockArgument newArg = addParameter(func, arg_ty);
          cur->setOperand(i, newArg);
        }
      } else {
        mlir::BlockArgument blk_arg = arg.cast<mlir::BlockArgument>();
        int arg_num = blk_arg.getArgNumber();
        if (arg_um.find(arg_num) == arg_um.end()) {
          arg_um.insert({arg_num, addParameter(func, arg_ty)});
        }
        cur->setOperand(i, arg_um[arg_num]);
      }
    }
  }

  while (!stk.empty()) {
    blk->push_front(stk.back());
    stk.pop_back();
  }

  for (auto op : needReturn) {
    if (isComb(op)) {
      for (auto res_it = op->result_begin(); res_it != op->result_end();
           ++res_it) {
        addResult(func, *res_it);
      }
    }
  }

  funcTy = mlir::FunctionType::get(&context, func.getArgumentTypes(),
                                   retOp.getOperation()->getOperandTypes());
  func.setFunctionType(funcTy);


  return func;
}

mlir::ModuleOp moveToModule(MLIRContext &context,
                            std::vector<mlir::Operation *> ops,
                            mlir::Location loc, unordered_map<mlir::Operation*, mlir::Operation*>& copyToOrigin){
  auto func= moveToFunc(context, ops, loc, copyToOrigin);
  auto moduleOp=ModuleOp::create(mlir::UnknownLoc::get(&context));
  moduleOp.getBodyRegion().getBlocks().front().push_front(func.getOperation());
  return moduleOp;
}

std::string funcToString(mlir::func::FuncOp func) {
  std::string result;
  llvm::raw_string_ostream os(result);
  func.print(os);
  return os.str();
}

std::vector<std::vector<mlir::Operation*>> sliceFunctions(ModuleOp moduleOp), extractFunctions(ModuleOp moduleOp);

void replaceConstantKB(mlir::Operation* op, mlir::OpBuilder& builder, mlir::OpBuilder& originalOpBuilder
                       , MLIRDemandedBits<mlir::func::FuncOp,mlir::func::ReturnOp>& db
                       ,unordered_map<mlir::Operation*, mlir::Operation*>& copyToOrigin){
  if(op->hasAttr("kb")&&!llvm::isa<circt::hw::ConstantOp>(op)){
    string kbAttr=op->getAttr("kb").cast<StringAttr>().str();
    llvm::KnownBits kb= fromString(kbAttr);
    if(kb.hasConflict()){
      op->getParentOp()->dump();
      exit(0);
    }
    if(kb.isConstant()){
      llvm::errs()<<"Constant Triggered\n";
      mlir::Operation* originalOp=copyToOrigin[op];
      assert(kb.getBitWidth()==originalOp->getResult(0).getType().getIntOrFloatBitWidth());
      for(auto userIt=originalOp->user_begin();userIt!=originalOp->user_end();++userIt){
        originalOpBuilder.setInsertionPoint(*userIt);
        auto constOp=originalOpBuilder.create<circt::hw::ConstantOp>(originalOp->getLoc(), kb.getConstant());
        userIt->replaceUsesOfWith(originalOp->getResult(0), constOp.getResult());
      }
    }
  }
}

void replaceConstantDB(mlir::Operation* op, mlir::OpBuilder& builder, mlir::OpBuilder& originalOpBuilder
                       , MLIRDemandedBits<mlir::func::FuncOp,mlir::func::ReturnOp>& db
                       ,unordered_map<mlir::Operation*, mlir::Operation*>& copyToOrigin){
  mlir::MLIRContext* context=op->getContext();
  for(int i=0;i<op->getNumOperands();++i){
    if(mlir::Operation* opop=op->getOperand(i).getDefiningOp();opop!=nullptr&&isComb(opop)){
      if(auto opdb=db.getDemandedBits(opop);opdb.isZero()&&opop->getResult(0).getType().isIntOrIndex()){
        llvm::errs()<<"DB triggered\n";
        builder.setInsertionPoint(op);
        auto const0 = builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(context), opop->getResult(0).getType(), 0);
        op->setOperand(i, const0);


        assert(copyToOrigin.find(op)!=copyToOrigin.end());
        mlir::Operation* originOp=copyToOrigin[op];
        //check if all users are comb dialect
        mlir::Operation* originOpop=originOp->getOperand(i).getDefiningOp();
        originalOpBuilder.setInsertionPoint(originOp);
        unsigned bitwidth=originOpop->getResult(0).getType().getIntOrFloatBitWidth();
        auto constOp=originalOpBuilder.create<circt::hw::ConstantOp>(originOp->getLoc(), llvm::APInt::getZero(bitwidth));
        originOpop->replaceUsesOfWith(originOp->getResult(0), constOp.getResult());

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

void replaceBitwiseBDCE(mlir::Operation* op, mlir::OpBuilder& builder, mlir::OpBuilder& originalOpBuilder
                       , MLIRDemandedBits<mlir::func::FuncOp,mlir::func::ReturnOp>& db
                       ,unordered_map<mlir::Operation*, mlir::Operation*>& copyToOrigin){
  if(isBinaryOperator(op)){
    mlir::Value op0 = op->getOperand(0), op1=op->getOperand(1);
    auto dbVal=db.getDemandedBits(op);
    if(!dbVal.isAllOnes()){
      bool canBeSimplified=false;
      if(mlir::Operation* op1op=op1.getDefiningOp(); op1op!=nullptr&&op1op->hasAttr("kb")){
        mlir::StringAttr attr=op1op->getAttr("kb").cast<mlir::StringAttr>();
        std::string knownBitsStr= attr.data();
        auto kb = fromString(knownBitsStr);
        if(kb.isConstant()){
          auto kbVal=kb.getConstant();
          if(llvm::isa<circt::comb::AndOp>(op)){
            canBeSimplified=dbVal.isSubsetOf(kbVal);
          }else if(llvm::isa<circt::comb::OrOp>(op)||llvm::isa<circt::comb::XorOp>(op)){
            canBeSimplified=!dbVal.intersects(kbVal);
          }
        }
      }
      if(canBeSimplified) {
        llvm::errs() << "BDCE triggered\n";
        mlir::Operation* originalOp=copyToOrigin[op];
        for(auto userIt=originalOp->user_begin();userIt!=originalOp->user_end();++userIt){
          userIt->replaceUsesOfWith(originalOp->getResult(0), originalOp->getOperand(0));
        }
        //op->replaceUsesOfWith(op->getResult(0), op0);
        return;
      }
    }
  }
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

static void instCombineSimplifyDB(mlir::Operation* op, mlir::OpBuilder& builder, mlir::OpBuilder& originalOpBuilder
                                  , MLIRDemandedBits<mlir::func::FuncOp,mlir::func::ReturnOp>& db
                                  ,unordered_map<mlir::Operation*, mlir::Operation*>& copyToOrigin){
  //if(false&&op->getResult(0).hasOneUse()){

  //}else if(isComb(op)){
  if(isComb(op)){
    auto dbVal=db.getDemandedBits(op);
    auto bitwidth=dbVal.getBitWidth();
    llvm::KnownBits LHSKnown(bitwidth),RHSKnown(bitwidth);
    llvm::KnownBits known(bitwidth);
    mlir::Type retTy=op->getResult(0).getType();
    mlir::Operation* originOp=copyToOrigin[op];

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
        //opBuilder.setInsertionPoint(op);
        //auto constVal=opBuilder.create<hw::ConstantOp>(op->getLoc(),known.One);
        //op->replaceUsesOfWith(op->getResult(0), constVal.getResult());

        originalOpBuilder.setInsertionPoint(originOp);
        auto constVal=originalOpBuilder.create<circt::hw::ConstantOp>(originOp->getLoc(),known.One);
        for(auto userIt=originOp->user_begin();userIt!=originOp->user_end();++userIt){
          userIt->replaceUsesOfWith(originOp->getResult(0), constVal.getResult());
        }
        llvm::errs()<<"And1 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(LHSKnown.Zero|RHSKnown.One)){
        //op->replaceUsesOfWith(op->getResult(0), op->getOperand(0));
        for(auto userIt=originOp->user_begin();userIt!=originOp->user_end();++userIt){
          userIt->replaceUsesOfWith(originOp->getResult(0), originOp->getOperand(0));
        }
        llvm::errs()<<"And2 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(RHSKnown.Zero| LHSKnown.One)){
        //op->replaceUsesOfWith(op->getResult(0), op->getOperand(1));
        for(auto userIt=originOp->user_begin();userIt!=originOp->user_end();++userIt){
          userIt->replaceUsesOfWith(originOp->getResult(0), originOp->getOperand(1));
        }
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
        //opBuilder.setInsertionPoint(op);
        //auto constVal=opBuilder.create<hw::ConstantOp>(op->getLoc(),known.One);
        //op->replaceUsesOfWith(op->getResult(0), constVal.getResult());

        originalOpBuilder.setInsertionPoint(originOp);
        auto constVal=originalOpBuilder.create<circt::hw::ConstantOp>(originOp->getLoc(),known.One);
        for(auto userIt=originOp->user_begin();userIt!=originOp->user_end();++userIt){
          userIt->replaceUsesOfWith(originOp->getResult(0), constVal.getResult());
        }
        llvm::errs()<<"Or1 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(LHSKnown.One|RHSKnown.Zero)){
        //op->replaceUsesOfWith(op->getResult(0), op->getOperand(0));
        for(auto userIt=originOp->user_begin();userIt!=originOp->user_end();++userIt){
          userIt->replaceUsesOfWith(originOp->getResult(0), originOp->getOperand(0));
        }
        llvm::errs()<<"Or2 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(RHSKnown.One| LHSKnown.Zero)){
        //op->replaceUsesOfWith(op->getResult(0), op->getOperand(1));
        for(auto userIt=originOp->user_begin();userIt!=originOp->user_end();++userIt){
          userIt->replaceUsesOfWith(originOp->getResult(0), originOp->getOperand(1));
        }
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
        //opBuilder.setInsertionPoint(op);
        //auto constVal=opBuilder.create<hw::ConstantOp>(op->getLoc(),known.One);
        //op->replaceUsesOfWith(op->getResult(0), constVal.getResult());

        originalOpBuilder.setInsertionPoint(originOp);
        auto constVal=originalOpBuilder.create<circt::hw::ConstantOp>(originOp->getLoc(),known.One);
        for(auto userIt=originOp->user_begin();userIt!=originOp->user_end();++userIt){
          userIt->replaceUsesOfWith(originOp->getResult(0), constVal.getResult());
        }
        llvm::errs()<<"Xor1 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(RHSKnown.Zero)){
        //op->replaceUsesOfWith(op->getResult(0), op->getOperand(0));
        for(auto userIt=originOp->user_begin();userIt!=originOp->user_end();++userIt){
          userIt->replaceUsesOfWith(originOp->getResult(0), originOp->getOperand(0));
        }
        llvm::errs()<<"Xor2 triggered\n";
        return;
      }
      if(dbVal.isSubsetOf(LHSKnown.Zero)){
        //op->replaceUsesOfWith(op->getResult(0), op->getOperand(1));
        for(auto userIt=originOp->user_begin();userIt!=originOp->user_end();++userIt){
          userIt->replaceUsesOfWith(originOp->getResult(0), originOp->getOperand(1));
        }
        llvm::errs()<<"Xor3 triggered\n";
        return;
      }
    }
  }
}

static void combineEqualityICmpWithKnownBitsAndConstant(
    circt::comb::ICmpOp cmpOp, const llvm::KnownBits &bitAnalysis, const llvm::APInt &rhsCst,
    mlir::OpBuilder& originalOpBuilder, unordered_map<mlir::Operation*, mlir::Operation*>& copyToOrigin) {

  llvm::errs()<<"CMP triggered!\n";
  mlir::Operation* op=cmpOp.getOperation();
  mlir::Operation* originalOp=copyToOrigin[op];
  APInt bitsKnown = bitAnalysis.Zero | bitAnalysis.One;
  if ((bitsKnown & rhsCst) != bitAnalysis.One) {
    // If we discover a mismatch then we know an "eq" comparison is false
    // and a "ne" comparison is true!
    bool result = cmpOp.getPredicate() == circt::comb::ICmpPredicate::ne;
    //replaceOpWithNewOpAndCopyName<hw::ConstantOp>(rewriter, cmpOp,
//                                                  APInt(1, result));
    originalOpBuilder.setInsertionPoint(originalOp);
    auto constOp=originalOpBuilder.create<circt::hw::ConstantOp>(originalOp->getLoc(), llvm::APInt(1, result));
    originalOp->getResult(0).replaceAllUsesWith(constOp.getResult());
    return;
  }
  originalOpBuilder.setInsertionPoint(originalOp);

  // Check to see if we can prove the result entirely of the comparison (in
  // which we bail out early), otherwise build a list of values to concat and a
  // smaller constant to compare against.
  SmallVector<mlir::Value> newConcatOperands;
  auto newConstant = APInt::getZeroWidth();

  // Ok, some (maybe all) bits are known and some others may be unknown.
  // Extract out segments of the operand and compare against the
  // corresponding bits.
  unsigned knownMSB = bitsKnown.countLeadingOnes();

  mlir::Value operand = cmpOp.getLhs();

  while (knownMSB != bitsKnown.getBitWidth()) {
    if (knownMSB)
      bitsKnown = bitsKnown.trunc(bitsKnown.getBitWidth() - knownMSB);

    unsigned unknownBits = bitsKnown.countLeadingZeros();
    unsigned lowBit = bitsKnown.getBitWidth() - unknownBits;
    auto spanOperand = originalOpBuilder.createOrFold<circt::comb::ExtractOp>(
        originalOp->getLoc(), originalOp->getOperand(0), /*lowBit=*/lowBit,
        /*bitWidth=*/unknownBits);
    auto spanConstant = rhsCst.lshr(lowBit).trunc(unknownBits);

    newConcatOperands.push_back(spanOperand);
    if (newConstant.getBitWidth() != 0)
      newConstant = newConstant.concat(spanConstant);
    else
      newConstant = spanConstant;

    unsigned newWidth = bitsKnown.getBitWidth() - unknownBits;
    bitsKnown = bitsKnown.trunc(newWidth);
    knownMSB = bitsKnown.countLeadingOnes();
  }

  if (newConcatOperands.empty()) {
    bool result = cmpOp.getPredicate() == circt::comb::ICmpPredicate::eq;
    //replaceOpWithNewOpAndCopyName<hw::ConstantOp>(rewriter, cmpOp,
    //                                              APInt(1, result));
    originalOpBuilder.setInsertionPoint(originalOp);
    auto constOp=originalOpBuilder.create<circt::hw::ConstantOp>(originalOp->getLoc(), llvm::APInt(1, result));
    originalOp->getResult(0).replaceAllUsesWith(constOp.getResult());
    return;
  }



  mlir::Value concatResult =
      originalOpBuilder.createOrFold<circt::comb::ConcatOp>(originalOp->getLoc(), newConcatOperands);

  auto newConstantOp = originalOpBuilder.create<circt::hw::ConstantOp>(
      originalOp->getOperand(1).getLoc(), newConstant);

  //replaceOpWithNewOpAndCopyName<ICmpOp>(rewriter, cmpOp, cmpOp.getPredicate(),
//                                        concatResult, newConstantOp,
//                                        cmpOp.getTwoState());

  auto newIcmpOp=originalOpBuilder.create<circt::comb::ICmpOp>(originalOp->getLoc(), cmpOp.getPredicate(),
                                                               concatResult, newConstantOp,
                                                               cmpOp.getTwoState());
  originalOp->getResult(0).replaceAllUsesWith(newIcmpOp.getResult());
}

static void combBuiltInOpt(mlir::Operation* op, mlir::OpBuilder& builder, mlir::OpBuilder& originalOpBuilder
                    , MLIRDemandedBits<mlir::func::FuncOp,mlir::func::ReturnOp>& db
                    ,unordered_map<mlir::Operation*, mlir::Operation*>& copyToOrigin){
  if(llvm::isa<circt::comb::ExtractOp>(op)){
    //constant replacement
  }else if(llvm::isa<circt::comb::ICmpOp>(op)){
    auto icmpOp=llvm::dyn_cast<circt::comb::ICmpOp>(op);
    if(icmpOp.getPredicate() == circt::comb::ICmpPredicate::eq ||
        icmpOp.getPredicate() == circt::comb::ICmpPredicate::ne){
      mlir::Value lhs_val=icmpOp.getLhs();
      mlir::Value rhs_val=icmpOp.getRhs();
      if(mlir::Operation* lhsOp=lhs_val.getDefiningOp();lhsOp){
        unsigned lhsBitwidth=lhs_val.getType().getIntOrFloatBitWidth();
        auto lhsKnownBits= getKnownBits(lhs_val, lhsBitwidth);
        auto rhsKnownBits =  getKnownBits(rhs_val, lhsBitwidth);
        if (!lhsKnownBits.isUnknown()&&rhsKnownBits.isConstant()){
          combineEqualityICmpWithKnownBitsAndConstant(icmpOp, lhsKnownBits, rhsKnownBits.getConstant(),
                                                      originalOpBuilder, copyToOrigin);
          //call the rhs function
        }
      }
    }
  }
}

void runOnOperation(mlir::Operation* op, mlir::OpBuilder& builder, mlir::OpBuilder& originalOpBuilder
                    , MLIRDemandedBits<mlir::func::FuncOp,mlir::func::ReturnOp>& db
                    ,unordered_map<mlir::Operation*, mlir::Operation*>& copyToOrigin){

  instCombineSimplifyDB(op, builder, originalOpBuilder, db, copyToOrigin);
}

extern std::pair<long long,long long> analyzeModule(ModuleOp m,bool debug=false);

void runOnModule(mlir::ModuleOp module, mlir::MLIRContext& context,  unordered_map<mlir::Operation*, mlir::Operation*>& copyToOrigin){
  mlir::OpBuilder opBuilder(&context), originalOpBuilder(&context);
  mlir::Operation* func=&module.getBodyRegion().front().front();
  assert(llvm::isa<mlir::func::FuncOp>(func));
  MLIRDemandedBits<mlir::func::FuncOp,mlir::func::ReturnOp> dbVal(llvm::dyn_cast<mlir::func::FuncOp>(func));
  analyzeModule(module);
  module->walk([&opBuilder, &originalOpBuilder, &dbVal, &copyToOrigin](mlir::Operation* op){
    runOnOperation(op, opBuilder, originalOpBuilder, dbVal, copyToOrigin);
  });
}

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::EnableDebugBuffering = true;

  llvm::cl::ParseCommandLineOptions(argc, argv);

  DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::emitc::EmitCDialect>();
  registry.insert<mlir::vector::VectorDialect>();

  circt::registerAllDialects(registry);
  circt::registerAllPasses();

  mlir::func::registerInlinerExtension(registry);

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerViewOpGraphPass();
  mlir::registerSymbolDCEPass();
  MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();

  // Register test passes
  circt::test::registerAnalysisTestPasses();
  if (!isValidInputPath()) {
    llvm::errs() << "Invalid input file!\n";
    return 1;
  }

  string errorMessage;
  auto src_file = openInputFile(filename_src, &errorMessage);

  if (!src_file) {
    llvm::errs() << errorMessage << "\n";
    return 66;
  }
  llvm::SourceMgr src_sourceMgr;
  ParserConfig parserConfig(&context);
  src_sourceMgr.AddNewSourceBuffer(move(src_file), llvm::SMLoc());
  auto ir_before = parseSourceFile<ModuleOp>(src_sourceMgr, parserConfig);
  ModuleOp moduleOp = ir_before.release();


  std::vector<std::vector<mlir::Operation*>> data;
  data= extractFunctions(moduleOp);

  std::vector<mlir::ModuleOp> modules;
  unordered_map<mlir::Operation*, mlir::Operation*> copyToOrigin;
  for (size_t i = 0; i < data.size(); ++i) {
    modules.push_back(moveToModule(context, data[i], moduleOp.getLoc(), copyToOrigin));
    runOnModule(modules[i],context,copyToOrigin);
  }

  if (!output_folder.empty()) {
    llvm::errs() << "Start writing to files\n";

    auto destFolder = std::filesystem::path(std::string(output_folder));

    std::error_code ec;
    if (!std::filesystem::is_directory(destFolder)) {
      std::filesystem::create_directory(destFolder);
    }
    std::string outputFileName = destFolder.string();
    if (outputFileName.back() != '/') {
      outputFileName.push_back('/');
    }
    outputFileName +=  "output.mlir";
    llvm::raw_fd_ostream fout(outputFileName, ec);
    moduleOp.print(fout);
    fout.close();

    llvm::errs() << "Writing files done\n";
  }


  return 0;
}

std::vector<std::vector<mlir::Operation*>> extractFunctions(ModuleOp moduleOp){
  std::unordered_set<mlir::Operation *> visited;
  std::vector<mlir::Operation *> tmp;
  std::vector<std::vector<mlir::Operation *>> data;
  int combOpCnt = 0;

  for (auto bit = moduleOp.getRegion().begin();
       bit != moduleOp.getRegion().end(); ++bit) {
    if (!bit->empty()) {
      for (auto iit = bit->begin(); iit != bit->end(); ++iit) {
        if (llvm::isa<circt::hw::HWModuleOp>(*iit) ||
            llvm::isa<circt::arc::DefineOp>(*iit)) {
          iit->walk([&visited, &data, &tmp, &combOpCnt](mlir::Operation *op) {
            if (isComb(op)) {
              ++combOpCnt;
              if (visited.find(op) == visited.end()) {
                visit(op, tmp, visited);
              }
              // We also consider DAGs with 1 operations
              if (tmp.size()) {
                data.push_back(tmp);
              }
              tmp.clear();
              }
          });
        }
      }
    }
  }
  llvm::errs() << "Extracted functions: " << data.size() << "\n";
  llvm::errs() << "Number of comb operations: " << combOpCnt << "\n";
  return data;
}


bool isValidInputPath() {
  bool result = filesystem::status(string(filename_src)).type() ==
                filesystem::file_type::regular;
  if (result) {
    inputPath = filesystem::path(string(filename_src));
  }
  return result;
}

bool isComb(mlir::Operation *op) {
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

void visit(mlir::Operation *op, std::vector<mlir::Operation *> &tmp,
           std::unordered_set<mlir::Operation *> &visited) {
  if (llvm::isa<circt::hw::ConstantOp>(op)) {
    tmp.push_back(op);
    return;
  }
  if (!isComb(op)) {
    return;
  }
  if (visited.find(op) == visited.end()) {
    visited.insert(op);
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::Operation *producer = operand.getDefiningOp()) {
        visit(producer, tmp, visited);
      }
    }
    tmp.push_back(op);
    mlir::OpResult result = op->getResult(0);
    for (mlir::Operation *userOp : result.getUsers()) {
      visit(userOp, tmp, visited);
    }
  }
}
