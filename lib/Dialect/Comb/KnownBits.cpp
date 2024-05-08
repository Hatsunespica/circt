#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include <optional>
#include "KnownBits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/Version.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

#define DEBUG_TYPE "int-range-analysis"

using llvm::KnownBits;
using namespace mlir;
using namespace mlir::dataflow;

KnownBitsRange KnownBitsRange::getMaxRange(Value value) {
    unsigned width =  ConstantIntRanges::getStorageBitwidth(value.getType());
    if (width == 0)
        return {};
    return KnownBitsRange{KnownBits(width)};
}

void KnownBitsRangeLattice::onUpdate(DataFlowSolver *solver) const {
    Lattice::onUpdate(solver);

    // If the integer range can be narrowed to a constant, update the constant
    // value of the SSA value.
    auto value = point.get<Value>();
    auto *cv = solver->getOrCreateState<Lattice<ConstantValue>>(value);
    if (!getValue().getValue().isConstant())
        return solver->propagateIfChanged(
                cv, cv->join(ConstantValue::getUnknownConstant()));

    APInt constant = getValue().getValue().getConstant();
    Dialect *dialect;
    if (auto *parent = value.getDefiningOp())
        dialect = parent->getDialect();
    else
        dialect = value.getParentBlock()->getParentOp()->getDialect();
    solver->propagateIfChanged(
            cv, cv->join(ConstantValue(IntegerAttr::get(value.getType(), constant),
                                       dialect)));
}

#include "transfer.h"

std::tuple<APInt,APInt> kbToTuple(KnownBits kb){
    return std::make_tuple(kb.Zero, kb.One);
}

KnownBits tupleToKb(std::tuple<APInt, APInt> t){
  APInt zero=std::get<0>(t);
  APInt one=std::get<1>(t);
  KnownBits res(zero.getBitWidth());
  res.Zero=zero;res.One=one;
  return res;
}

void KnownBitsRangeAnalysis::visitOperation(
        Operation *op, ArrayRef<const KnownBitsRangeLattice *> operands,
        ArrayRef<KnownBitsRangeLattice *> results) {
    // If the lattice on any operand is unitialized, bail out.
    if (llvm::any_of(operands, [](const KnownBitsRangeLattice *lattice) {
        return lattice->getValue().isUninitialized();
    })) {
        return;
    }

    // Ignore non-integer outputs - return early if the op has no scalar
    // integer results
    SmallVector<std::tuple<APInt, APInt>> argRanges(
            llvm::map_range(operands, [](const KnownBitsRangeLattice *val) {
                KnownBits knownBits= val->getValue().getValue();
                return std::make_tuple(knownBits.Zero, knownBits.One);
            }));
    bool hasIntegerResult = false;
    for (auto it : llvm::zip(results, op->getResults())) {
        Value value = std::get<1>(it);
        if (value.getType().isIntOrIndex()) {
            hasIntegerResult = true;
            //unsigned width=IndexType::kInternalStorageBitWidth;
            //if(!value.getType().isIndex()){
            //    width=value.getType().getIntOrFloatBitWidth();
            //}
            std::optional<std::tuple<APInt, APInt>> res;
            //self added
            if (auto castedOp=dyn_cast<circt::hw::ConstantOp>(op);castedOp){
                APInt apInt=castedOp.getValue();
                llvm::KnownBits knownBits= llvm::KnownBits::makeConstant(apInt);
                res= std::make_tuple(knownBits.Zero, knownBits.One);
            }else{
                bool useLLVM=false;
                if(auto castedOp = dyn_cast<circt::comb::ExtractOp>(op);castedOp){
                  int lowBit=castedOp.getLowBit();
                  auto resType=castedOp.getResult().getType().dyn_cast_or_null<mlir::IntegerType>();
                  int resLen=resType.getWidth();
                  APInt zero,one;
                  zero=APInt(32,~resLen), one=APInt( 32,resLen);
                  argRanges.push_back(std::make_tuple(zero,one));
                  zero=APInt(32,~lowBit), one=APInt(32,lowBit);
                  argRanges.push_back(std::make_tuple(zero,one));

                }

              if(auto casedOp = dyn_cast_or_null<circt::comb::ICmpOp>(op);casedOp){
                  IntegerAttr attr=op->getAttr("predicate").cast<IntegerAttr>();
                  int pred=attr.getValue().getZExtValue();
                  if(pred==0){
                    if(useLLVM){
                      std::optional<bool> calc_res=KnownBits::eq(tupleToKb(argRanges[0]), tupleToKb(argRanges[1]));
                      KnownBits kb(1);
                      if(calc_res.has_value()){
                        kb=KnownBits::makeConstant(APInt(1,*calc_res));
                      }
                      res= kbToTuple(kb);
                    }else{
                      res= EQImpl(argRanges[0], argRanges[1]);
                    }
                  }else if(pred==1) {
                    if(useLLVM){
                      std::optional<bool> calc_res=KnownBits::ne(tupleToKb(argRanges[0]), tupleToKb(argRanges[1]));
                      KnownBits kb(1);
                      if(calc_res.has_value()){
                        kb=KnownBits::makeConstant(APInt(1,*calc_res));
                      }
                      res= kbToTuple(kb);
                    }else{
                      res= NEImpl(argRanges[0], argRanges[1]);
                    }

                  }else{
                    APInt zero(1,0);
                    res=std::make_tuple(zero,zero);
                  }
                }else if(auto casedOp= dyn_cast_or_null<circt::comb::ReplicateOp>(op)){
                  IntegerType inputTy=op->getOperand(0).getType().cast<IntegerType>();
                  IntegerType outputTy=op->getResult(0).getType().cast<IntegerType>();
                  int inputWidth=inputTy.getWidth(),outputWidth=outputTy.getWidth();
                  APInt arg2(32, outputWidth/inputWidth);
                  APInt arg2Zero=arg2;
                  arg2Zero.flipAllBits();
                  argRanges.push_back(std::make_tuple(arg2Zero, arg2));
                  res= REPEATImpl(argRanges[0], argRanges[1]);
                }else{
                  if(useLLVM&&llvm::isa<circt::comb::ShlOp>(op)){
                    res= kbToTuple(KnownBits::shl(tupleToKb(argRanges[0]), tupleToKb(argRanges[1])));
                  }/*else if(useLLVM&&llvm::isa<circt::comb::ShrUOp>(op)){
                    res= kbToTuple(KnownBits::lshr(tupleToKb(argRanges[0]), tupleToKb(argRanges[1])));
                  }else if(useLLVM&&llvm::isa<circt::comb::ShrSOp>(op)){
                    res= kbToTuple(KnownBits::ashr(tupleToKb(argRanges[0]), tupleToKb(argRanges[1])));
                  }*/else{
                    res= naiveDispatcher(op, argRanges);
                  }
                }
            }

            KnownBitsRange knownBitsRange;
            if(!res.has_value()){
                knownBitsRange=KnownBitsRange::getMaxRange(value);
            }else{
                KnownBits result;
                result.Zero=std::get<0>(*res);
                result.One=std::get<1>(*res);
                knownBitsRange=KnownBitsRange(std::optional<KnownBits>(result));
            }
            //KnownBits knownBits(width);
            //knownBits.setAllOnes();
            //KnownBitsRange knownBitsRange=KnownBitsRange(std::optional<KnownBits>(knownBits));
            ChangeResult changed=results[0]->join(knownBitsRange);
            KnownBitsRangeLattice *lattice = std::get<0>(it);
            propagateIfChanged(lattice, changed);
        } else {
            KnownBitsRangeLattice *lattice = std::get<0>(it);
            propagateIfChanged(lattice,
                               lattice->join(KnownBitsRange::getMaxRange(value)));
        }
    }
    if (!hasIntegerResult)
        return;
    /*llvm::errs()<<"Visit operation\n";
    op->dump();
    llvm::errs()<<llvm::isa<mlir::arith::ConstantOp>(op)<<" "<<(dyn_cast<mlir::arith::ConstantOp>(op)!=nullptr)<<"\n";
    llvm::errs()<<operands.size()<<' '<<results.size()<<"\n";*/
    //setAllToEntryStates(results);
    /*
    auto inferrable = dyn_cast<InferIntRangeInterface>(op);
    if (!inferrable)
        return setAllToEntryStates(results);

    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
    SmallVector<KnownBitsRange> argRanges(
            llvm::map_range(operands, [](const KnownBitsRange *val) {
                return val->getValue().getValue();
            }));

    auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
        auto result = dyn_cast<OpResult>(v);
        if (!result)
            return;
        assert(llvm::is_contained(op->getResults(), result));

        LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
        KnownBitsRangeLattice *lattice = results[result.getResultNumber()];
        KnownBitsRange oldRange = lattice->getValue();

        ChangeResult changed = lattice->join(KnownBitsRange{attrs});

        // Catch loop results with loop variant bounds and conservatively make
        // them [-inf, inf] so we don't circle around infinitely often (because
        // the dataflow analysis in MLIR doesn't attempt to work out trip counts
        // and often can't).
        bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
            return op->hasTrait<OpTrait::IsTerminator>();
        });
        if (isYieldedResult && !oldRange.isUninitialized() &&
            !(lattice->getValue() == oldRange)) {
            LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
            changed |= lattice->join(IntegerValueRange::getMaxRange(v));
        }
        propagateIfChanged(lattice, changed);
    };

    inferrable.inferResultRanges(argRanges, joinCallback);*/
}

void KnownBitsRangeAnalysis::visitNonControlFlowArguments(
        Operation *op, const RegionSuccessor &successor,
        ArrayRef<KnownBitsRangeLattice *> argLattices, unsigned firstIndex) {
    //llvm::errs()<<"Current op\n";
    //op->dump();
    /*if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
        // If the lattice on any operand is unitialized, bail out.
        if (llvm::any_of(op->getOperands(), [&](Value value) {
            return getLatticeElementFor(op, value)->getValue().isUninitialized();
        }))
            return;
        SmallVector<ConstantIntRanges> argRanges(
                llvm::map_range(op->getOperands(), [&](Value value) {
                    return getLatticeElementFor(op, value)->getValue().getValue();
                }));

        auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
            auto arg = dyn_cast<BlockArgument>(v);
            if (!arg)
                return;
            if (!llvm::is_contained(successor.getSuccessor()->getArguments(), arg))
                return;

            LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
            IntegerValueRangeLattice *lattice = argLattices[arg.getArgNumber()];
            IntegerValueRange oldRange = lattice->getValue();

            ChangeResult changed = lattice->join(IntegerValueRange{attrs});

            // Catch loop results with loop variant bounds and conservatively make
            // them [-inf, inf] so we don't circle around infinitely often (because
            // the dataflow analysis in MLIR doesn't attempt to work out trip counts
            // and often can't).
            bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
                return op->hasTrait<OpTrait::IsTerminator>();
            });
            if (isYieldedValue && !oldRange.isUninitialized() &&
                !(lattice->getValue() == oldRange)) {
                LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
                changed |= lattice->join(IntegerValueRange::getMaxRange(v));
            }
            propagateIfChanged(lattice, changed);
        };

        inferrable.inferResultRanges(argRanges, joinCallback);
        return;
    }*/

    return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
            op, successor, argLattices, firstIndex);
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


static long long getSize(const llvm::KnownBits& kb){
    return kb.Zero.getBitWidth();
}

static long long getUnknownSize(const llvm::KnownBits& kb){
    unsigned sz=kb.Zero.getBitWidth(),result=0;
    for(unsigned i=0;i<sz;++i){
        if(!kb.Zero[i]&&!kb.One[i]){
            ++result;
        }
    }
    return result;
}

static std::string toString(const KnownBits& kb){
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

std::pair<long long,long long> analyzeModule(ModuleOp moduleOp, bool debug){
    MLIRContext& context=*moduleOp.getContext();
    long long totalSize=0,totalUnknown=0;
    for(auto& op:moduleOp.getBodyRegion().front().getOperations()){
        if(auto funcOp=llvm::dyn_cast<func::FuncOp>(op);!funcOp.isDeclaration()){

            DataFlowSolver solver;
            //dataflow::IntegerRangeAnalysis* analysis=solver.load<dataflow::IntegerRangeAnalysis>();
            KnownBitsRangeAnalysis* knownBitsRangeAnalysis=solver.load<KnownBitsRangeAnalysis>();
            //dataflow::SparseConstantPropagation* analysis=solver.load<dataflow::SparseConstantPropagation>();
            LogicalResult result=solver.initializeAndRun(funcOp);

            //auto tmp=solver.lookupState<dataflow::Executable>(ProgramPoint(&funcOp.getBody().front()));
            //((dataflow::Executable*)tmp)->setToLive();
            funcOp->walk([&](Operation* op){
              auto tmp=solver.lookupState<dataflow::Executable>(ProgramPoint(op->getBlock()));
              if(tmp){
                ((dataflow::Executable*)tmp)->setToLive();
              }
            });
            funcOp->walk([&](Operation* op){
              auto res=knownBitsRangeAnalysis->visit(ProgramPoint(op));
              //if(res.failed()){
              //    llvm::errs()<<"Error\n";
              //}
              //auto res=analysis->visit(ProgramPoint(op));
              if(res.failed()){
                llvm::errs()<<"Error\n";
              }
            });
            if(result.succeeded()){
                /*
                llvm::errs()<<"Success "<<solver.analysisStates.size()<<"\n";
                auto opit=funcOp->getBlock()->getOperations().begin();
                for(auto it=solver.analysisStates.begin();it!=solver.analysisStates.end();++it){
                    it->first.first.print(llvm::errs());
                    llvm::errs()<<"\n";
                    //opit->print(llvm::errs());
                    //llvm::errs()<<"\n";
                    llvm::errs()<<"\n";
                    it->second->print(llvm::errs());
                    llvm::errs()<<"\n";
                    llvm::errs()<<(it->first.second==TypeID::get<dataflow::IntegerValueRangeLattice>());
                    llvm::errs()<<"\n";
                }*/
                //llvm::errs()<<"Check args\n";
                for(auto it=funcOp.args_begin();it!=funcOp.args_end();++it){
                  ProgramPoint point(*it);
                  //point.print(llvm::errs());
                  //llvm::errs()<<"\n";
                  //auto res=solver.lookupState<dataflow::IntegerValueRangeLattice>(point);
                  auto res=solver.lookupState<KnownBitsRangeLattice>(point);
                  if(res!=nullptr){
                    //llvm::errs()<<"Known Bits: ";
                    //res->print(llvm::errs());
                    //llvm::errs()<<"\n";
                  }else {
                    llvm::errs() << res << "\n";
                  }

                }

                funcOp->walk([&](Operation* op){
                  if(op->getNumResults()==0){
                    return;
                  }
                  ProgramPoint point(op->getResult(0));
                  //auto intRes=solver.lookupState<dataflow::IntegerValueRangeLattice>(point);
                  auto res=solver.lookupState<KnownBitsRangeLattice>(point);
                  if(debug){
                    point.print(llvm::errs());
                    llvm::errs()<<"\n";
                  }
                  if(res!=nullptr){
                    std::string resStr=toString(res->getValue().getValue());
                    Twine tmpTwine(resStr);
                    StringAttr kb=StringAttr::get(&context, tmpTwine);
                    op->setAttr("kb", kb);
                    if(debug) {
                      llvm::errs()<<"Known Bits: ";
                      res->print(llvm::errs());
                      llvm::errs()<<"\n";
                    }

                    if(isComb(op)){
                      totalSize+= getSize(res->getValue().getValue());
                      totalUnknown+= getUnknownSize(res->getValue().getValue());
                    }

                  }else{
                    llvm::errs()<<res<<"\n";
                  }
                  //intRes->dump();
                  //llvm::errs()<<"\n";
                  //auto it = solver.analysisStates.find({ProgramPoint(op), TypeID::get<dataflow::IntegerValueRangeLattice>()});

                });
                //llvm::errs()<<totalSize<<' '<<totalUnknown<<' '<<filename_src<<"\n";
            }else{
                llvm::errs()<<"obtaining result failed\n";
            }
        }
    }
    return {totalSize, totalUnknown};
}