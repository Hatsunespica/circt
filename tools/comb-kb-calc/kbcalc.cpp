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
#include "llvm/Support/KnownBits.h"
#include <algorithm>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <vector>
#include <iostream>

using namespace std;
using namespace mlir;

llvm::cl::OptionCategory MlirTvCategory("mlir-tv options", "");
llvm::cl::OptionCategory MLIR_MUTATE_CAT("mlir-mutate-tv options", "");

llvm::cl::opt<string> filename_src(llvm::cl::Positional,
                                   llvm::cl::desc("first-mlir-file"),
                                   llvm::cl::Required,
                                   llvm::cl::value_desc("filename"),
                                   llvm::cl::cat(MLIR_MUTATE_CAT));
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

long long getSize(const llvm::KnownBits& kb){
  return kb.Zero.getBitWidth();
}

long long getUnknownSize(const llvm::KnownBits& kb){
  unsigned sz=kb.Zero.getBitWidth(),result=0;
  for(unsigned i=0;i<sz;++i){
    if(!kb.Zero[i]&&!kb.One[i]){
      ++result;
    }
  }
  return result;
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

  std::vector<llvm::KnownBits> kbs;

  for (auto bit = ir_before->getRegion().begin();
       bit != ir_before->getRegion().end(); ++bit) {
    if (!bit->empty()) {
      for (auto iit = bit->begin(); iit != bit->end(); ++iit) {
        if (llvm::isa<func::FuncOp>(*iit)) {
          iit->walk([&](mlir::Operation* op){
            if(isComb(op)){
              kbs.push_back(circt::comb::computeKnownBits(op->getResult(0)));
            }
          });
        }
      }
    }
  }
  long long totalSize=0,totalUnknown=0;
  for(const auto& kb:kbs){
    totalSize+= getSize(kb);
    totalUnknown+= getUnknownSize(kb);
  }
  llvm::errs()<<totalSize<<' '<<totalUnknown<<' '<<filename_src<<"\n";
  return 0;
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

