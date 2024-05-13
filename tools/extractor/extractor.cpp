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

llvm::cl::opt<int>
    depth("depth", llvm::cl::desc("Depth of search when performing slicing"),
          llvm::cl::Hidden, llvm::cl::init(5), llvm::cl::cat(MLIR_MUTATE_CAT));

// Defined in the test directory, no public header.
namespace circt {
namespace test {
void registerAnalysisTestPasses();
} // namespace test
} // namespace circt

filesystem::path inputPath, outputPath;
bool isValidInputPath(), isComb(mlir::Operation *op);
void visit(mlir::Operation *op, std::vector<mlir::Operation *> &tmp,
           std::unordered_set<mlir::Operation *> &visited, int depth);

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
                              mlir::Location loc) {

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

std::string funcToString(mlir::func::FuncOp func) {
  std::string result;
  llvm::raw_string_ostream os(result);
  func.print(os);
  return os.str();
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

  std::unordered_set<mlir::Operation *> visited;
  std::vector<mlir::Operation *> tmp;
  std::vector<std::vector<mlir::Operation *>> data;
  int combOpCnt = 0;

  for (auto bit = ir_before->getRegion().begin();
       bit != ir_before->getRegion().end(); ++bit) {
    if (!bit->empty()) {
      for (auto iit = bit->begin(); iit != bit->end(); ++iit) {
        if (llvm::isa<circt::hw::HWModuleOp>(*iit) ||
            llvm::isa<circt::arc::DefineOp>(*iit)) {
          int argDepth = depth;
          iit->walk([&visited, &data, &tmp, &combOpCnt,
                     &argDepth](mlir::Operation *op) {
            if (isComb(op)) {
              ++combOpCnt;
              if (visited.find(op) == visited.end()) {
                visit(op, tmp, visited, argDepth);
              }
              // We also consider DAGs with 1 operations
              if (tmp.size()) {
                data.push_back(tmp);
              }
              tmp.clear();
              // We clean visited to don't filter out operations
              visited.clear();
            }
          });
        }
      }
    }
  }
  llvm::errs() << "Sliced functions: " << data.size() << "\n";
  llvm::errs() << "Number of comb operations: " << combOpCnt << "\n";
  std::vector<pair<int, std::vector<mlir::Operation *>>> v;
  for (const auto &ele : data) {
    v.push_back({ele.size(), ele});
  }
  std::sort(v.begin(), v.end(),
            [](auto &a, auto &b) { return a.first > b.first; });
  llvm::errs() << "Max size: " << v[0].first << "\n";
  llvm::errs() << "Min size: " << v.back().first << "\n";
  /*
  llvm::errs() << "=======\n";
  for (const auto &x : v.back().second) {
    x->dump();
  }
  llvm::errs() << "=======\n";
  llvm::errs() << "Median size: " << v[v.size() / 2].first << "\n";
  llvm::errs() << "=======\n";
  for (const auto &x : v[v.size() / 2].second) {
    x->dump();
  }
  int i = 0;
  for (; i < v.size() && v[i].first > 2; ++i) {
  }
  llvm::errs() << "======\n";
  llvm::errs() << "Size >= 2:" << i << "\n";*/
  std::vector<mlir::func::FuncOp> funcs;
  for (size_t i = 0; i < v.size(); ++i) {
    funcs.push_back(moveToFunc(context, v[i].second, ir_before->getLoc()));
  }
  std::unordered_map<std::string, std::pair<mlir::func::FuncOp, int>> filter;
  for (int i = 0; i < funcs.size(); ++i) {
    std::string str = funcToString(funcs[i]);
    if (filter.find(str) == filter.end()) {
      filter.insert(std::make_pair(str, std::make_pair(funcs[i], 0)));
    }
    filter[str].second++;
  }
  std::vector<std::pair<mlir::func::FuncOp, int>> result;
  for (const auto &p : filter) {
    result.push_back(p.second);
  }
  sort(result.begin(), result.end(),
       [](std::pair<mlir::func::FuncOp, int> &a,
          std::pair<mlir::func::FuncOp, int> &b) {
         return a.second > b.second;
       });

  /*
  funcs.back().dump();
  llvm::errs() << "======\n";
  funcs[funcs.size()>>1].dump();
  llvm::errs() << "======\n";
  if(i<v.size()){
    funcs[i].dump();
    llvm::errs() << "======\n";
  }else{
    llvm::errs()<<"Cannot find a DAG with size larger than 2\n";
  }
  unordered_map<std::string,unordered_set<int>> um;
  unordered_map<std::string, int> cnt;
  unordered_map<int,int> pred;
  for(size_t i=0;i<v.size();++i){
    for(mlir::Operation* op:v[i].second){
      std::string str=op->getName().getStringRef().str();
      if(um.find(str)==um.end()){
        um.insert({str,unordered_set<int>()});
        cnt.insert({str,0});
      }
      if(llvm::isa<circt::comb::ICmpOp>(op)){
        mlir::Attribute attr=op->getAttr("predicate");
        int pred_val = attr.cast<mlir::IntegerAttr>().getValue().getZExtValue();
        if(pred.find(pred_val)==pred.end()){
          pred.insert({pred_val,0});
        }
        pred[pred_val]++;
      }
      um[str].insert(op->getNumOperands());
      cnt[str]++;
    }
  }
  */
  /*
  llvm::errs()<<"all ops: "<<um.size()<<"\n";
  int sum=0;
  for(const auto& x:um){
    sum+=cnt[x.first];
  }
  llvm::errs()<<"All count: "<<sum<<"\n";
  for(const auto& x:um){
    llvm::errs()<<x.first<<" ";
    llvm::errs()<<": "<<cnt[x.first]<<"\n";
  }*/
  // funcs.front().dump();
  llvm::errs() << "Final result size: " << result.size();
  llvm::errs() << "\n";

  if (!output_folder.empty()) {
    llvm::errs() << "Start writing to files\n";

    auto destFolder = std::filesystem::path(std::string(output_folder));

    for (size_t i = 0; i < result.size(); ++i) {
      std::error_code ec;
      if (!std::filesystem::is_directory(destFolder)) {
        std::filesystem::create_directory(destFolder);
      }
      std::string outputFileName = destFolder.string();
      if (outputFileName.back() != '/') {
        outputFileName.push_back('/');
      }
      outputFileName += to_string(i) + ".mlir";
      llvm::raw_fd_ostream fout(outputFileName, ec);
      fout << "// " << result[i].second << "\n";
      result[i].first.print(fout);
      fout.close();
      llvm::errs() << "file wrote to " << outputFileName << "\n";
    }

    llvm::errs() << "Writing files done\n";
  }

  int x;
  while (cin >> x && x != -1) {
    if (x > result.size()) {
      llvm::errs() << "out of range\n";
    } else {
      result[x].first.dump();
    }
  }
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

void visit(mlir::Operation *op, std::vector<mlir::Operation *> &tmp,
           std::unordered_set<mlir::Operation *> &visited, int depth) {
  if (llvm::isa<circt::hw::ConstantOp>(op)) {
    tmp.push_back(op);
    return;
  }
  if (depth == 0) {
    return;
  }
  if (!isComb(op)) {
    return;
  }
  if (visited.find(op) == visited.end()) {
    visited.insert(op);
    for (Value operand : op->getOperands()) {
      if (Operation *producer = operand.getDefiningOp()) {
        visit(producer, tmp, visited, depth - 1);
      }
    }
    tmp.push_back(op);
    /*mlir::OpResult result = op->getResult(0);
    for (Operation *userOp : result.getUsers()) {
      visit(userOp, tmp, visited);
    }*/
  }
}
