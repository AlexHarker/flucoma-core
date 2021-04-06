#pragma once

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/MLP.hpp"
#include "algorithms/SGD.hpp"
#include <string>

namespace fluid {
namespace client {

class MLPRegressorClient : public FluidBaseClient,
                           AudioIn,
                           ControlOut,
                           ModelObject,
                           public DataClient<algorithm::MLP> {

  enum {
    kHidden,
    kActivation,
    kOutputActivation,
    kInputTap,
    kOutputTap,
    kIter,
    kRate,
    kMomentum,
    kBatchSize,
    kVal,
    kInputBuffer,
    kOutputBuffer
  };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;

  static constexpr std::initializer_list<index> HiddenLayerDefaults = {3, 3};

  FLUID_DECLARE_PARAMS(
      LongArrayParam("hidden", "Hidden Layer Sizes", HiddenLayerDefaults),
      EnumParam("activation", "Activation Function", 2, "Identity", "Sigmoid",
                "ReLU", "Tanh"),
      EnumParam("outputActivation", "Output Activation Function", 0, "Identity", "Sigmoid",
                          "ReLU", "Tanh"),
      LongParam("tapIn", "Input Tap Index", 0, Min(0)),
      LongParam("tapOut", "Output Tap Index", -1, Min(-1)),
      LongParam("maxIter", "Maximum Number of Iterations", 1000, Min(1)),
      FloatParam("learnRate", "Learning Rate", 0.01, Min(0.0), Max(1.0)),
      FloatParam("momentum", "Momentum", 0.9, Min(0.0), Max(0.99)),
      LongParam("batchSize", "Batch Size", 50, Min(1)),
      FloatParam("validation", "Validation Amount", 0.2, Min(0), Max(0.9)),
      BufferParam("inputPointBuffer", "Input Point Buffer"),
      BufferParam("predictionBuffer", "Prediction Buffer"));

  MLPRegressorClient(ParamSetViewType &p) : mParams(p) {
    audioChannelsIn(1);
    controlChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &) {
    if (!mAlgorithm.trained())
      return;
    index inputTap = get<kInputTap>();
    index outputTap = get<kOutputTap>();
    if(inputTap >= mAlgorithm.size() - 1) return;
    if(outputTap >= mAlgorithm.size()) return;
    if(outputTap == 0) return;
    if(outputTap == -1) outputTap = mAlgorithm.size();

    index inputSize = mAlgorithm.inputSize(inputTap);
    index outputSize = mAlgorithm.outputSize(outputTap);

    InOutBuffersCheck bufCheck(inputSize);
    if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                              get<kOutputBuffer>().get()))
      return;
    auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
    if(outBuf.samps(0).size() != mAlgorithm.outputSize(outputTap)) return;

    RealVector src(inputSize);
    RealVector dest(outputSize);
    src =
        BufferAdaptor::ReadAccess(get<kInputBuffer>().get()).samps(0, inputSize, 0);
    mTrigger.process(input, output, [&]() {
      mAlgorithm.processFrame(src, dest, inputTap, outputTap);
      outBuf.samps(0) = dest;
    });
  }

  MessageResult<double> fit(DataSetClientRef source, DataSetClientRef target) {
    auto sourceClientPtr = source.get().lock();
    if (!sourceClientPtr)
      return Error<double>(NoDataSet);
    auto sourceDataSet = sourceClientPtr->getDataSet();
    if (sourceDataSet.size() == 0)
      return Error<double>(EmptyDataSet);
    if(mAlgorithm.initialized() && sourceDataSet.dims() != mAlgorithm.dims())
        return Error<double>(DimensionsDontMatch);

    auto targetClientPtr = target.get().lock();
    if (!targetClientPtr)
      return Error<double>(NoDataSet);
    auto targetDataSet = targetClientPtr->getDataSet();
    if (targetDataSet.size() == 0)
      return Error<double>(EmptyDataSet);
    if (sourceDataSet.size() != targetDataSet.size())
      return Error<double>(SizesDontMatch);
    if (!mAlgorithm.initialized() || mTracker.changed(get<kHidden>(), get<kActivation>())) {
      index outputAct = get<kOutputActivation>() == -1?get<kActivation>():get<kOutputActivation>();
      mAlgorithm.init(sourceDataSet.pointSize(), targetDataSet.pointSize(),
                      get<kHidden>(), get<kActivation>(), outputAct);
    }

    mAlgorithm.setTrained(false);
    DataSet result(1);
    auto data = sourceDataSet.getData();
    auto tgt = targetDataSet.getData();
    algorithm::SGD sgd;
    double error =
        sgd.train(mAlgorithm, data, tgt, get<kIter>(), get<kBatchSize>(),
                  get<kRate>(), get<kMomentum>(), get<kVal>());
    return error;
  }

  MessageResult<void> predict(DataSetClientRef srcClient,
                              DataSetClientRef destClient) {
    index inputTap = get<kInputTap>();
    index outputTap = get<kOutputTap>();
    if(inputTap >= mAlgorithm.size()) return Error("Input tap too large");
    if(outputTap > mAlgorithm.size()) return Error("Ouput tap too large");
    if(outputTap == 0) return Error("Ouput tap cannot be 0");
    if(outputTap == -1) outputTap = mAlgorithm.size();
    index inputSize = mAlgorithm.inputSize(inputTap);
    index outputSize = mAlgorithm.outputSize(outputTap);
    auto srcPtr = srcClient.get().lock();
    auto destPtr = destClient.get().lock();

    if (!srcPtr || !destPtr)
      return Error(NoDataSet);
    auto srcDataSet = srcPtr->getDataSet();
    if (srcDataSet.size() == 0)
      return Error(EmptyDataSet);
    if (!mAlgorithm.trained())
      return Error(NoDataFitted);
    if (srcDataSet.dims() != inputSize)
      return Error(WrongPointSize);

    StringVector ids{srcDataSet.getIds()};
    RealMatrix output(srcDataSet.size(), outputSize);
    mAlgorithm.process(srcDataSet.getData(), output, inputTap, outputTap);
    FluidDataSet<string, double, 1> result(ids, output);
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> predictPoint(BufferPtr in, BufferPtr out) {
    index inputTap = get<kInputTap>();
    index outputTap = get<kOutputTap>();
    if(inputTap >= mAlgorithm.size()) return Error("Input tap too large");
    if(outputTap > mAlgorithm.size()) return Error("Ouput tap too large");
    if(outputTap == 0) return Error("Ouput tap should be > 0 or -1");
    if(outputTap == -1) outputTap = mAlgorithm.size();
    index inputSize = mAlgorithm.inputSize(inputTap);
    index outputSize = mAlgorithm.outputSize(outputTap);

    if (!in || !out)
      return Error(NoBuffer);
    BufferAdaptor::Access inBuf(in.get());
    BufferAdaptor::Access outBuf(out.get());
    if (!inBuf.exists())
      return Error(InvalidBuffer);
    if (!outBuf.exists())
      return Error(InvalidBuffer);
    if (inBuf.numFrames() != inputSize)
      return Error(WrongPointSize);
    if (!mAlgorithm.trained())
      return Error(NoDataFitted);

    Result resizeResult =
        outBuf.resize(outputSize, 1, inBuf.sampleRate());
    if (!resizeResult.ok())
      return Error(BufferAlloc);
    RealVector src(inputSize);
    RealVector dest(outputSize);
    src = inBuf.samps(0,inputSize, 0);
    mAlgorithm.processFrame(src, dest, inputTap, outputTap);
    outBuf.samps(0) = dest;
    return OK();
  }

  index latency() { return 0; }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &MLPRegressorClient::fit),
                         makeMessage("predict", &MLPRegressorClient::predict),
                         makeMessage("predictPoint",
                                     &MLPRegressorClient::predictPoint),
                         makeMessage("clear", &MLPRegressorClient::clear),
                         makeMessage("cols", &MLPRegressorClient::dims),
                         makeMessage("size", &MLPRegressorClient::size),
                         makeMessage("load", &MLPRegressorClient::load),
                         makeMessage("dump", &MLPRegressorClient::dump),
                         makeMessage("write", &MLPRegressorClient::write),
                         makeMessage("read", &MLPRegressorClient::read));

private:
  FluidInputTrigger mTrigger;
  ParameterTrackChanges<IndexVector, index> mTracker;
};
constexpr std::initializer_list<index> MLPRegressorClient::HiddenLayerDefaults;
using RTMLPRegressorClient = ClientWrapper<MLPRegressorClient>;
} // namespace client
} // namespace fluid
