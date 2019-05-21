#pragma once

#include "../../algorithms/public/Loudness.hpp"
#include "../../data/TensorTypes.hpp"
#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../nrt/FluidNRTClientWrapper.hpp"
#include "../rt/BufferedProcess.hpp"
#include <tuple>

namespace fluid {
namespace client {

using algorithm::Loudness;

enum LoudnessParamIndex {
  kKWeighting,
  kTruePeak,
  kWindowSize,
  kHopSize,
  kMaxWindowSize
};

auto constexpr LoudnessParams = defineParameters(
    EnumParam("kWeighting", "Apply K-Weighting", 1, "Off","On"),
    EnumParam("truePeak", "Compute True Peak", 1, "Off","On"),
    LongParam("winSize", "Window Size", 1024, UpperLimit<kMaxWindowSize>()),
    LongParam("hopSize", "Hop Size", 512, Min(1)),
    LongParam<Fixed<true>>("maxWinSize", "Max Window Size",
              16384, Min(4), PowerOfTwo{})); // 17640 next power of two

template <typename T>
class LoudnessClient
    : public FluidBaseClient<decltype(LoudnessParams), LoudnessParams>,
      public AudioIn,
      public ControlOut {
  using HostVector = HostVector<T>;

public:
  LoudnessClient(ParamSetViewType &p) : FluidBaseClient(p) {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::controlChannelsOut(2);
    mDescriptors = FluidTensor<double, 1>(2);
  }

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output) {
    if (!input[0].data() || !output[0].data())
      return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() &&
           "Too few output channels");
    size_t hostVecSize = input[0].size();
    if (mBufferParamsTracker.changed(hostVecSize, get<kWindowSize>(),
                                     get<kHopSize>())) {
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(get<kWindowSize>(), get<kWindowSize>(),
                               FluidBaseClient::audioChannelsIn(),
                               FluidBaseClient::controlChannelsOut());
      mAlgorithm.init(get<kWindowSize>(), sampleRate());
    }
    RealMatrix in(1, hostVecSize);
    in.row(0) = input[0];
    mBufferedProcess.push(RealMatrixView(in));
    mBufferedProcess.processInput(
        get<kWindowSize>(), get<kHopSize>(), [&](RealMatrixView frame) {
          mAlgorithm.processFrame(frame.row(0), mDescriptors,
                                  get<kKWeighting>() == 1,
                                  get<kTruePeak>() == 1);
        });
    output[0](0) = mDescriptors(0);
    output[1](0) = mDescriptors(1);
  }

  size_t latency() { return get<kWindowSize>(); }

  size_t controlRate() { return get<kHopSize>(); }

private:
  Loudness mAlgorithm{get<kMaxWindowSize>()};
  ParameterTrackChanges<size_t, size_t, size_t> mBufferParamsTracker;
  BufferedProcess mBufferedProcess;
  FluidTensor<double, 1> mDescriptors;
};

auto constexpr NRTLoudnessParams =
    makeNRTParams<LoudnessClient>({BufferParam("source", "Source Buffer")},
                                  {BufferParam("features", "Features Buffer")});
template <typename T>
using NRTLoudnessClient =
    NRTControlAdaptor<LoudnessClient<T>, decltype(NRTLoudnessParams),
                      NRTLoudnessParams, 1, 1>;

} // namespace client
} // namespace fluid