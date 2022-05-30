/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/NoveltyFeature.hpp"

namespace fluid {
namespace client {

enum NoveltyDataParamIndex {kSource, kOffset, kNumFrames, kStartChan, kNumChans, kOutput, kKernelSize,kFilterSize};

auto constexpr NoveltyDataParams = defineParameters(
  InputBufferParam("source","Source Buffer"),
  LongParam("startFrame","Source Offset",0,Min(0)),
  LongParam("numFrames","Number of Frames",-1),
  LongParam("startChan","Start Channel",0,Min(0)),
  LongParam("numChans","Number of Channels",-1),
  BufferParam("output", "Output Buffer"),
  LongParam("kernelSize", "Kernel Size", 3, Min(3), Odd()),
  LongParam("filterSize", "Smoothing Filter Size", 1, Min(1))
 );


class NoveltyDataClient: public FluidBaseClient, public OfflineIn, public OfflineOut
{

public:
    
  using ParamDescType = decltype(NoveltyDataParams);
  using ParamSetViewType = ParameterSetView<ParamDescType>;

  NoveltyDataClient(ParamSetViewType& p)
  : mParams{p}
  {}

  template <std::size_t N>
  auto& get() noexcept
  {
    return mParams.get().template get<N>();
  }
    
  void setParams(ParamSetViewType& p)
  {
    mParams = p;
  }
    
  static constexpr auto& getParameterDescriptors() { return NoveltyDataParams; }

  template<typename T>
  Result process(FluidContext& c)
  {
    if(!get<kSource>().get())
      return {Result::Status::kError, "No input buffer supplied"};

    BufferAdaptor::ReadAccess source(get<kSource>().get());

    if(!source.exists())
        return {Result::Status::kError, "Input buffer not found"};

    if(!source.valid())
        return {Result::Status::kError, "Can't access input buffer"};

    {
        BufferAdaptor::Access idx(get<kOutput>().get());

        if(!idx.exists())
            return {Result::Status::kError, "Output buffer not found"};

    }

//    if(!idx.valid())
//        return {Result::Status::kError, "Can't access output buffer"};

    size_t nChannels = get<kNumChans>()  == -1 ? source.numChans() : get<kNumChans>();
    size_t nFrames   = get<kNumFrames>() == -1 ? source.numFrames(): get<kNumFrames>();

    index kernelSize = get<kKernelSize>();
    index filterSize = get<kFilterSize>();
      
    algorithm::NoveltyFeature processor(kernelSize, filterSize);

    auto inputData = FluidTensor<double,1>(nChannels);
    auto outputData = FluidTensor<double,1>(nFrames);
      
    processor.init(kernelSize, filterSize, nChannels);
    
    for (size_t i = 0; i < nFrames; i++)
    {
        // Get col/row

        // FIX - make this nicer
        
        for (size_t j = 0; j < nChannels; j++)
            inputData(j) = source.samps(get<kOffset>() + i, 1, get<kStartChan>() + j)(0);
        
        outputData(i) = processor.processFrame(inputData);
    }
    
    BufferAdaptor::Access idx(get<kOutput>().get());
        
    Result resizeResult =
    idx.resize(nFrames, 1, source.sampleRate());
    if (!resizeResult.ok()) return resizeResult;
        
    idx.samps(0) <<= outputData;
    
    return {Result::Status::kOk,""};
  }

  std::reference_wrapper<ParamSetViewType> mParams;
};

using NRTThreadedNoveltyDataClient = NRTThreadingAdaptor<ClientWrapper<NoveltyDataClient>>;

} // namespace client
} // namespace fluid
