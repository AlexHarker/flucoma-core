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

#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/HDBScan.hpp"
#include <string>

namespace fluid {
namespace client {
namespace hdbscan {

constexpr auto HDBScanParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("minPoints", "Min number of points in a cluster", 10, Min(1)),
    LongParam("minClusterSize", "Min cluster size", 0, Min(0)));

class HDBScanClient : public FluidBaseClient,
                     OfflineIn,
                     OfflineOut,
                     ModelObject,
                     public DataClient<algorithm::HDBScan>
{
  enum { kName, kMinPoints, kMinClusterSize };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using StringVectorView = FluidTensorView<string, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

  using ParamDescType = decltype(HDBScanParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return HDBScanParams; }

  HDBScanClient(ParamSetViewType& p) : mParams(p)
  {
    audioChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  MessageResult<IndexVector> fit(InputDataSetClientRef datasetClient)
  {
    index minPoints = get<kMinPoints>();
    index minClusterSize = get<kMinClusterSize>();
    auto  datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr) return Error<IndexVector>(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0) return Error<IndexVector>(EmptyDataSet);
    mAlgorithm.train(dataSet, minPoints, minClusterSize);
    IndexVector assignments(dataSet.size());
    mAlgorithm.getAssignments(assignments);
    return getCounts(assignments);
  }

  MessageResult<IndexVector> fitPredict(InputDataSetClientRef  datasetClient,
                                        LabelSetClientRef labelsetClient)
  {
    index minPoints = get<kMinPoints>();
    index minClusterSize = get<kMinClusterSize>();
    auto  datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr) return Error<IndexVector>(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0) return Error<IndexVector>(EmptyDataSet);
    auto labelsetClientPtr = labelsetClient.get().lock();
    if (!labelsetClientPtr) return Error<IndexVector>(NoLabelSet);
    mAlgorithm.train(dataSet, minPoints, minClusterSize);
    IndexVector assignments(dataSet.size());
    mAlgorithm.getAssignments(assignments);
    StringVectorView ids = dataSet.getIds();
    labelsetClientPtr->setLabelSet(getLabels(ids, assignments));
    return getCounts(assignments);
  }

  MessageResult<IndexVector> predict(InputDataSetClientRef  datasetClient,
                                     LabelSetClientRef labelClient) const
  {
    auto dataPtr = datasetClient.get().lock();
    if (!dataPtr) return Error<IndexVector>(NoDataSet);
    auto labelsetClientPtr = labelClient.get().lock();
    if (!labelsetClientPtr) return Error<IndexVector>(NoLabelSet);
    auto dataSet = dataPtr->getDataSet();
    if (dataSet.size() == 0) return Error<IndexVector>(EmptyDataSet);
    if (!mAlgorithm.initialized()) return Error<IndexVector>(NoDataFitted);
    if (dataSet.dims() != mAlgorithm.dims())
      return Error<IndexVector>(WrongPointSize);
    StringVectorView ids = dataSet.getIds();
    IndexVector      assignments(dataSet.size());
    RealVector       query(mAlgorithm.dims());
    for (index i = 0; i < dataSet.size(); i++)
    {
      dataSet.get(ids(i), query);
      assignments(i) = mAlgorithm.vq(query);
    }
    labelsetClientPtr->setLabelSet(getLabels(ids, assignments));
    return getCounts(assignments);
  }


  MessageResult<void> transform(InputDataSetClientRef srcClient,
                                DataSetClientRef dstClient) const
  {
    auto srcPtr = srcClient.get().lock();
    if (!srcPtr) return Error<void>(NoDataSet);
    auto destPtr = dstClient.get().lock();
    if (!destPtr) return Error<void>(NoDataSet);

    auto srcDataSet = srcPtr->getDataSet();
    if (srcDataSet.size() == 0) return Error<void>(EmptyDataSet);
    if (!mAlgorithm.initialized()) return Error<void>(NoDataFitted);
    if (srcDataSet.dims() != mAlgorithm.dims())
      return Error<void>(WrongPointSize);

    StringVectorView ids = srcDataSet.getIds();
    RealMatrix       output(srcDataSet.size(), mAlgorithm.size());
    mAlgorithm.getDistances(srcDataSet.getData(), output);
    FluidDataSet<string, double, 1> result(ids, output);
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<IndexVector> fitTransform(InputDataSetClientRef srcClient,
                                          DataSetClientRef dstClient)
  {
    index minPoints = get<kMinPoints>();
    index minClusterSize = get<kMinClusterSize>();
    auto  srcPtr = srcClient.get().lock();
    if (!srcPtr) return Error<IndexVector>(NoDataSet);
    auto destPtr = dstClient.get().lock();
    if (!destPtr) return Error<IndexVector>(NoDataSet);
    auto dataSet = srcPtr->getDataSet();
    if (dataSet.size() == 0) return Error<IndexVector>(EmptyDataSet);
    mAlgorithm.train(dataSet, minPoints, minClusterSize);
    IndexVector assignments(dataSet.size());
    mAlgorithm.getAssignments(assignments);
    transform(srcClient, dstClient);
    return getCounts(assignments);
  }

  MessageResult<index> predictPoint(InputBufferPtr data) const
  {
    if (!mAlgorithm.initialized()) return Error<index>(NoDataFitted);
    InBufferCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(data.get()))
      return Error<index>(bufCheck.error());
    RealVector point(mAlgorithm.dims());
    point <<=
        BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
    return mAlgorithm.vq(point);
  }

  MessageResult<void> getMeans(DataSetClientRef dstClient) const
  {
    auto destPtr = dstClient.get().lock();
    if (!destPtr) return Error<void>(NoDataSet);
    if (!mAlgorithm.initialized()) return Error<void>(NoDataFitted);
    RealMatrix output(mAlgorithm.size(), mAlgorithm.dims());
    mAlgorithm.getMeans(output);
    StringVector ids(mAlgorithm.size());
    std::generate(ids.begin(), ids.end(),
                  [n = 0]() mutable { return std::to_string(n++); });
    FluidDataSet<string, double, 1> result(ids, output);
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> setMeans(InputDataSetClientRef srcClient)
  {
    auto srcPtr = srcClient.get().lock();
    if (!srcPtr) return Error(NoDataSet);
    auto dataSet = srcPtr->getDataSet();
    if (dataSet.size() == 0) return Error(EmptyDataSet);
    if (dataSet.size() != mAlgorithm.numClusters()) return Error(WrongNumInitial);
    mAlgorithm.setMeans(dataSet.getData());
    return OK();
  }


  MessageResult<void> transformPoint(InputBufferPtr in, BufferPtr out) const
  {
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    InBufferCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(in.get())) return Error(bufCheck.error());
    BufferAdaptor::Access outBuf(out.get());
    Result                resizeResult =
        outBuf.resize(mAlgorithm.size(), 1, outBuf.sampleRate());
    if (!resizeResult.ok()) return Error(BufferAlloc);
    RealMatrix src(1, mAlgorithm.dims());
    RealMatrix dest(1, mAlgorithm.size());
    src.row(0) <<=
        BufferAdaptor::ReadAccess(in.get()).samps(0, mAlgorithm.dims(), 0);
    mAlgorithm.getDistances(src, dest);
    outBuf.allFrames()(Slice(0, 1), Slice(0, mAlgorithm.size())) <<= dest;
    return OK();
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &HDBScanClient::fit),
        makeMessage("predict", &HDBScanClient::predict),
        makeMessage("transform", &HDBScanClient::transform),
        makeMessage("predictPoint", &HDBScanClient::predictPoint),
        makeMessage("transformPoint", &HDBScanClient::transformPoint),
        makeMessage("fitTransform", &HDBScanClient::fitTransform),
        makeMessage("getMeans", &HDBScanClient::getMeans),
        makeMessage("setMeans", &HDBScanClient::setMeans),
        makeMessage("fitPredict", &HDBScanClient::fitPredict),
        makeMessage("cols", &HDBScanClient::dims),
        makeMessage("clear", &HDBScanClient::clear),
        makeMessage("size", &HDBScanClient::size),
        makeMessage("load", &HDBScanClient::load),
        makeMessage("dump", &HDBScanClient::dump),
        makeMessage("write", &HDBScanClient::write),
        makeMessage("read", &HDBScanClient::read));
  }


private:
  IndexVector getCounts(IndexVector assignments) const
  {
    IndexVector counts(mAlgorithm.numClusters() + 1);
    counts.fill(0);
    for (auto a : assignments)
        if (a < 0)
            counts[0]++;
        else if (a == 0)
            assert(false);
        else
            counts[a]++;
    return counts;
  }

  LabelSet getLabels(StringVectorView& ids, IndexVector assignments) const
  {
    LabelSet result(1);
    for (index i = 0; i < ids.size(); i++)
    {
      StringVector point = {std::to_string(assignments(i))};
      result.add(ids(i), point);
    }
    return result;
  }
};

using HDBScanRef = SharedClientRef<const HDBScanClient>;

constexpr auto HDBScanQueryParams =
    defineParameters(HDBScanRef::makeParam("kmeans", "Source KMeans model"),
                     InputBufferParam("inputPointBuffer", "Input Point Buffer"),
                     BufferParam("predictionBuffer", "Prediction Buffer"));

class HDBScanQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kModel, kInputBuffer, kOutputBuffer };

public:
  using ParamDescType = decltype(HDBScanQueryParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return HDBScanQueryParams; }

  HDBScanQuery(ParamSetViewType& p) : mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext&)
  {
    output[0] <<= input[0];
    if (input[0](0) > 0)
    {
      auto hdbscanPtr = get<kModel>().get().lock();
      if (!hdbscanPtr)
      {
        // report error?
        return;
      }
      if (!hdbscanPtr->initialized()) return;
      index             dims = hdbscanPtr->dims();
      InOutBuffersCheck bufCheck(dims);
      if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                                get<kOutputBuffer>().get()))
        return;
      auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
      if (outBuf.samps(0).size() < 1) return;
      RealVector point(dims);
      point <<= BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
                  .samps(0, dims, 0);
      outBuf.samps(0)[0] = hdbscanPtr->algorithm().vq(point);
    }
  }

  index latency() { return 0; }
};


} // namespace kmeans

using NRTThreadedHDBScanClient =
    NRTThreadingAdaptor<typename hdbscan::HDBScanRef::SharedType>;

using RTHDBScanQueryClient = ClientWrapper<hdbscan::HDBScanQuery>;

} // namespace client
} // namespace fluid
