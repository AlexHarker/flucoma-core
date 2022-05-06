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

#include "../util/DistanceFuncs.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <Hdbscan/hdbscan.hpp>
#include <Eigen/Core>
#include <queue>
#include <string>
#include <vector>

#include <Runner/hdbscanRunner.hpp>
#include <Runner/hdbscanParameters.hpp>
#include <Runner/hdbscanResult.hpp>
#include <HdbscanStar/outlierScore.hpp>


namespace fluid {
namespace algorithm {

class HDBScan
{

public:
  void clear()
  {
    mMeans.setZero();
    mNormalizedLabels.clear();
    mNoisyPoints = 0;
    mNumClusters = 0;
    mTrained = false;
  }

  bool initialized() const { return mTrained; }

  void train(const FluidDataSet<std::string, double, 1>& dataset, index minPoints,
             index minClusterSize)
  {
    assert(!mTrained);
    auto data = dataset.getData();
    
    std::vector<vector<double>> dataCopy;
      
    for (index i = 0; i < data.extent(0); i++)
    {
      auto row = data.row(i);
        dataCopy.push_back(std::vector<double>());
        dataCopy.back().reserve(row.size());
      for (index j = 0; j < row.size(); j++)
          dataCopy.back().push_back(row[j]);
    }
      
    hdbscanRunner runner;
    hdbscanParameters parameters;
    uint32_t noisyPoints = 0;
    set<int> numClustersSet;
    map<int, int> clustersMap;
      
    parameters.dataset = dataCopy;
    parameters.minPoints = minPoints;
    parameters.minClusterSize = minClusterSize;
    parameters.distanceFunction = "";
      
    auto result = runner.run(parameters);
      
    for (uint32_t i = 0; i < result.labels.size(); i++)
    {
      if (result.labels[i] == 0)
        noisyPoints++;
      else
        numClustersSet.insert(result.labels[i]);
    }
    
    mNumClusters = numClustersSet.size();
    mNoisyPoints = noisyPoints;
     
    int idx = 1;
    for (auto it = numClustersSet.begin(); it != numClustersSet.end(); it++)
        clustersMap[*it] = idx++;
    
    for (int i = 0; i < result.labels.size(); i++)
    {
      if (result.labels[i] != 0)
        mNormalizedLabels.push_back(clustersMap[result.labels[i]]);
      else if (result.labels[i] == 0)
        mNormalizedLabels.push_back(-1);
    }
          
    mTrained = true;
  }

  index getClusterSize(index cluster) const
  {
    index count = 0;
    for (index i = 0; i < mNormalizedLabels.size(); i++)
    {
      if (mNormalizedLabels
          [i] == cluster) count++;
    }
    return count;
  }

  index vq(RealVectorView point) const
  {
    assert(point.size() == mDims);
    return assignPoint(_impl::asEigen<Eigen::Array>(point));
  }

  void getMeans(RealMatrixView out) const
  {
    if (mTrained) out <<= _impl::asFluid(mMeans);
  }

  void setMeans(RealMatrixView means)
  {
    mMeans = _impl::asEigen<Eigen::Array>(means);
    mDims = mMeans.cols();
    mK = mMeans.rows();
    mTrained = true;
  }

  index dims() const { return mMeans.cols(); }
  index size() const { return mMeans.rows(); }
  index getK() const { return mMeans.rows(); }
  index nAssigned() const { return mNormalizedLabels.size(); }

  void getAssignments(FluidTensorView<index, 1> out) const
  {
    for (index i = 0; i < mNormalizedLabels.size(); i++)
      out[i] = mNormalizedLabels[i];
  }

  index numClusters() const
  {
    return mNumClusters;
  }
  
  void getDistances(RealMatrixView data, RealMatrixView out) const
  {
    Eigen::ArrayXXd points = _impl::asEigen<Eigen::Array>(data);
    Eigen::ArrayXXd D = fluid::algorithm::DistanceMatrix(points, 2);
    Eigen::MatrixXd means = mMeans.matrix();
    D = fluid::algorithm::DistanceMatrix<Eigen::ArrayXXd>(points, mMeans, 2);
    out <<= _impl::asFluid(D);
  }

private:
  double distance(Eigen::ArrayXd v1, Eigen::ArrayXd v2) const
  {
    return (v1 - v2).matrix().norm();
  }

  index assignPoint(Eigen::ArrayXd point) const
  {
    double minDistance = std::numeric_limits<double>::infinity();
    index  minK;
    for (index k = 0; k < mK; k++)
    {
      double dist = distance(point, mMeans.row(k));
      if (dist < minDistance)
      {
        minK = k;
        minDistance = dist;
      }
    }
    return minK;
  }
    
  std::vector<int> mNormalizedLabels;
  uint32_t mNoisyPoints;
  uint32_t mNumClusters;

  index             mK{0};
  index             mDims{0};
  Eigen::ArrayXXd   mMeans;
  bool              mTrained{false};
};
} // namespace algorithm
} // namespace fluid
