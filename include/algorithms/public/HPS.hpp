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

#include "../util/FluidEigenMappings.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class HPS
{

public:
  void processFrame(const RealVectorView& input, RealVectorView output,
                    index nHarmonics, double minFreq, double maxFreq,
                    double sampleRate)
  {
    using namespace Eigen;
    ArrayXd::Index maxIndex;

    ArrayXd mag = _impl::asEigen<Array>(input);
    ArrayXd hps = mag;
    index   nBins = mag.size();
    double  binHz = sampleRate / ((nBins - 1) * 2);
    index   minBin = std::lrint(minFreq / binHz);
    index   maxBin = std::lrint(maxFreq / binHz);
    double  f0 = minBin * binHz;
    double  confidence = 0;

    for (index i = 2; i < nHarmonics; i++)
    {
      index   hBins = nBins / i;
      ArrayXd h = ArrayXd::Zero(hBins);
      for (index j = 0; j < hBins; j++) h(j) = mag(j * i);
      ArrayXd hp = ArrayXd::Zero(nBins);
      hp.segment(0, hBins) = h;
      hps = hps * hp;
    }

    if (maxBin > minBin)
    {
      hps = hps.segment(minBin, maxBin - minBin);
      double maxVal =  hps.maxCoeff(&maxIndex);
      double sum = hps.sum();
      confidence = sum == 0 ? 0 : maxVal / sum;
      f0 = (minBin + maxIndex) * binHz;
    }
    output(0) = f0;
    output(1) = confidence;
  }
};
} // namespace algorithm
} // namespace fluid
