#pragma once

#include "HISSTools_FFT/HISSTools_FFT.h"
#include <Eigen/Core>
#include <vector>

namespace fluid {
namespace fft {

using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::Ref;
using std::complex;
using std::vector;

class FFT {
public:
  FFT(size_t size)
      : mSize(size), mFrameSize(size / 2 + 1), mLog2Size(log2(size)),
        mOutputBuffer(mFrameSize) {
    hisstools_create_setup(&mSetup, mLog2Size);
    mSplit.realp = new double[(1 << (mLog2Size - 1)) + 1];
    mSplit.imagp = new double[(1 << (mLog2Size - 1)) + 1];
  }

  ~FFT() {
    if (mSplit.realp)
      delete[] mSplit.realp;
    if (mSplit.imagp)
      delete[] mSplit.imagp;
  }

  Ref<ArrayXcd> process(const Ref<const ArrayXd> &input) {
    hisstools_rfft(mSetup, input.data(), &mSplit, input.size(), mLog2Size);
    mSplit.realp[mFrameSize - 1] = mSplit.imagp[0];
    mSplit.imagp[mFrameSize - 1] = 0;
    mSplit.imagp[0] = 0;
    for (int i = 0; i < mFrameSize; i++) {
      mOutputBuffer(i) =
          0.5 * complex<double>(mSplit.realp[i], mSplit.imagp[i]);
    }
    return mOutputBuffer;
  }

protected:
  size_t mSize;
  size_t mFrameSize;
  size_t mLog2Size;

  FFT_SETUP_D mSetup;
  FFT_SPLIT_COMPLEX_D mSplit;

private:
  ArrayXcd mOutputBuffer;
};

class IFFT : FFT {
public:
  IFFT(size_t size) : FFT(size), mOutputBuffer(size) {}

  Ref<ArrayXd> process(const Ref<const ArrayXcd> &input) {
    for (int i = 0; i < input.size(); i++) {
      mSplit.realp[i] = input[i].real();
      mSplit.imagp[i] = input[i].imag();
    }
    mSplit.imagp[0] = mSplit.realp[mFrameSize - 1];
    hisstools_rifft(mSetup, &mSplit, mOutputBuffer.data(), mLog2Size);
    return mOutputBuffer;
  }

private:
  ArrayXd mOutputBuffer;
};
} // namespace fft
} // namespace fluid