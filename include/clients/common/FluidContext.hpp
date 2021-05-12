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

#include "../common/FluidTask.hpp"
#include "../common/Result.hpp"

namespace fluid {
namespace client {


class FluidContext
{

public:
  //  addError()

  FluidContext(FluidTask& t) : mTask{&t} {}
  FluidContext() = default;

  FluidTask* task() { return mTask; }
  void       task(FluidTask* t) { mTask = t; }

private:
  FluidTask*  mTask{nullptr};
  MessageList mMessages;
};

} // namespace client
} // namespace fluid
