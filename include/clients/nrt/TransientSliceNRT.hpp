#pragma once


#include "algorithms/TransientSegmentation.hpp"

#include "clients/common/FluidParams.hpp"
#include "data/FluidTensor.hpp"
#include "data/FluidBuffers.hpp"



#include <algorithm> //for max_element
#include <string>
#include <sstream> //for ostringstream
#include <utility> //for std make_pair
#include <unordered_set>
#include <vector> //for containers of params, and for checking things


namespace fluid {
  namespace segmentation{
    
    /**
     Integration class for doing NMF filtering and resynthesis
     **/
    class TransientSliceNRT
    {
      using desc_type = parameter::Descriptor;
      using param_type = parameter::Instance;
    public:
      
      struct ProcessModel
      {

        bool returnTransients;
        bool returnResidual;
        size_t windowSize;
        size_t frames;
        size_t offset;
        size_t channels;
        size_t channelOffset;
        parameter::BufferAdaptor* src = 0;
        parameter::BufferAdaptor* trans = 0;
        parameter::BufferAdaptor* res = 0;
        
        size_t order;
        size_t blocksize;
        size_t halfWindow;
        double skew;
        size_t padding;
        double fwdThresh;
        double backThresh;
        double debounce;
        unsigned iterations = 3;
        double robustFactor = 3.0;
      };
      
      static const std::vector<parameter::Descriptor>& getParamDescriptors()
      {
        static std::vector<desc_type> params;
        if(params.empty())
        {
          params.emplace_back(desc_type{"src","Source Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(true);
          
          params.emplace_back(desc_type{"offsetframes","Source Offset", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setDefault(0);

          params.emplace_back(desc_type{"numframes","Source Frames", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);
          
          params.emplace_back(desc_type{"offsetchans","Source Channel Offset", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setDefault(0);
          
          params.emplace_back(desc_type{"numchans","Source Channels", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);
          
          params.emplace_back(desc_type{"transbuf","Indices Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(false);
          
          params.emplace_back("order", "Order", parameter::Type::Long);
          params.back().setInstantiation(false).setMin(20).setDefault(200);
          //order min > paramDetectHalfWindow, or ~40-50 ms,
       
//          int paramBlockSize = 2048;  // The main block size for processing (higher == longer processing times N^2 but better quality)
          params.emplace_back("blocksize","Block Size", parameter::Type::Long);
          params.back().setInstantiation(false).setMin(100).setDefault(2048);
          
          //must be greater than model order
          
//          int paramPad = 1024;        // The analysis is done on a longer segment than the block, with this many extra values on either side
          //padding min 0
          params.emplace_back("padding","Padding", parameter::Type::Long);
          params.back().setInstantiation(false).setMin(0).setDefault(1024);
          
          
          // This ensures the analysis is valid across the whole block (some padding is a good idea, but not too much)
          
          // The detection parameters
          
          // Detection is based on absolute forward and backwards prediction errors in relation to the estimated deviation of the AR model - these predictions are smoothed with a window and subjected to an on and off threshold - higher on thresholds make detection less likely and the reset threshold is used (along with a hold time) to ensure that the detection does not switch off before the end of a transient
          
          
          //'skew', do 2^n -10, 10
          
//          double paramDetectPower = 1.0;           // The power factor used when windowing - higher makes detection more likely
          params.emplace_back("skew","Skew", parameter::Type::Float);
          params.back().setInstantiation(false).setMin(-10).setMax(10).setDefault(0);

          
//          double paramDetectThreshHi = 3.0;        // The threshold for detection (in multiples of the model deviation)
          //
          params.emplace_back("threshfwd","Forward Threshold", parameter::Type::Float);
          params.back().setInstantiation(false).setMin(0).setDefault(3);
          
          
//          double paramDetectThreshLo = 1.1;        // The reset threshold to end a detected segment (in multiples of the model deviation)
          params.emplace_back("threshback","Backward Threshold", parameter::Type::Float);
          params.back().setInstantiation(false).setMin(0).setDefault(1.1);
          
          
//          double paramDetectHalfWindow = 7;        // Half the window size used to smooth detection functions (in samples)
          //up to model order ~40 = 1ms, 15 default sampples for whole window
          //
          params.emplace_back("windowsize","Window Size(ms)", parameter::Type::Float);
          params.back().setInstantiation(false).setMin(0).setDefault(14);

          
//          int paramDetectHold = 25;               // The hold time for detection (in samples)
          //prevents onsets within n samples of an offset, min 0,
          params.emplace_back("debounce","Debounce(ms)", parameter::Type::Float);
          params.back().setInstantiation(false).setMin(0).setDefault(25);

          
//          // This is broken right now - SET FALSE AND DO NOT EXPOSE - turning it on will produce worse results (should make things better)
//
//          const bool paramRefine = false;
//
//          // These parameters relate to the way we estimate the AR parameters robustly from data
//          // They probably aren't worth exposing to the end user (DO NOT EXPOSE THE ROBUST FACTOR)
//
//          int paramIterations = 3;           // How many times to iterate over the data to robustify it (can be 0)
//          double paramRobustFactor = 3.0;    // Data futher than this * deviation from the expected value it will be clipped
          
        }
        return params;
      }
      
      void newParameterSet()
      {
        mParams.clear();
        mParams.reserve(getParamDescriptors().size());
        //Note: I'm pretty sure I want auto's copy behaviour here
        for(auto p: getParamDescriptors())
          mParams.emplace_back( parameter::Instance(p));
      }
      
      /**
       Go over the supplied parameter values and ensure that they are sensible  
       
       Args
       source buffer [offset numframes offsetchan numchan] transient_buff residual_buff
       
       
      **/
      std::tuple<bool,std::string,ProcessModel> sanityCheck()
      {
        ProcessModel model;
        const std::vector<parameter::Descriptor>& desc = getParamDescriptors();
        //First, let's make sure that we have a complete of parameters of the right sort
        bool sensible = std::equal(mParams.begin(), mParams.end(),desc.begin(),
          [](const param_type& i, const parameter::Descriptor& d)
          {
            return i.getDescriptor() == d;
          });
        
        if(! sensible || (desc.size() != mParams.size()))
        {
          return {false, "Invalid params passed. Were these generated with newParameterSet()?", model };
        }
        
        size_t bufCount = 0;
        std::unordered_set<parameter::BufferAdaptor*> uniqueBuffers;
        //First round of buffer checks
        //Source buffer is mandatory, and should exist
        parameter::BufferAdaptor::Access src(mParams[0].getBuffer());
        
        if(!src.valid())
        {
          return  { false, "Source buffer doesn't exist or can't be accessed.", model };
        }
        
        if(src.numFrames() == 0)
        {
          return {false, "Source buffer is empty",model}; 
        }
        
      
        for(auto&& p: mParams)
        {
          switch(p.getDescriptor().getType())
          {
            case parameter::Type::Buffer:
              //If we've been handed a buffer that we're expecting, then it should exist
              if(p.hasChanged() && p.getBuffer())
              {
                parameter::BufferAdaptor::Access b(p.getBuffer());
                if(!b.valid())
                 {
                   std::ostringstream ss;
                   ss << "Buffer given for " << p.getDescriptor().getName() << " doesn't exist.";
                
                   return {false, ss.str(), model};
                 }
                ++bufCount;
                uniqueBuffers.insert(p.getBuffer());
              }
            default:
              continue;
          }
        }
 
        if(bufCount < 2)
        {
          return { false, "Expecting two valid buffers", model};
        }
        
        if(bufCount > uniqueBuffers.size())
        {
          return {false, "Buffers are the same. They all need to be distinct", model};
        }
        
        
        //Now scan everything for range, until we hit a problem
        //TODO Factor into parameter::instance
        for(auto&& p: mParams)
        {
          parameter::Descriptor d = p.getDescriptor();
          bool rangeOk;
          parameter::Instance::RangeErrorType errorType;
          std::tie(rangeOk, errorType) = p.checkRange();
          if (!rangeOk)
          {
            std::ostringstream msg;
            msg << "Parameter " << d.getName();
            switch (errorType)
            {
              case parameter::Instance::RangeErrorType::Min:
                msg << " value below minimum(" << d.getMin() << ")";
                break;
              case parameter::Instance::RangeErrorType::Max:
                msg << " value above maximum(" << d.getMin() << ")";
              default:
                assert(false && "This should be unreachable");
            }
            return { false, msg.str(), model};
          }

        }
        
  
        //Check the size of our buffers
//        parameter::BufferAdaptor* src= params[0].getBuffer();
        
        long srcOffset     = parameter::lookupParam("offsetframes",         mParams).getLong();
        long srcFrames     = parameter::lookupParam("numframes",         mParams).getLong();
        long srcChanOffset = parameter::lookupParam("offsetchans", mParams).getLong();
        long srcChans      = parameter::lookupParam("numchans",       mParams).getLong();
        
        //Ensure that the source buffer can deliver
        if(srcFrames > 0 ? (src.numFrames() < (srcOffset + srcFrames)) : (src.numFrames() < srcOffset))
        {
          return  { false, "Source buffer not long enough for given offset and frame count",model};
        }
        
        if((srcChans > 0) ? (src.numChans() < (srcChanOffset + srcChans)) : (src.numChans() < srcChanOffset))
        {
          return {false, "Source buffer doesn't have enough channels for given offset and channel count", model};
        }
        
        //At this point, we're happy with the source buffer
        model.src           = mParams[0].getBuffer();
        model.offset        = srcOffset;
        model.frames        = srcFrames > 0 ? srcFrames : src.numFrames() - model.offset;
        model.channelOffset = srcChanOffset;
        model.channels      = srcChans >  0 ? srcChans  : src.numChans() - model.channelOffset;
        
      
        
        parameter::Instance& transBufParam =  mParams[5];
        parameter::BufferAdaptor::Access transBuf(transBufParam.getBuffer());
        
        if(transBufParam.hasChanged() && (!transBuf.valid()))
        {
          return {false, "Invalid transients buffer supplied", model};
        }
        
//        model.returnTransients = transBufParam.hasChanged();
        model.trans = transBufParam.getBuffer();
        
        model.halfWindow = std::round(parameter::lookupParam("windowsize", mParams).getFloat() / 2);
        
        long order = parameter::lookupParam("order", mParams).getLong();
        
        if(order < model.halfWindow)
        {
          return {false, "Model order must be more than half the window size", model};
        }
        
        long blocksize = parameter::lookupParam("blocksize", mParams).getLong();
        
        if(blocksize < order)
        {
          return {false, "Block size must be greater than model order",model};
        }
        
        
        model.skew = std::pow(2, parameter::lookupParam("skew",mParams).getFloat());
        
        model.padding = parameter::lookupParam("padding",mParams).getLong();
        model.fwdThresh = parameter::lookupParam("threshfwd",mParams).getFloat();
        model.backThresh = parameter::lookupParam("threshback",mParams).getFloat();;
        model.debounce = parameter::lookupParam("debounce",mParams).getFloat();;
        model.blocksize = blocksize;
        model.order = order; 

        //We made it
        return {true, "Everything is lovely",model};
      }
      
      
      //No, you may not  copy this, or move this
      TransientSliceNRT(TransientSliceNRT&)=delete;
      TransientSliceNRT(TransientSliceNRT&&)=delete;
      TransientSliceNRT operator=(TransientSliceNRT&)=delete;
      TransientSliceNRT operator=(TransientSliceNRT&&)=delete;
      
      TransientSliceNRT(){
        newParameterSet(); 
      }
   
      void process(ProcessModel model)
      {
        parameter::BufferAdaptor::Access src(model.src);
        parameter::BufferAdaptor::Access trans(model.trans);
        
        FluidTensor<double,1> transientFrames(model.frames + model.blocksize);

        FluidTensor<double,1> monoSource(model.frames);
        
        //Make a mono sum;
        for(size_t i = 0; i < model.channels; ++i)
        {
          monoSource.apply(src.samps(i), [](double& x, double y){
            x += y;
          });
        }
        
        
        segmentation::TransientSegmentation segmentor(model.order, model.iterations, model.robustFactor);
        segmentor.prepareStream(model.blocksize,model.padding);
        segmentor.setDetectionParameters(model.skew, model.fwdThresh, model.backThresh, model.halfWindow, model.debounce); 
  

        size_t hopsize = segmentor.hopSize();
        assert(hopsize > 0);
        for(size_t i = 0; i < model.frames; i+= hopsize)
        {
          size_t size = std::min<size_t>(segmentor.inputSize(), model.frames - i);
          FluidTensorView<const double,1> markers(segmentor.process(monoSource.data() + i, size),0, hopsize);
          transientFrames(fluid::slice(i,hopsize)) = markers;
        }
        
        size_t num_spikes = std::accumulate(transientFrames.begin(), transientFrames.end() , 0);
        
        //Arg sort
        std::vector<size_t> indices(transientFrames.size());
        std::iota(indices.begin(),indices.end(),0);
        std::sort(indices.begin(), indices.end(),[&](size_t i1, size_t i2){
          return transientFrames[i1] > transientFrames[i2];
        });

        //Now put the gathered indicies into ascending order
        std::sort(indices.begin(), indices.begin() + num_spikes);
        
        //Add model offset
        std::transform(indices.begin(), indices.begin() + num_spikes, indices.begin(),[&](size_t x)->size_t {
          return x + model.offset;
        });
        
        //insert leading <offset> and num_frames
        indices.insert(indices.begin() + num_spikes, model.offset + model.frames);
        indices.insert(indices.begin(), model.offset);
        
        trans.resize(num_spikes + 2,1,1);
        trans.samps().col(0) = FluidTensorView<size_t,1>{indices.data(),0,num_spikes + 2};
      }
      
      
      std::vector<parameter::Instance>& getParams()
      {
        return mParams;
      }
      
    private:
      std::vector<parameter::Instance> mParams;
    };
  } //namespace max
} //namesapce fluid