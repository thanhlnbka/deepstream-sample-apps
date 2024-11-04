/**
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <unistd.h>
#include <dlfcn.h>
#include <thread>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cmath>

#include "convbufmanager.h"
#include "nvtracker_proc.h"
#include "logging.h"
#include "nvtx_helper.h"
#include "nvds_dewarper_meta.h"
#include "nvdspreprocess_meta.h"
#include "gstnvdspreprocess_allocator.h"

using namespace std;

extern "C"
{
#define GST_NVDSPREPROCESS_MEMORY_TYPE "nvdspreprocess"

/** Structure allocated internally by the allocator. */
typedef struct
{
  /** Should be the first member of a structure extending GstMemory. */
  GstMemory mem;
  /** Custom Gst memory for preprocess plugin */
  GstNvDsPreProcessMemory mem_preprocess;
} GstNvDsPreProcessMem;

static gpointer copy_nvtracker2_past_frame_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data)
  {
    NvDsTargetMiscDataBatch *pPastFrameObjBatch =
      (NvDsTargetMiscDataBatch *) user_meta->user_meta_data;
    gst_mini_object_ref( (GstMiniObject *) pPastFrameObjBatch->priv_data);
    return (gpointer) user_meta->user_meta_data;
  }
  return NULL;
}

static void
free_nvtracker2_past_frame_meta(gpointer meta_data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) meta_data;
  if (user_meta && user_meta->user_meta_data)
  {
    NvDsTargetMiscDataBatch *pPastFrameObjBatch =
      (NvDsTargetMiscDataBatch *) user_meta->user_meta_data;
    gst_mini_object_unref (GST_MINI_OBJECT(pPastFrameObjBatch->priv_data));
  }
}

static gpointer copy_nvtracker2_batch_reid_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data)
  {
    NvDsReidTensorBatch *pReidTensor =
      (NvDsReidTensorBatch *) user_meta->user_meta_data;
    gst_mini_object_ref( (GstMiniObject *) pReidTensor->priv_data);
    return (gpointer) user_meta->user_meta_data;
  }
  return NULL;
}

static void
free_nvtracker2_batch_reid_meta(gpointer meta_data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) meta_data;
  if (user_meta && user_meta->user_meta_data)
  {
    NvDsReidTensorBatch *pPastFrameObjBatch =
      (NvDsReidTensorBatch *) user_meta->user_meta_data;
    gst_mini_object_unref (GST_MINI_OBJECT(pPastFrameObjBatch->priv_data));
  }
}

static gpointer copy_nvtracker2_obj_reid_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data)
  {
    int32_t *pNewReidInd = new int32_t;
    *pNewReidInd = *((int32_t *) user_meta->user_meta_data);
    return (gpointer) pNewReidInd; //user_meta->user_meta_data;
  }
  return NULL;
}

static void
free_nvtracker2_obj_reid_meta(gpointer meta_data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) meta_data;
  if (user_meta && user_meta->user_meta_data)
  {
    int32_t *pReidInd = (int32_t *) user_meta->user_meta_data;
    delete pReidInd;
  }
}

static gpointer copy_nvtracker2_terminated_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data)
  {
    NvDsTargetMiscDataBatch *pTerminatedObjBatch =
      (NvDsTargetMiscDataBatch *) user_meta->user_meta_data;
    gst_mini_object_ref( (GstMiniObject *) pTerminatedObjBatch->priv_data);
    return (gpointer) user_meta->user_meta_data;
  }
  return NULL;
}

static void
free_nvtracker2_terminated_meta(gpointer meta_data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) meta_data;
  if (user_meta && user_meta->user_meta_data)
  {
    NvDsTargetMiscDataBatch *pTerminatdObjBatch =
      (NvDsTargetMiscDataBatch *) user_meta->user_meta_data;
    gst_mini_object_unref (GST_MINI_OBJECT(pTerminatdObjBatch->priv_data));
  }
}

static gpointer copy_nvtracker2_shadowTracks_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data)
  {
    NvDsTargetMiscDataBatch *pShadowTracks =
      (NvDsTargetMiscDataBatch *) user_meta->user_meta_data;
    gst_mini_object_ref( (GstMiniObject *) pShadowTracks->priv_data);
    return (gpointer) user_meta->user_meta_data;
  }
  return NULL;
}

static void
free_nvtracker2_shadowTracks_meta(gpointer meta_data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) meta_data;
  if (user_meta && user_meta->user_meta_data)
  {
    NvDsTargetMiscDataBatch *pShadowTracks =
      (NvDsTargetMiscDataBatch *) user_meta->user_meta_data;
    gst_mini_object_unref (GST_MINI_OBJECT(pShadowTracks->priv_data));
  }
}

static gpointer copy_nvtracker2_obj_visibility_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data)
  {
    float *pNewVis = new float;
    *pNewVis = *((float *) user_meta->user_meta_data);
    return (gpointer) pNewVis; //user_meta->user_meta_data;
  }
  return NULL;
}

static void
free_nvtracker2_obj_visibility_meta(gpointer meta_data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) meta_data;
  if (user_meta && user_meta->user_meta_data)
  {
    float *pVis = (float *) user_meta->user_meta_data;
    delete pVis;
  }
}

static gpointer copy_nvtracker2_obj_foot_loc_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data)
  {
    float *pPtFeet = new float [2];
    pPtFeet[0] = ((float *) user_meta->user_meta_data)[0];
    pPtFeet[1] = ((float *) user_meta->user_meta_data)[1];
    return (gpointer) pPtFeet; //user_meta->user_meta_data;
  }
  return NULL;
}

static void
free_nvtracker2_obj_foot_loc_meta(gpointer meta_data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) meta_data;
  if (user_meta && user_meta->user_meta_data)
  {
    float *pPtFeet = (float *) user_meta->user_meta_data;
    delete [] pPtFeet;
  }
}

static gpointer copy_nvtracker2_obj_convex_hull_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data)
  {
    NvDsObjConvexHull* pConvexHull = (NvDsObjConvexHull *) user_meta->user_meta_data;
    if (pConvexHull->numPointsAllocated == 0)
    {
      return NULL;
    }

    NvDsObjConvexHull* pNewConvexHull = new NvDsObjConvexHull;
    pNewConvexHull->list = new int [pConvexHull->numPointsAllocated * 2];
    memcpy(pNewConvexHull->list, pConvexHull->list,
        pConvexHull->numPoints * 2 * sizeof(int));
    pNewConvexHull->numPoints = pConvexHull->numPoints;
    pNewConvexHull->numPointsAllocated = pConvexHull->numPointsAllocated;
    return (gpointer) pNewConvexHull; //user_meta->user_meta_data;
  }
  return NULL;
}

static void
free_nvtracker2_obj_convex_hull_meta(gpointer meta_data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) meta_data;
  if (user_meta && user_meta->user_meta_data)
  {
    NvDsObjConvexHull *pConvexHull = (NvDsObjConvexHull *) user_meta->user_meta_data;
    if (pConvexHull->list)
    {
      delete [] pConvexHull->list;
    }
    delete pConvexHull;
  }
}

static void
return_nvtracker_buffer_meta(GstMiniObject * obj)
{
  GstNvTrackerMiscDataObject *pGstObj =
    (GstNvTrackerMiscDataObject *) obj;
  pGstObj->misc_data_manager->returnBuffer(pGstObj->misc_data_buffer);
  delete pGstObj;
}
}

NvTrackerProc::NvTrackerProc()
{
}

NvTrackerProc:: ~NvTrackerProc()
{
}

bool NvTrackerProc::init(const TrackerConfig& config)
{
  bool ret;

  m_Config = config;

  /** Load the low-level tracker lib */
  ret = initTrackerLib();
  if (!ret) {
    LOG_ERROR("gstnvtracker: Failed to initilaize low level lib.\n");
    return false;
  }

  /** Initialize the buffer pool based on config */
  if (m_Config.inputTensorMeta == false){
    ret = initConvBufPool();
    if (!ret) {
      LOG_ERROR("gstnvtracker: Failed to initilaize surface transform buffer pool.\n");
      deInit();
      return false;
    }
  }

  ret = initMiscDataPool();
  if (!ret) {
    LOG_ERROR("gstnvtracker: Failed to initilaize user meta buffer pool.\n");
    deInit();
    return false;
  }

  m_Running = true;
  m_TrackerLibError = false;

  if (m_Config.subBatchesConfig.size() == 0)
  {
  m_BatchContextHandle = initTrackerContext();
  if (nullptr == m_BatchContextHandle) {
    LOG_ERROR("gstnvtracker:: Failed to create batch context. Shutting down processing.\n");
    deInit();
    return false;
  }
  m_ProcessBatchThread = thread(&NvTrackerProc::processBatch, this);
  // A single batch, hence pad_index and seq_index will have a 1:1 correspondence
  for(uint32_t i = 0; i < m_Config.batchSize ; i++)
  {
      m_PadIndexSeqIndexMap[i] = i;
  }
  }
  else
  {
    for (uint32_t i =0; i < m_Config.subBatchesConfig.size(); i++)
    {
        /** This is a new sub-batch, create its thread and record dispatch info. */
        DispatchInfo *pDispatchInfo = new DispatchInfo;
        //DispatchInfo &newDispatchInfo = *pDispatchInfo;
        m_DispatchMap[i] = *pDispatchInfo;
        pDispatchInfo = &m_DispatchMap[i];
        pDispatchInfo->sbId = i;
        pDispatchInfo->running = true;
        /** This looks circular since thread is part of the info, but the thread
         * processing does not use this part so it's ok. */
        pDispatchInfo->sbThread = thread(&NvTrackerProc::processSubBatch, this, pDispatchInfo);

        // Create a mapping from pad index (source id) to seq_index for each sub batch
        // This is required since seq_index needs to range from 0 to (max batch size -1 )
        // where as, pad index of a stream in a sub batch may not satisfy this requirement
        std::vector<int> &subBatch =  m_Config.subBatchesConfig[i];
        for(uint32_t j = 0; j < subBatch.size(); j++)
        {
          int padIndex = subBatch[j];
          if (m_PadIndexSeqIndexMap.find(padIndex) != m_PadIndexSeqIndexMap.end())
          {
            LOG_ERROR("gstnvtracker:: Check the config \"sub-batches\"!!  Source id %d has re-occured!!", padIndex);
            deInit();
            return false;
          }
          m_PadIndexSeqIndexMap[padIndex] = j;
        }
    }
  }

  return true;
}

void NvTrackerProc::deInit()
{
  /** Clear out all pending process requests and return surface buffer
    * and notify all threads waiting for these requests */
  unique_lock<mutex> lkProc(m_ProcQueueLock);
  if (m_Config.numTransforms > 0 && m_Config.inputTensorMeta == false) {
    while (!m_ConvBufMgr.isQueueFull())
    {
      /** printf("m_ConvBufMgr.getFreeQueueSize() %d, m_ConvBufMgr.getActualPoolSize() %d\n",
        * m_ConvBufMgr.getFreeQueueSize(), m_ConvBufMgr.getActualPoolSize()); */
      m_BufQueueCond.wait(lkProc);
    }
  }
  if (m_Config.inputTensorMeta == false)
    deInitConvBufPool();

  /** Allow threads to finish */
  m_Running = false;

  m_ProcQueueCond.notify_all();
  lkProc.unlock();

  /** Wait for all process threads to finish */
  if (m_ProcessBatchThread.joinable()) {
    m_ProcessBatchThread.join();
  }

  {
    /** Shut down all sub-batch threads */
    unique_lock<mutex> lkDispatch(m_DispatchMapLock);
    std::map<SubBatchId, DispatchInfo>::iterator it;
    for (it = m_DispatchMap.begin(); it != m_DispatchMap.end();)
    {
      //SubBatchId sbId = it->first;
      DispatchInfo &dispatchInfo = it->second;
      dispatchInfo.running = false;
      dispatchInfo.reqQueueCond.notify_one();
      if (dispatchInfo.sbThread.joinable())
      {
        dispatchInfo.sbThread.join();
      }
      /** Remove this entry from the map */
      it = m_DispatchMap.erase(it);
    }
    m_DispatchMap.clear();
  }
  m_SsidSubBatchMap.clear();

  if (m_BatchContextHandle != nullptr) {
    m_TrackerLibDeInit(m_BatchContextHandle);
    m_BatchContextHandle = nullptr;
  }

  deInitMiscDataPool();

  /** Clear out all pending completions
    * and notify all threads waiting for completions */
  unique_lock<mutex> lkCompletion(m_CompletionQueueLock);
  m_CompletionQueue = queue<InputParams>();
  lkCompletion.unlock();
  m_CompletionQueueCond.notify_all();

  /** Clear out pending batch info */
  unique_lock<mutex> pendingBatchLock(m_PendingBatchLock);
  m_PendingBatch.clear();
  m_ConvBufComplete.clear();
  pendingBatchLock.unlock();

  /** Clean up resources */
  deInitTrackerLib();

  /** Clean up the class info map */
  m_ClassInfoMap.clear();
  /** Clean up the map from pad index to sub batch sequence index */
  m_PadIndexSeqIndexMap.clear();
}

bool NvTrackerProc::addSource(uint32_t sourceId)
{
  /** The sourceId alone is insufficient to do anything.
    * Defer processing to when frames start showing up from this source.
    * We will have the needed surface info then. */
  return true;
}

bool NvTrackerProc::removeSource(uint32_t sourceId, bool removeObjectIdMapping)
{
  if (m_Config.subBatchesConfig.size() == 0)
  {
  /** Clean up resources for this source. */
  if (m_TrackerLibRemoveStreams) {
    /** Given streamId, need to find all its SurfaceStreamIds */
    for (auto it = m_ObjectIdOffsetMap.begin(); it != m_ObjectIdOffsetMap.end(); it++)
    {
      SurfaceStreamId ssId = it->first;
      if (getStreamId(ssId) == sourceId)
      {
        NvMOTStatus status = m_TrackerLibRemoveStreams(m_BatchContextHandle, ssId);
        if (status != NvMOTStatus_OK)
        {
          LOG_ERROR("gstnvtracker: Fail to remove stream %d.\n", sourceId);
          m_TrackerLibError = true;
          return false;
        }
      }
    }
  }
  }
  else
  {
    SurfaceStreamId ssId;
    /** Given streamId, first fiind the sub batch containing the stream
     * and then instruct the sub-batch thread to remove the source */
    for (auto it = m_ObjectIdOffsetMap.begin(); it != m_ObjectIdOffsetMap.end(); it++)
    {
      ssId = it->first;
      if (getStreamId(ssId) == sourceId)
      {
        SubBatchId sbId;
        if (m_SsidSubBatchMap.find(ssId) == m_SsidSubBatchMap.end())
        {
          LOG_ERROR("removeSource : Could not find a sub batch contaning the Ssid = %lx\n", ssId);
          return false;
        }
        else
        {
          sbId = m_SsidSubBatchMap[ssId];
          /** Since the source is going to be removed, erase the entry from mapping too */
          auto it = m_SsidSubBatchMap.find(ssId);
          m_SsidSubBatchMap.erase(it);
        }
        DispatchReq req;
        req.removeSource = true;
        req.ssId = ssId;
        unique_lock<mutex> lk(m_DispatchMapLock);
        map<SubBatchId, DispatchInfo>::iterator it = m_DispatchMap.find(sbId);
        DispatchInfo *pDispatchInfo;
        if (it != m_DispatchMap.end())
        {
          pDispatchInfo = &it->second;
        }
        else
        {
          LOG_ERROR("gstnvtracker: Running thread not found corresponfing to sub-batch %d\n", sbId);
        }
        /** We can release the dispatch map lock now because a dispatch info is kept valid
         * until its thread joins. */
        lk.unlock();

        /** Queue up the req for stream removal */
        unique_lock<mutex> lkReq(pDispatchInfo->reqQueueLock);
        pDispatchInfo->reqQueue.push(req);
        lkReq.unlock();
        pDispatchInfo->reqQueueCond.notify_one();
      }
    }
  }

  if (removeObjectIdMapping)
  {
    for (auto it = m_ObjectIdOffsetMap.begin(); it != m_ObjectIdOffsetMap.end();)
    {
      if (getStreamId(it->first) == sourceId) /** SurfaceStreamId has source id to delete */
      {
        it = m_ObjectIdOffsetMap.erase(it);
      }
      else
      {
        it++;
      }
    }
  }
  return true;
}

bool NvTrackerProc::resetSource(uint32_t sourceId)
{
  /** Mark the next frame from this source as reset frame
    * so the low-level tracker lib can act accordingly.
    * Serialize with submitInput().
    * So far nothing needs doing */
  return true;
}

bool NvTrackerProc::submitInput(const InputParams& inputParams)
{
  if (m_TrackerLibError.load())
  {
    return false;
  }

  ProcParams procParams;

  char contextName[100];
  snprintf(contextName, sizeof(contextName), "%s_nvtracker_convert_buffer(Frame=%u)",
       m_Config.gstName, m_BatchId);
  nvtx_helper_push_pop(contextName);
  /** Convert to format/size required by low-level tracker lib. */
  procParams.pConvBuf = nullptr;
  if (m_Config.numTransforms > 0)
  {
    while (m_Running)
    {
      unique_lock<mutex> lk(m_ProcQueueLock);

      if (m_Config.inputTensorMeta==false)  {
        if (m_ConvBufMgr.isQueueEmpty()) {
          m_BufQueueCond.wait(lk);
        }
      }
      /** Check for shutdown signal or spurious wakeup */
      if (!m_Running)
      {
        break;
      }
      else if (m_ConvBufMgr.isQueueEmpty()  && m_Config.inputTensorMeta==false)
      {
        continue;
      }

      /** Buffer conversion is asynchronous. Call syncBufferSet before low level library accessing transformed surface */
      if (m_Config.inputTensorMeta == false) {
        procParams.pConvBuf = m_ConvBufMgr.convertBatchAsync(inputParams.pSurfaceBatch, &procParams.bufSetSyncObjs);
      }
      else {
        GstBuffer *conv_gst_buf = nullptr;
        NvDsBatchMeta *batch_meta = inputParams.pBatchMeta;
        GstMemory *mem;
        if (inputParams.pBatchMeta == nullptr) {
          LOG_ERROR("gstnvtracker: Batch meta is nullptr.\n");
          return false;
        }

        /** Storing all the preprocess batch info in single NvBufSurface*/
        NvBufSurface* preprocSurface =  new NvBufSurface;
        memset(preprocSurface, 0, sizeof(NvBufSurface));
        preprocSurface->batchSize = inputParams.pSurfaceBatch->batchSize;
        preprocSurface->numFilled = 0;
        preprocSurface->isContiguous = false;
        preprocSurface->surfaceList = new NvBufSurfaceParams[preprocSurface->batchSize];
        memset(preprocSurface->surfaceList, 0, preprocSurface->batchSize * sizeof(NvBufSurfaceParams));

        for (NvDsMetaList * l_user = batch_meta->batch_user_meta_list; l_user != NULL;
            l_user = l_user->next) {

          NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user->data);
          GstNvDsPreProcessMemory * preproc_buf = nullptr;

          if (user_meta->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META)
            continue;
          GstNvDsPreProcessBatchMeta *preproc_meta =
            (GstNvDsPreProcessBatchMeta *) user_meta->user_meta_data;
          const auto & uids = preproc_meta->target_unique_ids;
          if (std::find (uids.begin (), uids.end (),
                m_Config.tensorMetaGieId) == uids.end ())
            continue;

          if (!preproc_meta->private_data)
            continue;

          if (preproc_meta->tensor_meta->gpu_id != (guint)m_Config.gpuId) {
            LOG_ERROR("gstnvtracker: GPU ID mismatch of tensor meta and tracker.\n");
            return false;
          }
          conv_gst_buf = static_cast <GstBuffer*>(preproc_meta->private_data);
          mem = gst_buffer_peek_memory (conv_gst_buf, 0);
          if (!mem || !gst_memory_is_type (mem, GST_NVDSPREPROCESS_MEMORY_TYPE))
          {
            LOG_ERROR("gstnvtracker: preprocess memory is null.\n");
            return false;
          }
          preproc_buf =  &(((GstNvDsPreProcessMem *) mem)->mem_preprocess);
          if (procParams.pConvBuf == nullptr){
            preprocSurface->gpuId = preproc_buf->surf->gpuId;
            preprocSurface->memType = preproc_buf->surf->memType;
            /* Adding all the surfaces of the preprocess batch to surfaceList */
            for (uint32_t i=0; (i < preproc_buf->surf->batchSize) && (preprocSurface->numFilled < inputParams.pSurfaceBatch->batchSize); i++) {
              preprocSurface->surfaceList[preprocSurface->numFilled] = preproc_buf->surf->surfaceList[i];
              preprocSurface->numFilled += 1;
            }
          }
        }

        /* Copy the combined NvBufSurface info to pConvBuf */
        procParams.pConvBuf = preprocSurface;

        if (procParams.pConvBuf == nullptr){
          LOG_ERROR("gstnvtracker: no tensor meta found with buffer please"\
          " check tensor-meta-gie-id is set correctly or ensure preprocess plugin is used"\
          " when input-tensor-meta=1\n");
          return false;
        }

      }
      lk.unlock();
      break;
    }
  }
  nvtx_helper_push_pop(NULL);
  if (procParams.pConvBuf == nullptr && m_Config.numTransforms > 0)
  {
    LOG_ERROR("gstnvtracker: Failed to convert input batch.\n");
    return false;
  }
  procParams.useConvBuf = true;
  procParams.input = inputParams;

  if (m_Config.subBatchesConfig.size() == 0)
  {
    /** Track this batch as pending processing for event synchronization.
     * There is no need to record the streams here so the ssId vector is unused. */
    std::vector<SubBatchId> nextBatch;
    unique_lock<mutex> pendingBatchLock(m_PendingBatchLock);
    if (m_PendingBatch.find(m_BatchId) != m_PendingBatch.end())
    {
      LOG_ERROR("gstnvtracker: Batch %d already active!\n", m_BatchId);
      pendingBatchLock.unlock();
      return false;
    }
    m_PendingBatch[m_BatchId] = nextBatch;
    procParams.batchId = m_BatchId;
    m_BatchId++;
    pendingBatchLock.unlock();

    /** Queue up and signal the processing thread */
    unique_lock<mutex> lk(m_ProcQueueLock);
    m_ProcQueue.push(procParams);
    lk.unlock();
    m_ProcQueueCond.notify_one();
    return true;
  }
  else
  {
    /** The sub-batch dispatch function will also track this pending batch. */
    return dispatchSubBatches(procParams);
  }
}

CompletionStatus NvTrackerProc::waitForCompletion(InputParams& inputParams)
{
  while (m_Running)
  {
    unique_lock<mutex> lk(m_CompletionQueueLock);
    if (m_CompletionQueue.empty()) {
      m_CompletionQueueCond.wait(lk);
      /** Make sure this is not just a spurious wakeup */
      if (m_CompletionQueue.empty()) {
        continue;
      }
    }

    inputParams = m_CompletionQueue.front();
    m_CompletionQueue.pop();
    return CompletionStatus_OK;
  }

  return CompletionStatus_Exit;
}

bool NvTrackerProc::flushReqs()
{
  /** Wait for pending batches to clear. */
  unique_lock<mutex> pendingBatchLock(m_PendingBatchLock);
  while (!m_PendingBatch.empty())
  {
    pendingBatchLock.unlock();
    usleep(1000);
    pendingBatchLock.lock();
  }
  pendingBatchLock.unlock();

  /** All pending batches are cleared.
    * Insert flush marker completion to notify the caller of
    * waitForCompletion() */
  InputParams params;
  params.pSurfaceBatch = nullptr;
  params.pBatchMeta = nullptr;
  params.pPreservedData = nullptr;
  params.eventMarker = true;

  unique_lock<mutex> lkComp(m_CompletionQueueLock);
  m_CompletionQueue.push(params);
  lkComp.unlock();
  m_CompletionQueueCond.notify_one();

  return true;
}

/******************
 * Private methods
 *****************/

bool NvTrackerProc::initTrackerLib()
{
  char* error;

  LOG_INFO("gstnvtracker: Loading low-level lib at %s\n", m_Config.trackerLibFile);

  m_TrackerLibHandle = dlopen(m_Config.trackerLibFile, RTLD_NOW);
  if (!m_TrackerLibHandle) {
    LOG_ERROR("gstnvtracker: Failed to open low-level lib at %s\n", m_Config.trackerLibFile);
    if ((error = dlerror()) != NULL)
    {
      LOG_ERROR(" dlopen error: %s\n", error);
    }
    return false;
  }

  *(void **)(&m_TrackerLibInit) = dlsym(m_TrackerLibHandle, "NvMOT_Init");
  if ((error = dlerror()) != NULL)
  {
    LOG_ERROR("gstnvtracker: Failed to load NvMOT_Init\n");
    return false;
  }

  *(void **)(&m_TrackerLibDeInit) = dlsym(m_TrackerLibHandle, "NvMOT_DeInit");
  if ((error = dlerror()) != NULL)
  {
    LOG_ERROR("gstnvtracker: Failed to load NvMOT_DeInit\n");
    return false;
  }

  *(void **)(&m_TrackerLibProcess) = dlsym(m_TrackerLibHandle, "NvMOT_Process");
  if ((error = dlerror()) != NULL)
  {
    LOG_ERROR("gstnvtracker: Failed to load NvMOT_Process\n");
    return false;
  }

  *(void **)(&m_TrackerLibRetrieveMiscData) = dlsym(m_TrackerLibHandle, "NvMOT_RetrieveMiscData");
  if ((error = dlerror()) != NULL)
  {
    LOG_INFO("gstnvtracker: Optional NvMOT_RetrieveMiscData not implemented\n");
  }

  *(void **)(&m_TrackerLibRemoveStreams) = dlsym(m_TrackerLibHandle, "NvMOT_RemoveStreams");
  if ((error = dlerror()) != NULL)
  {
    LOG_INFO("gstnvtracker: Optional NvMOT_RemoveStreams not implemented\n");
  }

  /** Query the low-level tracker lib to complete the configs */
  *(void **)(&m_TrackerLibQuery) = dlsym(m_TrackerLibHandle, "NvMOT_Query");
  if ((error = dlerror()) != NULL)
  {
    LOG_WARNING("gstnvtracker: Failed to load NvMOT_Query. Using default parameters for now.\n");
  }
  else
  {
    NvMOTQuery query;
    /** Initialize NvMOTQuery with default values. */
    memset(&query, 0, sizeof(query));
    query.maxTargetsPerStream = 150;

    uint16_t configFilePathSize = 0;
    if (m_Config.trackerConfigFile != nullptr)
    {
      configFilePathSize = strlen(m_Config.trackerConfigFile);
    }
    NvMOTStatus status = m_TrackerLibQuery(configFilePathSize, m_Config.trackerConfigFile, &query);
    if (status != NvMOTStatus_OK)
    {
      LOG_ERROR("gstnvtracker: Got error querying low-level tracker for caps.\n");
      if (status == NvMOTStatus_Invalid_Path)
      {
        LOG_ERROR("  Config file path is invalid: %s\n",
              (m_Config.trackerConfigFile == nullptr) ? "nullptr" : m_Config.trackerConfigFile);
      }
      return false;
    }

    if (query.numTransforms > 1){
      LOG_ERROR("gstnvtracker: query.numTransforms must be 0 or 1\n");
      return false;
    }

    if (query.batchMode != NvMOTBatchMode_Batch)
    {
      LOG_ERROR("gstnvtracker: Only batch processing mode is supported. query.batchMode must be set as NvMOTBatchMode_Batch\n");
      return false;
    }

    if (m_Config.inputTensorMeta==true)  {
      query.colorFormats[0] = NVBUF_COLOR_FORMAT_RGBA;
      LOG_INFO("gstnvtracker: Forcing format RGBA for tracker \n");
    }

    m_Config.numTransforms = query.numTransforms;
    m_Config.computeTarget = query.computeConfig;
    m_Config.colorFormat = query.colorFormats[0];
    m_Config.memType = query.memType;
    m_Config.maxTargetsPerStream = query.maxTargetsPerStream;
    m_Config.maxShadowTrackingAge = query.maxShadowTrackingAge;
    m_Config.reidFeatureSize = query.reidFeatureSize;
    m_Config.outputReidTensor = query.outputReidTensor;
    m_Config.pastFrame = query.supportPastFrame;
    m_Config.outputTerminatedTracks = query.outputTerminatedTracks;
    m_Config.maxTrajectoryBufferLength = query.maxTrajectoryBufferLength;
    m_Config.outputShadowTracks = query.outputShadowTracks;
    m_Config.maxConvexHullSize = query.maxConvexHullSize;
    m_Config.outputVisibility = query.outputVisibility;
    m_Config.outputFootLocation = query.outputFootLocation;
    m_Config.outputConvexHull = query.outputConvexHull;
  }

  return true;
}

void NvTrackerProc::deInitTrackerLib()
{
  if (m_TrackerLibHandle)
  {
    dlclose(m_TrackerLibHandle);
  }
}

bool NvTrackerProc::initConvBufPool()
{
  NvBufSurfaceCreateParams createParams;
  if (m_Config.numTransforms > 0)
  {
    createParams.gpuId = m_Config.gpuId;
    createParams.width = m_Config.trackerWidth;
    createParams.height = m_Config.trackerHeight;
    createParams.size = 0;
    createParams.colorFormat = m_Config.colorFormat;
    createParams.layout = NVBUF_LAYOUT_PITCH;
    createParams.memType = m_Config.memType;
  }

  /** Create the pool of buffers for color/size-converted
    * frames for low-level tracker lib consumption. */
  bool ret = m_ConvBufMgr.init(m_Config.batchSize, m_Config.gpuId, m_Config.compute_hw,
    createParams, m_Config.numTransforms == 0);
  if (!ret)
  {
    LOG_ERROR("gstnvtracker: Failed to initialize ConvBufManager\n");
    return false;
  }

  m_Config.maxConvBufPoolSize = (uint32_t) (m_ConvBufMgr.getMaxPoolSize());
  return true;
}

void NvTrackerProc::deInitConvBufPool()
{
  m_ConvBufMgr.deInit();
}

bool NvTrackerProc::initMiscDataPool()
{
  bool ret = m_MiscDataMgr.init(m_Config.batchSize, m_Config.gpuId,
    m_Config.maxTargetsPerStream, m_Config.maxShadowTrackingAge,
    m_Config.reidFeatureSize, m_Config.maxMiscDataPoolSize, m_Config.pastFrame,
    m_Config.outputReidTensor,
    m_Config.outputTerminatedTracks,
    m_Config.outputShadowTracks,
    m_Config.maxTrajectoryBufferLength);

  if (!ret)
  {
    LOG_ERROR("gstnvtracker: Failed to initialize MiscDataManager\n");
    return false;
  }

  return true;
}

void NvTrackerProc::deInitMiscDataPool()
{
  m_MiscDataMgr.deInit();
}

NvMOTContextHandle NvTrackerProc::initTrackerContext(uint32_t sbId)
{
  NvMOTConfig config;
  NvMOTConfigResponse configResp;

  config.computeConfig = m_Config.computeTarget;

  //config.maxStreams = m_Config.batchSize;
  config.maxStreams = (m_Config.subBatchesConfig.size() == 0) ? m_Config.batchSize : (m_Config.subBatchesConfig[sbId]).size();
  config.maxBufSurfAddrSize = m_Config.maxConvBufPoolSize * m_Config.batchSize; /** max num of addresses in buffer pool */
  config.numTransforms = m_Config.numTransforms;

  NvMOTPerTransformBatchConfig batchConfig;
  memset(&batchConfig, 0, sizeof(batchConfig));
  config.perTransformBatchConfig = &batchConfig;

  if (config.numTransforms > 0)
  {
    batchConfig.bufferType = m_Config.memType;
    batchConfig.maxWidth = m_Config.trackerWidth;
    batchConfig.maxHeight = m_Config.trackerHeight;
    batchConfig.maxPitch = m_Config.trackerWidth * 2 * 4;
    batchConfig.maxSize = batchConfig.maxPitch * m_Config.trackerHeight;
    batchConfig.colorFormat = m_Config.colorFormat;
  }

  config.miscConfig.gpuId = m_Config.gpuId;
  config.miscConfig.maxObjPerStream = 0;
  config.miscConfig.maxObjPerBatch = 0;

  std::string customConfigFilePath = {};
  if(m_Config.trackerConfigFilePerSubBatch.size() > sbId){
     customConfigFilePath = m_Config.trackerConfigFilePerSubBatch[sbId];
  }
  else{
    if(m_Config.trackerConfigFilePerSubBatch.size())
    {
      /** If config file is not specified for this sub-batch, use the last config file in the list */
      customConfigFilePath = m_Config.trackerConfigFilePerSubBatch.back();
    }
    else
    {
      /** If no config file is specified, set the path as an empty string */
      customConfigFilePath = {};
    }
  }
  if(customConfigFilePath.empty()){
    LOG_WARNING("gstnvtracker: ll-config-file not specified for sub batch : %d\n", sbId);
  }
  config.customConfigFilePath = (char*)customConfigFilePath.c_str();
  if (config.customConfigFilePath != nullptr) {
    config.customConfigFilePathSize = strlen(customConfigFilePath.c_str());
  } else {
    config.customConfigFilePathSize = 0;
  }

  NvMOTContextHandle contextHandle = nullptr;
  NvMOTStatus status = m_TrackerLibInit (&config, &contextHandle, &configResp);
  if (NvMOTStatus_OK != status) {
    LOG_ERROR("gstnvtracker: Failed to initialize tracker context!\n");
  }

  return contextHandle;
}

void NvTrackerProc::queueFrames(const NvDsBatchMeta& batchMeta,
  std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>>& batchList)
{
  std::map<SurfaceStreamId, uint32_t> frameCount;

  for (NvDsFrameMetaList *l = batchMeta.frame_meta_list; l != nullptr; l = l->next)
  {
    NvDsFrameMeta *pFrameMeta = (NvDsFrameMeta*)(l->data);

    /** Skip surface types not selected for tracking */
    if ((pFrameMeta == nullptr) || (pFrameMeta->surface_type != NVDS_META_SURFACE_NONE &&
      m_Config.trackingSurfTypeFromConfig &&
       pFrameMeta->surface_type != m_Config.trackingSurfType) )
    {
      continue;
    }

    SurfaceStreamId ssId = getSurfaceStreamId(pFrameMeta->pad_index,
      pFrameMeta->surface_index);
    if (frameCount.find(ssId) == frameCount.end())
    {
      frameCount[ssId] = 0;
    }

    while (frameCount.at(ssId) >= batchList.size())
    {
      batchList.push_back({});
    }

    batchList[frameCount[ssId]++][ssId] = pFrameMeta;
  }
}

void NvTrackerProc::fillMOTFrame(SurfaceStreamId ssId,
                 const ProcParams& procParams,
                 const NvDsFrameMeta& frameMeta,
                 NvMOTFrame& motFrame,
                 NvMOTTrackedObjList& trackedObjList)
{
  uint32_t i = 0;
  NvMOTObjToTrackList *pObjList = &motFrame.objectsIn;

  NvBufSurfaceParams *pInputBuf = &procParams.input.pSurfaceBatch->surfaceList[frameMeta.batch_id];
  float scaleWidth = ((float)m_Config.trackerWidth/pInputBuf->width);
  float scaleHeight = ((float)m_Config.trackerHeight/pInputBuf->height);

  pObjList->numFilled = 0;
  NvDsObjectMetaList *l = NULL;
  NvDsObjectMeta *objectMeta = NULL;
  NvMOTObjToTrack *pObjs = pObjList->list;

  for (i = 0, l = frameMeta.obj_meta_list;
     i < pObjList->numAllocated && l != NULL;
     i++, l = l->next)
  {
    objectMeta = (NvDsObjectMeta *)(l->data);
    NvOSD_RectParams *rectParams = &objectMeta->rect_params;

    pObjs[i].bbox.x = rectParams->left * scaleWidth;
    pObjs[i].bbox.y = rectParams->top * scaleHeight;
    pObjs[i].bbox.width = rectParams->width * scaleWidth;
    pObjs[i].bbox.height = rectParams->height * scaleHeight;
    pObjs[i].classId = objectMeta->class_id;
    pObjs[i].confidence = objectMeta->confidence;
    pObjs[i].doTracking = true;
    pObjs[i].pPreservedData = objectMeta;
    pObjList->numFilled++;

    /** Store info for this class if previously unseen. */
    auto it = m_ClassInfoMap.find(objectMeta->class_id);
    if (it == m_ClassInfoMap.end())
    {
      ClassInfo newClassInfo;
      m_ClassInfoMap[objectMeta->class_id] = newClassInfo;

      ClassInfo &classInfo = m_ClassInfoMap[objectMeta->class_id];
      classInfo.rectParams = *rectParams;
      classInfo.textParams = objectMeta->text_params;
      classInfo.uniqueComponentId = objectMeta->unique_component_id;
      if (objectMeta->text_params.display_text != nullptr) {
        classInfo.displayTextString = objectMeta->text_params.display_text;
      }
      if (objectMeta->obj_label[0] != '\0') {
        classInfo.objLabel = objectMeta->obj_label;
      }
    }
  }
  pObjList->detectionDone = frameMeta.bInferDone ? true : false;

  motFrame.streamID = ssId;
  motFrame.seq_index = m_PadIndexSeqIndexMap[frameMeta.pad_index];
  motFrame.frameNum = frameMeta.frame_num;
  motFrame.srcFrameWidth = pInputBuf->width;
  motFrame.srcFrameHeight = pInputBuf->height;
  /**motFrame.timeStamp = frameMeta.ntp_timestamp; */
  motFrame.timeStampValid = false;
  motFrame.doTracking = true;
  motFrame.reset = false;
  if (procParams.pConvBuf != nullptr) {
    motFrame.numBuffers = 1;
    *(motFrame.bufferList) = procParams.pConvBuf->surfaceList + frameMeta.batch_id;
  } else {
    motFrame.numBuffers = 0;
  }

  trackedObjList.streamID = ssId;
  trackedObjList.frameNum = frameMeta.frame_num;
}

bool NvTrackerProc::clipBBox(uint32_t frameWidth, uint32_t frameHeight,
               float &left, float &top, float &width, float &height)
{
  if (left < 0)
  {
    width += left;
    left = 0;
  }

  if (top < 0)
  {
    height += top;
    top = 0;
  }

  if (left >= frameWidth || top >= frameHeight ||
    width <= 0 || height <= 0)
  {
    return false;
  }

  /** Clip bbox to be within the frame. */
  if (left + width > frameWidth)
  {
    width = frameWidth - left;
  }
  if (top + height > frameHeight)
  {
    height = frameHeight - top;
  }
  return true;
}

/**
 * Attach past frame object data to batch meta.
 */
void NvTrackerProc::updatePastFrameMeta(
                    const std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>>& batchList,
                    GstNvTrackerMiscDataObject *pGstObj,
                    ProcParams& procParams)
{
  if (!m_Config.pastFrame)
  {
    return;
  }

  NvDsTargetMiscDataBatch *pPastFrameObjBatch = &pGstObj->misc_data_buffer->pastFrameObjBatch;
  if (!batchList.empty())
  {
    const std::map<SurfaceStreamId, NvDsFrameMeta *>& frameMap = batchList.front();
    for (uint32_t si = 0; si < pPastFrameObjBatch->numFilled; si++) {
      /** Iterate to assign plugin-level info to past frame data, inclusing
        * stream ID, class names and box rescale */
      NvDsTargetMiscDataStream *pPastFrameObjStream = pPastFrameObjBatch->list + si;
      SurfaceStreamId ssId = pPastFrameObjStream->surfaceStreamID;
      pPastFrameObjStream->streamID = getStreamId(ssId);
      if (frameMap.find(ssId) == frameMap.end())
      {
        LOG_WARNING("gstnvtracker: past frame data contains wrong stream ID");
        continue;
      }
      NvBufSurfaceParams *pInputBuf = &procParams.input.pSurfaceBatch->surfaceList[frameMap.at(ssId)->batch_id];
      float scaleWidth = (float)pInputBuf->width / m_Config.trackerWidth;
      float scaleHeight = (float)pInputBuf->height / m_Config.trackerHeight;

      for (uint32_t li = 0; li < pPastFrameObjStream->numFilled; li++) {
        NvDsTargetMiscDataObject *pPastFrameObjList = pPastFrameObjStream->list + li;
        g_strlcpy(pPastFrameObjList->objLabel,
          m_ClassInfoMap[pPastFrameObjList->classId].objLabel.c_str(), MAX_LABEL_SIZE);
        for (uint32_t oi = 0; oi < pPastFrameObjList->numObj; oi++) {
          /** Reshape bounding boxes from tracker scale to buffer scale */
          NvDsTargetMiscDataFrame *pPastFrameObj = pPastFrameObjList->list + oi;
          pPastFrameObj->tBbox.left *= scaleWidth;
          pPastFrameObj->tBbox.width *= scaleWidth;
          pPastFrameObj->tBbox.top *= scaleHeight;
          pPastFrameObj->tBbox.height *= scaleHeight;
        }
      }
    }
  }

  /** Attach past frame data to user meta */
  pPastFrameObjBatch->priv_data = (void *) gst_mini_object_ref((GstMiniObject *) pGstObj);

  NvDsBatchMeta *pBatchMeta = procParams.input.pBatchMeta;
  NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (pBatchMeta);
  user_meta->user_meta_data = (void *)pPastFrameObjBatch;
  user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_TRACKER_PAST_FRAME_META;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_nvtracker2_past_frame_meta;
  user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) free_nvtracker2_past_frame_meta;
  user_meta->base_meta.batch_meta = pBatchMeta;
  nvds_add_user_meta_to_batch (pBatchMeta, user_meta);
}

void NvTrackerProc::updateBatchReidMeta(GstNvTrackerMiscDataObject *pGstObj,
  ProcParams& procParams)
{
  if (!m_Config.outputReidTensor)
  {
    return;
  }

  NvDsReidTensorBatch *pReidTensor = &pGstObj->misc_data_buffer->reidTensorBatch;
  pReidTensor->priv_data = (void *) gst_mini_object_ref((GstMiniObject *) pGstObj);

  NvDsBatchMeta *pBatchMeta = procParams.input.pBatchMeta;
  NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (pBatchMeta);
  user_meta->user_meta_data = (void *)pReidTensor;
  user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_TRACKER_BATCH_REID_META;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_nvtracker2_batch_reid_meta;
  user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) free_nvtracker2_batch_reid_meta;
  user_meta->base_meta.batch_meta = pBatchMeta;
  nvds_add_user_meta_to_batch (pBatchMeta, user_meta);
}

void NvTrackerProc::updateTerminatedTrackMeta(GstNvTrackerMiscDataObject *pGstObj, ProcParams& procParams)
{
  if (!m_Config.outputTerminatedTracks)
  {
    return;
  }

  NvDsTargetMiscDataBatch *pTerminatedListBatch = &pGstObj->misc_data_buffer->terminatedTrackBatch;
  if (pTerminatedListBatch && pTerminatedListBatch->list)
  {
    for (uint32_t si = 0; si < pTerminatedListBatch->numFilled; si++)
    {
        NvDsTargetMiscDataStream *pTerminatedListStream = pTerminatedListBatch->list + si;
        SurfaceStreamId ssId = pTerminatedListStream->surfaceStreamID;
        pTerminatedListStream->streamID = getStreamId(ssId);
    }
  }

  pTerminatedListBatch->priv_data = (void *) gst_mini_object_ref((GstMiniObject *) pGstObj);

  NvDsBatchMeta *pBatchMeta = procParams.input.pBatchMeta;
  NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (pBatchMeta);
  user_meta->user_meta_data = (void *)pTerminatedListBatch;
  user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_TRACKER_TERMINATED_LIST_META;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_nvtracker2_terminated_meta;
  user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) free_nvtracker2_terminated_meta;
  user_meta->base_meta.batch_meta = pBatchMeta;
  nvds_add_user_meta_to_batch (pBatchMeta, user_meta);
}


void NvTrackerProc::updateShadowTrackMeta(GstNvTrackerMiscDataObject *pGstObj, ProcParams& procParams)
{
  if (!m_Config.outputShadowTracks)
  {
    return;
  }

  NvDsTargetMiscDataBatch *pShadowTrackBatch = &pGstObj->misc_data_buffer->shadowTracksBatch;
  if (pShadowTrackBatch && pShadowTrackBatch->list)
  {
    for (uint32_t si = 0; si < pShadowTrackBatch->numFilled; si++)
    {
        NvDsTargetMiscDataStream *pShadowTrackStream = pShadowTrackBatch->list + si;
        SurfaceStreamId ssId = pShadowTrackStream->surfaceStreamID;
        pShadowTrackStream->streamID = getStreamId(ssId);

    }
  }

  pShadowTrackBatch->priv_data = (void *) gst_mini_object_ref((GstMiniObject *) pGstObj);

  NvDsBatchMeta *pBatchMeta = procParams.input.pBatchMeta;
  NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (pBatchMeta);
  user_meta->user_meta_data = (void *)pShadowTrackBatch;
  user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_TRACKER_SHADOW_LIST_META;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_nvtracker2_shadowTracks_meta;
  user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) free_nvtracker2_shadowTracks_meta;
  user_meta->base_meta.batch_meta = pBatchMeta;
  nvds_add_user_meta_to_batch (pBatchMeta, user_meta);
}

void NvTrackerProc::updateUserMeta(
           const std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>>& batchList,
           ProcParams& procParams, NvTrackerMiscDataBuffer *pMiscDataBuf)
{
  if (pMiscDataBuf == nullptr)
  {
    return;
  }

  if (m_TrackerLibError.load())
  {
    m_MiscDataMgr.returnBuffer(pMiscDataBuf);
    return;
  }

  GstNvTrackerMiscDataObject *pGstObj = new GstNvTrackerMiscDataObject;
  pGstObj->misc_data_buffer = pMiscDataBuf;
  pGstObj->misc_data_manager = &m_MiscDataMgr;

  gst_mini_object_init (GST_MINI_OBJECT (pGstObj), 0, G_TYPE_POINTER,
    NULL, NULL, return_nvtracker_buffer_meta);
  updatePastFrameMeta(batchList, pGstObj, procParams);
  updateBatchReidMeta(pGstObj, procParams);
  updateTerminatedTrackMeta(pGstObj, procParams);
  updateShadowTrackMeta(pGstObj, procParams);

  /** mini_obj_ref number is alreadt 1 after initialization, so descrease by 1 here. */
  gst_mini_object_unref (GST_MINI_OBJECT(pGstObj));
}

void NvTrackerProc::updateUserMeta(
           const std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>>& batchList,
           ProcParams& procParams, NvTrackerMiscDataBuffer *pMiscDataBuf, TrackerMiscDataManager& miscDataMgr)
{
  if (pMiscDataBuf == nullptr)
  {
    return;
  }

  if (m_TrackerLibError.load())
  {
    miscDataMgr.returnBuffer(pMiscDataBuf);
    return;
  }

  GstNvTrackerMiscDataObject *pGstObj = new GstNvTrackerMiscDataObject;
  pGstObj->misc_data_buffer = pMiscDataBuf;
  pGstObj->misc_data_manager = &miscDataMgr;

  gst_mini_object_init (GST_MINI_OBJECT (pGstObj), 0, G_TYPE_POINTER,
    NULL, NULL, return_nvtracker_buffer_meta);
  updatePastFrameMeta(batchList, pGstObj, procParams);
  updateBatchReidMeta(pGstObj, procParams);
  updateTerminatedTrackMeta(pGstObj, procParams);
  updateShadowTrackMeta(pGstObj, procParams);

  /** mini_obj_ref number is alreadt 1 after initialization, so descrease by 1 here. */
  gst_mini_object_unref (GST_MINI_OBJECT(pGstObj));
}

void NvTrackerProc::updateBatchMeta(const NvMOTTrackedObjBatch& procResult,
                  const ProcParams& procParams,
                  std::map<SurfaceStreamId, NvDsFrameMeta *>& frameMap)
{
  /* Update metadata with output. */
  for (uint32_t i = 0; i < procResult.numFilled; i++)
  {
    const NvMOTTrackedObjList& trackedObjList = procResult.list[i];
    SurfaceStreamId ssId = trackedObjList.streamID;
    if (frameMap.find(ssId) == frameMap.end())
    {
      LOG_ERROR("gstnvtracker: Got output for unknown stream %lx\n", ssId);
      continue;
    }
    NvDsFrameMeta *pFrameMeta = frameMap[ssId];
    updateFrameMeta(pFrameMeta, trackedObjList, procParams);
  } /* Loop over trackedObjBatch. */

  for (auto it = frameMap.begin(); it != frameMap.end(); ++it)
  {
    NvDsFrameMeta *pFrameMeta = it->second;
    removeUntrackedObjects(pFrameMeta);
  }
}

void NvTrackerProc::updateFrameMeta(NvDsFrameMeta* pFrameMeta,
                  const NvMOTTrackedObjList& trackedObjList,
                  const ProcParams& procParams)
{
  NvDsBatchMeta *pBatchMeta = procParams.input.pBatchMeta;
  NvBufSurfaceParams *pInputBuf = &procParams.input.pSurfaceBatch->surfaceList[pFrameMeta->batch_id];
  float scaleWidth = ((float)pInputBuf->width/m_Config.trackerWidth);
  float scaleHeight = ((float)pInputBuf->height/m_Config.trackerHeight);

  for (uint32_t i = 0; i < trackedObjList.numFilled; i++)
  {
    /** Find the input object associated with this tracked object
      * If none is found, then this is tracked from previous frames and possibly missed
      * by the detector in this frame. */
    NvMOTTrackedObj *pTrackedObj = &trackedObjList.list[i];
    NvMOTObjToTrack *pInObj = pTrackedObj->associatedObjectIn;
    NvDsObjectMeta *pObjectMeta;
    NvOSD_RectParams *pRectParams;
    NvOSD_TextParams *pTextParams = nullptr;

    /** Save unclipped boxes from low-level tracker */
    float left_unclipped = pTrackedObj->bbox.x * scaleWidth;
    float top_unclipped = pTrackedObj->bbox.y * scaleHeight;
    float width_unclipped = pTrackedObj->bbox.width * scaleWidth;
    float height_unclipped = pTrackedObj->bbox.height * scaleHeight;

    /** Clipping boxes */
    float left = left_unclipped;
    float top = top_unclipped;
    float width = width_unclipped;
    float height = height_unclipped;
    bool clipRet = clipBBox(pInputBuf->width, pInputBuf->height, left, top, width, height);
    if (NULL != pInObj)
    {
      pObjectMeta = (NvDsObjectMeta*)pInObj->pPreservedData;
      pRectParams = &pObjectMeta->rect_params;
      pTextParams = &pObjectMeta->text_params;
      pObjectMeta->object_id = objectIdMapping(pTrackedObj->trackingId, trackedObjList.streamID);
      if (!clipRet) {
        nvds_remove_obj_meta_from_frame(pFrameMeta, pObjectMeta);
        pTextParams = nullptr;
      }
      else if (pObjectMeta->class_id != pTrackedObj->classId) {
        LOG_WARNING("gstnvtracker: obj %lu Class mismatch! %d -> %d\n",
              pObjectMeta->object_id,
              pObjectMeta->class_id, pTrackedObj->classId);
      } else {
        pRectParams->left = left;
        pRectParams->top = top;
        pRectParams->width = width;
        pRectParams->height = height;
        /** Fill tracker_bbox_info with unclipped bbox */
        pObjectMeta->tracker_bbox_info.org_bbox_coords.left = left_unclipped;
        pObjectMeta->tracker_bbox_info.org_bbox_coords.top = top_unclipped;
        pObjectMeta->tracker_bbox_info.org_bbox_coords.width = width_unclipped;
        pObjectMeta->tracker_bbox_info.org_bbox_coords.height = height_unclipped;
        /** Fill tracker_confidence from low-level tracker */
        pObjectMeta->tracker_confidence = pTrackedObj->confidence;
        /** Fill tracker reid meta. */
        updateObjectReidMeta(pObjectMeta, pTrackedObj, pBatchMeta);
        updateObjectProjectionMeta(pObjectMeta, pTrackedObj, pBatchMeta, scaleWidth,
          scaleHeight);
      }
    } else if (clipRet) {
      /** Need to add this object to metadata */
      pObjectMeta = nvds_acquire_obj_meta_from_pool(pBatchMeta);
      pRectParams = &pObjectMeta->rect_params;
      pTextParams = &pObjectMeta->text_params;

      /** First fill in unknown parts of rect_params and text_params from
        * Class Info Map. */
      auto it = m_ClassInfoMap.find(pTrackedObj->classId);
      if (it != m_ClassInfoMap.end())
      {
        *pRectParams = it->second.rectParams;
        *pTextParams = it->second.textParams;
        pTextParams->display_text = strdup(it->second.displayTextString.c_str());
        pObjectMeta->unique_component_id = it->second.uniqueComponentId;
        /** Fill in obj_label with the cached class label */
        g_strlcpy(pObjectMeta->obj_label, it->second.objLabel.c_str(), MAX_LABEL_SIZE);
      }


      /** Set detector_bbox_info as 0 */
      pObjectMeta->detector_bbox_info.org_bbox_coords.left = 0.0;
      pObjectMeta->detector_bbox_info.org_bbox_coords.top = 0.0;
      pObjectMeta->detector_bbox_info.org_bbox_coords.width = 0.0;
      pObjectMeta->detector_bbox_info.org_bbox_coords.height = 0.0;
      /** Set detector confidence as -0.1 */
      pObjectMeta->confidence = -0.1;

      /** Update the parts returned by the low-level tracker */
      pObjectMeta->class_id = pTrackedObj->classId;
      pObjectMeta->object_id = objectIdMapping(pTrackedObj->trackingId, trackedObjList.streamID);
      pRectParams->left = left;
      pRectParams->top = top;
      pRectParams->width = width;
      pRectParams->height = height;
      /** Set tracker_bbox_info with unclipped box */
      pObjectMeta->tracker_bbox_info.org_bbox_coords.left = left_unclipped;
      pObjectMeta->tracker_bbox_info.org_bbox_coords.top = top_unclipped;
      pObjectMeta->tracker_bbox_info.org_bbox_coords.width = width_unclipped;
      pObjectMeta->tracker_bbox_info.org_bbox_coords.height = height_unclipped;
      /** Set tracker_confidence from low-level tracker */
      pObjectMeta->tracker_confidence = pTrackedObj->confidence;
      /** Fill tracker reid meta. */
      updateObjectReidMeta(pObjectMeta, pTrackedObj, pBatchMeta);
      updateObjectProjectionMeta(pObjectMeta, pTrackedObj, pBatchMeta, scaleWidth,
          scaleHeight);
      nvds_add_obj_meta_to_frame(pFrameMeta, pObjectMeta, NULL);
    }

    /** Set display text to support gst-launch type of app that
      * does not compose its own display text. */
    const int MAX_GST_STR_LEN = 128;
    char displayText[MAX_GST_STR_LEN] = {0};
    int offset = 0;

    if(pTextParams)
    {
      pTextParams->font_params.font_name = (char*)"Serif";
      pTextParams->font_params.font_size = 10;
      pTextParams->font_params.font_color.red = 1.0;
      pTextParams->font_params.font_color.green = 1.0;
      pTextParams->font_params.font_color.blue = 1.0;
      pTextParams->font_params.font_color.alpha = 1.0;

      pTextParams->set_bg_clr = 1;
      pTextParams->text_bg_clr = (NvOSD_ColorParams) {
        0, 0, 0, 1};

      if (m_Config.displayTrackingId) {
        offset = snprintf(displayText, MAX_GST_STR_LEN, "%s ", (char *)pTextParams->display_text);
        free(pTextParams->display_text);

        offset = snprintf(displayText + offset, MAX_GST_STR_LEN, "%lu", pObjectMeta->object_id);
        pTextParams->display_text = g_strdup (displayText);
      }
      pTextParams->x_offset = pRectParams->left;
      pTextParams->y_offset = pRectParams->top - 20;
    }
  }
}

void NvTrackerProc::updateObjectReidMeta(NvDsObjectMeta *pObjectMeta, NvMOTTrackedObj *pTrackedObj,
            NvDsBatchMeta *pBatchMeta)
{
  if (m_Config.outputReidTensor && pTrackedObj->reidInd >= 0)
  {
      NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (pBatchMeta);
      int32_t* pReidInd = new int32_t;
      *pReidInd = pTrackedObj->reidInd;
      user_meta->user_meta_data = (void *)pReidInd;
      user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_TRACKER_OBJ_REID_META;
      user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_nvtracker2_obj_reid_meta;
      user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) free_nvtracker2_obj_reid_meta;
      nvds_add_user_meta_to_obj (pObjectMeta, user_meta);
  }
}

void NvTrackerProc::updateObjectProjectionMeta(NvDsObjectMeta *pObjectMeta, NvMOTTrackedObj *pTrackedObj,
            NvDsBatchMeta *pBatchMeta, float scaleWidth, float scaleHeight)
{
  if (m_Config.outputVisibility)
  {
      NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (pBatchMeta);
      float* pVis = new float;
      *pVis = pTrackedObj->visibility;
      user_meta->user_meta_data = (void *)pVis;
      user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_OBJ_VISIBILITY;
      user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_nvtracker2_obj_visibility_meta;
      user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) free_nvtracker2_obj_visibility_meta;
      nvds_add_user_meta_to_obj (pObjectMeta, user_meta);
  }
  if (m_Config.outputFootLocation)
  {
    {
      /** Output estimated foot location in frame coordinates. */
      NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (pBatchMeta);
      float* pPtImgFeet = new float [2];
      pPtImgFeet[0] = pTrackedObj->ptImgFeet[0] * scaleWidth;
      pPtImgFeet[1] = pTrackedObj->ptImgFeet[1] * scaleHeight;
      user_meta->user_meta_data = (void *)pPtImgFeet;
      user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_OBJ_IMAGE_FOOT_LOCATION;
      user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_nvtracker2_obj_foot_loc_meta;
      user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) free_nvtracker2_obj_foot_loc_meta;
      nvds_add_user_meta_to_obj (pObjectMeta, user_meta);
    }

    {
      /** Output estimated foot location in 2D coordinates on the estimated world ground. */
      NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (pBatchMeta);
      float* pPtWorldFeet = new float [2];
      pPtWorldFeet[0] = pTrackedObj->ptWorldFeet[0];
      pPtWorldFeet[1] = pTrackedObj->ptWorldFeet[1];
      user_meta->user_meta_data = (void *)pPtWorldFeet;
      user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_OBJ_WORLD_FOOT_LOCATION;
      user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_nvtracker2_obj_foot_loc_meta;
      user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) free_nvtracker2_obj_foot_loc_meta;
      nvds_add_user_meta_to_obj (pObjectMeta, user_meta);
    }
  }
  if (m_Config.outputConvexHull && pTrackedObj->convexHull.numPointsAllocated != 0)
  {
      NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (pBatchMeta);
      NvDsObjConvexHull* pConvexHull = new NvDsObjConvexHull;
      pConvexHull->list = new int [pTrackedObj->convexHull.numPointsAllocated * 2];
      for (uint32_t i=0; i < pTrackedObj->convexHull.numPoints; i++)
      {
        pConvexHull->list[2*i] = (int) (pTrackedObj->convexHull.list[2*i] * scaleWidth);
        pConvexHull->list[2*i+1] = (int) (pTrackedObj->convexHull.list[2*i+1] * scaleHeight);
      }
      pConvexHull->numPoints = pTrackedObj->convexHull.numPoints;
      pConvexHull->numPointsAllocated = pTrackedObj->convexHull.numPointsAllocated;
      user_meta->user_meta_data = (void *)pConvexHull;
      user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_OBJ_IMAGE_CONVEX_HULL;
      user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_nvtracker2_obj_convex_hull_meta;
      user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) free_nvtracker2_obj_convex_hull_meta;
      nvds_add_user_meta_to_obj (pObjectMeta, user_meta);
  }
}

void NvTrackerProc::removeUntrackedObjects(NvDsFrameMeta* pFrameMeta)
{
  vector<NvDsObjectMeta*> removalList;
  uint32_t removalCount = 0;
  NvDsObjectMetaList *l = NULL;
  NvDsObjectMeta *pObjectMeta = NULL;

  /** Go through the object metas to collect untracked objects for removal */
  for (l = pFrameMeta->obj_meta_list; l != NULL; l = l->next)
  {
    pObjectMeta = (NvDsObjectMeta *)(l->data);
    if (UNTRACKED_OBJECT_ID == pObjectMeta->object_id) {
      removalList.push_back(pObjectMeta);
    }
  }

  /** Remove the untracked objects. This should be a short list. */
  removalCount = removalList.size();
  for (uint32_t i = 0; i < removalCount; i++)
  {
    nvds_remove_obj_meta_from_frame(pFrameMeta, removalList[i]);
  }
}

guint64 NvTrackerProc::objectIdMapping(const guint64& objectId, const SurfaceStreamId& ssId)
{
  /** If a new stream added or an existing stream reaches EOS,
    * record the first new object id as offset.
    * Still keeps ssId for stream removal when don't reset object id */
  if (m_ObjectIdOffsetMap.find(ssId) == m_ObjectIdOffsetMap.end())
  {
    m_ObjectIdOffsetMap[ssId] = objectId;
  }

  if (m_Config.trackingIdResetMode & TrackingIdResetMode_FromZeroAfterEOS)
  {
    /** New id higher 32 bits unchanged, lower 32 bits reduces by an offset */
    return ( objectId & 0xffffffff00000000 ) + (guint32)(objectId - m_ObjectIdOffsetMap.at(ssId));
  }
  else
  {
    return objectId;
  }
}

void NvTrackerProc::allocateConvexHullMemory(NvMOTTrackedObj* list, const uint32_t numAllocated)
{
  if ( (!m_Config.outputConvexHull) || m_Config.maxConvexHullSize == 0 || list == nullptr)
  {
    return;
  }

  for (uint32_t i = 0; i < numAllocated; i++)
  {
    list[i].convexHull.list = new int [2 * m_Config.maxConvexHullSize];
    list[i].convexHull.numPointsAllocated = m_Config.maxConvexHullSize;
    list[i].convexHull.numPoints = 0;
  }
}

void NvTrackerProc::releaseConvexHullMemory(NvMOTTrackedObj* list, const uint32_t numAllocated)
{
  if ( (!m_Config.outputConvexHull) || m_Config.maxConvexHullSize == 0 || list == nullptr)
  {
    return;
  }

  for (uint32_t i = 0; i < numAllocated; i++)
  {
    delete [] (list[i].convexHull.list);
    list[i].convexHull.numPoints = 0;
  }
}

void NvTrackerProc::allocateProcessMemory(NvMOTProcessParams& procInput,
  NvMOTTrackedObjBatch& procResult)
{
  procInput.frameList = new NvMOTFrame[m_Config.batchSize];
  if (procInput.frameList == nullptr)
  {
    LOG_ERROR("gstnvtracker: Failed to allocate resources (frame) for batch processing.\n");
    m_TrackerLibError = true;
  }
  else
  {
    for (uint32_t i = 0; i < m_Config.batchSize; i++)
    {
      procInput.frameList[i].objectsIn.list = new NvMOTObjToTrack[m_Config.maxTargetsPerStream];
      procInput.frameList[i].bufferList = new (NvBufSurfaceParams*);
      if (procInput.frameList[i].objectsIn.list == nullptr ||
        procInput.frameList[i].bufferList == nullptr)
      {
        LOG_ERROR("gstnvtracker: Failed to allocate resources (objToTrack) for batch processing.\n");
        m_TrackerLibError = true;
      }
      procInput.frameList[i].objectsIn.numAllocated = m_Config.maxTargetsPerStream;
    }
  }

  /** Allocate memory for low level library output*/
  procResult.list = new NvMOTTrackedObjList[m_Config.batchSize];
  if (procResult.list == nullptr)
  {
    LOG_ERROR("gstnvtracker: Failed to allocate resources (trackedObjList) for batch processing.\n");
    m_TrackerLibError = true;
  }
  else
  {
    for (uint32_t i = 0; i < m_Config.batchSize; i++)
    {
      procResult.list[i].list = new NvMOTTrackedObj[m_Config.maxTargetsPerStream];
      if (procResult.list[i].list == nullptr)
      {
        LOG_ERROR("gstnvtracker: Failed to allocate resources (trackedObj) for batch processing.\n");
        m_TrackerLibError = true;
      }
      procResult.list[i].numAllocated = m_Config.maxTargetsPerStream;
      allocateConvexHullMemory(procResult.list[i].list, procResult.list[i].numAllocated);
    }
  }

  procResult.pReidTensorBatch = nullptr;
}

void NvTrackerProc::releaseProcessMemory(NvMOTProcessParams& procInput,
                        NvMOTTrackedObjBatch& procResult)
{
  if (procInput.frameList != nullptr)
  {
    for (uint32_t i = 0; i < m_Config.batchSize; i++)
    {
      if (procInput.frameList[i].objectsIn.list != nullptr)
      {
        delete [] (procInput.frameList[i].objectsIn.list);
      }
      if (procInput.frameList[i].bufferList != nullptr)
      {
        delete procInput.frameList[i].bufferList;
      }
    }
    delete [] (procInput.frameList);
  }

  /** Release output memory */
  if (procResult.list != nullptr)
  {
    for (uint32_t i = 0; i < m_Config.batchSize; i++)
    {
      releaseConvexHullMemory(procResult.list[i].list, procResult.list[i].numAllocated);
      if (procResult.list[i].list != nullptr) {
        delete [] (procResult.list[i].list);
      }
    }
    delete [] (procResult.list);
  }
}

void NvTrackerProc::allocateProcessMemory(NvMOTProcessParams& procInput,
  NvMOTTrackedObjBatch& procResult, uint32_t batchSize)
{
  procInput.frameList = new NvMOTFrame[batchSize];
  if (procInput.frameList == nullptr)
  {
    LOG_ERROR("gstnvtracker: Failed to allocate resources (frame) for batch processing.\n");
    m_TrackerLibError = true;
  }
  else
  {
    for (uint32_t i = 0; i < batchSize; i++)
    {
      procInput.frameList[i].objectsIn.list = new NvMOTObjToTrack[m_Config.maxTargetsPerStream];
      procInput.frameList[i].bufferList = new (NvBufSurfaceParams*);
      if (procInput.frameList[i].objectsIn.list == nullptr ||
        procInput.frameList[i].bufferList == nullptr)
      {
        LOG_ERROR("gstnvtracker: Failed to allocate resources (objToTrack) for batch processing.\n");
        m_TrackerLibError = true;
      }
      procInput.frameList[i].objectsIn.numAllocated = m_Config.maxTargetsPerStream;
    }
  }

  /** Allocate memory for low level library output*/
  procResult.list = new NvMOTTrackedObjList[batchSize];
  if (procResult.list == nullptr)
  {
    LOG_ERROR("gstnvtracker: Failed to allocate resources (trackedObjList) for batch processing.\n");
    m_TrackerLibError = true;
  }
  else
  {
    for (uint32_t i = 0; i < batchSize; i++)
    {
      procResult.list[i].list = new NvMOTTrackedObj[m_Config.maxTargetsPerStream];
      if (procResult.list[i].list == nullptr)
      {
        LOG_ERROR("gstnvtracker: Failed to allocate resources (trackedObj) for batch processing.\n");
        m_TrackerLibError = true;
      }
      procResult.list[i].numAllocated = m_Config.maxTargetsPerStream;
      allocateConvexHullMemory(procResult.list[i].list, procResult.list[i].numAllocated);
    }
  }

  procResult.pReidTensorBatch = nullptr;
}

void NvTrackerProc::releaseProcessMemory(NvMOTProcessParams& procInput,
                        NvMOTTrackedObjBatch& procResult, uint32_t batchSize)
{
  if (procInput.frameList != nullptr)
  {
    for (uint32_t i = 0; i < batchSize; i++)
    {
      if (procInput.frameList[i].objectsIn.list != nullptr)
      {
        delete [] (procInput.frameList[i].objectsIn.list);
      }
      if (procInput.frameList[i].bufferList != nullptr)
      {
        delete procInput.frameList[i].bufferList;
      }
    }
    delete [] (procInput.frameList);
  }

  /** Release output memory */
  if (procResult.list != nullptr)
  {
    for (uint32_t i = 0; i < batchSize; i++)
    {
      releaseConvexHullMemory(procResult.list[i].list, procResult.list[i].numAllocated);
      if (procResult.list[i].list != nullptr) {
        delete [] (procResult.list[i].list);
      }
    }
    delete [] (procResult.list);
  }
}

void NvTrackerProc::processBatch()
{
  NvMOTProcessParams procInput;
  NvMOTTrackedObjBatch procResult;
  allocateProcessMemory(procInput, procResult);

  cudaError_t cudaReturn = cudaSetDevice(m_Config.gpuId);
  if (cudaReturn != cudaSuccess) {
    LOG_ERROR ("gstnvtracker: Failed to set gpu-id with error: %s\n",
      cudaGetErrorName (cudaReturn));
    m_TrackerLibError = true;
  }

  unique_lock<mutex> lkProc(m_ProcQueueLock);
  while (m_Running)
  {
    if (m_ProcQueue.empty()) {
      m_ProcQueueCond.wait(lkProc);
      continue;
    }
    ProcParams procParams = m_ProcQueue.front();
    m_ProcQueue.pop();
    m_ProcQueueCond.notify_one();
    lkProc.unlock();

    /* Only process when low level tracker works fine; otherwise send unprocessed as output.
     * The gst pipeline will send quit signal and call tracker plugin stop function
     * to terminate all other threads. */
    if (!m_TrackerLibError.load())
    {
      /* In case the batch contains two frames with the same ssId,
       * sort the frames based on frameNum in ascending order.
       * such as {(ssId 1, frameNum 1); (ssId 2, frameNum 1); (ssId 1, frameNum 2)}
       * In this case, low level lib is called multiple times to track based on
       * frameNum in ascending order. */
      std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>> batchList;
      queueFrames(*(procParams.input.pBatchMeta), batchList);

      NvTrackerMiscDataBuffer *pMiscDataBuf = m_MiscDataMgr.pop();
      NvMOTTrackerMiscData miscDataResult;
      if (pMiscDataBuf)
      {
        miscDataResult.pPastFrameObjBatch = &pMiscDataBuf->pastFrameObjBatch;

        miscDataResult.pTerminatedTrackBatch = m_Config.outputTerminatedTracks? \
            &pMiscDataBuf->terminatedTrackBatch : nullptr;

        miscDataResult.pShadowTrackBatch = m_Config.outputShadowTracks? \
          &pMiscDataBuf->shadowTracksBatch : nullptr;

        procResult.pReidTensorBatch = m_Config.outputReidTensor ? \
          (&pMiscDataBuf->reidTensorBatch) : nullptr;

      }
      else
      {
        miscDataResult.pPastFrameObjBatch = nullptr;
        procResult.pReidTensorBatch = nullptr;
        miscDataResult.pTerminatedTrackBatch = nullptr;

        /** Print warning if user meta is required but unavailable. */
        if (m_Config.pastFrame || m_Config.outputReidTensor)
        {
          LOG_WARNING(
            "gstnvtracker: Unable to acquire a user meta buffer. Try increasing user-meta-pool-size\n");
        }
      }

      for (uint listInd = 0; listInd < batchList.size(); listInd++)
      {
        std::map<SurfaceStreamId, NvDsFrameMeta *>& frameMap = batchList.at(listInd);

        int frameInd = 0;
        for (auto it = frameMap.begin(); it != frameMap.end(); it++, frameInd++)
        {
            fillMOTFrame(it->first, procParams, *it->second, procInput.frameList[frameInd],
                procResult.list[frameInd]);
        }
        procInput.numFrames = frameMap.size();
        procResult.numFilled = frameMap.size();
        procResult.numAllocated = frameMap.size();

        /* Wait for buffer surface transform to finish. */
        if (m_Config.inputTensorMeta == false) {
           m_ConvBufMgr.syncBuffer(procParams.pConvBuf, &procParams.bufSetSyncObjs);
        }

        char contextName[100];
        snprintf(contextName, sizeof(contextName), "%s_nvtracker_process(Batch=%u)",
            m_Config.gstName, procParams.batchId);
        nvtx_helper_push_pop(contextName);
        NvMOTStatus status  = m_TrackerLibProcess(m_BatchContextHandle, &procInput, &procResult);
        nvtx_helper_push_pop(NULL);
        if (NvMOTStatus_OK != status)
        {
          LOG_ERROR("gstnvtracker: Low-level tracker lib returned error %d\n", status);
          m_TrackerLibError = true;
        }

        if (m_TrackerLibRetrieveMiscData){
          status = m_TrackerLibRetrieveMiscData(m_BatchContextHandle, &procInput, &miscDataResult);
          if (NvMOTStatus_OK != status)
          {
            LOG_ERROR("gstnvtracker: When flushing previous frames, low-level tracker lib returned error %d\n", status);
            m_TrackerLibError = true;
          }
        }
        updateBatchMeta(procResult, procParams, frameMap);
      }
      updateUserMeta(batchList, procParams, pMiscDataBuf);
    }

    /** Mark this request complete. */
    unique_lock<mutex> lkBatch(m_PendingBatchLock);
    auto it = m_PendingBatch.find(procParams.batchId);
    if (it != m_PendingBatch.end()) {
      m_PendingBatch.erase(it);
      lkBatch.unlock();
    } else {
      lkBatch.unlock();
      LOG_ERROR("gstnvtracker: completed req is not in active batch!\n");
      m_TrackerLibError = true;
    }

    /** Done with the batch. Send a completion signal. */
    unique_lock<mutex> lkComp(m_CompletionQueueLock);
    m_CompletionQueue.push(procParams.input);
    lkComp.unlock();
    m_CompletionQueueCond.notify_one();

    /** Return the convert buffer set. OK if it's null. */
    lkProc.lock();
    if (m_Config.inputTensorMeta == false)  {
      m_ConvBufMgr.returnBuffer(procParams.pConvBuf);
    }
    else {
      delete [] procParams.pConvBuf->surfaceList;
      delete procParams.pConvBuf;
      procParams.pConvBuf = nullptr;
    }
    m_BufQueueCond.notify_one();

  } /* while (m_Running) */
  lkProc.unlock();

  releaseProcessMemory(procInput, procResult);

  unique_lock<mutex> lk(m_CompletionQueueLock);
  m_CompletionQueueCond.notify_all();
  lk.unlock();
}

bool NvTrackerProc::dispatchSubBatches(ProcParams procParams)
{
  NvDsBatchMeta *pBatchMeta = procParams.input.pBatchMeta;
  std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>> batchList;
  queueFrames(*pBatchMeta, batchList);
  map<SurfaceStreamId, NvDsFrameMeta *>::iterator it;

  std::vector<SubBatchId> nextBatch;

  unique_lock<mutex> pendingBatchLock(m_PendingBatchLock);
  if (m_PendingBatch.find(m_BatchId) != m_PendingBatch.end())
  {
    LOG_ERROR("gstnvtracker: Batch %d already active!\n", m_BatchId);
    pendingBatchLock.unlock();
    return false;
  }
  m_PendingBatch[m_BatchId] = nextBatch;
  m_ConvBufComplete[m_BatchId] = false;
  std::vector<SubBatchId> &nextBatchRef = m_PendingBatch[m_BatchId];
  procParams.batchId = m_BatchId;
  pendingBatchLock.unlock();

  for (uint listInd = 0; listInd < batchList.size(); listInd++)
  {
    /** For each entry in batchList create a subbBatchList
     * and pass it to the corresponding sub-batch thread for processing */
    std::map<SurfaceStreamId, NvDsFrameMeta *> &frameMap = batchList.at(listInd);
    std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>> subbBatchList(m_Config.subBatchesConfig.size());

    for (it = frameMap.begin(); it != frameMap.end(); ++it)
    {
        SurfaceStreamId ssId = it->first;
        NvDsFrameMeta *pFrameMeta = it->second;
        SubBatchId sbId = 0;
        if (m_SsidSubBatchMap.find(ssId) == m_SsidSubBatchMap.end())
        {
          // Find the sub batch id to which this stream is mapped
          for (sbId = 0; sbId < m_Config.subBatchesConfig.size(); sbId++)
          {
              std::vector<int> &subBatch = m_Config.subBatchesConfig.at(sbId);
              auto jt = find(subBatch.begin(), subBatch.end(), pFrameMeta->pad_index);
              if(jt != subBatch.end())
              {
                break;
              }
          }
          if (sbId >= m_Config.subBatchesConfig.size())
          {
            LOG_ERROR("gstnvtracker: Source index %d is not mapped to any sub batch !! Check \"sub-batches\" configuraion. \n", pFrameMeta->pad_index);
          }
          m_SsidSubBatchMap[ssId] = sbId;
        }
        else
        {
          sbId = m_SsidSubBatchMap[ssId];
        }
        subbBatchList[sbId][ssId] = pFrameMeta;
    }
    /** Now that subbBatchList is created, create a DispatchReq for each entry
     * and push it in the thread queue */
    for (uint sbInd = 0; sbInd < subbBatchList.size(); sbInd++)
    {
      std::map<SurfaceStreamId, NvDsFrameMeta *> &subBatch = subbBatchList.at(sbInd);
      if(subBatch.size() == 0)
        continue;
      DispatchReq req;
      /** Perform a deep copy of the "subBatch" i.e. deep copy implementation of
       * req.subBatchFrameMap = subBatch */
      for (const auto& entry : subBatch) {
        req.subBatchFrameMap[entry.first] = entry.second;
      }
      req.procParams = procParams;
      req.batchId = m_BatchId;
      req.removeSource = false;

      unique_lock<mutex> lk(m_DispatchMapLock);
      map<SubBatchId, DispatchInfo>::iterator it = m_DispatchMap.find(sbInd);
      DispatchInfo *pDispatchInfo;
      if (it != m_DispatchMap.end())
      {
        pDispatchInfo = &it->second;
      }
      else
      {
        LOG_ERROR("gstnvtracker: No thread corresponding to sub-batch %d\n", sbInd);
      }

      /** We can release the dispatch map lock now because a dispatch info is kept valid
       * until its thread joins. */
      lk.unlock();

      /** record the subbatch as pending in the next batch. */
      nextBatchRef.push_back(sbInd);

      /** Queue up the req. */
      unique_lock<mutex> lkReq(pDispatchInfo->reqQueueLock);
      pDispatchInfo->reqQueue.push(req);
      lkReq.unlock();
      pDispatchInfo->reqQueueCond.notify_one();
    }
  }

  m_BatchId++;

  return true;
}

void NvTrackerProc::processSubBatch(DispatchInfo *pDispatchInfo)
{
  SubBatchId sbId = pDispatchInfo->sbId;
  queue<DispatchReq> &reqQueue = pDispatchInfo->reqQueue;
  uint32_t subBatchSize = (m_Config.subBatchesConfig[sbId]).size();

  // Current implementation of sub batches doesn't support handling of miscellaneous data
  if((sbId > 0) && ((m_Config.outputReidTensor == true) ||
  (m_Config.outputTerminatedTracks == true) || (m_Config.outputShadowTracks)))
  {
    LOG_ERROR("gstnvtracker: Current implementation of sub batches doesn't support handling of miscellaneous data.\
    Either use a single batch or disable the configs outputReidTensor, outputShadowTracks and outputTerminatedTracks.");
    m_TrackerLibError = true;
  }

  // Create context with low-level lib
  NvMOTContextHandle contextHandle = initTrackerContext(sbId);
  if (NULL == contextHandle)
  {
    LOG_ERROR("gstnvtracker: Failed to init context for sub batch %x\n", sbId);
    m_TrackerLibError = true;
  }

  NvMOTProcessParams procInput;
  NvMOTTrackedObjBatch procResult;
  allocateProcessMemory(procInput, procResult, subBatchSize);

  TrackerMiscDataManager miscDataMgr;
  if(!miscDataMgr.init(subBatchSize, m_Config.gpuId,
    m_Config.maxTargetsPerStream, m_Config.maxShadowTrackingAge,
    m_Config.reidFeatureSize, m_Config.maxMiscDataPoolSize, m_Config.pastFrame,
    m_Config.outputReidTensor,
    m_Config.outputTerminatedTracks,
    m_Config.outputShadowTracks,
    m_Config.maxTrajectoryBufferLength))
    {
      LOG_ERROR("gstnvtracker: Failed to init TrackerMiscDataManager context for sub batch %x\n", sbId);
      m_TrackerLibError = true;
    }

  cudaError_t cudaReturn = cudaSetDevice(m_Config.gpuId);
  if (cudaReturn != cudaSuccess)
  {
    LOG_ERROR("gstnvtracker: Failed to set gpu-id with error: %s\n",
              cudaGetErrorName(cudaReturn));
    m_TrackerLibError = true;
  }

  while (pDispatchInfo->running)
  {
    unique_lock<mutex> lk(pDispatchInfo->reqQueueLock);
    if (reqQueue.empty())
    {
      pDispatchInfo->reqQueueCond.wait(lk);
    }

    /** Check for shutdown signal or spurious wakeup */
    if (!pDispatchInfo->running)
    {
      break;
    }
    else if (reqQueue.empty())
    {
      continue;
    }

    DispatchReq req = reqQueue.front();
    reqQueue.pop();
    lk.unlock();

    /** Check if the req is for stream removal */
    if(req.removeSource == true)
    {
      if(m_TrackerLibRemoveStreams)
      {
        SurfaceStreamId ssId = req.ssId;
        NvMOTStatus status = m_TrackerLibRemoveStreams(contextHandle, ssId);
        if (status != NvMOTStatus_OK)
        {
          LOG_ERROR("gstnvtracker: Fail to remove stream with Id %lx.\n", ssId);
          m_TrackerLibError = true;
        }
      }
      continue;
    }

    NvTrackerMiscDataBuffer *pMiscDataBuf = miscDataMgr.pop();
    NvMOTTrackerMiscData miscDataResult;
    if (pMiscDataBuf)
    {
      miscDataResult.pPastFrameObjBatch = &pMiscDataBuf->pastFrameObjBatch;
      miscDataResult.pTerminatedTrackBatch = m_Config.outputTerminatedTracks? \
            &pMiscDataBuf->terminatedTrackBatch : nullptr;
      miscDataResult.pShadowTrackBatch = m_Config.outputShadowTracks? \
          &pMiscDataBuf->shadowTracksBatch : nullptr;
      procResult.pReidTensorBatch = m_Config.outputReidTensor ? (&pMiscDataBuf->reidTensorBatch) : nullptr;
    }
    else
    {
      miscDataResult.pPastFrameObjBatch = nullptr;
      procResult.pReidTensorBatch = nullptr;
      miscDataResult.pTerminatedTrackBatch = nullptr;

      /** Print warning if user meta is required but unavailable. */
      if (m_Config.pastFrame || m_Config.outputReidTensor)
      {
        LOG_WARNING(
            "gstnvtracker: Unable to acquire a user meta buffer. Try increasing user-meta-pool-size\n");
      }
    }
    ProcParams procParams = req.procParams;
    /* Only process when low level tracker works fine; otherwise send unprocessed as output.
     * The gst pipeline will send quit signal and call tracker plugin stop function
     * to terminate all other threads. */
    if (!m_TrackerLibError.load())
    {
      std::map<SurfaceStreamId, NvDsFrameMeta *> &frameMap = req.subBatchFrameMap;
      int frameInd = 0;
      for (auto it = frameMap.begin(); it != frameMap.end(); it++, frameInd++)
      {
        fillMOTFrame(it->first, procParams, *it->second, procInput.frameList[frameInd],
                     procResult.list[frameInd]);
      }

      procInput.numFrames = frameMap.size();
      procResult.numFilled = frameMap.size();
      procResult.numAllocated = frameMap.size();
      //printf("Batch num : %d, sub batch number : %d, sub-batch size = %ld\n",
        //req.batchId, sbId, frameMap.size());

      /** Wait for buffer surface transform to finish */
      if (m_Config.inputTensorMeta == false)
      {
        unique_lock<mutex> lkBatch(m_ConvBufLock);
        auto it = m_ConvBufComplete.find(req.batchId);
        if (it != m_ConvBufComplete.end())
        {
          if (m_ConvBufComplete[req.batchId] == false)
          {
            //LOG_DEBUG("gstnvtracker: Syncing for ConvBuf completion for batch %d.!\n", req.batchId);
            m_ConvBufMgr.syncBuffer(procParams.pConvBuf, &procParams.bufSetSyncObjs);
            m_ConvBufComplete[req.batchId] = true;
          }
          else
          {
            //LOG_DEBUG("gstnvtracker: ConvBuf transformation is already complete for batch %d.!\n", req.batchId);
          }
          lkBatch.unlock();
        }
        else
        {
          lkBatch.unlock();
          LOG_ERROR("gstnvtracker: ConvBuf status is not available for active batch!\n");
          m_TrackerLibError = true;
        }
      }
      char contextName[100];
      snprintf(contextName, sizeof(contextName), "%s_nvtracker_process(SubBatchId=%u Frame=%u)",
               m_Config.gstName, sbId, procParams.batchId);
      nvtx_helper_push_pop(contextName);
      // Send request to low-level lib
      NvMOTStatus status = m_TrackerLibProcess(contextHandle, &procInput, &procResult);
      nvtx_helper_push_pop(NULL);
      if (NvMOTStatus_OK != status)
      {
        LOG_ERROR("gstnvtracker: Low-level tracker lib returned error %d for sub batch %x\n", status, sbId);
        m_TrackerLibError = true;
        break;
      }
      if (m_TrackerLibRetrieveMiscData)
      {
        status = m_TrackerLibRetrieveMiscData(contextHandle, &procInput, &miscDataResult);
        if (NvMOTStatus_OK != status)
        {
          LOG_ERROR("gstnvtracker: When flushing previous frames, low-level tracker lib returned error %d\n", status);
          m_TrackerLibError = true;
        }
      }

      updateBatchMeta(procResult, procParams, frameMap);
      {
        std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>> batchList;
        batchList.push_back({});
        batchList[0] = frameMap;
        updateUserMeta(batchList, procParams, pMiscDataBuf, miscDataMgr);
      }
    }
    /** Mark this request complete. */
    /** If the whole batch is done, signal completion. */
    unique_lock<mutex> lkBatch(m_PendingBatchLock);
    auto it = m_PendingBatch.find(req.batchId);
    if (it != m_PendingBatch.end())
    {
      auto itSB = find(it->second.begin(), it->second.end(), sbId);
      if (itSB != it->second.end())
      {
        it->second.erase(itSB);
        if (it->second.empty())
        {
          /** Return the convert buffer set. OK if it's null. */
          unique_lock<mutex> lkProc(m_ProcQueueLock);

          if (m_Config.inputTensorMeta == false)
          {
            m_ConvBufMgr.returnBuffer(procParams.pConvBuf);
          }
          else {
            delete [] procParams.pConvBuf->surfaceList;
            delete procParams.pConvBuf;
            procParams.pConvBuf = nullptr;
          }
          lkProc.unlock();
          m_BufQueueCond.notify_one();

          m_PendingBatch.erase(it);
          m_ConvBufComplete.erase(req.batchId);
          unique_lock<mutex> lkComp(m_CompletionQueueLock);
          m_CompletionQueue.push(procParams.input);
          lkComp.unlock();
          m_CompletionQueueCond.notify_one();
        }
      }
      else
      {
        LOG_ERROR("gstnvtracker: completed sub-batch req is not in a pending batch!\n");
        m_TrackerLibError = true;
      }
      lkBatch.unlock();
    }
    else
    {
      lkBatch.unlock();
      LOG_ERROR("gstnvtracker: completed req is not in active batch!\n");
      m_TrackerLibError = true;
    }
  } /** while (pDispatchInfo->running) */

  /** Cleanup for both normal exit and exception exit */
  if (contextHandle != nullptr)
  {
    m_TrackerLibDeInit(contextHandle);
    contextHandle = nullptr;
  }

  releaseProcessMemory(procInput, procResult, subBatchSize);
  miscDataMgr.deInit();
}
