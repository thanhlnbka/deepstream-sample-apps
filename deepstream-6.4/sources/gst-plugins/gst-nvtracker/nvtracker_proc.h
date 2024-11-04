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

#ifndef _NVTRACKERPROC_H
#define _NVTRACKERPROC_H

#include <queue>
#include <map>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

#include "nvdsmeta.h"
#include "invtracker_proc.h"
#include "convbufmanager.h"
#include "miscdatabufmanager.h"
#include "nvdstracker.h"

/** Tracker process class. */
class NvTrackerProc : public INvTrackerProc
{
public:
  NvTrackerProc();
  virtual ~NvTrackerProc();

  virtual bool init(const TrackerConfig& config);
  virtual void deInit();

  /** Tracker actions when a source is added to the pipeline. */
  virtual bool addSource(uint32_t sourceId);
  /** Tracker actions when a source is removed to the pipeline. */
  virtual bool removeSource(uint32_t sourceId, bool removeObjectIdMapping=true);
  /** Tracker actions when a source is reset. */
  virtual bool resetSource(uint32_t sourceId);
  /** Submit an input batch to tracker process queue. */
  virtual bool submitInput(const InputParams& inputParams);
  /** Wait until a batch's process is done.*/
  virtual CompletionStatus waitForCompletion(InputParams& inputParams);
  /** Flush the request to send the batch downstream. */
  virtual bool flushReqs();

protected:
  /** Object class information. */
  struct ClassInfo
  {
    NvOSD_RectParams rectParams;
    NvOSD_TextParams textParams;
    std::string displayTextString;
    std::string objLabel;
    uint uniqueComponentId;
  };

  /** Info mapped per class. The key is class ID. */
  std::map<int, ClassInfo> m_ClassInfoMap;

  /** Current batch ID. */
  typedef uint32_t BatchId;

  /** Composition of unique ID for each surface from each source
   *  Some sources (e.g. 360d) produce multiple surfaces per frame due to dewarping etc.
   */
  typedef uint64_t SurfaceStreamId;
  /** Source stream ID. */
  typedef uint32_t StreamId;
  /** Surface type ID. */
  typedef uint32_t SurfaceId;
  /** Sub batch ID. */
  typedef uint32_t SubBatchId;
  /** Tracker processing config. */
  TrackerConfig m_Config;
  /** Tracker lib is running. */
  bool m_Running = false;
  /** Tracker lib encounters an error. */
  std::atomic<bool> m_TrackerLibError;

  /** Buffer manager for surface transformation. */
  ConvBufManager m_ConvBufMgr;

  /** Buffer manager for misc data (user meta). */
  TrackerMiscDataManager m_MiscDataMgr;

  /** Tracker data submitted for proceeding. */
  struct ProcParams
  {
    InputParams input;
    bool useConvBuf;
    NvBufSurface *pConvBuf;
    BatchId batchId;
    NvBufSurfTransformSyncObj_t bufSetSyncObjs;
  };

  /** Input queue waiting for tracker proceeding. */
  std::queue<ProcParams> m_ProcQueue;
  std::mutex m_ProcQueueLock;
  std::condition_variable m_ProcQueueCond;
  std::condition_variable m_BufQueueCond;

  /** Stores batches not finishing tracker proceeding. */
  std::map< BatchId, std::vector<SubBatchId> > m_PendingBatch;
  BatchId m_BatchId = 0;
  std::mutex m_PendingBatchLock;

  /** Stores status of pending batches for buffer conversion. */
  std::map< BatchId, bool > m_ConvBufComplete;
  std::mutex m_ConvBufLock;

  /** Batch dispatch to low-level tracker lib */
  NvMOTContextHandle m_BatchContextHandle = nullptr;
  std::thread m_ProcessBatchThread;

  /** Per-sub-batch dispatch to low-level tracker lib */
  struct DispatchReq
  {
    BatchId batchId;
    /** Map for frames present in a sub-batch */
    std::map<SurfaceStreamId, NvDsFrameMeta *> subBatchFrameMap;
    /* Tracker data for processing */
    ProcParams procParams;
    /** Information for stream removal */
    /** Set below flag to true if the request is for stream removal */
    bool removeSource;
    /** Id of the stream to be removed */
    SurfaceStreamId ssId;
  };

  /** Info corresponding to each sub-batch thread */
  struct DispatchInfo
  {
    SubBatchId sbId;
    std::thread sbThread;
    bool running;
    std::queue<DispatchReq> reqQueue;
    std::mutex reqQueueLock;
    std::condition_variable reqQueueCond;

    // FIXME: operator= needed but not used in meaningful way
    DispatchInfo &operator=(DispatchInfo const &rhs)
    {
      return *this;
    }
  };

  /** Mapping fron sub-batch id to corresponding thread info */
  std::map<SubBatchId, DispatchInfo> m_DispatchMap;
  std::mutex m_DispatchMapLock;
  /** Mapping for streamID => SubBatchId */
  std::map<SurfaceStreamId, SubBatchId> m_SsidSubBatchMap;
  /** Pad index(Source id) mapping: {Pad index(source id): sequence index in the sub-batch} */
  std::map<uint32_t, uint32_t> m_PadIndexSeqIndexMap;
  /** Function to pop sub-batch from queue, call low level lib to process
   *  and update meta data with tracking results for the sub-batch
   */
  void processSubBatch(DispatchInfo *pDispatchInfo);
  /** Function to create sub-batches from a batch and submit to
   * corresponding sub-batch queues */
  bool dispatchSubBatches(ProcParams procParams);

  /** Completions */
  std::queue<InputParams> m_CompletionQueue;
  std::mutex m_CompletionQueueLock;
  std::condition_variable m_CompletionQueueCond;

  /** Object id mapping: {surface stream ID: object id offset} */
  std::map<SurfaceStreamId, guint64> m_ObjectIdOffsetMap;

  /** Low-level tracker lib API and support functions. */
  void* m_TrackerLibHandle;

  /** Internal initializing and release function. */

  /** initialize the batch context with low-level tracker lib.*/
  NvMOTContextHandle initTrackerContext(uint32_t sbId = 0);
  /** Initialize surface transform buffer pool. */
  bool initConvBufPool();
  /** Deinitialize surface transform buffer pool. */
  void deInitConvBufPool();
  /** Initialize misc data (user meta) buffer pool. */
  bool initMiscDataPool();
  /** Deinitialize misc data (user meta) buffer pool. */
  void deInitMiscDataPool();
  /** Initialize low-level tracker lib. */
  bool initTrackerLib();
  /** Deinitialize low-level tracker lib. */
  void deInitTrackerLib();
  /** Allocate memory for low level library input*/
  void allocateProcessMemory(NvMOTProcessParams &procInput,
                             NvMOTTrackedObjBatch &procResult);
  void allocateProcessMemory(NvMOTProcessParams &procInput,
                             NvMOTTrackedObjBatch &procResult, uint32_t batchSize);
  /** Release input memory */
  void releaseProcessMemory(NvMOTProcessParams &procInput,
                            NvMOTTrackedObjBatch &procResult);
  void releaseProcessMemory(NvMOTProcessParams &procInput,
                            NvMOTTrackedObjBatch &procResult, uint32_t batchSize);

  void allocateConvexHullMemory(NvMOTTrackedObj *list, const uint32_t numAllocated);
  void releaseConvexHullMemory(NvMOTTrackedObj *list, const uint32_t numAllocated);

  /** Functions to process a batch. */

  /** In case the batch contains two frames with the same ssId,
   * sort the frames based on frameNum in ascending order.*/
  void queueFrames(const NvDsBatchMeta& batchMeta,
                   std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>>& batchList);
  /** Fill input to low-level tracker lib. */
  void fillMOTFrame(SurfaceStreamId ssId,
          const ProcParams& procParams,
          const NvDsFrameMeta& frameMeta,
          NvMOTFrame& motFrame,
          NvMOTTrackedObjList& trackedObjList);
  /** Update batch meta with tracking results. */
  void updateBatchMeta(const NvMOTTrackedObjBatch& procResult,
           const ProcParams& procParams,
           std::map<SurfaceStreamId, NvDsFrameMeta *>& frameMap);
  /** Update each stream's frame meta with tracking results. */
  void updateFrameMeta(NvDsFrameMeta* pFrameMeta,
           const NvMOTTrackedObjList& objList,
           const ProcParams& procParams);
  /** Update past frame data. */
  void updatePastFrameMeta(
           const std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>>& batchList,
           GstNvTrackerMiscDataObject *pGstObj,
           ProcParams& procParams);
  /** Update tracker user meta. */
  void updateUserMeta(
           const std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>>& batchList,
           ProcParams& procParams, NvTrackerMiscDataBuffer *pMiscDataBuf);
  void updateUserMeta(
      const std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>> &batchList,
      ProcParams &procParams, NvTrackerMiscDataBuffer *pMiscDataBuf, TrackerMiscDataManager &miscDataMgr);
  /** Update each object's reid meta. */
  void updateObjectReidMeta(NvDsObjectMeta *pObjectMeta, NvMOTTrackedObj *pTrackedObj,
            NvDsBatchMeta *pBatchMeta);
  /** Update batch reid meta. */
  void updateBatchReidMeta(GstNvTrackerMiscDataObject *pGstObj,
                    ProcParams& procParams);
  /** Update each object's model projection meta. */
  void updateObjectProjectionMeta(NvDsObjectMeta *pObjectMeta, NvMOTTrackedObj *pTrackedObj,
            NvDsBatchMeta *pBatchMeta, float scaleWidth, float scaleHeight);

  void updateTerminatedTrackMeta(GstNvTrackerMiscDataObject *pGstObj,
                    ProcParams& procParams);

  void updateShadowTrackMeta(GstNvTrackerMiscDataObject *pGstObj,
                     ProcParams& procParams);

  /** Function to pop batch from input queue, call low level lib to process
   *  and update meta data with tracking results.
   */
  void processBatch();

  /** Getter */
  SurfaceStreamId getSurfaceStreamId(StreamId streamId, SurfaceId surfaceId)
  { return ( ((uint64_t)streamId << 32) | surfaceId ); };
  StreamId getStreamId(SurfaceStreamId ssId) { return (StreamId)( ssId >> 32 ); };
  SurfaceId getSurfaceId(SurfaceStreamId ssId) { return (SurfaceId)(ssId); };

  /** Object ID mapper to ensure ID starts from 0 after reset. */
  guint64 objectIdMapping(const guint64& objectId, const SurfaceStreamId& ssId);
  /** Remove objects that don't need an ID. */
  void removeUntrackedObjects(NvDsFrameMeta* pFrameMeta);
  /**
   * Clip the bbox to be within the frame. If the bbox is entirely
   * outside, or shrinks to 0 in either dimension, then return false
   * so it can be removed from the object list.
   */
  bool clipBBox(uint32_t frameWidth, uint32_t frameHeight, float &left, float &top, float &width,  float &height);
  /** Tracker low level library functions. */
  /** Low level lib init. */
  NvMOTStatus (*m_TrackerLibInit) (NvMOTConfig *pConfigIn,
           NvMOTContextHandle *pContextHandle,
           NvMOTConfigResponse *pConfigResponse);
  /** Low level lib deinit. */
  void (*m_TrackerLibDeInit) (NvMOTContextHandle contextHandle);
  /** Low level lib process. */
  NvMOTStatus (*m_TrackerLibProcess) (NvMOTContextHandle contextHandle,
            NvMOTProcessParams *pParams,
            NvMOTTrackedObjBatch *pTrackedObjectsBatch);
  /** Low level lib process past frame. */
  NvMOTStatus (*m_TrackerLibRetrieveMiscData) (NvMOTContextHandle contextHandle,
            NvMOTProcessParams *pParams,
            NvMOTTrackerMiscData *pTrackerMiscData);
  /** Low level lib query. */
  NvMOTStatus (*m_TrackerLibQuery) (uint16_t customConfigFilePathSize,
            char* pCustomConfigFilePath,
            NvMOTQuery *pQuery);
  /** Low level lib remove streams. */
  NvMOTStatus (*m_TrackerLibRemoveStreams) (NvMOTContextHandle contextHandle,
            NvMOTStreamId streamIdMask);
  NvDsTargetMiscDataBatch* allocatePastFrameMemory();
};

#endif
