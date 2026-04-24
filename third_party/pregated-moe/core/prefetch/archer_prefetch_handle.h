// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "aio/archer_tensor_handle.h"
#include "model/model_topology.h"
#include "parallel/expert_dispatcher.h"

class ArcherPrefetchHandle {
 public:
  ArcherPrefetchHandle(const std::string& prefix,
                       const double device_memory_ratio = 0.8);
  ~ArcherPrefetchHandle();

  bool IsTensorOffloaded(const std::uint32_t tensor_id);

  void AcquireTensor(std::uint64_t& request_id, torch::Tensor& buffer);
  void ReleaseTensor(std::uint64_t& request_id, torch::Tensor& buffer);
  void PrefetchTensors(std::uint64_t& request_id,
                       const std::vector<std::uint32_t>& buffer);
  void FetchTensors(std::uint64_t& request_id,
                    const std::vector<std::uint32_t>& buffer);

  void ReplaceCacheCandidates(const std::vector<std::uint32_t>& tensor_ids);
  void EnqueuePrefetch(const uint32_t tensor_id, int gpu_id);
  void BeginExactRequest(std::uint64_t request_id);
  void SubmitExactLayer(std::uint64_t request_id, std::int64_t layer_id,
                        const std::vector<std::uint32_t>& tensor_ids);
  void WaitLayerReady(std::uint64_t request_id, std::int64_t layer_id);
  void ReleaseLayer(std::uint64_t request_id, std::int64_t layer_id);
  void EndExactRequest(std::uint64_t request_id);
  std::unordered_map<std::string, std::int64_t> GetExactPrepareStats() const;

  void OffloadTensor(torch::Tensor& tensor, const std::uint32_t tensor_id);
  void RegisterTensor(torch::Tensor& tensor, const std::uint32_t tensor_id);
  void RegisterModule(torch::nn::Module& module);
  void RegisterTensor(torch::Tensor* tensor);

  int GetNodeDefaultDevice(std::vector<std::uint32_t> tensor_ids) const;
  int GetNodeDevice(std::vector<std::uint32_t> tensor_ids) const;

  void SetTensorDevice(torch::Tensor& tensor, torch::Device device) const;

  torch::Tensor GetTrace();
  torch::Tensor GetHitRate();
  void SetTrace(const torch::Tensor& trace);
  void TraceRequest(const std::uint64_t request_id, const TensorID tensor_id);
  void SetTopology(const std::vector<
                   std::tuple<std::string, std::vector<std::vector<TensorID>>>>&
                       topology);
  void UpdateTensorMap(std::uint64_t old_ptr, std::uint64_t new_ptr);
  bool IsTensorIndexInitialized() const;
  bool IsTensorOnDevice(const torch::Tensor& tensor) const;
  bool IsTensorOnDevice(const TensorID tensor_id) const;

  void CleanUpResources();

  // void SetNodeCachePriority(const std::uint64_t corr_id, const float
  // priority);

 private:
  std::string prefix_;
  std::unordered_map<std::size_t, std::unordered_set<std::uint32_t>>
      node_id_to_tensor_ids_;
  std::unordered_set<std::uint32_t> tensors_to_delete_;
  uint64_t last_layer_id_;
  NodePtr last_node_;
  bool has_cleaned_up_resources_;

  std::unordered_map<std::uint64_t, std::unordered_set<NodePtr>>
      request_id_to_nodes_;

  std::mutex mutex_;

  struct ExactLayerKey {
    std::uint64_t request_id;
    std::int64_t layer_id;

    bool operator==(const ExactLayerKey& other) const noexcept {
      return request_id == other.request_id && layer_id == other.layer_id;
    }
  };

  struct ExactLayerKeyHash {
    std::size_t operator()(const ExactLayerKey& key) const noexcept {
      auto h1 = std::hash<std::uint64_t>{}(key.request_id);
      auto h2 = std::hash<std::int64_t>{}(key.layer_id);
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
  };

  struct LayerPrepareState {
    std::vector<NodePtr> nodes;
    std::unordered_set<std::size_t> node_ids;
    std::size_t ready_count = 0;
    bool released = false;
    std::mutex mutex;
    std::condition_variable cv;
  };

  struct ExactTask {
    ExactLayerKey key;
    NodePtr node;
    int gpu_id = -1;
    bool stop = false;
  };

  void ExactPrepareWorker(int gpu_id);
  bool IsNodeResidentOnDefaultDevice(const NodePtr& node) const;
  void EnsureExactQueueCapacity(int gpu_id, const NodePtr& node);
  NodePtr FindEvictableNode(int gpu_id, const NodePtr& protected_node,
                            std::int64_t* live_bytes = nullptr,
                            std::size_t* evictable_nodes = nullptr) const;
  std::unordered_set<std::size_t> SnapshotInflightNodeIds() const;
  NodePtrList GetLiveResidentNodes(int gpu_id,
                                   std::int64_t* total_bytes = nullptr) const;
  void MarkLayerNodeReady(const ExactLayerKey& key, const NodePtr& node);
  std::shared_ptr<LayerPrepareState> GetOrCreateLayerState(
      const ExactLayerKey& key);
  std::shared_ptr<LayerPrepareState> FindLayerState(const ExactLayerKey& key);
  void EraseLayerState(const ExactLayerKey& key);

  std::atomic<bool> exact_stop_{false};
  std::vector<std::thread> exact_workers_;
  std::vector<std::deque<ExactTask>> exact_queues_;
  std::vector<std::mutex> exact_queue_mutexes_;
  std::vector<std::condition_variable> exact_queue_cvs_;
  std::vector<std::int64_t> resident_limits_;
  std::unordered_set<std::size_t> inflight_node_ids_;
  std::unordered_map<ExactLayerKey, std::shared_ptr<LayerPrepareState>,
                     ExactLayerKeyHash>
      exact_layer_states_;
  std::unordered_map<std::uint64_t, std::vector<ExactLayerKey>> request_layers_;
  mutable std::mutex exact_mutex_;

  std::atomic<std::int64_t> exact_submitted_{0};
  std::atomic<std::int64_t> exact_skipped_resident_{0};
  std::atomic<std::int64_t> exact_loaded_{0};
  std::atomic<std::int64_t> exact_evicted_{0};
  std::atomic<std::int64_t> exact_sync_fallback_{0};
};
