// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "archer_prefetch_handle.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <torch/extension.h>
#include "aio/archer_tensor_handle.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "memory/memory_pool.h"
#include "task_scheduler.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"

namespace {

bool PregatedDebugEnabled() {
  static bool enabled = [] {
    const char* value = std::getenv("PREGATED_DEBUG");
    if (value == nullptr) {
      return false;
    }
    std::string flag(value);
    std::transform(flag.begin(), flag.end(), flag.begin(), ::tolower);
    return flag == "1" || flag == "true" || flag == "yes" || flag == "on";
  }();
  return enabled;
}

constexpr auto kWaitLayerReadyTimeout = std::chrono::minutes(3);

}  // namespace

ArcherPrefetchHandle::ArcherPrefetchHandle(const std::string& prefix,
                                           const double device_memory_ratio)
    : prefix_(prefix), last_layer_id_(0), has_cleaned_up_resources_(false) {
  // InitLogger();
  int num_io_threads = 0;
  const char* io_threads_env = std::getenv("MOE_IO_THREADS");
  if (io_threads_env != nullptr) {
    num_io_threads = std::atoi(io_threads_env);
  }
  kTensorIndex = std::make_unique<ArcherTensorIndex>();
  kArcherTensorHandle =
      std::make_unique<ArcherTensorHandle>(prefix, num_io_threads);
  kTopologyHandle = std::make_unique<ArcherTopologyHandle>();
  kTaskPool = std::make_unique<ArcherTaskPool>();
  kDeviceMemoryPool = std::make_unique<DeviceMemoryPool>();
  kHostMemoryPool = std::make_unique<HostMemoryPool>();
  kDeviceMemoryPool->SetMemoryRatio(device_memory_ratio);
  DLOG_TRACE("Free Device Memory ",
             kDeviceMemoryPool->GetFreeMemory(CUDA_DEVICE(0)));

  if (prefix_.back() != '/') {
    prefix_ += '/';
  }

  // enable peer access for kernels
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  DLOG_INFO("Device count ", device_count);

  for (int i = 0; i < device_count; i++) {
    for (int j = 0; j < device_count; j++) {
      if (i != j) {
        int can_access = 0;
        cudaDeviceCanAccessPeer(&can_access, i, j);
        if (can_access == 1) {
          cudaSetDevice(i);
          cudaError_t status = cudaDeviceEnablePeerAccess(j, 0);
          if (status == cudaErrorPeerAccessAlreadyEnabled) {
            DLOG_INFO("Peer access already enabled between device ", i, j);
            cudaGetLastError();  // clear error
          } else if (status != cudaSuccess) {
            DLOG_ERROR("Failed to enable peer access between device ", i, j);
          } else {
            DLOG_INFO("Enabled peer access between device ", i, j);
          }
        }
      }
    }
  }

  DLOG_INFO("Enabled peer access for all devices");

  exact_queues_.resize(device_count);
  exact_queue_mutexes_ = std::vector<std::mutex>(device_count);
  exact_queue_cvs_ = std::vector<std::condition_variable>(device_count);
  resident_limits_.assign(device_count, 0);
  exact_workers_.reserve(device_count);
  for (int gpu_id = 0; gpu_id < device_count; ++gpu_id) {
    exact_workers_.emplace_back(
        &ArcherPrefetchHandle::ExactPrepareWorker, this, gpu_id);
  }
}

ArcherPrefetchHandle::~ArcherPrefetchHandle() {
  // served as a global manager for order of destruction
  if (!has_cleaned_up_resources_) {
    CleanUpResources();
  }
}

void ArcherPrefetchHandle::CleanUpResources() {
  exact_stop_.store(true);
  for (size_t gpu_id = 0; gpu_id < exact_queue_cvs_.size(); ++gpu_id) {
    exact_queue_cvs_[gpu_id].notify_all();
  }
  for (auto& worker : exact_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  exact_workers_.clear();

  kTaskPool.reset();
  kArcherTensorHandle.reset();
  kTensorIndex.reset();
  kTopologyHandle.reset();
  kDeviceMemoryPool.reset();
  kHostMemoryPool.reset();
  has_cleaned_up_resources_ = true;
}

void ArcherPrefetchHandle::AcquireTensor(std::uint64_t& request_id,
                                         torch::Tensor& buffer) {
  auto tensor_id = kArcherTensorHandle->GetTensorId((void*)buffer.data_ptr());
  void* old_ptr = (void*)buffer.data_ptr();
  DLOG_TRACE("Acquire tensor ", tensor_id, old_ptr);

  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
  node->state = 1;

  // add node tensor_ids to node_id_to_tensor_ids_
  if (node_id_to_tensor_ids_.find(node->id) == node_id_to_tensor_ids_.end() ||
      node_id_to_tensor_ids_[node->id].size() == 0) {
    node_id_to_tensor_ids_[node->id] = std::unordered_set<std::uint32_t>();
    for (auto& tensor_id : node->tensor_ids) {
      node_id_to_tensor_ids_[node->id].insert(tensor_id);
    }

    auto node_body = kTopologyHandle->GetNodeBodyFromCorrID(node->corr_id);
    if (node->device.is_cuda()) {
      node_body->gpu_hit_cnt++;
    }

    // always lock node, wait for previous prefetch task to finish
    node->mutex.lock();
    std::unique_lock<std::mutex> lock(node->mutex, std::adopt_lock);

    if (node->is_sparse) {
      bool success = kTaskPool->RemoveCachedSparseNode(node);
      if (!success) node->is_overflow = true;
    } else {
      kTaskPool->RemoveCachedDenseNode(node);
    }
    kTaskPool->StartExec(request_id, node);
    node->cv.wait(lock, [node] { return node->state == 0; });
  }

  kArcherTensorHandle->SetTensor(tensor_id, buffer);
  kArcherTensorHandle->UpdateTensorMap(old_ptr, (void*)buffer.data_ptr());
}
void ArcherPrefetchHandle::ReleaseTensor(std::uint64_t& request_id,
                                         torch::Tensor& buffer) {
  auto tensor_id = kArcherTensorHandle->GetTensorId((void*)buffer.data_ptr());
  void* old_ptr = (void*)buffer.data_ptr();
  DLOG_TRACE("Release tensor ", tensor_id, old_ptr);

  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
  // node->state = 1;

  if (node_id_to_tensor_ids_.find(node->id) == node_id_to_tensor_ids_.end()) {
    DLOG_ERROR("Node not found in node_id_to_tensor_ids_", node->str());
    return;
  }

  /*  This needs to go after Release, default host can be changed in
   * TraceRequest Faulty case: node -> default_host = cpu, node -> default_host
   * = cuda; tensor already released
   */
  // if (node != last_node_) {
  //     // kTaskPool->Prefetch(request_id, node);
  //     TraceRequest(request_id, tensor_id);
  // }
  // TraceRequest(request_id, tensor_id);

  auto current_layer_id = node->corr_id & 0xFFFFFFFF;
  if (current_layer_id != last_layer_id_ &&
      node_id_to_tensor_ids_[last_node_->id].size() != 0) {
    node_id_to_tensor_ids_[last_node_->id].clear();
    kTaskPool->StopExec(request_id,
                        last_node_);  // evict last node to cpu or disk
    last_node_->mutex.unlock();
  }
  last_layer_id_ = current_layer_id;
  last_node_ = node;

  node_id_to_tensor_ids_[node->id].erase(tensor_id);
  // DLOG_TRACE(
  //     "Node {} tensor_ids size {}", node->id,
  //     node_id_to_tensor_ids_[node->id].size());

  if (node_id_to_tensor_ids_[node->id].size() == 0) {
    kTaskPool->StopExec(request_id,
                        node);  // FIXME: change api to add request id
    // always unlock node here since, exec queue do not unlock automatically
    node->mutex.unlock();
  }

  if (kTopologyHandle->IsLastNode(node)) {
    DLOG_TRACE("Node is last, clean up", node->str());
    request_id_to_nodes_.erase(request_id);
  }

  at::TensorOptions options;
  options = options.device(torch::kCPU);
  options = options.dtype(buffer.dtype());
  auto zero_tensor = torch::zeros({1}, options);
  buffer.set_data(zero_tensor);
  kArcherTensorHandle->UpdateTensorMap(old_ptr, (void*)buffer.data_ptr());
}

void ArcherPrefetchHandle::PrefetchTensors(
    std::uint64_t& request_id, const std::vector<std::uint32_t>& buffer) {
  std::vector<NodePtr> candidates;
  for (std::uint32_t tensor_id : buffer) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    candidates.push_back(node);
  }

  if (candidates.size() == 0) {
    return;
  }
}

void ArcherPrefetchHandle::ReplaceCacheCandidates(
    const std::vector<std::uint32_t>& tensor_ids) {
  std::vector<NodePtr> candidates;
  for (std::uint32_t tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    node->mutex.try_lock();
    candidates.push_back(node);
  }

  kTaskPool->ReplaceCacheCandidates(candidates);
}
void ArcherPrefetchHandle::EnqueuePrefetch(const uint32_t tensor_id,
                                           int gpu_id) {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);

  auto task = std::make_shared<Task>();
  task->priority = 1;
  task->node = node;
  task->on_demand = false;
  task->src_device = node->device;
  // task->dst_device = CUDA_DEVICE(gpu_id); // use default device for now
  task->dst_device = node->default_device;
  kTaskPool->EnqueueTask(task);
}

void ArcherPrefetchHandle::FetchTensors(
    std::uint64_t& request_id, const std::vector<std::uint32_t>& buffer) {
  // std::vector<NodePtr> candidates;
  for (std::uint32_t tensor_id : buffer) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    kTaskPool->FetchExec(request_id, node);
  }
}

void ArcherPrefetchHandle::OffloadTensor(torch::Tensor& tensor,
                                         const std::uint32_t tensor_id) {
  kArcherTensorHandle->StoreTensor(tensor_id, tensor);

  auto ckpt_index_path = prefix_ + std::string(ARCHER_IHDEX_NAME);

  std::unique_lock<std::mutex> lock(mutex_);
  kTensorIndex->Serialize(ckpt_index_path.c_str());
}

void ArcherPrefetchHandle::RegisterTensor(torch::Tensor& tensor,
                                          const std::uint32_t tensor_id) {
  kArcherTensorHandle->RegisterTensor(tensor_id, tensor);
}

void ArcherPrefetchHandle::RegisterModule(torch::nn::Module& module) {
  for (auto it = module.parameters().begin(); it != module.parameters().end();
       ++it) {
    auto tensor_id =
        kArcherTensorHandle->GetTensorId((void*)(*it).unsafeGetTensorImpl());
    kArcherTensorHandle->RegisterTensor(tensor_id, *it);
  }

  for (auto it = module.buffers().begin(); it != module.buffers().end(); ++it) {
    auto tensor_id =
        kArcherTensorHandle->GetTensorId((void*)(*it).unsafeGetTensorImpl());
    kArcherTensorHandle->RegisterTensor(tensor_id, *it);
  }
}

void ArcherPrefetchHandle::RegisterTensor(torch::Tensor* tensor) {
  DLOG_TRACE("Register tensor: is view ", (void*)tensor, tensor->is_view());
}

torch::Tensor ArcherPrefetchHandle::GetTrace() {
  const auto& child_visit_cnts = kTopologyHandle->GetChildVisitCounts();
  const auto num_layers_and_experts = kTopologyHandle->GetNumLayersAndExperts();
  const auto num_layers = std::get<0>(num_layers_and_experts);
  const auto num_experts = std::get<1>(num_layers_and_experts);

  std::vector<int64_t> trace_vec(child_visit_cnts.begin(),
                                 child_visit_cnts.end());
  torch::Tensor trace = torch::from_blob(trace_vec.data(),
                                         {static_cast<int64_t>(num_layers - 1),
                                          static_cast<int64_t>(num_experts),
                                          static_cast<int64_t>(num_experts)},
                                         torch::kInt64)
                            .clone();

  return trace;
}

torch::Tensor ArcherPrefetchHandle::GetHitRate() {
  const auto& node_visit_cnts = kTopologyHandle->GetNodeVisitCounts();

  // flatten vector of vectors
  std::vector<int64_t> node_visit_cnts_vec;
  for (auto& node_visit_cnt : node_visit_cnts) {
    node_visit_cnts_vec.insert(node_visit_cnts_vec.end(),
                               node_visit_cnt.begin(), node_visit_cnt.end());
  }

  torch::Tensor trace =
      torch::from_blob(node_visit_cnts_vec.data(),
                       {node_visit_cnts.size(), node_visit_cnts[0].size()},
                       torch::kInt64)
          .clone();
  return trace;
}

void ArcherPrefetchHandle::SetTrace(const torch::Tensor& trace) {
  if (trace.dim() != 3 || !trace.is_contiguous() || !trace.is_cpu()) {
    DLOG_ERROR("Trace should be a contiguous 3D tensor on CPU");
    return;
  }

  std::vector<std::size_t> child_visit_cnts(
      trace.data_ptr<int64_t>(), trace.data_ptr<int64_t>() + trace.numel());
  kTopologyHandle->SetChildVisitCounts(child_visit_cnts);
}

void ArcherPrefetchHandle::TraceRequest(const std::uint64_t request_id,
                                        const TensorID tensor_id) {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);

  auto it = request_id_to_nodes_.find(request_id);
  if (it == request_id_to_nodes_.end()) {
    request_id_to_nodes_[request_id] = std::unordered_set<NodePtr>();
  }

  auto node_it = request_id_to_nodes_[request_id].find(node);
  if (node_it != request_id_to_nodes_[request_id].end()) {
    DLOG_TRACE("Node already traced for request ", request_id, node->str());
    return;
  }

  request_id_to_nodes_[request_id].insert(node);
}

void ArcherPrefetchHandle::SetTopology(
    const std::vector<
        std::tuple<std::string, std::vector<std::vector<TensorID>>>>&
        topology) {
  kTopologyHandle->InitializeTopology(topology);

  {
    std::lock_guard<std::mutex> lock(exact_mutex_);
    inflight_node_ids_.clear();
    exact_layer_states_.clear();
    request_layers_.clear();
  }

  for (size_t gpu_id = 0; gpu_id < resident_limits_.size(); ++gpu_id) {
    resident_limits_[gpu_id] =
        kTopologyHandle->GetSparseCacheLimit(CUDA_DEVICE(gpu_id));
    std::lock_guard<std::mutex> queue_lock(exact_queue_mutexes_[gpu_id]);
    exact_queues_[gpu_id].clear();
  }
}

std::shared_ptr<ArcherPrefetchHandle::LayerPrepareState>
ArcherPrefetchHandle::GetOrCreateLayerState(const ExactLayerKey& key) {
  std::lock_guard<std::mutex> lock(exact_mutex_);
  auto it = exact_layer_states_.find(key);
  if (it != exact_layer_states_.end()) {
    return it->second;
  }

  auto state = std::make_shared<LayerPrepareState>();
  exact_layer_states_.emplace(key, state);
  request_layers_[key.request_id].push_back(key);
  return state;
}

std::shared_ptr<ArcherPrefetchHandle::LayerPrepareState>
ArcherPrefetchHandle::FindLayerState(const ExactLayerKey& key) {
  std::lock_guard<std::mutex> lock(exact_mutex_);
  auto it = exact_layer_states_.find(key);
  if (it == exact_layer_states_.end()) {
    return nullptr;
  }
  return it->second;
}

void ArcherPrefetchHandle::EraseLayerState(const ExactLayerKey& key) {
  std::lock_guard<std::mutex> lock(exact_mutex_);
  exact_layer_states_.erase(key);
}

bool ArcherPrefetchHandle::IsNodeResidentOnDefaultDevice(
    const NodePtr& node) const {
  return node != nullptr && node->device.is_cuda() &&
         node->default_device.is_cuda() &&
         node->device.index() == node->default_device.index();
}

std::unordered_set<std::size_t> ArcherPrefetchHandle::SnapshotInflightNodeIds()
    const {
  std::lock_guard<std::mutex> lock(exact_mutex_);
  return inflight_node_ids_;
}

NodePtrList ArcherPrefetchHandle::GetLiveResidentNodes(
    int gpu_id, std::int64_t* total_bytes) const {
  NodePtrList live_nodes;
  std::int64_t live_bytes = 0;

  for (auto& node : kTopologyHandle->GetSparseNodes()) {
    if (node == nullptr || !node->device.is_cuda() ||
        node->device.index() != gpu_id) {
      continue;
    }
    live_nodes.push_back(node);
    live_bytes += node->byte_size;
  }

  if (total_bytes != nullptr) {
    *total_bytes = live_bytes;
  }
  return live_nodes;
}

void ArcherPrefetchHandle::BeginExactRequest(std::uint64_t request_id) {
  std::lock_guard<std::mutex> lock(exact_mutex_);
  request_layers_.erase(request_id);
}

void ArcherPrefetchHandle::SubmitExactLayer(
    std::uint64_t request_id, std::int64_t layer_id,
    const std::vector<std::uint32_t>& tensor_ids) {
  ExactLayerKey key{request_id, layer_id};
  auto state = GetOrCreateLayerState(key);
  std::vector<std::vector<ExactTask>> pending_tasks(exact_queues_.size());
  bool notify_ready = false;

  {
    std::lock_guard<std::mutex> exact_lock(exact_mutex_);
    std::lock_guard<std::mutex> state_lock(state->mutex);

    for (auto tensor_id : tensor_ids) {
      auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
      if (node == nullptr) {
        continue;
      }
      if (!state->node_ids.insert(node->id).second) {
        continue;
      }

      state->nodes.push_back(node);
      node->pin_count.fetch_add(1);

      if (IsNodeResidentOnDefaultDevice(node)) {
        state->ready_count += 1;
        exact_skipped_resident_.fetch_add(1);
        continue;
      }

      if (!inflight_node_ids_.insert(node->id).second) {
        continue;
      }

      int gpu_id = node->default_device.index();
      if (gpu_id < 0 || gpu_id >= static_cast<int>(pending_tasks.size())) {
        inflight_node_ids_.erase(node->id);
        DLOG_FATAL("SubmitExactLayer: invalid gpu_id ", gpu_id, " for node ",
                   node->str());
      }

      pending_tasks[gpu_id].push_back(ExactTask{key, node, gpu_id, false});
      exact_submitted_.fetch_add(1);
    }

    notify_ready = state->ready_count >= state->node_ids.size();
  }

  if (PregatedDebugEnabled()) {
    DLOG_INFO("SubmitExactLayer request ", request_id, " layer ", layer_id,
              " tensors ", tensor_ids.size(), " nodes ",
              state->node_ids.size(), " ready ", state->ready_count);
    for (auto& node : state->nodes) {
      if (node == nullptr) {
        continue;
      }
      DLOG_INFO("  submit node ", node->str(), " pin ",
                node->pin_count.load(), " active ",
                node->active_users.load());
    }
  }

  for (size_t gpu_id = 0; gpu_id < pending_tasks.size(); ++gpu_id) {
    if (pending_tasks[gpu_id].empty()) {
      continue;
    }

    {
      std::lock_guard<std::mutex> queue_lock(exact_queue_mutexes_[gpu_id]);
      for (auto& task : pending_tasks[gpu_id]) {
        exact_queues_[gpu_id].push_back(task);
      }
    }
    exact_queue_cvs_[gpu_id].notify_one();
  }

  if (notify_ready) {
    state->cv.notify_all();
  }
}

void ArcherPrefetchHandle::WaitLayerReady(std::uint64_t request_id,
                                          std::int64_t layer_id) {
  ExactLayerKey key{request_id, layer_id};
  auto state = FindLayerState(key);
  if (state == nullptr) {
    throw std::runtime_error(
        std::string("WaitLayerReady called before SubmitExactLayer ") +
        std::to_string(request_id) + ":" + std::to_string(layer_id));
  }

  std::vector<NodePtr> nodes_to_verify;
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    auto total_nodes = state->node_ids.size();
    if (PregatedDebugEnabled()) {
      DLOG_INFO("WaitLayerReady begin request ", request_id, " layer ",
                layer_id, " ready ", state->ready_count, " total ",
                total_nodes);
    }
    bool ready = state->cv.wait_for(lock, kWaitLayerReadyTimeout, [&state] {
      return state->ready_count >= state->node_ids.size();
    });
    if (!ready) {
      auto ready_count = state->ready_count;
      DLOG_ERROR("WaitLayerReady timeout after ",
                 std::chrono::duration_cast<std::chrono::seconds>(
                     kWaitLayerReadyTimeout)
                     .count(),
                 " seconds for request ", request_id, " layer ", layer_id,
                 " ready ", ready_count, " total ", total_nodes);
      throw std::runtime_error(
          std::string("Pregated-MoE exact prepare timed out after 180 seconds "
                      "for request ") +
          std::to_string(request_id) + ", layer " +
          std::to_string(layer_id) + " (ready " +
          std::to_string(ready_count) + "/" + std::to_string(total_nodes) +
          "). Please retry.");
    }
    nodes_to_verify = state->nodes;
  }

  for (auto& node : nodes_to_verify) {
    if (node == nullptr || IsNodeResidentOnDefaultDevice(node)) {
      continue;
    }

    auto gpu_id = node->default_device.index();
    if (gpu_id < 0 || gpu_id >= static_cast<int>(resident_limits_.size())) {
      throw std::runtime_error(
          std::string("WaitLayerReady: invalid default gpu for node ") +
          node->str());
    }

    std::unique_lock<std::mutex> node_lock(node->mutex);
    if (IsNodeResidentOnDefaultDevice(node)) {
      continue;
    }

    EnsureExactQueueCapacity(gpu_id, node);
    if (node->device.is_cuda() && node->device.index() != gpu_id) {
      node->SetDevice(node->default_host, true, nullptr);
    }
    node->SetDevice(node->default_device, true, nullptr);
    node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
    {
      std::lock_guard<std::mutex> exact_lock(exact_mutex_);
      inflight_node_ids_.erase(node->id);
    }
    exact_sync_fallback_.fetch_add(1);
    if (PregatedDebugEnabled()) {
      std::int64_t live_bytes = 0;
      GetLiveResidentNodes(gpu_id, &live_bytes);
      DLOG_INFO("WaitLayerReady fallback-loaded node ", node->str(), " gpu ",
                gpu_id, " live_bytes ", live_bytes, " limit ",
                resident_limits_[gpu_id]);
    }
    DLOG_WARN("WaitLayerReady fallback-loaded node ", node->str(),
              " for request ", request_id, " layer ", layer_id);
  }

  if (PregatedDebugEnabled()) {
    for (auto& node : nodes_to_verify) {
      if (node == nullptr) {
        continue;
      }
      DLOG_INFO("WaitLayerReady done request ", request_id, " layer ",
                layer_id, " node ", node->str(), " pin ",
                node->pin_count.load(), " active ",
                node->active_users.load());
    }
  }
}

void ArcherPrefetchHandle::ReleaseLayer(std::uint64_t request_id,
                                        std::int64_t layer_id) {
  ExactLayerKey key{request_id, layer_id};
  std::shared_ptr<LayerPrepareState> state;
  {
    std::lock_guard<std::mutex> exact_lock(exact_mutex_);
    auto it = exact_layer_states_.find(key);
    if (it == exact_layer_states_.end()) {
      return;
    }
    state = it->second;
  }

  std::vector<NodePtr> nodes_to_release;
  {
    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (state->released) {
      return;
    }
    state->released = true;
    nodes_to_release = state->nodes;
  }

  for (auto& node : nodes_to_release) {
    if (node == nullptr) {
      continue;
    }
    auto previous = node->pin_count.fetch_sub(1);
    if (previous == 0) {
      node->pin_count.store(0);
    }
  }

  EraseLayerState(key);
}

void ArcherPrefetchHandle::EndExactRequest(std::uint64_t request_id) {
  std::vector<ExactLayerKey> layer_keys;
  {
    std::lock_guard<std::mutex> exact_lock(exact_mutex_);
    auto it = request_layers_.find(request_id);
    if (it == request_layers_.end()) {
      return;
    }
    layer_keys = it->second;
    request_layers_.erase(it);
  }

  for (auto& key : layer_keys) {
    ReleaseLayer(key.request_id, key.layer_id);
  }
}

NodePtr ArcherPrefetchHandle::FindEvictableNode(
    int gpu_id, const NodePtr& protected_node, std::int64_t* live_bytes,
    std::size_t* evictable_nodes) const {
  auto inflight_node_ids = SnapshotInflightNodeIds();
  auto live_resident_nodes = GetLiveResidentNodes(gpu_id, live_bytes);
  NodePtr evict_node = nullptr;
  std::size_t evictable_count = 0;

  for (auto& node : live_resident_nodes) {
    if (protected_node != nullptr && node->id == protected_node->id) {
      continue;
    }
    if (inflight_node_ids.find(node->id) != inflight_node_ids.end()) {
      continue;
    }
    if (node->active_users.load() > 0 || node->pin_count.load() > 0) {
      continue;
    }
    evictable_count += 1;

    if (evict_node == nullptr ||
        node->last_access_time < evict_node->last_access_time) {
      evict_node = node;
    }
  }

  if (evictable_nodes != nullptr) {
    *evictable_nodes = evictable_count;
  }
  return evict_node;
}

void ArcherPrefetchHandle::EnsureExactQueueCapacity(int gpu_id,
                                                    const NodePtr& node) {
  bool logged_wait = false;
  while (resident_limits_[gpu_id] > 0) {
    std::int64_t live_bytes = 0;
    std::size_t evictable_nodes = 0;
    auto evict_node =
        FindEvictableNode(gpu_id, node, &live_bytes, &evictable_nodes);
    if (live_bytes + node->byte_size <= resident_limits_[gpu_id]) {
      return;
    }

    if (evict_node == nullptr) {
      if (PregatedDebugEnabled() && !logged_wait) {
        auto inflight_node_ids = SnapshotInflightNodeIds();
        DLOG_WARN("EnsureExactQueueCapacity waiting gpu ", gpu_id,
                  " live_bytes ", live_bytes, " limit ",
                  resident_limits_[gpu_id], " target_bytes ",
                  node->byte_size, " inflight ", inflight_node_ids.size(),
                  " evictable ", evictable_nodes, " protected ", node->str());
        for (auto& live_node : GetLiveResidentNodes(gpu_id)) {
          DLOG_WARN("  resident node ", live_node->str(), " pin ",
                    live_node->pin_count.load(), " active ",
                    live_node->active_users.load(), " inflight ",
                    inflight_node_ids.find(live_node->id) !=
                        inflight_node_ids.end());
        }
        logged_wait = true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      continue;
    }
    logged_wait = false;

    std::unique_lock<std::mutex> evict_lock(evict_node->mutex,
                                            std::try_to_lock);
    if (!evict_lock.owns_lock()) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
      continue;
    }

    bool inflight = false;
    {
      std::lock_guard<std::mutex> exact_lock(exact_mutex_);
      inflight = inflight_node_ids_.find(evict_node->id) !=
                 inflight_node_ids_.end();
    }

    if (!evict_node->device.is_cuda() || evict_node->device.index() != gpu_id ||
        evict_node->active_users.load() > 0 ||
        evict_node->pin_count.load() > 0 || inflight) {
      continue;
    }

    evict_node->SetDevice(evict_node->default_host, true, nullptr);
    exact_evicted_.fetch_add(1);
    if (PregatedDebugEnabled()) {
      std::int64_t live_bytes_after = 0;
      GetLiveResidentNodes(gpu_id, &live_bytes_after);
      DLOG_INFO("EnsureExactQueueCapacity evicted node ", evict_node->str(),
                " gpu ", gpu_id, " live_bytes ", live_bytes_after, " limit ",
                resident_limits_[gpu_id]);
    }
  }
}

void ArcherPrefetchHandle::MarkLayerNodeReady(const ExactLayerKey& key,
                                              const NodePtr& node) {
  std::shared_ptr<LayerPrepareState> state;
  {
    std::lock_guard<std::mutex> exact_lock(exact_mutex_);
    inflight_node_ids_.erase(node->id);
    auto it = exact_layer_states_.find(key);
    if (it == exact_layer_states_.end()) {
      return;
    }
    state = it->second;
  }

  {
    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (state->node_ids.find(node->id) != state->node_ids.end()) {
      state->ready_count += 1;
    }
  }
  state->cv.notify_all();
}

void ArcherPrefetchHandle::ExactPrepareWorker(int gpu_id) {
  cudaSetDevice(gpu_id);
  while (!exact_stop_.load()) {
    ExactTask task;
    {
      std::unique_lock<std::mutex> queue_lock(exact_queue_mutexes_[gpu_id]);
      exact_queue_cvs_[gpu_id].wait(queue_lock, [&] {
        return exact_stop_.load() || !exact_queues_[gpu_id].empty();
      });
      if (exact_stop_.load() && exact_queues_[gpu_id].empty()) {
        return;
      }

      task = std::move(exact_queues_[gpu_id].front());
      exact_queues_[gpu_id].pop_front();
    }

    if (task.stop || task.node == nullptr) {
      continue;
    }

    auto node = task.node;
    {
      std::unique_lock<std::mutex> node_lock(node->mutex);
      if (!IsNodeResidentOnDefaultDevice(node)) {
        EnsureExactQueueCapacity(gpu_id, node);
        if (node->device.is_cuda() && node->device.index() != gpu_id) {
          node->SetDevice(node->default_host, true, nullptr);
        }
        node->SetDevice(node->default_device, true, nullptr);
        node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
        exact_loaded_.fetch_add(1);
        if (PregatedDebugEnabled()) {
          std::int64_t live_bytes = 0;
          GetLiveResidentNodes(gpu_id, &live_bytes);
          DLOG_INFO("ExactPrepareWorker loaded node ", node->str(), " gpu ",
                    gpu_id, " live_bytes ", live_bytes, " limit ",
                    resident_limits_[gpu_id]);
        }
      }
    }

    MarkLayerNodeReady(task.key, node);
  }
}

std::unordered_map<std::string, std::int64_t>
ArcherPrefetchHandle::GetExactPrepareStats() const {
  std::unordered_map<std::string, std::int64_t> stats = {
      {"submitted", exact_submitted_.load()},
      {"skipped_resident", exact_skipped_resident_.load()},
      {"loaded", exact_loaded_.load()},
      {"evicted", exact_evicted_.load()},
      {"sync_fallback", exact_sync_fallback_.load()},
  };
  for (size_t gpu_id = 0; gpu_id < resident_limits_.size(); ++gpu_id) {
    std::int64_t live_bytes = 0;
    GetLiveResidentNodes(static_cast<int>(gpu_id), &live_bytes);
    stats["resident_bytes_gpu_" + std::to_string(gpu_id)] =
        live_bytes;
    stats["resident_limit_gpu_" + std::to_string(gpu_id)] =
        resident_limits_[gpu_id];
  }
  return stats;
}

bool ArcherPrefetchHandle::IsTensorOffloaded(const std::uint32_t tensor_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = kTensorIndex->find(tensor_id);
  // DLOG_TRACE("Check tensor {} {}", tensor_id, it == kTensorIndex->end());
  bool is_offloaded = it != kTensorIndex->end();
  if (is_offloaded) {
    it->second.id = tensor_id;
  }
  return is_offloaded;
}

void ArcherPrefetchHandle::SetTensorDevice(torch::Tensor& tensor,
                                           torch::Device device) const {
  void* device_ptr = nullptr;
  auto byte_size = tensor.element_size() * tensor.numel();

  DLOG_TRACE("Set tensor to device ", (void*)tensor.data_ptr(), device.str());

  // then copy to target device
  cudaSetDevice(device.index());
  cudaMalloc(&device_ptr, byte_size);

  CudaMemcpy(device_ptr, tensor.data_ptr(), byte_size,
             cudaMemcpyDeviceToDevice);

  auto new_tensor = torch::from_blob(
      device_ptr, tensor.sizes(), [](void* ptr) { cudaFree(ptr); },
      tensor.options().device(device).pinned_memory(false));
  tensor.set_data(new_tensor);
}

bool ArcherPrefetchHandle::IsTensorIndexInitialized() const {
  return kArcherTensorHandle->IsTensorIndexInitialized();
}

bool ArcherPrefetchHandle::IsTensorOnDevice(const torch::Tensor& tensor) const {
  auto tensor_id = kArcherTensorHandle->GetTensorId((void*)tensor.data_ptr());
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
  return node->device.is_cuda();
}

void ArcherPrefetchHandle::UpdateTensorMap(std::uint64_t old_data_ptr,
                                           std::uint64_t new_data_ptr) {
  kArcherTensorHandle->UpdateTensorMap((void*)old_data_ptr,
                                       (void*)new_data_ptr);
}

bool ArcherPrefetchHandle::IsTensorOnDevice(const TensorID tensor_id) const {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
  return node->device.is_cuda();
}

int ArcherPrefetchHandle::GetNodeDefaultDevice(
    std::vector<std::uint32_t> tensor_ids) const {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_ids[0]);
  // DLOG_TRACE("Get node {} default device {}", node->str(),
  return node->default_device.index();
}

int ArcherPrefetchHandle::GetNodeDevice(
    std::vector<std::uint32_t> tensor_ids) const {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_ids[0]);
  // DLOG_TRACE("Get node {} device {}", node->str(), node->device.str());
  return node->device.index();
}

// void ArcherPrefetchHandle::SetNodeCachePriority(const std::uint32_t
// tensor_id, const float priority) {
//     auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
//     node->cache_priority = priority;
// }
