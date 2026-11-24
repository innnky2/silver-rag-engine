#pragma once
// threadpool.h — persistent pinned thread pool with chunked parallel_for.
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace ours {

inline void pin_thread_to_core(int core) {
  if (core < 0) return;
#if defined(_WIN32)
  if (core < (int)(sizeof(DWORD_PTR) * 8)) {
    DWORD_PTR mask = (DWORD_PTR)1 << core;
    if (mask) SetThreadAffinityMask(GetCurrentThread(), mask);
  }
#else
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

inline int default_threads(int want) {
  if (want > 0) return want > 64 ? 64 : want;
  unsigned hw = std::thread::hardware_concurrency();
  int n = hw ? (int)hw : 4;
  return n > 64 ? 64 : n;
}

class ThreadPool {
 public:
  explicit ThreadPool(int nthreads = 0, bool pin = true) : stop_(false), pin_(pin) {
    start_(default_threads(nthreads));
  }
  ~ThreadPool() { shutdown(); }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  // Move ops deleted: worker threads capture `this`; moving thread objects
  // would dangle those pointers (silent pool death). Use reset() instead.
  ThreadPool(ThreadPool&& o) noexcept = delete;
  ThreadPool& operator=(ThreadPool&& o) noexcept = delete;

  int size() const { return (int)workers_.size(); }

  void reset(int nthreads, bool pin) {
    shutdown();
    pin_ = pin;
    stop_ = false;
    active_ = 0;
    start_(default_threads(nthreads));
  }

  // Chunked parallel for: one job per worker (no per-row job flood).
  void parallel_for(size_t n, const std::function<void(size_t)>& fn) {
    if (n == 0) return;
    if (workers_.empty()) {
      for (size_t i = 0; i < n; ++i) fn(i);
      return;
    }
    size_t W = workers_.size();
    size_t chunk = (n + W - 1) / W;
    {
      std::unique_lock<std::mutex> lk(done_mu_);
      active_ = (int)W;
    }
    {
      std::unique_lock<std::mutex> lk(mu_);
      for (size_t w = 0; w < W; ++w) {
        size_t lo = w * chunk, hi = lo + chunk < n ? lo + chunk : n;
        jobs_.push([fn, lo, hi]() {
          for (size_t i = lo; i < hi; ++i) fn(i);
        });
      }
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lk(done_mu_);
    done_cv_.wait(lk, [this]() { return active_ == 0; });
  }

 private:
  void start_(int n) {
    for (int i = 0; i < n; ++i) {
      workers_.emplace_back([this, i, n]() {
        if (pin_) pin_thread_to_core(n > 0 ? (i % n) : i);
        for (;;) {
          std::function<void()> job;
          {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]() { return stop_ || !jobs_.empty(); });
            if (stop_ && jobs_.empty()) return;
            job = std::move(jobs_.front());
            jobs_.pop();
          }
          job();
          {
            std::unique_lock<std::mutex> lk(done_mu_);
            if (--active_ == 0) done_cv_.notify_all();
          }
        }
      });
    }
  }
  void shutdown() {
    {
      std::unique_lock<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) if (t.joinable()) t.join();
    workers_.clear();
    std::queue<std::function<void()>> empty;
    std::swap(jobs_, empty);
  }
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> jobs_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::mutex done_mu_;
  std::condition_variable done_cv_;
  int active_ = 0;
  bool stop_ = false;
  bool pin_ = true;
};

}  // namespace ours
