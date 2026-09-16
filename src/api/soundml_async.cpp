#include "soundml_async.h"
#include "host_class.h"
#include "object_builder.h"

#include <chrono>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace brosoundml::api {

namespace {

struct AsyncJob {
    std::shared_ptr<HostAsyncHandle> handle;
    AsyncWorkFn work;
    AsyncPollFn poll;
    AsyncDoneFn done;
    std::atomic<bool> workDone{false};
    std::string error;       // written by the work thread before workDone
    std::thread worker;

    ~AsyncJob() {
        if (worker.joinable()) worker.join();
    }
};

struct Registry {
    std::mutex m;
    std::vector<std::shared_ptr<AsyncJob>> jobs;
};

Registry& registry() {
    static thread_local Registry r;
    return r;
}

HostClass g_asyncHandleClass;

std::shared_ptr<HostAsyncHandle>* handleOf(Value v) {
    return static_cast<std::shared_ptr<HostAsyncHandle>*>(g_asyncHandleClass.unwrap(v));
}

// Finish one job on the JS thread: join, final poll, done(). The job has
// already been removed from the registry by the caller.
void finishJob(const std::shared_ptr<AsyncJob>& job) {
    if (job->worker.joinable()) job->worker.join();
    if (job->poll) job->poll();
    const bool cancelled = job->handle->cancelled.load(std::memory_order_acquire);
    if (job->done) job->done(cancelled, job->error);
    job->handle->finished.store(true, std::memory_order_release);
}

}  // namespace

void installAsyncHandleClass() {
    if (g_asyncHandleClass.installed()) return;
    g_asyncHandleClass.install("AsyncHandle", 0, nullptr, [](ObjectBuilder& b) {
        b.def("cancel", 0, [](Value self, std::span<const Value>) -> Value {
            if (auto* h = handleOf(self); h && *h) {
                (*h)->cancelled.store(true, std::memory_order_release);
            }
            return ev::undefined();
        });
        // Block the JS thread until the job has delivered its callbacks.
        // For scripts that want the synchronous answer; ticks the registry
        // itself so the frame pump is not needed.
        b.def("wait", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = handleOf(self);
            if (!h || !*h) return ev::undefined();
            while (!(*h)->finished.load(std::memory_order_acquire)) {
                tickAsyncJobs();
                if ((*h)->finished.load(std::memory_order_acquire)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            return ev::undefined();
        });
        b.accessor("finished", [](Value self, std::span<const Value>) -> Value {
            auto* h = handleOf(self);
            return ev::fromBool(h && *h && (*h)->finished.load(std::memory_order_acquire));
        });
        b.accessor("cancelled", [](Value self, std::span<const Value>) -> Value {
            auto* h = handleOf(self);
            return ev::fromBool(h && *h && (*h)->cancelled.load(std::memory_order_acquire));
        });
    }, /*global=*/false);
}

Value launchAsyncJob(AsyncWorkFn work, AsyncPollFn poll, AsyncDoneFn done) {
    installAsyncHandleClass();
    auto job = std::make_shared<AsyncJob>();
    job->handle = std::make_shared<HostAsyncHandle>();
    job->work = std::move(work);
    job->poll = std::move(poll);
    job->done = std::move(done);

    {
        std::lock_guard<std::mutex> lock(registry().m);
        registry().jobs.push_back(job);
    }

    // The thread captures the job by shared_ptr: the registry may drop its
    // reference (shutdown) while the work is still running.
    job->worker = std::thread([job] {
        try {
            if (job->work) job->work(job->handle->cancelled);
        } catch (const std::exception& e) {
            job->error = e.what();
            if (job->error.empty()) job->error = "operation failed";
        } catch (...) {
            job->error = "operation failed (unknown exception)";
        }
        job->workDone.store(true, std::memory_order_release);
    });

    auto* payload = new std::shared_ptr<HostAsyncHandle>(job->handle);
    return g_asyncHandleClass.make(payload, [](void* p) {
        delete static_cast<std::shared_ptr<HostAsyncHandle>*>(p);
    });
}

void tickAsyncJobs() {
    Registry& reg = registry();
    std::vector<std::shared_ptr<AsyncJob>> snapshot;
    {
        std::lock_guard<std::mutex> lock(reg.m);
        if (reg.jobs.empty()) return;
        snapshot = reg.jobs;
    }
    for (auto& job : snapshot) {
        if (job->poll) job->poll();
        if (!job->workDone.load(std::memory_order_acquire)) continue;
        {
            // Remove BEFORE done(): done() may launch a new job on this
            // model, and must never see this one still registered.
            std::lock_guard<std::mutex> lock(reg.m);
            auto& v = reg.jobs;
            for (size_t i = 0; i < v.size(); ++i) {
                if (v[i] == job) { v.erase(v.begin() + static_cast<std::ptrdiff_t>(i)); break; }
            }
        }
        finishJob(job);
    }
}

bool hasAsyncJobs() {
    std::lock_guard<std::mutex> lock(registry().m);
    return !registry().jobs.empty();
}

void shutdownAsyncJobs() {
    Registry& reg = registry();
    std::vector<std::shared_ptr<AsyncJob>> all;
    {
        std::lock_guard<std::mutex> lock(reg.m);
        all.swap(reg.jobs);
    }
    for (auto& job : all) job->handle->cancelled.store(true, std::memory_order_release);
    for (auto& job : all) finishJob(job);
}

} // namespace brosoundml::api
