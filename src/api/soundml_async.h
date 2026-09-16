#pragma once

// ─── Async-job runner for the brosoundml JS API ──────────────────────────────
//
// Runs a blocking model operation (a load, a transcription, a synthesis) on a
// background thread and delivers its result back onto the JS thread on that
// thread's next tick. The JS side stays non-blocking and every job is
// cancellable through the AsyncHandle it returns.
//
// A job is three lambdas:
//   work(cancel)  BACKGROUND thread. Pure C++: it may touch brotensor and the
//                 model, never the bronze heap. Reads `cancel` to stop early;
//                 may throw — the text reaches done() as `error`.
//   poll()        JS thread, once per tick WHILE the job runs and once more
//                 right after the work thread has finished (streaming: drain
//                 tokens / steps published lock-free by work()).
//   done(cancelled, error)
//                 JS thread, exactly once when the job completes, was
//                 cancelled, or threw. Builds the result values and fires the
//                 JS callbacks the launcher rooted in Persistents.
//
// The data plane is lock-free: work() is the sole writer of a committed
// prefix published through an atomic, poll() the sole reader. The registry
// of in-flight jobs (control plane) is a mutex-guarded vector. The registry
// is per thread — a realm runs on exactly one thread — so tickAsyncJobs()
// on the engine's frame pump only ever sees the main realm's jobs.
//
// A model handle's single-owner gate (ModelGate, host_soundml_internal.h) is
// claimed by the launcher and released in done() BEFORE the callbacks fire,
// so an onDone that synchronously starts the next op on the same model (a
// serialized queue) succeeds instead of tripping the in-flight guard.

#include "embed/embed.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>

namespace brosoundml::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

struct HostAsyncHandle {
    std::atomic<bool> cancelled{false};
    std::atomic<bool> finished{false};
};

using AsyncWorkFn = std::function<void(const std::atomic<bool>& cancel)>;
using AsyncPollFn = std::function<void()>;
using AsyncDoneFn = std::function<void(bool cancelled, const std::string& error)>;

// Install the AsyncHandle class (once per realm). Not published as a global:
// brolm owns the `AsyncHandle` global name; instances here are duck-typed
// ({ cancel(), wait() }) exactly like brolm's.
void installAsyncHandleClass();

// Spawn `work` on a background thread, register the job on the calling
// thread's registry, and return the AsyncHandle JS object ({ cancel(),
// wait() }). `poll` may be empty.
Value launchAsyncJob(AsyncWorkFn work, AsyncPollFn poll, AsyncDoneFn done);

// Drain streaming output and finish completed jobs. Call once per frame on
// the thread that owns the realm (the engine pumps it every frame, headless
// advanceTime() included). Cheap no-op with nothing in flight. Re-entrant
// with respect to launches: a done() that starts a new job is fine.
void tickAsyncJobs();

// True while any job launched from this thread is still running.
bool hasAsyncJobs();

// Cancel and join every in-flight job, then run their done() so rooted
// callbacks are released. Call at realm teardown on the owning thread.
void shutdownAsyncJobs();

} // namespace brosoundml::api
