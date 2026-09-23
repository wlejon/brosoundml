// Shared model-loader plumbing for the bro.* namespaces: the argument
// prologue (path + device, GPU by default) and the sync-or-async build runner
// (opts.onReady / opts.onError turn a loader into a background job).
#pragma once

#include "host_soundml_internal.h"

namespace brosoundml::api {

// Run a build on the JS thread (sync) or on a work thread (opts.onReady is a
// function), wrapping the result in `cls`. The build touches no JS state.
// `opts` must be fresh (read since the caller's last allocation); it is
// rooted here before the first property read.
template <typename W>
Value runModelLoader(const char* fn, Value opts, const HostClass& cls,
                     std::function<std::unique_ptr<W>()> build) {
    struct State {
        std::string fn;
        const HostClass* cls = nullptr;
        std::function<std::unique_ptr<W>()> build;
        std::unique_ptr<W> w;
        ev::Persistent onReady, onError;
    };
    ev::Persistent optsRoot(opts);
    auto st = std::make_shared<State>();
    st->fn = fn;
    st->cls = &cls;
    st->build = std::move(build);
    st->onReady = getFunctionOpt(optsRoot.get(), "onReady");
    st->onError = getFunctionOpt(optsRoot.get(), "onError");
    if (!ev::isFunction(st->onReady.get())) {
        try {
            return cls.createInstance(st->build());
        } catch (const std::exception& e) {
            return ev::throwError(st->fn + ": " + e.what());
        }
    }
    auto work = [st](const std::atomic<bool>&) { st->w = st->build(); };
    auto done = [st](bool, const std::string& error) {
        if (!error.empty() || !st->w) {
            if (ev::isFunction(st->onError.get())) {
                ev::Persistent msg(ev::fromUtf8(error.empty() ? st->fn + " failed" : error));
                callCallback1(st->onError.get(), msg.get());
            }
            return;
        }
        ev::Persistent inst(st->cls->createInstance(std::move(st->w)));
        callCallback1(st->onReady.get(), inst.get());
    };
    return launchAsyncJob(std::move(work), nullptr, std::move(done));
}

// Device + opts prologue shared by every loader once its path arguments are
// checked: brotensor init, GPU-first auto device, strict opts.device parse.
// False with a pending TypeError on a bad device option. `opts` comes back
// rooted (the device parse allocates; a raw Value would be stale).
inline bool loaderDevice(const char* fn, std::span<const Value> args, std::size_t optsIndex,
                         brotensor::Device& dev, ev::Persistent& opts, bool* explicitDevice = nullptr) {
    brotensor::init();
    dev = autoDevice();
    opts.set(isObjectArg(args, optsIndex) ? args[optsIndex] : ev::undefined());
    std::string err;
    if (!parseDeviceOpt(opts.get(), dev, err, explicitDevice)) {
        ev::throwTypeError(std::string(fn) + ": " + err);
        return false;
    }
    return true;
}

// The one-path form: `fn(modelDir, opts?)`.
inline bool modelLoaderArgs(const char* fn, std::span<const Value> args, std::string& dir,
                            brotensor::Device& dev, ev::Persistent& opts, bool* explicitDevice = nullptr) {
    if (!isStringArg(args, 0)) {
        ev::throwTypeError(std::string(fn) + "(modelDir, opts?): path required");
        return false;
    }
    dir = resolvePath(strAt(args, 0));
    return loaderDevice(fn, args, 1, dev, opts, explicitDevice);
}

} // namespace brosoundml::api
