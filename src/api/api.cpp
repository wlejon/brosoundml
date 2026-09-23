#include "api.h"
#include "host_soundml_internal.h"
#include "soundml_listen_internal.h"

#include <cstdio>
#include <utility>

namespace brosoundml::api {

namespace {

std::function<std::string(const std::string&)>& pathResolver() {
    static std::function<std::string(const std::string&)> r;
    return r;
}

std::function<void(const std::string&)>& logHook() {
    static std::function<void(const std::string&)> h;
    return h;
}

broaudio::Engine*& audioEngineSlot() {
    static broaudio::Engine* e = nullptr;
    return e;
}

InferenceScheduler& schedulerSlot() {
    static InferenceScheduler s;
    return s;
}

}  // namespace

broaudio::Engine* audioEngine() { return audioEngineSlot(); }

const InferenceScheduler& inferenceScheduler() { return schedulerSlot(); }

void setAudioEngine(broaudio::Engine* engine) { audioEngineSlot() = engine; }

void setInferenceScheduler(InferenceScheduler scheduler) {
    schedulerSlot() = std::move(scheduler);
}

std::string resolvePath(const std::string& path) {
    auto& r = pathResolver();
    return r ? r(path) : path;
}

void logInfo(const std::string& line) {
    auto& h = logHook();
    if (h) {
        h(line);
        return;
    }
    std::fprintf(stderr, "[INFO] %s\n", line.c_str());
}

void setPathResolver(std::function<std::string(const std::string&)> resolver) {
    pathResolver() = std::move(resolver);
}

void setLogHook(std::function<void(const std::string&)> hook) {
    logHook() = std::move(hook);
}

void tickSoundML() {
    tickAsyncJobs();
    tickWake();
    tickKws();
    tickGesture();
    tickVoiceAgent();
}

void tickSoundMLAsync() {
    tickAsyncJobs();
}

void shutdownSoundML() {
    shutdownAsyncJobs();
    // Tenants first (they detach from their streams — a scheduler barrier —
    // and release their rooted callbacks), then the streams and their
    // sources, while the audio engine and the scheduler are still alive.
    cleanupWake();
    cleanupKws();
    cleanupSense();
    cleanupGesture();
    shutdownListenHost();
}

namespace {

// The `bro` root of the calling thread's realm, created and registered when
// the host has not published one, then the installers over it. `compute`
// leaves out the listen tenants (api.h: installSoundMLCompute).
void installSoundMLOnto(bool compute) {
    // Every Value below that outlives an allocating call rides in a
    // Persistent (embed.h GC contract): getProperty, createObject,
    // registerGlobal and setProperty may each move everything.
    ev::Persistent globalRoot;
    {
        auto gt = ev::globalValue("globalThis");
        if (gt.found && ev::isObject(gt.value)) globalRoot.set(gt.value);
    }

    ev::Persistent broRoot;
    {
        auto bg = ev::globalValue("bro");
        if (bg.found && ev::isObject(bg.value)) broRoot.set(bg.value);
    }
    if (!ev::isObject(broRoot.get()) && ev::isObject(globalRoot.get())) {
        Value candidate = ev::getProperty(globalRoot.get(), "bro");
        if (ev::isObject(candidate)) broRoot.set(candidate);
    }
    if (!ev::isObject(broRoot.get())) {
        broRoot.set(ev::createObject());
        ev::registerGlobal("bro", broRoot.get());
        if (ev::isObject(globalRoot.get())) {
            ev::setProperty(globalRoot.get(), "bro", broRoot.get());
        }
    }

    ObjectBuilder bro(broRoot.get());

    installAsyncHandleClass();

    // Mount the audio/voice AI subsystems onto bro
    installStt(bro);
    installTts(bro);
    installDiar(bro);
    installRave(bro);
    installVoiceAgent(bro);
    if (!compute) {
        installWake(bro);
        installKws(bro);
        installSense(bro);
        installGesture(bro);
        installListen(bro);
    }

    // Keep updated bro global
    ev::registerGlobal("bro", bro.get());
    if (!ev::isUndefined(globalRoot.get())) {
        ev::setProperty(globalRoot.get(), "bro", bro.get());
    }
}

}  // namespace

void installSoundML() {
    installSoundMLOnto(/*compute=*/false);
}

void installSoundMLCompute() {
    installSoundMLOnto(/*compute=*/true);
}

} // namespace brosoundml::api
