#include "api.h"
#include "host_soundml_internal.h"

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

}  // namespace

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
}

void shutdownSoundML() {
    shutdownAsyncJobs();
}

void installSoundML() {
    Value globalThisVal = ev::undefined();
    auto gt = ev::globalValue("globalThis");
    if (gt.found && ev::isObject(gt.value)) {
        globalThisVal = gt.value;
    }
    ev::Persistent globalRoot(globalThisVal);

    Value broVal = ev::globalValue("bro").found ? ev::globalValue("bro").value : ev::undefined();
    if (!ev::isObject(broVal)) {
        if (!ev::isUndefined(globalRoot.get())) {
            Value candidate = ev::getProperty(globalRoot.get(), "bro");
            if (ev::isObject(candidate)) {
                broVal = candidate;
            }
        }
    }
    if (!ev::isObject(broVal)) {
        broVal = ev::createObject();
        ev::registerGlobal("bro", broVal);
        if (!ev::isUndefined(globalRoot.get())) {
            ev::setProperty(globalRoot.get(), "bro", broVal);
        }
    }

    ObjectBuilder bro(broVal);

    installAsyncHandleClass();

    // Mount the 9 audio/voice AI subsystems onto bro
    installStt(bro);
    installTts(bro);
    installDiar(bro);
    installRave(bro);
    installWakeAndKws(bro);
    installListenSenseGesture(bro);

    // Keep updated bro global
    ev::registerGlobal("bro", bro.get());
    if (!ev::isUndefined(globalRoot.get())) {
        ev::setProperty(globalRoot.get(), "bro", bro.get());
    }
}

} // namespace brosoundml::api
