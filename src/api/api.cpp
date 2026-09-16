#include "api.h"
#include "host_soundml_internal.h"

namespace brosoundml::api {

void installSoundML() {
    Value globalThisVal = ev::undefined();
    auto gt = ev::globalValue("globalThis");
    if (gt.found && ev::isObject(gt.value)) {
        globalThisVal = gt.value;
    }

    Value broVal = ev::globalValue("bro").found ? ev::globalValue("bro").value : ev::undefined();
    if (!ev::isObject(broVal)) {
        if (!ev::isUndefined(globalThisVal)) {
            Value candidate = ev::getProperty(globalThisVal, "bro");
            if (ev::isObject(candidate)) {
                broVal = candidate;
            }
        }
    }
    if (!ev::isObject(broVal)) {
        broVal = ev::createObject();
        ev::registerGlobal("bro", broVal);
        if (!ev::isUndefined(globalThisVal)) {
            ev::setProperty(globalThisVal, "bro", broVal);
        }
    }

    ObjectBuilder bro(broVal);

    // Mount the 9 audio/voice AI subsystems onto bro
    installStt(bro);
    installTts(bro);
    installDiar(bro);
    installRave(bro);
    installWakeAndKws(bro);
    installListenSenseGesture(bro);

    // Keep updated bro global
    ev::registerGlobal("bro", bro.get());
    if (!ev::isUndefined(globalThisVal)) {
        ev::setProperty(globalThisVal, "bro", bro.get());
    }
}

} // namespace brosoundml::api
