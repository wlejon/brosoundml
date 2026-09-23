#pragma once

#include "embed/embed.h"
#include "object_builder.h"

#include <functional>
#include <memory>
#include <span>
#include <string>

namespace brosoundml::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

// A host class: one constructor function + one shared prototype, with
// instances created by make()/createInstance() as handles born on that
// prototype (so `x instanceof Ctor` holds and methods live once, on the
// prototype). A class whose `body` is null is not constructible from JS —
// `new Ctor()` throws a TypeError naming it — which is what every model class
// wants: instances come from the loaders.
class HostClass {
public:
    // `global`: also publish the constructor under `name` in bronze's
    // host-global registry and on globalThis. Off for a class whose name
    // another sibling already owns (brolm registers `AsyncHandle`; a second
    // registration would replace it and break instanceof against the first).
    void install(const char* name, uint32_t arity, ev::NativeFn body,
                 const std::function<void(ObjectBuilder&)>& decorate = nullptr,
                 bool global = true);

    void init(const char* name, const std::function<void(ObjectBuilder&)>& decorate) {
        install(name, 0, nullptr, decorate);
    }

    template <typename T>
    Value createInstance(std::unique_ptr<T> ptr) const {
        return make(ptr.release(), [](void* p) { delete static_cast<T*>(p); });
    }

    void alias(const char* name) const;
    void inherit(const HostClass& base) const;

    Value make(void* data, ev::HandleDestructor dtor,
               ev::Finalize when = ev::Finalize::InSweep) const;

    void setStatic(const char* name, Value v) const;

    // The payload of a handle THIS class made; nullptr for anything else,
    // including a handle of another class (host_class.cpp, brands).
    void* unwrap(Value val) const;

    // True when `val` is a handle THIS class made, so a method body can tell
    // a Whisper handle from a Parakeet handle before it casts the payload.
    // Allocates nothing.
    bool isInstance(Value val) const;

    // Whether install() has run on the CALLING thread.
    bool installed() const;

    Value prototype() const;
    Value constructor() const;

    // The class objects are process-global (`HostClass g_whisperModelClass`),
    // but what they hold is PER THREAD: bronze's runtime is per-thread, and
    // a Persistent is a slot in its creating thread's registry, so a
    // constructor made on the main thread means nothing to a Worker's realm.
    // Every accessor reads the CALLING thread's slots and install() fills
    // the calling thread's; a realm installs each class once, and
    // installed() answers for the calling thread.
    struct Slots {
        ev::Persistent* proto = nullptr;
        ev::Persistent* ctor = nullptr;
    };

private:
    Slots& slots() const;
    const Slots* slotsIfAny() const;
};

Value hostArrayOf(size_t count, const std::function<Value(size_t)>& make);

} // namespace brosoundml::api
