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

    void* unwrap(Value val) const { return ev::handleData(val); }

    // True when `val` is an instance of THIS class (its prototype chain
    // reaches this class's prototype), so a method body can tell a Whisper
    // handle from a Parakeet handle before it casts the payload.
    bool isInstance(Value val) const;

    bool installed() const { return proto_ != nullptr; }

    Value prototype() const;
    Value constructor() const;

private:
    ev::Persistent* proto_ = nullptr;
    ev::Persistent* ctor_ = nullptr;
};

Value hostArrayOf(size_t count, const std::function<Value(size_t)>& make);

} // namespace brosoundml::api
