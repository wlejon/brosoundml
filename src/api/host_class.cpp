#include "host_class.h"
#include "object_builder.h"

namespace brosoundml::api {

void HostClass::install(const char* name, uint32_t arity, ev::NativeFn body,
                        const std::function<void(ObjectBuilder&)>& decorate) {
    ev::NativeFn ctorBody = body;
    if (!ctorBody) {
        std::string msg = std::string("TypeError: ") + name + " is not a constructor";
        ctorBody = [msg](Value, std::span<const Value>) { return ev::throwTypeError(msg); };
    }

    ev::Persistent ctor(ev::makeFunction(std::move(ctorBody), arity, name));
    ctor_ = new ev::Persistent(ctor.get());

    {
        ObjectBuilder proto(ev::getProperty(ctor.get(), "prototype"));
        proto.set("constructor", ctor.get());
        if (decorate) decorate(proto);
        proto_ = new ev::Persistent(proto.get());
    }

    ev::registerGlobal(name, ctor_->get());
    ev::GlobalValue gt = ev::globalValue("globalThis");
    if (gt.found && !gt.value.isUndefined() && ev::isObject(gt.value)) {
        ev::setProperty(gt.value, name, ctor_->get());
    }
}

void HostClass::alias(const char* name) const {
    if (!ctor_) return;
    ev::registerGlobal(name, ctor_->get());
    ev::GlobalValue gt = ev::globalValue("globalThis");
    if (gt.found && !gt.value.isUndefined() && ev::isObject(gt.value)) {
        ev::setProperty(gt.value, name, ctor_->get());
    }
}

void HostClass::inherit(const HostClass& base) const {
    if (!proto_ || !base.proto_) return;
    ev::GlobalValue objectCtor = ev::globalValue("Object");
    if (!objectCtor.found || !ev::isObject(objectCtor.value)) return;
    ev::Persistent objectNs(objectCtor.value);
    ev::Persistent setProto(ev::getProperty(objectNs.get(), "setPrototypeOf"));
    if (!ev::isFunction(setProto.get())) return;
    const Value args[2] = {proto_->get(), base.proto_->get()};
    ev::call(setProto.get(), ev::undefined(), std::span<const Value>(args, 2));
}

Value HostClass::make(void* data, ev::HandleDestructor dtor, ev::Finalize when) const {
    if (!proto_) return ev::makeHandle(data, dtor, when);
    return ev::makeHandle(data, dtor, when, proto_->get());
}

void HostClass::setStatic(const char* name, Value v) const {
    if (!ctor_) return;
    ctor_->set(ev::setProperty(ctor_->get(), name, v));
}

Value HostClass::prototype() const {
    return proto_ ? proto_->get() : ev::undefined();
}

Value HostClass::constructor() const {
    return ctor_ ? ctor_->get() : ev::undefined();
}

Value hostArrayOf(size_t count, const std::function<Value(size_t)>& make) {
    ev::CallResult parsed = ev::parseJson("[]");
    if (parsed.thrown) {
        return ev::undefined();
    }
    ev::Persistent arr(parsed.value);
    if (count == 0) return arr.get();

    ev::Persistent push(ev::getProperty(arr.get(), "push"));
    if (!ev::isFunction(push.get())) {
        return arr.get();
    }
    for (size_t i = 0; i < count; ++i) {
        Value v = make(i);
        ev::CallResult r = ev::call(push.get(), arr.get(),
                                    std::span<const Value>(&v, 1));
        if (r.thrown) {
            break;
        }
    }
    return arr.get();
}

} // namespace brosoundml::api
