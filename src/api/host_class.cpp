#include "host_class.h"
#include "object_builder.h"

#include <unordered_map>

namespace brosoundml::api {

namespace {

// One table per thread, keyed by the class object (host_class.h): the
// constructor and prototype a class holds are Persistents of the thread
// that installed it, so a second thread's realm gets a second, independent
// pair. Never freed — a thread's runtime lives until the thread does.
std::unordered_map<const HostClass*, HostClass::Slots>& threadSlots() {
    static thread_local std::unordered_map<const HostClass*, HostClass::Slots> t;
    return t;
}

}  // namespace

HostClass::Slots& HostClass::slots() const {
    return threadSlots()[this];
}

const HostClass::Slots* HostClass::slotsIfAny() const {
    auto& t = threadSlots();
    auto it = t.find(this);
    return it == t.end() ? nullptr : &it->second;
}

bool HostClass::installed() const {
    const Slots* s = slotsIfAny();
    return s && s->proto;
}

void HostClass::install(const char* name, uint32_t arity, ev::NativeFn body,
                        const std::function<void(ObjectBuilder&)>& decorate,
                        bool global) {
    ev::NativeFn ctorBody = body;
    if (!ctorBody) {
        std::string msg = std::string(name) +
            " is not constructible: instances come from the loader that returns one";
        ctorBody = [msg](Value, std::span<const Value>) { return ev::throwTypeError(msg); };
    }

    Slots& s = slots();
    ev::Persistent ctor(ev::makeFunction(std::move(ctorBody), arity, name));
    s.ctor = new ev::Persistent(ctor.get());

    {
        ObjectBuilder proto(ev::getProperty(ctor.get(), "prototype"));
        proto.set("constructor", ctor.get());
        if (decorate) decorate(proto);
        s.proto = new ev::Persistent(proto.get());
    }

    if (!global) return;
    ev::registerGlobal(name, s.ctor->get());
    ev::GlobalValue gt = ev::globalValue("globalThis");
    if (gt.found && !gt.value.isUndefined() && ev::isObject(gt.value)) {
        ev::setProperty(gt.value, name, s.ctor->get());
    }
}

void HostClass::alias(const char* name) const {
    const Slots* s = slotsIfAny();
    if (!s || !s->ctor) return;
    ev::registerGlobal(name, s->ctor->get());
    ev::GlobalValue gt = ev::globalValue("globalThis");
    if (gt.found && !gt.value.isUndefined() && ev::isObject(gt.value)) {
        ev::setProperty(gt.value, name, s->ctor->get());
    }
}

void HostClass::inherit(const HostClass& base) const {
    const Slots* s = slotsIfAny();
    const Slots* b = base.slotsIfAny();
    if (!s || !s->proto || !b || !b->proto) return;
    ev::GlobalValue objectCtor = ev::globalValue("Object");
    if (!objectCtor.found || !ev::isObject(objectCtor.value)) return;
    ev::Persistent objectNs(objectCtor.value);
    ev::Persistent setProto(ev::getProperty(objectNs.get(), "setPrototypeOf"));
    if (!ev::isFunction(setProto.get())) return;
    const Value args[2] = {s->proto->get(), b->proto->get()};
    ev::call(setProto.get(), ev::undefined(), std::span<const Value>(args, 2));
}

bool HostClass::isInstance(Value val) const {
    const Slots* s = slotsIfAny();
    if (!s || !s->proto || !ev::isObject(val)) return false;
    ev::Persistent target(val);
    ev::GlobalValue objectCtor = ev::globalValue("Object");
    if (!objectCtor.found || !ev::isObject(objectCtor.value)) return false;
    ev::Persistent objectNs(objectCtor.value);
    ev::Persistent objectProto(ev::getProperty(objectNs.get(), "prototype"));
    ev::Persistent isProtoOf(ev::getProperty(objectProto.get(), "isPrototypeOf"));
    if (!ev::isFunction(isProtoOf.get())) return false;
    const Value args[1] = {target.get()};
    ev::CallResult r = ev::call(isProtoOf.get(), s->proto->get(), std::span<const Value>(args, 1));
    if (r.thrown) return false;
    return ev::isBool(r.value) && ev::toBool(r.value);
}

Value HostClass::make(void* data, ev::HandleDestructor dtor, ev::Finalize when) const {
    const Slots* s = slotsIfAny();
    if (!s || !s->proto) return ev::makeHandle(data, dtor, when);
    return ev::makeHandle(data, dtor, when, s->proto->get());
}

void HostClass::setStatic(const char* name, Value v) const {
    const Slots* s = slotsIfAny();
    if (!s || !s->ctor) return;
    s->ctor->set(ev::setProperty(s->ctor->get(), name, v));
}

Value HostClass::prototype() const {
    const Slots* s = slotsIfAny();
    return (s && s->proto) ? s->proto->get() : ev::undefined();
}

Value HostClass::constructor() const {
    const Slots* s = slotsIfAny();
    return (s && s->ctor) ? s->ctor->get() : ev::undefined();
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
