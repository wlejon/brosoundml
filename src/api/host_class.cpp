#include "host_class.h"
#include "object_builder.h"

#include <mutex>
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

// registerGlobal + globalThis[name]. The value and globalThis are rooted:
// registerGlobal and setProperty both allocate.
void publishGlobal(const char* name, Value val) {
    ev::Persistent v(val);
    ev::registerGlobal(name, v.get());
    ev::GlobalValue gt = ev::globalValue("globalThis");
    if (gt.found && ev::isObject(gt.value)) {
        ev::Persistent global(gt.value);
        ev::setProperty(global.get(), name, v.get());
    }
}

// ── Brands ──────────────────────────────────────────────────────────────────
// ev::handleData answers the payload of ANY handle, and a prototype-chain test
// is fooled by Object.setPrototypeOf. Every payload made here is registered
// with the class that made it; unwrap()/isInstance() answer only for their
// own. The lookup allocates nothing, so a caller's raw Values stay valid.
struct Brand {
    const HostClass* cls;
    ev::HandleDestructor dtor;
};

std::mutex& brandMutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<const void*, Brand>& brands() {
    static auto* m = new std::unordered_map<const void*, Brand>();  // outlives every sweep
    return *m;
}

// The destructor every branded handle carries: unregister, then run the
// class's own destructor.
void brandedDestroy(void* data) {
    ev::HandleDestructor dtor = nullptr;
    {
        std::lock_guard<std::mutex> lk(brandMutex());
        auto& m = brands();
        auto it = m.find(data);
        if (it != m.end()) {
            dtor = it->second.dtor;
            m.erase(it);
        }
    }
    if (dtor) dtor(data);
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
    publishGlobal(name, s.ctor->get());
}

void HostClass::alias(const char* name) const {
    const Slots* s = slotsIfAny();
    if (!s || !s->ctor) return;
    publishGlobal(name, s->ctor->get());
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
    return unwrap(val) != nullptr;
}

void* HostClass::unwrap(Value val) const {
    void* data = ev::handleData(val);
    if (!data) return nullptr;
    std::lock_guard<std::mutex> lk(brandMutex());
    auto& m = brands();
    auto it = m.find(data);
    return (it != m.end() && it->second.cls == this) ? data : nullptr;
}

Value HostClass::make(void* data, ev::HandleDestructor dtor, ev::Finalize when) const {
    if (data) {
        std::lock_guard<std::mutex> lk(brandMutex());
        brands()[data] = Brand{this, dtor};
    }
    const Slots* s = slotsIfAny();
    if (!s || !s->proto) return ev::makeHandle(data, brandedDestroy, when);
    return ev::makeHandle(data, brandedDestroy, when, s->proto->get());
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
