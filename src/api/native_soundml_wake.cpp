#include "soundml_legacy_compat.h"

#include <brosoundml/wake.h>
#include <brosoundml/phoneme_spotter.h>
#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace brosoundml::api {

namespace {

HostClass g_wakeViewClass;
HostClass g_kwsViewClass;

struct HostWakeState {
    std::shared_ptr<brosoundml::WakeWord> wake;
    bool active = false;
    bool suspended = false;
    bool loaded = false;
    double threshold = 0.85;
    double lastScore = 0.0;
    double rollingPeak = 0.0;
    double scoreMax = 0.0;
    uint64_t framesDelivered = 0;
    uint64_t samplesDelivered = 0;
};

struct HostKwsState {
    std::shared_ptr<brosoundml::PhonemeSpotter> spotter;
    bool active = false;
    bool suspended = false;
    bool loaded = false;
    int sampleRate = 16000;
    double prefixProgress = 0.0;
    double rollingPeak = 0.0;
    uint64_t framesDelivered = 0;
    uint64_t samplesDelivered = 0;
    std::vector<std::string> templates;
};

static HostWakeState g_defaultWake;
static HostKwsState  g_defaultKws;

Value makeWakeStats(const HostWakeState& s) {
    if (!s.active && !s.loaded) return ev::null();
    ObjectBuilder st;
    st.set("framesDelivered", static_cast<double>(s.framesDelivered));
    st.set("samplesDelivered", static_cast<double>(s.samplesDelivered));
    st.set("rollingPeak", s.rollingPeak);
    st.set("scoreMax", s.scoreMax);
    return st.get();
}

Value makeKwsStats(const HostKwsState& s) {
    if (!s.active && !s.loaded) return ev::null();
    ObjectBuilder st;
    st.set("framesDelivered", static_cast<double>(s.framesDelivered));
    st.set("samplesDelivered", static_cast<double>(s.samplesDelivered));
    st.set("rollingPeak", s.rollingPeak);
    return st.get();
}

void feedWake(HostWakeState& s, const std::vector<float>& samples) {
    if (samples.empty()) return;
    s.samplesDelivered += samples.size();
    s.framesDelivered += samples.size() / 160;
    float pk = 0.0f;
    for (float f : samples) {
        float af = std::abs(f);
        if (af > pk) pk = af;
    }
    s.rollingPeak = pk;
    if (s.wake && s.loaded) {
        try {
            s.wake->feed(samples.data(), static_cast<int>(samples.size()));
            float sc = s.wake->last_score();
            s.lastScore = sc;
            if (sc > s.scoreMax) s.scoreMax = sc;
        } catch (...) {}
    }
}

void feedKws(HostKwsState& s, const std::vector<float>& samples) {
    if (samples.empty()) return;
    s.samplesDelivered += samples.size();
    s.framesDelivered += samples.size() / 160;
    float pk = 0.0f;
    for (float f : samples) {
        float af = std::abs(f);
        if (af > pk) pk = af;
    }
    s.rollingPeak = pk;
}

} // namespace

Value createWakeStreamViewHandle() {
    auto* s = new HostWakeState();
    return g_wakeViewClass.make(s, [](void* p) { delete static_cast<HostWakeState*>(p); });
}

Value createKwsStreamViewHandle() {
    auto* s = new HostKwsState();
    return g_kwsViewClass.make(s, [](void* p) { delete static_cast<HostKwsState*>(p); });
}

void installWakeAndKws(ObjectBuilder& bro) {
    // -----------------------------------------------------------------------
    // WakeStreamView
    // -----------------------------------------------------------------------
    g_wakeViewClass.install("WakeStreamView", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* s = new HostWakeState();
            return g_wakeViewClass.make(s, [](void* p) { delete static_cast<HostWakeState*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("active", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                return ev::fromBool(s && s->active);
            });
            b.def("listen", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                if (s) {
                    s->active = true;
                    s->suspended = false;
                    Value opts = argAt(a, 0);
                    s->threshold = getPropertyDouble(opts, "threshold", s->threshold);
                    if (s->wake) s->wake->set_threshold(static_cast<float>(s->threshold));
                }
                return ev::undefined();
            });
            b.def("stop", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                if (s) s->active = false;
                return ev::undefined();
            });
            b.def("suspend", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                if (s) s->suspended = true;
                return ev::undefined();
            });
            b.def("resume", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                if (s) s->suspended = false;
                return ev::undefined();
            });
            b.def("lastScore", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                return ev::fromDouble(s ? s->lastScore : 0.0);
            });
            b.def("isActive", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                return ev::fromBool(s && s->active);
            });
            b.def("isSuspended", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                return ev::fromBool(s && s->suspended);
            });
            b.def("isLoaded", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                return ev::fromBool(s && s->loaded);
            });
            b.def("setThreshold", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                if (s) {
                    s->threshold = numAt(a, 0, 0.85);
                    if (s->wake) s->wake->set_threshold(static_cast<float>(s->threshold));
                }
                return ev::undefined();
            });
            b.def("stats", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                return s ? makeWakeStats(*s) : ev::null();
            });
            b.def("feed", 2, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostWakeState*>(g_wakeViewClass.unwrap(self));
                auto samples = extractFloatAudio(argAt(a, 0));
                if (s) feedWake(*s, samples);
                return ev::fromDouble(s ? s->lastScore : 0.0);
            });
        });

    // -----------------------------------------------------------------------
    // KwsStreamView
    // -----------------------------------------------------------------------
    g_kwsViewClass.install("KwsStreamView", 0,
        [](Value, std::span<const Value>) -> Value {
            auto* s = new HostKwsState();
            return g_kwsViewClass.make(s, [](void* p) { delete static_cast<HostKwsState*>(p); });
        },
        [](ObjectBuilder& b) {
            b.accessor("active", [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                return ev::fromBool(s && s->active);
            });
            b.def("enroll", 3, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                std::string name = strAt(a, 0);
                auto phonemes = extractInt32Array(argAt(a, 1));
                if (s && !name.empty()) s->templates.push_back(name);
                return ev::fromDouble(static_cast<double>(phonemes.size()));
            });
            b.def("enrollFromAudio", 3, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                std::string name = strAt(a, 0);
                auto samples = extractFloatAudio(argAt(a, 1));
                if (s && !name.empty()) s->templates.push_back(name);
                return ev::fromDouble(static_cast<double>(samples.size() / 160));
            });
            b.def("enrollFromClasses", 3, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                std::string name = strAt(a, 0);
                auto classes = extractInt32Array(argAt(a, 1));
                if (s && !name.empty()) s->templates.push_back(name);
                return ev::fromDouble(static_cast<double>(classes.size()));
            });
            b.def("inspect", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                std::string name = strAt(a, 0);
                if (!s || name.empty()) return ev::null();
                ObjectBuilder insp;
                insp.set("name", name);
                insp.set("threshold", 0.5);
                insp.set("frameMs", 10.0);
                insp.set("hasGaps", false);
                return insp.get();
            });
            b.def("remove", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                std::string name = strAt(a, 0);
                if (!s) return ev::fromBool(false);
                auto it = std::find(s->templates.begin(), s->templates.end(), name);
                if (it != s->templates.end()) {
                    s->templates.erase(it);
                    return ev::fromBool(true);
                }
                return ev::fromBool(false);
            });
            b.def("clear", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                if (s) s->templates.clear();
                return ev::undefined();
            });
            b.def("templates", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                if (!s) return ev::parseJson("[]").value;
                return hostArrayOf(s->templates.size(), [s](size_t i) {
                    return ev::fromUtf8(s->templates[i]);
                });
            });
            b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                if (s && s->spotter) s->spotter->reset();
                return ev::undefined();
            });
            b.def("listen", 1, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                if (s) { s->active = true; s->suspended = false; }
                return ev::undefined();
            });
            b.def("stop", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                if (s) s->active = false;
                return ev::undefined();
            });
            b.def("suspend", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                if (s) s->suspended = true;
                return ev::undefined();
            });
            b.def("resume", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                if (s) s->suspended = false;
                return ev::undefined();
            });
            b.def("isActive", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                return ev::fromBool(s && s->active);
            });
            b.def("isSuspended", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                return ev::fromBool(s && s->suspended);
            });
            b.def("isLoaded", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                return ev::fromBool(s && s->loaded);
            });
            b.def("sampleRate", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                return ev::fromDouble(s ? s->sampleRate : 16000);
            });
            b.def("prefixProgress", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                return ev::fromDouble(s ? s->prefixProgress : 0.0);
            });
            b.def("progress", 0, [](Value, std::span<const Value>) -> Value {
                return ev::null();
            });
            b.def("posterior", 1, [](Value, std::span<const Value>) -> Value {
                return ev::null();
            });
            b.def("stats", 0, [](Value self, std::span<const Value>) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                return s ? makeKwsStats(*s) : ev::null();
            });
            b.def("feed", 1, [](Value self, std::span<const Value> a) -> Value {
                auto* s = static_cast<HostKwsState*>(g_kwsViewClass.unwrap(self));
                auto samples = extractFloatAudio(argAt(a, 0));
                if (s) feedKws(*s, samples);
                return ev::null();
            });
        });

    // -----------------------------------------------------------------------
    // bro.wake namespace
    // -----------------------------------------------------------------------
    ObjectBuilder wake;

    wake.def("init", 0, [](Value, std::span<const Value>) -> Value {
        brotensor::init();
        return ev::undefined();
    });
    wake.def("load", 1, [](Value, std::span<const Value> a) -> Value {
        Value opts = argAt(a, 0);
        std::string weights = getPropertyString(opts, "weights");
        g_defaultWake.loaded = true;
        if (!weights.empty()) {
            try {
                g_defaultWake.wake = std::make_shared<brosoundml::WakeWord>();
                g_defaultWake.wake->load(weights);
            } catch (...) {}
        }
        return ev::undefined();
    });
    wake.def("unload", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultWake.wake.reset();
        g_defaultWake.loaded = false;
        g_defaultWake.active = false;
        return ev::undefined();
    });
    wake.def("listen", 1, [](Value, std::span<const Value> a) -> Value {
        g_defaultWake.active = true;
        g_defaultWake.suspended = false;
        Value opts = argAt(a, 0);
        g_defaultWake.threshold = getPropertyDouble(opts, "threshold", g_defaultWake.threshold);
        if (g_defaultWake.wake) g_defaultWake.wake->set_threshold(static_cast<float>(g_defaultWake.threshold));
        return ev::undefined();
    });
    wake.def("stop", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultWake.active = false;
        return ev::undefined();
    });
    wake.def("suspend", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultWake.suspended = true;
        return ev::undefined();
    });
    wake.def("resume", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultWake.suspended = false;
        return ev::undefined();
    });
    wake.def("lastScore", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromDouble(g_defaultWake.lastScore);
    });
    wake.def("isActive", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromBool(g_defaultWake.active);
    });
    wake.def("isSuspended", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromBool(g_defaultWake.suspended);
    });
    wake.def("isLoaded", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromBool(g_defaultWake.loaded);
    });
    wake.def("setThreshold", 1, [](Value, std::span<const Value> a) -> Value {
        g_defaultWake.threshold = numAt(a, 0, 0.85);
        if (g_defaultWake.wake) g_defaultWake.wake->set_threshold(static_cast<float>(g_defaultWake.threshold));
        return ev::undefined();
    });
    wake.def("stats", 0, [](Value, std::span<const Value>) -> Value {
        return makeWakeStats(g_defaultWake);
    });
    wake.def("feed", 2, [](Value, std::span<const Value> a) -> Value {
        auto samples = extractFloatAudio(argAt(a, 0));
        feedWake(g_defaultWake, samples);
        return ev::fromDouble(g_defaultWake.lastScore);
    });

    wake.set("WakeStreamView", g_wakeViewClass.constructor());
    bro.set("wake", wake.get());

    // -----------------------------------------------------------------------
    // bro.kws namespace
    // -----------------------------------------------------------------------
    ObjectBuilder kws;

    kws.def("init", 0, [](Value, std::span<const Value>) -> Value {
        brotensor::init();
        return ev::undefined();
    });
    kws.def("load", 1, [](Value, std::span<const Value>) -> Value {
        g_defaultKws.loaded = true;
        return ev::undefined();
    });
    kws.def("unload", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultKws.loaded = false;
        g_defaultKws.active = false;
        g_defaultKws.templates.clear();
        return ev::undefined();
    });
    kws.def("enroll", 3, [](Value, std::span<const Value> a) -> Value {
        std::string name = strAt(a, 0);
        auto phonemes = extractInt32Array(argAt(a, 1));
        if (!name.empty()) g_defaultKws.templates.push_back(name);
        return ev::fromDouble(static_cast<double>(phonemes.size()));
    });
    kws.def("enrollFromAudio", 3, [](Value, std::span<const Value> a) -> Value {
        std::string name = strAt(a, 0);
        auto samples = extractFloatAudio(argAt(a, 1));
        if (!name.empty()) g_defaultKws.templates.push_back(name);
        return ev::fromDouble(static_cast<double>(samples.size() / 160));
    });
    kws.def("enrollFromClasses", 3, [](Value, std::span<const Value> a) -> Value {
        std::string name = strAt(a, 0);
        auto classes = extractInt32Array(argAt(a, 1));
        if (!name.empty()) g_defaultKws.templates.push_back(name);
        return ev::fromDouble(static_cast<double>(classes.size()));
    });
    kws.def("inspect", 1, [](Value, std::span<const Value> a) -> Value {
        std::string name = strAt(a, 0);
        if (name.empty()) return ev::null();
        ObjectBuilder insp;
        insp.set("name", name);
        insp.set("threshold", 0.5);
        insp.set("frameMs", 10.0);
        insp.set("hasGaps", false);
        return insp.get();
    });
    kws.def("remove", 1, [](Value, std::span<const Value> a) -> Value {
        std::string name = strAt(a, 0);
        auto it = std::find(g_defaultKws.templates.begin(), g_defaultKws.templates.end(), name);
        if (it != g_defaultKws.templates.end()) {
            g_defaultKws.templates.erase(it);
            return ev::fromBool(true);
        }
        return ev::fromBool(false);
    });
    kws.def("clear", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultKws.templates.clear();
        return ev::undefined();
    });
    kws.def("templates", 0, [](Value, std::span<const Value>) -> Value {
        return hostArrayOf(g_defaultKws.templates.size(), [](size_t i) {
            return ev::fromUtf8(g_defaultKws.templates[i]);
        });
    });
    kws.def("reset", 0, [](Value, std::span<const Value>) -> Value {
        if (g_defaultKws.spotter) g_defaultKws.spotter->reset();
        return ev::undefined();
    });
    kws.def("listen", 1, [](Value, std::span<const Value>) -> Value {
        g_defaultKws.active = true;
        g_defaultKws.suspended = false;
        return ev::undefined();
    });
    kws.def("stop", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultKws.active = false;
        return ev::undefined();
    });
    kws.def("suspend", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultKws.suspended = true;
        return ev::undefined();
    });
    kws.def("resume", 0, [](Value, std::span<const Value>) -> Value {
        g_defaultKws.suspended = false;
        return ev::undefined();
    });
    kws.def("isActive", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromBool(g_defaultKws.active);
    });
    kws.def("isSuspended", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromBool(g_defaultKws.suspended);
    });
    kws.def("isLoaded", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromBool(g_defaultKws.loaded);
    });
    kws.def("sampleRate", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromDouble(g_defaultKws.sampleRate);
    });
    kws.def("prefixProgress", 0, [](Value, std::span<const Value>) -> Value {
        return ev::fromDouble(g_defaultKws.prefixProgress);
    });
    kws.def("progress", 0, [](Value, std::span<const Value>) -> Value {
        return ev::null();
    });
    kws.def("posterior", 1, [](Value, std::span<const Value>) -> Value {
        return ev::null();
    });
    kws.def("stats", 0, [](Value, std::span<const Value>) -> Value {
        return makeKwsStats(g_defaultKws);
    });
    kws.def("feed", 1, [](Value, std::span<const Value> a) -> Value {
        auto samples = extractFloatAudio(argAt(a, 0));
        feedKws(g_defaultKws, samples);
        return ev::null();
    });

    kws.set("KwsStreamView", g_kwsViewClass.constructor());
    bro.set("kws", kws.get());
}

} // namespace brosoundml::api
