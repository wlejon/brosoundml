// Stage-3 plBERT + bert_encoder + text_encoder vs upstream Kokoro reference.
//
// Loads weights/kokoro/model.safetensors and the per-stage reference
// activation dumps produced by scripts/dump-reference.py, runs the three
// submodules through their forward passes, and compares each output
// element-wise against the upstream Python reference.
//
// Silently SKIPS (returns 0) when the weights or reference dumps are missing —
// fresh clones still pass this test; the diagnostic only fires when the
// upstream artifacts are present and a stage drifts.
#include "brosoundml/kokoro.h"
#include "brosoundml/kokoro_modules.h"

#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace bt = brotensor;
namespace stf = brotensor::safetensors;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

// Element-wise max-abs / mean-abs diff against a reference safetensors entry.
// Reports both metrics (so a single noisy element doesn't drown out the
// overall agreement) and fails the test when either exceeds tolerance.
static void compare(const bt::Tensor& got, const stf::TensorView& want,
                    int elem_count, const char* label,
                    float max_tol, float mean_tol) {
    if (want.dtype != stf::Dtype::F32) {
        std::fprintf(stderr, "FAIL: %s reference dtype is not F32\n", label);
        ++failures;
        return;
    }
    if (want.numel() != elem_count || got.size() != elem_count) {
        std::fprintf(stderr,
                     "FAIL: %s element count mismatch — got=%d want=%lld expected=%d\n",
                     label, got.size(), static_cast<long long>(want.numel()),
                     elem_count);
        ++failures;
        return;
    }
    const float* g = got.host_f32();
    const float* w = reinterpret_cast<const float*>(want.data);
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    for (int i = 0; i < elem_count; ++i) {
        const float d = std::abs(g[i] - w[i]);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
    }
    const float mean_abs = static_cast<float>(sum_abs / elem_count);
    const bool ok = max_abs <= max_tol && mean_abs <= mean_tol;
    std::fprintf(stderr,
                 "  %s:%s max=%.3e mean=%.3e (tol max=%.1e mean=%.1e)\n",
                 label, ok ? " ok " : " FAIL", max_abs, mean_abs,
                 max_tol, mean_tol);
    if (!ok) ++failures;
}

int main() {
    const fs::path root = fs::path(BROSOUNDML_REPO_DIR) / "weights" / "kokoro";
    const fs::path model_path = root / "model.safetensors";
    const fs::path ref_dir    = root / "reference";
    if (!fs::exists(model_path) || !fs::exists(ref_dir / "01_bert.safetensors")) {
        std::printf("test_kokoro_modules: weights/reference missing under %s — "
                    "run scripts/download-kokoro.sh && scripts/convert-kokoro.py && "
                    "scripts/dump-reference.py to enable, skipping\n",
                    root.string().c_str());
        return 0;
    }

    // ─── Load config + weights ─────────────────────────────────────────────
    brosoundml::Kokoro k;
    k.load(root.string());
    const auto& cfg = k.config();

    stf::File weights = stf::File::open(model_path.string());
    stf::File ref_in  = stf::File::open((ref_dir / "00_inputs.safetensors").string());
    stf::File ref_bert= stf::File::open((ref_dir / "01_bert.safetensors").string());
    stf::File ref_be  = stf::File::open((ref_dir / "02_bert_encoder.safetensors").string());
    stf::File ref_te  = stf::File::open((ref_dir / "06_text_encoder.safetensors").string());

    // input_ids were saved as FP32 (safetensors quirk in the dumper). Cast
    // back to int32 here.
    const stf::TensorView& ids_v = ref_in.get("input_ids");
    const int L = static_cast<int>(ids_v.numel());
    std::vector<int32_t> input_ids(L);
    {
        const float* idf = reinterpret_cast<const float*>(ids_v.data);
        for (int i = 0; i < L; ++i) input_ids[i] = static_cast<int32_t>(idf[i]);
    }
    // No padding in our fixed test input; pass an empty mask.
    const std::vector<int> attention_mask;
    const std::vector<int> text_mask_pad;

    // ─── PLBert ────────────────────────────────────────────────────────────
    brosoundml::PLBert bert;
    bert.load_from(weights, cfg.plbert);

    bt::Tensor bert_dur;
    bert.forward(input_ids, attention_mask, bert_dur);
    // bert_dur shape: (L, hidden_size=768). Reference is (1, L, 768) flat.
    CHECK(bert_dur.rows == L && bert_dur.cols == cfg.plbert.hidden_size,
          "PLBert output shape (L, hidden_size)");
    compare(bert_dur, ref_bert.get("bert_dur"),
            L * cfg.plbert.hidden_size, "bert_dur", 1e-3f, 1e-4f);

    // ─── BertEncoder ───────────────────────────────────────────────────────
    brosoundml::BertEncoder be;
    be.load_from(weights, cfg.plbert.hidden_size, cfg.hidden_dim);

    bt::Tensor d_en;
    be.forward(bert_dur, d_en);
    CHECK(d_en.size() == cfg.hidden_dim * L,
          "BertEncoder output element count (hidden_dim * L)");
    compare(d_en, ref_be.get("d_en"),
            cfg.hidden_dim * L, "d_en", 1e-3f, 1e-4f);

    // ─── TextEncoder ───────────────────────────────────────────────────────
    brosoundml::TextEncoder te;
    te.load_from(weights, cfg);

    bt::Tensor t_en;
    te.forward(input_ids, text_mask_pad, t_en);
    CHECK(t_en.size() == cfg.hidden_dim * L,
          "TextEncoder output element count (hidden_dim * L)");
    compare(t_en, ref_te.get("t_en"),
            cfg.hidden_dim * L, "t_en", 1e-3f, 1e-4f);

    // ─── Predictor ─────────────────────────────────────────────────────────
    stf::File ref_pre = stf::File::open((ref_dir / "03_predictor_pre.safetensors").string());
    stf::File ref_lr  = stf::File::open((ref_dir / "04_length_reg.safetensors").string());
    stf::File ref_f0n = stf::File::open((ref_dir / "05_f0_energy.safetensors").string());

    // Reconstruct ref_s = a (1, 2*style_dim) row vector from the inputs dump.
    const stf::TensorView& ref_s_view = ref_in.get("ref_s");
    bt::Tensor ref_s = bt::Tensor::from_host_on(
        bt::Device::CPU,
        reinterpret_cast<const float*>(ref_s_view.data),
        1, static_cast<int>(ref_s_view.numel()));

    brosoundml::Predictor pred;
    pred.load_from(weights, cfg);
    brosoundml::Predictor::Output po;
    pred.forward(d_en, ref_s, L, /*speed=*/1.0f, po);

    compare(po.d,        ref_pre.get("d"),
            L * (cfg.hidden_dim + cfg.style_dim), "predictor.d",      1e-3f, 1e-4f);
    compare(po.lstm_x,   ref_pre.get("lstm_x"),
            L * cfg.hidden_dim,                   "predictor.lstm_x", 1e-3f, 1e-4f);
    compare(po.duration, ref_pre.get("duration"),
            L * cfg.max_dur,                      "predictor.duration", 1e-3f, 1e-4f);

    // pred_dur is integer-valued — compare exact (after casting the FP32-saved
    // reference to int). A mismatch here means the duration LSTM drifted enough
    // to cross a rounding boundary somewhere.
    const stf::TensorView& pdv = ref_pre.get("pred_dur");
    const float* pdv_f = reinterpret_cast<const float*>(pdv.data);
    bool pdur_ok = (pdv.numel() == L);
    for (int l = 0; l < L && pdur_ok; ++l) {
        if (po.pred_dur[l] != static_cast<int32_t>(pdv_f[l])) pdur_ok = false;
    }
    CHECK(pdur_ok, "predictor.pred_dur matches reference exactly");

    const int total = std::accumulate(po.pred_dur.begin(), po.pred_dur.end(), 0);
    compare(po.en,      ref_lr.get("en"),
            (cfg.hidden_dim + cfg.style_dim) * total, "predictor.en", 1e-3f, 1e-4f);
    compare(po.F0_pred, ref_f0n.get("F0_pred"),
            2 * total, "predictor.F0_pred", 5e-3f, 5e-4f);
    compare(po.N_pred,  ref_f0n.get("N_pred"),
            2 * total, "predictor.N_pred",  5e-3f, 5e-4f);

    if (failures == 0) {
        std::printf("test_kokoro_modules: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_kokoro_modules: %d check(s) failed\n", failures);
    return 1;
}
