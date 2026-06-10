// Unit tests for brosoundml::phoneme_data — the data layer of open-vocabulary
// phoneme-posterior keyword spotting. Pure host DSP/IO: no Kokoro, no model
// weights, so this runs anywhere.

#define _CRT_SECURE_NO_WARNINGS   // std::fopen in the class-map IO round-trip

#include "brosoundml/phoneme_data.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using brosoundml::PhonemeClassMap;
using brosoundml::PhonemeClip;
using brosoundml::PhonemeDataset;
using brosoundml::PhonemeDatasetHeader;
using brosoundml::PhonemeDatasetWriter;
using brosoundml::PhonemeValidationConfig;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                \
            ++failures;                                               \
        }                                                             \
    } while (0)

static int class_index(const PhonemeClassMap& cm, const std::string& name) {
    for (int c = 0; c < cm.num_classes; ++c)
        if (cm.class_names[static_cast<std::size_t>(c)] == name) return c;
    return -1;
}

int main() {
    // ─── 1. Class-map build over a toy vocab ───────────────────────────────
    {
        // IPA string -> id. Mix of named phonemes, an unnamed spoken symbol,
        // and non-spoken (space / punctuation / digit) tokens.
        // Source is compiled /utf-8, so plain literals carry UTF-8 IPA bytes.
        std::unordered_map<std::string, int> vocab = {
            {"p", 1}, {"b", 2}, {"æ", 3}, {"ə", 4},
            {"s", 5}, {"k", 6}, {" ", 7}, {".", 8},
            {"ʤ", 9}, {"ʒ", 10},
            {"ʡ", 11}, {"5", 12},   // ʡ: spoken but unnamed -> 'other'
        };
        auto cm  = brosoundml::build_default_english_classmap(vocab);
        auto cm2 = brosoundml::build_default_english_classmap(vocab);

        CHECK(cm.num_classes >= 3, "classmap: at least sil+phonemes+other");
        CHECK(cm.class_names[0] == "sil", "classmap: class 0 is silence 'sil'");
        CHECK(cm.silence_class() == 0, "classmap: silence_class() == 0");
        CHECK(cm.class_names.back() == "other", "classmap: last class is 'other'");

        // Expected groupings.
        CHECK(class_index(cm, "P") == cm.class_for_id(1), "classmap: p -> P");
        CHECK(class_index(cm, "B") == cm.class_for_id(2), "classmap: b -> B");
        CHECK(class_index(cm, "AE") == cm.class_for_id(3), "classmap: æ -> AE");
        CHECK(class_index(cm, "AX") == cm.class_for_id(4), "classmap: ə -> AX");
        CHECK(class_index(cm, "S") == cm.class_for_id(5), "classmap: s -> S");
        CHECK(class_index(cm, "K") == cm.class_for_id(6), "classmap: k -> K");
        CHECK(class_index(cm, "JH") == cm.class_for_id(9), "classmap: ʤ -> JH");
        CHECK(class_index(cm, "ZH") == cm.class_for_id(10), "classmap: ʒ -> ZH");

        // Non-spoken tokens -> silence.
        CHECK(cm.class_for_id(7) == 0, "classmap: space -> silence");
        CHECK(cm.class_for_id(8) == 0, "classmap: '.' -> silence");
        CHECK(cm.class_for_id(12) == 0, "classmap: digit -> silence");

        // Unnamed spoken symbol -> 'other'.
        CHECK(cm.class_for_id(11) == class_index(cm, "other"),
              "classmap: unnamed spoken -> other");

        // Full coverage: class_for_id succeeds for every assigned id.
        bool all_covered = true;
        for (const auto& kv : vocab) {
            try { (void)cm.class_for_id(kv.second); }
            catch (...) { all_covered = false; }
        }
        CHECK(all_covered, "classmap: every vocab id is covered");

        // Truly-unknown id throws.
        bool threw = false;
        try { (void)cm.class_for_id(99999); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw, "classmap: unknown id throws");

        // Determinism: two builds are byte-identical.
        CHECK(cm == cm2, "classmap: deterministic across builds");
    }

    // ─── 2. Class-map serialize round-trip ─────────────────────────────────
    {
        std::unordered_map<std::string, int> vocab = {
            {"p", 1}, {"b", 2}, {"æ", 3}, {" ", 4},
            {"ʃ", 5}, {"ŋ", 6},
        };
        auto cm = brosoundml::build_default_english_classmap(vocab);

        fs::path tmp = fs::temp_directory_path() / "brosoundml_classmap.bin";
        {
            std::FILE* f = std::fopen(tmp.string().c_str(), "wb");
            CHECK(f != nullptr, "classmap io: open for write");
            if (f) { brosoundml::write_classmap(f, cm); std::fclose(f); }
        }
        PhonemeClassMap rd;
        {
            std::FILE* f = std::fopen(tmp.string().c_str(), "rb");
            CHECK(f != nullptr, "classmap io: open for read");
            if (f) { rd = brosoundml::read_classmap(f); std::fclose(f); }
        }
        CHECK(rd == cm, "classmap io: round-trip equal");
        // Inverse still works after deserialization.
        CHECK(rd.class_for_id(1) == cm.class_for_id(1),
              "classmap io: inverse rebuilt after read");
        std::error_code ec; fs::remove(tmp, ec);
    }

    // ─── 3. build_frame_labels (hand-computed) ─────────────────────────────
    {
        // Tiny custom map: sil + two phoneme classes.
        PhonemeClassMap cm;
        cm.num_classes  = 3;
        cm.class_names  = {"sil", "AA", "BB"};
        cm.class_to_ids = {{}, {101}, {202}};

        // pred_dur {BOS=5, 20, 20, EOS=5}, sum=50; 5000 samples -> 100 spk.
        std::vector<int32_t> pred_dur = {5, 20, 20, 5};
        std::vector<int32_t> ids      = {101, 202};
        const int n_samples = 5000, win = 400, hop = 160;

        auto labels = brosoundml::build_frame_labels(pred_dur, ids, cm,
                                                     n_samples, win, hop);
        // Framing: 1 + (5000-400)/160 = 29 frames.
        CHECK(labels.size() == 29, "labels: framing frame count == 29");

        bool ok = labels.size() == 29;
        if (ok) {
            for (int t = 0; t < 2 && ok; ++t)
                ok = labels[static_cast<std::size_t>(t)] == 0;   // BOS edge
            for (int t = 2; t <= 14 && ok; ++t)
                ok = labels[static_cast<std::size_t>(t)] == 1;   // class AA
            for (int t = 15; t <= 26 && ok; ++t)
                ok = labels[static_cast<std::size_t>(t)] == 2;   // class BB
            for (int t = 27; t <= 28 && ok; ++t)
                ok = labels[static_cast<std::size_t>(t)] == 0;   // EOS edge
        }
        CHECK(ok, "labels: hand-computed segmentation matches");

        // Edge-silent.
        CHECK(labels.front() == 0 && labels.back() == 0,
              "labels: silent at both edges (BOS/EOS)");

        // Monotone-segmented: exactly three transitions (0->1, 1->2, 2->0).
        int transitions = 0;
        for (std::size_t i = 1; i < labels.size(); ++i)
            if (labels[i] != labels[i - 1]) ++transitions;
        CHECK(transitions == 3, "labels: three contiguous segment boundaries");

        // Degenerate total duration -> all silence.
        auto sil = brosoundml::build_frame_labels({0, 0, 0, 0}, ids, cm,
                                                  n_samples, win, hop);
        bool all_sil = sil.size() == 29;
        for (auto v : sil) if (v != 0) all_sil = false;
        CHECK(all_sil, "labels: zero total duration -> all silence");

        // Wrong wrapped length throws.
        bool threw = false;
        try { brosoundml::build_frame_labels({5, 20}, ids, cm, n_samples, win, hop); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw, "labels: wrong pred_dur length throws");
    }

    // ─── 3b. resample_labels_nn + crop_or_pad_labels_centered ──────────────
    {
        std::vector<int16_t> base = {0, 1, 2};
        auto up = brosoundml::resample_labels_nn(base, 6);
        std::vector<int16_t> up_exp = {0, 0, 1, 1, 2, 2};
        CHECK(up == up_exp, "resample_labels_nn: 2x upsample center-aligned");
        auto down = brosoundml::resample_labels_nn(up, 3);
        CHECK(down == base, "resample_labels_nn: 2x downsample recovers base");

        // Pad: {1,1,1} -> len 7, silence 0 -> {0,0,1,1,1,0,0}.
        auto pad = brosoundml::crop_or_pad_labels_centered({1, 1, 1}, 7, 0);
        std::vector<int16_t> pad_exp = {0, 0, 1, 1, 1, 0, 0};
        CHECK(pad == pad_exp, "crop_or_pad_labels: centered pad with silence");
        // Crop: {1,2,3,4,5} -> len 3 -> {2,3,4}.
        auto crop = brosoundml::crop_or_pad_labels_centered({1, 2, 3, 4, 5}, 3, 0);
        std::vector<int16_t> crop_exp = {2, 3, 4};
        CHECK(crop == crop_exp, "crop_or_pad_labels: centered crop");
    }

    // ─── 4. Dataset writer -> reader round-trip ────────────────────────────
    PhonemeClassMap ds_cm;
    ds_cm.num_classes  = 3;
    ds_cm.class_names  = {"sil", "AA", "BB"};
    ds_cm.class_to_ids = {{}, {101}, {202}};

    PhonemeDatasetHeader hdr;   // defaults: 16000 / 512 / 400 / 160 / 40

    std::vector<std::vector<float>>   pcms;
    std::vector<std::vector<int16_t>> labs;
    {
        // Clip A: 5000 samples (29 frames), labels from build_frame_labels.
        std::vector<float> a(5000);
        for (int n = 0; n < 5000; ++n)
            a[static_cast<std::size_t>(n)] =
                0.3f * std::sin(2.f * 3.14159265f * 220.f * n / 16000.f);
        auto la = brosoundml::build_frame_labels({5, 20, 20, 5}, {101, 202},
                                                 ds_cm, 5000, 400, 160);
        // Clip B: 8000 samples -> 1 + (8000-400)/160 = 48 frames; cover all 3.
        std::vector<float> b(8000);
        for (int n = 0; n < 8000; ++n)
            b[static_cast<std::size_t>(n)] =
                0.5f * std::sin(2.f * 3.14159265f * 440.f * n / 16000.f);
        std::vector<int16_t> lb(48);
        for (int i = 0; i < 48; ++i)
            lb[static_cast<std::size_t>(i)] =
                static_cast<int16_t>(i < 16 ? 0 : (i < 32 ? 1 : 2));
        pcms = {a, b};
        labs = {la, lb};
    }

    fs::path ds_path = fs::temp_directory_path() / "brosoundml_phoneme.bpds";
    {
        PhonemeDatasetWriter w(ds_path.string(), hdr, ds_cm);
        w.append(pcms[0], labs[0]);
        w.append(pcms[1], labs[1]);
        w.finalize();
        CHECK(w.clips() == 2, "writer: clip count tracked");
    }

    PhonemeDataset ds = brosoundml::read_phoneme_dataset(ds_path.string());
    CHECK(ds.header.sample_rate == 16000 && ds.header.n_fft == 512 &&
          ds.header.win_length == 400 && ds.header.hop_length == 160 &&
          ds.header.n_mels == 40, "reader: header fields preserved");
    CHECK(ds.class_map == ds_cm, "reader: class map preserved");
    CHECK(ds.clips.size() == 2, "reader: clip count preserved");
    {
        bool pcm_ok = true, lab_ok = true, inv_ok = true;
        for (std::size_t c = 0; c < ds.clips.size(); ++c) {
            const auto& clip = ds.clips[c];
            if (clip.pcm.size() != pcms[c].size()) pcm_ok = false;
            auto fpcm = clip.pcm_float();
            for (std::size_t i = 0; i < fpcm.size() && pcm_ok; ++i)
                if (std::fabs(fpcm[i] - pcms[c][i]) > 2.0f / 32767.0f)
                    pcm_ok = false;
            if (clip.labels != labs[c]) lab_ok = false;
            const int expect = 1 + (static_cast<int>(clip.pcm.size()) - 400) / 160;
            if (static_cast<int>(clip.labels.size()) != expect) inv_ok = false;
        }
        CHECK(pcm_ok, "reader: PCM within int16 quantization");
        CHECK(lab_ok, "reader: labels match exactly");
        CHECK(inv_ok, "reader: per-clip framing invariant holds");
    }

    // ─── 5. Validator ──────────────────────────────────────────────────────
    {
        PhonemeValidationConfig cfg;   // 16000 / min 1 per class / 0.95 silence
        auto rep = brosoundml::validate_phoneme_dataset(ds, cfg);
        CHECK(rep.total_clips == 2, "validate: total clips");
        CHECK(rep.length_mismatch_clips == 0, "validate: no length mismatch");
        CHECK(rep.label_out_of_range == 0, "validate: labels in range");
        std::string reason;
        CHECK(brosoundml::report_passes(rep, ds.class_map, cfg, &reason),
              ("validate: clean dataset passes (" + reason + ")").c_str());

        // Broken: label out of range.
        PhonemeDataset bad = ds;
        bad.clips[0].labels[0] = 99;
        auto bad_rep = brosoundml::validate_phoneme_dataset(bad, cfg);
        CHECK(bad_rep.label_out_of_range > 0,
              "validate: out-of-range label detected");
        std::string br;
        CHECK(!brosoundml::report_passes(bad_rep, bad.class_map, cfg, &br),
              "validate: out-of-range dataset fails");
        CHECK(!br.empty(), "validate: failure has a reason");

        // Broken: length-invariant violation.
        PhonemeDataset bad2 = ds;
        bad2.clips[1].labels.pop_back();   // now != framing(8000)
        auto bad2_rep = brosoundml::validate_phoneme_dataset(bad2, cfg);
        CHECK(bad2_rep.length_mismatch_clips > 0,
              "validate: length mismatch detected");
        std::string br2;
        CHECK(!brosoundml::report_passes(bad2_rep, bad2.class_map, cfg, &br2),
              "validate: length-mismatch dataset fails");
    }

    {
        std::error_code ec; fs::remove(ds_path, ec);
    }

    if (failures) {
        std::fprintf(stderr, "test_phoneme_data: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_phoneme_data: OK\n");
    return 0;
}
