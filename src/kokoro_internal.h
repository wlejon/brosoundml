#pragma once

#include "brosoundml/kokoro_modules.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace brosoundml {

namespace bt  = brotensor;
namespace stf = brotensor::safetensors;

extern bt::Device g_kokoro_load_device;

bool kokoro_profile_enabled();
void kokoro_profile_mark(bt::Device dev, const char* name);

bt::Tensor upload_int32_idx(bt::Device dev, const std::int32_t* host_idx, int n);
[[noreturn]] void fail(const std::string& where, const std::string& msg);

void upload(const stf::File& f, const std::string& key,
            int rows, int cols, bt::Tensor& dst,
            const std::string& where);

void layernorm_rows(int N, int D,
                    const bt::Tensor& gamma, const bt::Tensor& beta,
                    float eps, const bt::Tensor& X, bt::Tensor& Y);

void layernorm_1d_ncl(const bt::Tensor& X,
                      const bt::Tensor& gamma, const bt::Tensor& beta,
                      int N, int C, int L, float eps,
                      bt::Tensor& Y);

void load_lstm_cell(const stf::File& f, const std::string& prefix,
                    int input_size, int hidden, bool reverse,
                    LSTMCellWeights& cell, const std::string& where);

void compute_style_affine(const Linear& fc, int C,
                          const bt::Tensor& style,
                          bt::Tensor& gamma, bt::Tensor& beta);

void ada_layernorm(const AdaLayerNormWeights& w, int L,
                   const bt::Tensor& x_lc, const bt::Tensor& style,
                   bt::Tensor& y_lc);

void ada_in_1d_styled(const AdaIN1dWeights& w, int N, int C, int L,
                      const bt::Tensor& x_ncl, const bt::Tensor& style,
                      bt::Tensor& y_ncl);

void load_ada_in_1d(const stf::File& f, const std::string& prefix,
                    int C, int style_dim, AdaIN1dWeights& w,
                    const std::string& where);

void leaky_relu_ncl(bt::Tensor& y, float slope = 0.2f);

void upsample_nearest_2x_ncl(const bt::Tensor& x, int N, int C, int L_in,
                             bt::Tensor& y);

void load_adain_resblk(const stf::File& f, const std::string& prefix,
                       int dim_in, int dim_out, int style_dim,
                       bool upsample, AdainResBlk1dWeights& w,
                       const std::string& where);

void adain_resblk_1d_forward(const AdainResBlk1dWeights& w,
                             const bt::Tensor& x, int L_in,
                             const bt::Tensor& style,
                             int& L_out, bt::Tensor& y);

}  // namespace brosoundml
