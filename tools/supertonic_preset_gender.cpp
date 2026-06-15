// Build the masc<->fem axis the realizable way: directly from the model's own
// labeled presets (voice_styles/F* female, M* male), which ARE valid style_ttl
// voices — not by inverting a class-mean x-vector, which has no audible preimage
// and collapses to near-silence (the failure the verifier caught). The axis is
// the labeled female-mean minus male-mean in STYLE space (the within-group
// contrast done on realizable points). We synthesize the endpoints + midpoint,
// embed them, and report rendered separation + RMS so audibility is proven, then
// write masc_fem_basis.json.
//
//   supertonic_preset_gender <model_dir> <enc_dir> <out_wav_dir> <out_basis.json>

#define _CRT_SECURE_NO_WARNINGS
#include "brosoundml/audio.h"
#include "brosoundml/speaker_encoder.h"
#include "brosoundml/supertonic.h"
#include <brotensor/runtime.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using brosoundml::AudioBuffer;
using brosoundml::VoiceStyle;

static std::vector<float> load_f32(const std::string& p, std::size_t n) {
    std::ifstream f(p, std::ios::binary); std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n*sizeof(float))); if (!f) v.clear(); return v;
}
static double ccos(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& mean) {
    double dot=0,na=0,nb=0; for (std::size_t i=0;i<a.size();++i){const double ai=a[i]-mean[i],bi=b[i]-mean[i];dot+=ai*bi;na+=ai*ai;nb+=bi*bi;}
    return dot/(std::sqrt(std::max(na,1e-20))*std::sqrt(std::max(nb,1e-20)));
}
static void stats(const AudioBuffer& a, double& rms, double& zcr){
    double acc=0; long zc=0; for(std::size_t i=0;i<a.samples.size();++i){acc+=double(a.samples[i])*a.samples[i]; if(i&&((a.samples[i]<0)!=(a.samples[i-1]<0)))++zc;}
    rms=a.samples.empty()?0:std::sqrt(acc/a.samples.size()); zcr=a.samples.empty()?0:double(zc)/a.samples.size()*a.sample_rate;
}
static void save_basis(const std::string& path, const std::vector<float>& mid, const std::vector<float>& half,
                       const std::vector<float>& fem, const std::vector<float>& masc, const std::vector<float>& dp) {
    std::ofstream o(path);
    auto arr=[&](const char* k,const std::vector<float>& d){o<<"\""<<k<<"\": [";for(std::size_t i=0;i<d.size();++i){if(i)o<<", ";o<<d[i];}o<<"]";};
    double mag=0; for(float h:half) mag+=double(h)*h; mag=std::sqrt(mag);
    o<<"{\n  \"axis\": \"masc_fem\",\n  \"source\": \"preset_means\",\n";
    o<<"  \"note\": \"style(t) = mid_ttl + t*half_axis_ttl, t in [-1(masc)..+1(fem)]; ttl-only, dp travels with base voice\",\n";
    o<<"  \"dims\": [1, 50, 256],\n  \"magnitude\": "<<mag<<",\n  "; arr("mid_ttl",mid); o<<",\n  "; arr("half_axis_ttl",half);
    o<<",\n  "; arr("fem_ttl",fem); o<<",\n  "; arr("masc_ttl",masc); o<<",\n  "; arr("dp",dp); o<<"\n}\n";
}

int main(int argc, char** argv){
    brotensor::init();
    const bool gpu = brotensor::is_available(brotensor::Device::CUDA);
    brotensor::set_default_device(gpu?brotensor::Device::CUDA:brotensor::Device::CPU);
    if(argc<5){std::printf("usage: %s <model> <enc> <out_wav> <out_basis.json>\n",argv[0]);return 2;}
    const std::string model_dir=argv[1], enc_dir=argv[2], outw=argv[3], outb=argv[4];
    fs::create_directories(outw);
    const std::string vsdir=(fs::path(model_dir)/"voice_styles").string();

    brosoundml::Supertonic model; brosoundml::SpeakerEncoder enc;
    model.load(model_dir, gpu?brotensor::Device::CUDA:brotensor::Device::CPU);
    enc.load(enc_dir);
    const std::size_t E=std::size_t(enc.enc_dim());
    std::vector<float> mean=load_f32((fs::path(enc_dir)/"xvector_mean.f32").string(),E);
    if(mean.size()!=E) mean.assign(E,0.0f);

    // collect F* and M* presets
    std::vector<VoiceStyle> fem, masc;
    for(auto& e: fs::directory_iterator(vsdir)){
        if(e.path().extension()!=".json") continue;
        std::string stem=e.path().stem().string();
        VoiceStyle v=model.load_voice_style(e.path().string());
        if(stem.size() && (stem[0]=='F'||stem[0]=='f')) fem.push_back(v);
        else if(stem.size() && (stem[0]=='M'||stem[0]=='m')) masc.push_back(v);
    }
    std::printf("presets: %zu female, %zu male\n", fem.size(), masc.size());
    if(fem.empty()||masc.empty()){std::printf("need both F* and M* presets\n");return 1;}

    auto meanvec=[](const std::vector<VoiceStyle>& g, bool ttl)->std::vector<float>{
        const std::vector<float>& s0 = ttl? g[0].ttl : g[0].dp;
        std::vector<double> acc(s0.size(),0.0);
        for(auto& v: g){const std::vector<float>& s = ttl? v.ttl : v.dp; for(std::size_t i=0;i<acc.size();++i) acc[i]+=s[i];}
        std::vector<float> m(acc.size()); for(std::size_t i=0;i<acc.size();++i) m[i]=float(acc[i]/g.size()); return m;
    };
    std::vector<float> fT=meanvec(fem,true), mT=meanvec(masc,true);
    std::vector<float> fD=meanvec(fem,false), mD=meanvec(masc,false);
    const std::size_t N=fT.size();
    std::vector<float> mid(N), half(N); for(std::size_t i=0;i<N;++i){mid[i]=(fT[i]+mT[i])*0.5f; half[i]=(fT[i]-mT[i])*0.5f;}
    std::vector<float> midD(fD.size()); for(std::size_t i=0;i<fD.size();++i) midD[i]=(fD[i]+mD[i])*0.5f;

    auto synth_embed=[&](const std::vector<float>& ttl, const std::vector<float>& dp, const char* wav)->std::vector<float>{
        VoiceStyle v; v.ttl=ttl; v.dp=dp;
        AudioBuffer a=model.synthesize("Hello there, this is a test of the voice.","en",v,8,1.05f,7777,3.0f);
        double r,z; stats(a,r,z); a.write_wav((fs::path(outw)/wav).string());
        std::printf("    [%s] rms=%.3f zcr=%.0fHz\n",wav,r,z); return enc.embed(a);
    };

    std::printf("\n== preset-mean gender axis ==\n");
    // t in {-1 (masc), 0 (mid), +1 (fem)} plus partials
    std::vector<float> tfem(N), tmasc(N);
    for(std::size_t i=0;i<N;++i){tfem[i]=mid[i]+half[i]; tmasc[i]=mid[i]-half[i];}
    std::vector<float> ef=synth_embed(tfem,midD,"pg_fem.wav");
    std::vector<float> e0=synth_embed(mid, midD,"pg_mid.wav");
    std::vector<float> em=synth_embed(tmasc,midD,"pg_masc.wav");
    std::printf("  fem-vs-masc rendered separation (centered cos): %.3f  (lower=more distinct)\n", ccos(ef,em,mean));
    std::printf("  fem-vs-mid: %.3f   masc-vs-mid: %.3f\n", ccos(ef,e0,mean), ccos(em,e0,mean));

    save_basis(outb, mid, half, tfem, tmasc, midD);
    std::printf("wrote %s  (|half-axis|)\n", outb.c_str());
    return 0;
}
