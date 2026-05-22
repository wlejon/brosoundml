// Smoke test: brosoundml links, versions, and can reach brotensor.
#include "brosoundml/version.h"

#include <brotensor/tensor.h>

#include <cstdio>
#include <cstring>

int main() {
    const char* v = brosoundml::version_string();
    if (!v || std::strlen(v) == 0) {
        std::fprintf(stderr, "version_string() returned empty\n");
        return 1;
    }

    // Prove the brotensor dependency is wired in and usable: a 2x3 host tensor.
    brotensor::Tensor t = brotensor::Tensor::mat(2, 3);
    if (t.rows != 2 || t.cols != 3) {
        std::fprintf(stderr, "brotensor tensor has wrong shape: %dx%d\n",
                     t.rows, t.cols);
        return 1;
    }

    std::printf("brosoundml %s (brotensor reachable)\n", v);
    return 0;
}
