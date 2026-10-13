// ffmpeg -i video.mp4 -f rawvideo -pix_fmt bgr24 - | ./detector model.xml <w> <h> <max_frames> [threads] [async]
#include <openvino/openvino.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr float kConf = 0.1f;
constexpr float kIou = 0.5f;

struct Box { float x1, y1, x2, y2, score; };
struct Geometry { float scale; int padx, pady; };

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

Geometry letterbox(const uint8_t* src, int w, int h, uint8_t* dst, int size) {
    Geometry g{};
    g.scale = std::min(static_cast<float>(size) / w, static_cast<float>(size) / h);
    const int nw = static_cast<int>(w * g.scale + 0.5f), nh = static_cast<int>(h * g.scale + 0.5f);
    g.padx = (size - nw) / 2;
    g.pady = (size - nh) / 2;
    std::fill(dst, dst + static_cast<size_t>(size) * size * 3, uint8_t{114});
    std::vector<int> x0s(nw), x1s(nw);
    std::vector<float> fxs(nw);
    for (int x = 0; x < nw; ++x) {
        const float sx = std::max(0.0f, (x + 0.5f) / g.scale - 0.5f);
        x0s[x] = std::min(static_cast<int>(sx), w - 1);
        x1s[x] = std::min(x0s[x] + 1, w - 1);
        fxs[x] = sx - x0s[x];
    }
    for (int y = 0; y < nh; ++y) {
        const float sy = std::max(0.0f, (y + 0.5f) / g.scale - 0.5f);
        const int y0 = std::min(static_cast<int>(sy), h - 1), y1 = std::min(y0 + 1, h - 1);
        const float fy = sy - y0;
        const uint8_t* r0 = src + static_cast<size_t>(y0) * w * 3;
        const uint8_t* r1 = src + static_cast<size_t>(y1) * w * 3;
        uint8_t* out = dst + (static_cast<size_t>(y + g.pady) * size + g.padx) * 3;
        for (int x = 0; x < nw; ++x) {
            const int a = x0s[x] * 3, b = x1s[x] * 3;
            const float fx = fxs[x];
            for (int c = 0; c < 3; ++c) {
                const float top = r0[a + c] + (r0[b + c] - r0[a + c]) * fx;
                const float bot = r1[a + c] + (r1[b + c] - r1[a + c]) * fx;
                out[x * 3 + c] = static_cast<uint8_t>(top + (bot - top) * fy + 0.5f);
            }
        }
    }
    return g;
}

float iou(const Box& a, const Box& b) {
    const float ix = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float iy = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = ix * iy;
    return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-6f);
}

std::vector<Box> decode(const float* out, size_t n, const Geometry& g) {
    std::vector<Box> cand;
    for (size_t i = 0; i < n; ++i) {
        const float s = out[4 * n + i];
        if (s < kConf) continue;
        const float cx = (out[i] - g.padx) / g.scale, cy = (out[n + i] - g.pady) / g.scale;
        const float bw = out[2 * n + i] / g.scale, bh = out[3 * n + i] / g.scale;
        cand.push_back({cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, s});
    }
    std::sort(cand.begin(), cand.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
    std::vector<Box> keep;
    std::vector<bool> removed(cand.size(), false);
    for (size_t i = 0; i < cand.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(cand[i]);
        for (size_t j = i + 1; j < cand.size(); ++j)
            if (!removed[j] && iou(cand[i], cand[j]) > kIou) removed[j] = true;
    }
    return keep;
}

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(p * (v.size() - 1))];
}

double mean(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: detector <model.xml> <width> <height> <max_frames> [threads] [async 0|1]\n";
        return 2;
    }
    const std::string model_path = argv[1];
    const int w = std::stoi(argv[2]), h = std::stoi(argv[3]), max_frames = std::stoi(argv[4]);
    const int threads = argc > 5 ? std::stoi(argv[5]) : 8;
    const bool async = argc > 6 && std::stoi(argv[6]) != 0;

    ov::Core core;
    auto model = core.read_model(model_path);
    const int size = static_cast<int>(model->input().get_shape()[2]);
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(
        ov::preprocess::ColorFormat::BGR);
    ppp.input().model().set_layout("NCHW");
    ppp.input().preprocess().convert_color(ov::preprocess::ColorFormat::RGB).convert_element_type(
        ov::element::f32).scale(255.0f);
    auto compiled = core.compile_model(ppp.build(), "CPU", ov::inference_num_threads(threads),
                                       ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

    const int nreq = async ? 2 : 1;
    std::vector<ov::InferRequest> reqs;
    for (int i = 0; i < nreq; ++i) reqs.push_back(compiled.create_infer_request());
    std::array<Geometry, 2> geo{};

    std::vector<uint8_t> frame(static_cast<size_t>(w) * h * 3);
    std::vector<double> t_read, t_pre, t_wait, t_post;
    size_t detections = 0;
    int submitted = 0, done = 0;

    auto read_and_prepare = [&](int slot) -> bool {
        const auto a = Clock::now();
        if (submitted >= max_frames || std::fread(frame.data(), 1, frame.size(), stdin) != frame.size()) return false;
        const auto b = Clock::now();
        geo[slot] = letterbox(frame.data(), w, h, reqs[slot].get_input_tensor().data<uint8_t>(), size);
        t_read.push_back(ms(a, b));
        t_pre.push_back(ms(b, Clock::now()));
        ++submitted;
        return true;
    };
    auto finish = [&](int slot) {
        const auto a = Clock::now();
        reqs[slot].wait();
        const auto b = Clock::now();
        const ov::Tensor out = reqs[slot].get_output_tensor(0);
        detections += decode(out.data<const float>(), out.get_shape()[2], geo[slot]).size();
        t_wait.push_back(ms(a, b));
        t_post.push_back(ms(b, Clock::now()));
        ++done;
    };

    const auto start = Clock::now();
    if (!async) {
        while (read_and_prepare(0)) {
            reqs[0].start_async();
            finish(0);
        }
    } else {
        int cur = 0;
        bool have = read_and_prepare(cur);
        if (have) reqs[cur].start_async();
        while (have) {
            const int next = 1 - cur;
            const bool more = read_and_prepare(next);
            finish(cur);
            if (more) reqs[next].start_async();
            have = more;
            cur = next;
        }
    }
    const double total = ms(start, Clock::now());
    std::printf("{\"frames\": %d, \"fps\": %.2f, \"threads\": %d, \"async\": %d, \"mean_detections\": %.2f, "
                "\"ms\": {\"read\": %.3f, \"pre\": %.3f, \"infer_wait\": %.3f, \"post\": %.3f, "
                "\"wait_p50\": %.3f, \"wait_p95\": %.3f}}\n",
                done, done / (total / 1000.0), threads, async ? 1 : 0,
                done ? static_cast<double>(detections) / done : 0.0, mean(t_read), mean(t_pre), mean(t_wait),
                mean(t_post), pct(t_wait, 0.5), pct(t_wait, 0.95));
    return done > 0 ? 0 : 1;
}
