# Edge deployment results (CPU, OpenVINO)

All CPU numbers pinned to 8 cores of the server (an edge-box budget). Input 640x640.

## Accuracy and speed per model and runtime

| model | runtime | mAP50 | mAP50-95 | latency p50 | p95 | throughput (OpenVINO streams) |
|---|---|---|---|---|---|---|
| yolov8x | pytorch_fp32 | 0.963 | 0.794 | 222.6 ms | 1017.4 ms | - |
| yolov8x | onnxruntime_fp32 | 0.950 | 0.750 | 228.0 ms | 1063.2 ms | - |
| yolov8x | openvino_fp32 | 0.950 | 0.750 | 34.0 ms | 202.3 ms | 24.4 FPS |
| yolov8x | openvino_int8 | 0.947 | 0.751 | 33.2 ms | 197.9 ms | 28.6 FPS |
| yolov8s | pytorch_fp32 | 0.954 | 0.778 | 39.0 ms | 186.1 ms | - |
| yolov8s | onnxruntime_fp32 | 0.945 | 0.760 | 37.5 ms | 198.1 ms | - |
| yolov8s | openvino_fp32 | 0.945 | 0.761 | 7.4 ms | 7.9 ms | 148.0 FPS |
| yolov8s | openvino_int8 | 0.944 | 0.751 | 7.7 ms | 88.5 ms | 171.1 FPS |
| yolov8n | pytorch_fp32 | 0.925 | 0.720 | 19.3 ms | 103.8 ms | - |
| yolov8n | onnxruntime_fp32 | 0.915 | 0.701 | 21.5 ms | 21.8 ms | - |
| yolov8n | openvino_fp32 | 0.915 | 0.702 | 4.0 ms | 4.1 ms | 321.6 FPS |
| yolov8n | openvino_int8 | 0.913 | 0.692 | 4.9 ms | 5.1 ms | 289.8 FPS |

mAP on a fixed 1000-image subset of the validation split; INT8 calibrated on 300 training frames.

## Counting on the test videos

| model | format | test_video.mp4 | test_video2.mp4 | test_video3.mp4 | FPS (8 cores) |
|---|---|---|---|---|---|
| yolov8x PyTorch GPU + TTA (original) | | 0 | 4 | 7 | |
| yolov8x PyTorch GPU, no TTA | | 0 | 4 | 4 | |
| yolov8x | OpenVINO fp32 | 0 | 4 | 3 | 17.6 |
| yolov8x | OpenVINO int8 | 0 | 5 | 4 | 18.2 |
| yolov8s | OpenVINO fp32 | 0 | 4 | 6 | 61.5 |
| yolov8s | OpenVINO int8 | 0 | 5 | 6 | 61.8 |
| yolov8n | OpenVINO fp32 | 0 | 4 | 3 | 78.2 |
| yolov8n | OpenVINO int8 | 0 | 7 | 5 | 81.4 |

**Read with care.** These counts use the old fixed rule (horizontal line at 60% of the height, both directions). test_video.mp4 is a handheld promo clip (moving camera, bags travel sideways) and test_video2.mp4 an edited promo with scene cuts: neither is a valid counting test, so this table only shows runtime agreement. On test_video3.mp4 the 60% line sits in the truck where workers handle bags. The service's accuracy is measured instead against hand-labelled crossings with a user-drawn line: see README (9/9 on the chute clip, tests/integration).

## Live RTSP, 45 s per run (MediaMTX, H.264 ~30 FPS per camera, newest-frame policy)

| model | cameras | processed FPS / camera | frames dropped | latency p50 | p95 | counts per camera |
|---|---|---|---|---|---|---|
| yolov8n INT8 | 1 | 20.3 | 19% | 26 ms | 123 ms | [5] |
| yolov8n INT8 | 2 | 19.3 | 23% | 29 ms | 119 ms | [5, 5] |
| yolov8n INT8 | 4 | 17.9 | 27% | 48 ms | 118 ms | [4, 4, 2, 4] |
| yolov8s INT8 | 1 | 18.9 | 24% | 59 ms | 200 ms | [3] |
| yolov8s INT8 | 2 | 17.9 | 28% | 61 ms | 200 ms | [3, 3] |
| yolov8s INT8 | 4 | 14.8 | 39% | 62 ms | 206 ms | [4, 4, 2, 3] |
| yolov8x INT8 | 1 | 3.9 | 84% | 212 ms | 806 ms | [3] |
| yolov8x INT8 | 2 | 4.0 | 84% | 207 ms | 811 ms | [3, 3] |
| yolov8x INT8 | 4 | 3.7 | 85% | 205 ms | 902 ms | [3, 3, 2, 3] |

## C++ vs Python (same INT8 model, same 900 frames of test_video2, 8 cores)

| model | C++ sync FPS | C++ async FPS | Python FPS | C++ pre / infer / post ms | Python pre / infer / post ms |
|---|---|---|---|---|---|
| yolov8x | 19.4 | 19.6 | 17.2 | 1.2 / 49.9 / 0.02 | 4.3 / 51.7 / 0.8 (incl. tracking) |
| yolov8s | 85.7 | 88.7 | 58.2 | 1.1 / 10.1 / 0.01 | 3.1 / 12.2 / 0.6 (incl. tracking) |
| yolov8n | 112.2 | 171.6 | 74.5 | 1.1 / 7.4 / 0.01 | 3.0 / 8.5 / 0.5 (incl. tracking) |

C++ letterboxes into a u8 buffer and lets OpenVINO's PrePostProcessor do BGR->RGB, u8->f32 and /255 inside the compiled model; async = two infer requests, the next frame is read and letterboxed while the current one infers. An earlier version of this table showed C++ slower than Python: e6_cpp.py passed 1280x720 for a 640x360 video, so C++ read four frames as one. Fixed (size from ffprobe, frame count asserted).

