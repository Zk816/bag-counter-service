#!/usr/bin/env bash
# Build against the OpenVINO C++ runtime that ships inside the pip wheel (headers + CMake config).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OV=$(../../../venv/bin/python -c "import openvino, os; print(os.path.dirname(openvino.__file__))")
cmake -S . -B build -DOpenVINO_DIR="$OV/cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
echo "built $(pwd)/build/detector (run with LD_LIBRARY_PATH=$OV/libs)"
