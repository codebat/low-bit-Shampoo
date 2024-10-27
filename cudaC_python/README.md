## Compilation of libqtensor

**Requirements:** gcc + nvcc + cmake for Linux; msvc + nvcc + cmake for Windows.

To generate .so for Linux, run:

```bash
cmake -B build . && cmake --build build
```

To generate .dll for Windows, run:

```bash
cmake -B build .
cmake --build build --config Release
```

After compilation, the dynamic library can be found in path "./build/qtensor".

Our C++ code is mainly based on the implementation of "bitsandbytes" (see https://github.com/TimDettmers/bitsandbytes/tree/main/csrc).