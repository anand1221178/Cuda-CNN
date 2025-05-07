# CUDA Image Convolution

This CUDA program performs 2D convolution on grayscale `.pgm` images using **Sharpen**, **Average**, or **Emboss** filters of sizes **3x3**, **5x5**, or **7x7**, and runs in three modes:

- ✅ **CPU (Serial)**
- 🚀 **GPU using Global Memory**
- ⚡ **GPU using Shared Memory**

---

## ✅ Usage

```bash
make clean && make && ./imageConvolution -input=your_image.pgm -filter=TYPE -size=SIZE
```

### 🔧 Arguments

| Flag      | Description                    | Values                          |
|-----------|--------------------------------|---------------------------------|
| `-input`  | Path to `.pgm` grayscale image | e.g. `teapot512.pgm`            |
| `-filter` | Filter type                    | `sharpen`, `average`, `emboss` |
| `-size`   | Mask size for the filter       | `3`, `5`, or `7`                |

---

## 💡 Examples

```bash
./imageConvolution -input=teapot512.pgm -filter=emboss -size=3   # 3x3 Emboss
./imageConvolution -input=teapot512.pgm -filter=sharpen -size=5  # 5x5 Sharpen
./imageConvolution -input=teapot512.pgm -filter=average -size=7  # 7x7 Average
```

---

## 💾 Output Files

After execution, the following files are generated:

- `serial_output.pgm` – output from CPU serial implementation  
- `global_output.pgm` – output from GPU global memory implementation  
- `shared_output.pgm` – output from GPU shared memory implementation  

Additionally, performance metrics and speedups are printed in the terminal.

---

## 🧼 Cleaning & Rebuilding

To clean and recompile:

```bash
make clean && make
```

---

## 🖥️ Running on Wits Cluster

To run the CUDA program on the Wits cluster, use a job submission script. Ensure that `run_cuda.sh` is in the same directory as the `.cu` file and then run:

```bash
sbatch run_cuda.sh
```

Make sure the script correctly references your CUDA executable (e.g., `imageConvolution`) and sets the appropriate resource requests (e.g., GPU, time, memory).

---

## 📂 Notes

- Input must be a grayscale `.pgm` image.
- Make sure your system supports CUDA and has an NVIDIA GPU.
- The program uses CUDA C++ and must be compiled with `nvcc`.

---

## 👨‍💻 Author

Anand Patel