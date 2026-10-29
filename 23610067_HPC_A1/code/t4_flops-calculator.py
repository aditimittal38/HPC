import subprocess
import re

def get_lscpu():
    output = subprocess.check_output(["lscpu"], text=True)
    return output

def get_value(output, key):
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", output, re.MULTILINE)
    return match.group(1).strip() if match else "Unknown"

# Get CPU information
cpu_info = get_lscpu()

model = get_value(cpu_info, "Model name")

# Physical cores
cores = int(get_value(cpu_info, "Core(s) per socket"))

# Number of sockets
sockets = int(get_value(cpu_info, "Socket(s)"))

physical_cores = cores * sockets

# Logical processors
logical_processors = int(get_value(cpu_info, "CPU(s)"))

# Maximum frequency
max_freq_match = re.search(r"CPU max MHz:\s*([\d.]+)", cpu_info)

if max_freq_match:
    max_frequency_mhz = float(max_freq_match.group(1))
else:
    # Fallback: read /proc/cpuinfo
    freq_match = re.search(r"cpu MHz\s*:\s*([\d.]+)", 
                           subprocess.check_output(
                               ["bash", "-c", "grep 'cpu MHz' /proc/cpuinfo | head -1"],
                               text=True))
    max_frequency_mhz = float(freq_match.group(1)) if freq_match else 0

frequency_ghz = max_frequency_mhz / 1000
frequency_hz = max_frequency_mhz * 1_000_000

# Detect SIMD instruction set
flags = get_value(cpu_info, "Flags")

if "avx512f" in flags:
    simd_width_bits = 512
    simd_type = "AVX-512"
elif "avx2" in flags:
    simd_width_bits = 256
    simd_type = "AVX2"
elif "avx" in flags:
    simd_width_bits = 256
    simd_type = "AVX"
elif "sse" in flags:
    simd_width_bits = 128
    simd_type = "SSE"
else:
    simd_width_bits = 32
    simd_type = "Scalar"

# FP32 = 32 bits
fp32_bits = 32
simd_elements = simd_width_bits // fp32_bits

# FMA = 2 FLOPs
fma_flops = 2

# Calculate theoretical peak FLOPS
peak_flops = (
    physical_cores
    * frequency_hz
    * simd_elements
    * fma_flops
)

peak_gflops = peak_flops / 1e9
peak_tflops = peak_flops / 1e12

# Print report
print("=" * 70)
print("       THEORETICAL PEAK FLOPS CALCULATION")
print("=" * 70)

print("\n1. CPU INFORMATION")
print("-" * 70)
print(f"Processor Model       : {model}")
print(f"Physical CPU Cores    : {physical_cores}")
print(f"Logical Processors    : {logical_processors}")
# print(f"Maximum Frequency     : {frequency_mhz:.2f} MHz")
print(f"Maximum Frequency     : {frequency_ghz:.2f} GHz")
print(f"SIMD Instruction Set  : {simd_type}")
print(f"SIMD Register Width   : {simd_width_bits} bits")
print(f"FP32 Values/Instruction: {simd_elements}")

print("\n2. VALUES USED")
print("-" * 70)
print(f"Number of Cores       = {physical_cores}")
print(f"Frequency             = {frequency_ghz:.2f} GHz")
print(f"Frequency in Hz       = {frequency_hz:.2e} Hz")
print(f"SIMD Width            = {simd_elements} FP32 values")
print(f"FMA                   = {fma_flops} FLOPs")

print("\n3. FORMULA")
print("-" * 70)
print("Peak FLOPS = Cores × Frequency × SIMD Width × Operations per Instruction")

print("\nFor FMA:")
print("1 FMA = 2 FLOPs")
print("Therefore:")
print("Peak FLOPS = Cores × Frequency × SIMD Width × 2")

print("\n4. SUBSTITUTION")
print("-" * 70)
print(
    f"Peak FLOPS = {physical_cores} × "
    f"{frequency_hz:.2e} × "
    f"{simd_elements} × 2"
)

print("\n5. CALCULATION")
print("-" * 70)

step1 = physical_cores * frequency_hz
print(f"{physical_cores} × {frequency_hz:.2e}")
print(f"= {step1:.2e}")

step2 = step1 * simd_elements
print(f"\n{step1:.2e} × {simd_elements}")
print(f"= {step2:.2e}")

step3 = step2 * 2
print(f"\n{step2:.2e} × 2")
print(f"= {step3:.2e} FLOPS")

print("\n6. FINAL RESULT")
print("=" * 70)
print(f"Theoretical Peak Performance = {peak_flops:.2e} FLOPS")
print(f"Theoretical Peak Performance = {peak_gflops:.2f} GFLOPS")
print(f"Theoretical Peak Performance = {peak_tflops:.4f} TFLOPS")
print("=" * 70)

print("\nNOTE:")
print("This is a theoretical peak calculation assuming:")
print("• All physical CPU cores are operating at the selected frequency")
print("• SIMD instructions are fully utilized")
print("• FMA instructions are being used")
print("• FP32 floating-point operations are being performed")
print("Actual performance will normally be lower.")