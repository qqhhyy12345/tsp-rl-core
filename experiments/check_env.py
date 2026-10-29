"""环境自检脚本 —— 验证 ROCm / PyTorch 是否正确识别 RX 9060 XT。

用法：
    .venv\\Scripts\\python.exe scripts\\check_env.py

背景（本机实测结论，2026-10）：
    本机同时存在两块 AMD GPU：
      [0] AMD Radeon(TM) Graphics   —— CPU 核显，RDNA 3.5 (gfx115x)
      [1] AMD Radeon RX 9060 XT     —— 独立显卡，RDNA 4 (gfx1200)
    安装的 wheel 是 gfx120X-all，只为 RDNA 4 编译 kernel。
    因此核显虽被 torch 枚举到，但一算就报 hipErrorInvalidImage。
    解决办法：设置环境变量 HIP_VISIBLE_DEVICES=1，把独显变成 device 0。
"""

import os
import platform
import sys
import time

# 必须在 import torch 之前设置，否则不生效
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    os.environ["HIP_VISIBLE_DEVICES"] = sys.argv[1]

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def probe_device(idx: int) -> tuple[bool, str]:
    """在指定设备上跑一次矩阵乘，确认 kernel 真的能执行。"""
    try:
        a = torch.randn(256, 256, device=f"cuda:{idx}")
        b = a @ a
        torch.cuda.synchronize()
        return bool(torch.isfinite(b).all()), ""
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0]
        return False, msg


def main() -> int:
    ok = True

    section("1. 基础环境")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"解释器      : {sys.executable}")
    print(f"操作系统    : {platform.system()} {platform.release()}")
    if sys.version_info[:2] != (3, 12):
        print("  [X] ROCm 的 Windows wheel 按 cp312 构建")
        ok = False

    section("2. PyTorch 与 ROCm")
    print(f"torch 版本  : {torch.__version__}")
    has_rocm = "+rocm" in torch.__version__
    print(f"ROCm 构建   : {'是' if has_rocm else '否'}")
    if not has_rocm:
        ok = False

    cuda_ok = torch.cuda.is_available()
    print(f"GPU 可用    : {cuda_ok}")
    if not cuda_ok:
        print("  [X] 检查驱动版本是否与 ROCm 7.13.0 匹配")
        return 1

    section("3. 设备枚举与逐设备实算探测")
    n = torch.cuda.device_count()
    visible = os.environ.get("HIP_VISIBLE_DEVICES", "(未设置)")
    print(f"HIP_VISIBLE_DEVICES = {visible}")
    print(f"可见设备数 = {n}\n")

    usable = []
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        good, err = probe_device(i)
        mark = "可用" if good else "不可用"
        print(f"  [{i}] {props.name}")
        print(f"      显存 {props.total_memory / 1024**3:5.1f} GB   状态: {mark}")
        if good:
            usable.append(i)
        else:
            print(f"      原因: {err[:100]}")

    if not usable:
        print("\n  [X] 没有任何设备能执行 kernel")
        return 1

    target = usable[-1]
    print(f"\n  → 建议使用设备索引 {target}"
          f"（{torch.cuda.get_device_name(target)}）")

    section("4. 目标设备性能实测")
    secs = 20
    a = torch.randn(1024, 1024, device=f"cuda:{target}")
    b = torch.randn(1024, 1024, device=f"cuda:{target}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(secs):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = secs * 2 * 1024**3 / dt / 1e12
    print(f"1024x1024 矩阵乘 x{secs} : {dt * 1000:7.1f} ms   约 {tflops:.2f} TFLOPS")
    print(f"结果有限性校验           : {bool(torch.isfinite(c).all())}")

    section("5. DQN 小网络 CPU vs GPU 耗时对比")
    print("  验证「Track A 固定跑 CPU」这一结论")
    net = nn.Sequential(
        nn.Linear(40, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, 128), nn.Tanh(),
        nn.Linear(128, 20),
    ).eval()
    x = torch.randn(1024, 40)

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(50):
            net(x)
        cpu_ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"  CPU : {cpu_ms:7.3f} ms / batch")

    net_g = net.to(f"cuda:{target}")
    xg = x.to(f"cuda:{target}")
    with torch.no_grad():
        for _ in range(5):
            net_g(xg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            net_g(xg)
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"  GPU : {gpu_ms:7.3f} ms / batch")
    ratio = gpu_ms / cpu_ms
    print(f"  比值: GPU/CPU = {ratio:.1f}x  → "
          f"{'GPU 明显更慢，小网络应固定 CPU' if ratio > 1 else 'GPU 更快'}")

    section("自检结论")
    if ok and usable:
        print("环境就绪")
        if n > 1 and target != 0:
            print(f"\n注意：本机有 {n} 块 GPU，可用的是索引 {target}。")
            print("  启动训练前请设置环境变量：")
            print(f"    PowerShell : $env:HIP_VISIBLE_DEVICES=\"{target}\"")
            print(f"    CMD        : set HIP_VISIBLE_DEVICES={target}")
            print("  否则 torch 默认选中索引 0，会因架构不匹配而报 hipErrorInvalidImage。")
        return 0
    print("环境未就绪，请排查上面标 [X] 的项")
    return 1


if __name__ == "__main__":
    sys.exit(main())
