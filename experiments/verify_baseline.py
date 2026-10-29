"""端到端验证：基线按钮 → 对比表分组 → 画布图层。

用系统 Edge（不下载 Chromium）驱动真实页面：
  1. 生成实例
  2. 点「OR-Tools」算基准
  3. 点两个规则基线按钮
  4. 检查对比表是否按「学习方法 / 规则基线 / 参照」分组
  5. 检查画布上是否画出 baseline 图层

用法：
    .venv\\Scripts\\python.exe scripts\\verify_baseline.py [--port 8000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "data" / "shots"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    errors: list[str] = []
    failed = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failed
        if not cond:
            failed += 1
        print(f"  {'通过' if cond else '失败'}  {name}")
        if detail:
            print(f"        {detail}")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 950})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        print("=" * 68)
        print("  基线按钮与对比表分组验证")
        print("=" * 68)

        page.goto(base, wait_until="networkidle")
        page.wait_for_timeout(1000)

        print("\n[1] 生成实例")
        page.eval_on_selector(
            "#nRange",
            """(el, v) => { el.value = v;
                el.dispatchEvent(new Event('input', {bubbles: true})); }""",
            args.n,
        )
        page.click("#btnInstance")
        page.wait_for_timeout(1500)
        info = page.locator("#instInfo").inner_text().strip()
        check("实例已生成", "城市" in info, info)
        check("画布空状态消失", page.locator("#canvasEmpty").is_hidden())

        print("\n[2] 求解 OR-Tools 基准")
        page.click("#btnOrtools")
        page.wait_for_timeout(4000)
        log = page.locator("#logBox").inner_text()
        check("OR-Tools 完成", "OR-Tools 完成" in log,
              [ln for ln in log.splitlines() if "OR-Tools" in ln][-1:])

        print("\n[3] 求解两个规则基线")
        page.click("#btnRr2opt")
        page.wait_for_timeout(6000)
        log = page.locator("#logBox").inner_text()
        check("随机重启+2-opt 完成", "随机重启+2-opt 完成" in log,
              [ln for ln in log.splitlines() if "随机重启" in ln][-1:])

        page.click("#btnMs2opt")
        page.wait_for_timeout(6000)
        log = page.locator("#logBox").inner_text()
        check("多起点+2-opt 完成", "多起点+2-opt 完成" in log,
              [ln for ln in log.splitlines() if "多起点" in ln][-1:])

        print("\n[4] 训练一次（让「学习方法」分组有内容）")
        page.fill("#episodesInp", "200")
        page.fill("#earlyStopInp", "150")
        page.click("#btnTrain")
        page.wait_for_timeout(3000)
        # 等训练结束
        for _ in range(90):
            if not page.locator("#btnTrain").is_disabled():
                break
            page.wait_for_timeout(1000)
        page.wait_for_timeout(2000)
        log_txt = page.locator("#logBox").inner_text()
        check("训练已结束", page.locator("#btnTrain").is_enabled(),
              [ln for ln in log_txt.splitlines() if "训练" in ln][-2:])

        print("\n[5] 对比表分组")
        groups = page.locator("#resultBody tr.grp")
        n_grp = groups.count()
        check("有 3 个分组标题", n_grp == 3, f"实际 {n_grp}")
        if n_grp == 3:
            titles = [groups.nth(i).locator(".grp-title").inner_text().strip() for i in range(n_grp)]
            check("分组为 学习方法/规则基线/参照",
                  titles == ["学习方法", "规则基线（不需要学习）", "参照"], str(titles))

        tbl = page.locator("#resultBody").inner_text()
        check("含 DQN 训练最优", "DQN 训练最优" in tbl)
        check("含 随机重启 + 2-opt", "随机重启 + 2-opt" in tbl)
        check("含 OR-Tools 基准", "OR-Tools 基准" in tbl)

        print("\n[6] 数值检查")
        rows_txt = page.locator("#resultBody").inner_text()
        # 提取随机重启基线那一行的长度与 gap
        import re

        m = re.search(r"随机重启 \+ 2-opt\s+([\d.]+)\s+([\d.]+)%", rows_txt)
        if m:
            length, gap = float(m.group(1)), float(m.group(2))
            check("随机基线 gap 接近 0%（说明它确实很强）", abs(gap) < 1.0,
                  f"长度 {length}，gap {gap}%")
        else:
            check("能解析出随机基线行", False, rows_txt[:120])

        print("\n[7] 画布图层")
        # baseline 图层是否真的画上去了（通过内部 paths 判断）
        drawn = page.evaluate(
            """() => {
            const c = document.getElementById('tourCanvas');
            const ctx = c.getContext('2d');
            const d = ctx.getImageData(0, 0, c.width, c.height).data;
            // 统计青色像素（baseline 图层颜色 #0891b2）
            let cyan = 0;
            for (let i = 0; i < d.length; i += 4) {
                if (Math.abs(d[i] - 8) < 30 && Math.abs(d[i+1] - 145) < 30 && Math.abs(d[i+2] - 178) < 30) cyan++;
            }
            return cyan;
        }"""
        )
        check("画布上有 baseline 图层像素（青色）", drawn > 50, f"{drawn} 个青色像素")

        print("\n[8] 截图")
        shot = SHOTS / "baseline_table.png"
        page.screenshot(path=str(shot), full_page=False)
        print(f"        已保存 {shot}")

        browser.close()

    print("\n" + "=" * 68)
    print("  全部通过" if failed == 0 else f"  {failed} 项失败")
    if errors:
        print("  JS 报错：")
        for e in errors[:5]:
            print("   ", e)
    print("=" * 68)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
