# 面向 RTX 5060 的现代 GPU 编程

**从 Triton 到 FlashAttention：一套中文教程与可复现的 GPU 实验。**

本书沿着硬件、编程模型、GEMM、注意力四条递进主线，面向 GeForce RTX 5060 Laptop GPU（SM120）重新组织讲解和代码。适合会 Python、PyTorch 和基本线性代数，希望理解 GPU 内核的读者。每章附源码、运行方法、正确性检查和练习。

## 从哪里开始

- [阅读完整教程](docs/index.md)
- [安装、硬件与第一组实验](docs/01-environment.md)
- [原教程映射与参考资料](docs/appendix/sources.md)
- [实测结果与性能目标状态](docs/15-results.md)

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[docs,test]'
# 未安装 CUDA 版 PyTorch / Triton 时，先按第一章安装匹配版本。
python -m fa5060.probe --output results/environment.json
python -m fa5060.experiments basics --output results/basics.json
python -m fa5060.experiments gemm --output results/gemm.json
python -m fa5060.experiments attention --output results/attention.json
pytest -q
sphinx-build -W --keep-going -b html docs _build/html
python -m http.server --directory _build/html 8000
```

`gpu` 可选依赖安装 PyTorch/Triton；文档构建不需要 GPU。CI 只验证文档和 CPU 检查，GPU 结果由实际硬件实验生成。

GitHub [Actions](https://github.com/Justi255/modern-gpu-programming/actions/workflows/verify.yml) 成功运行后提供可下载的 `rtx5060-textbook-html` 站点产物。本仓库不默认启用 GitHub Pages。

本次已实测冻结的 312 项注意力用例，所选自定义配置全部通过正确性。离线调优后的 prefill/decode 几何平均加速分别为 **0.920× / 1.130×**，最差为 **0.688× / 0.748×**，因此**尚未达到约定的整体性能目标**。完整数据、114 个超过 5% 的回退项及基线数值失败记录见[结果章](docs/15-results.md)，不能将 decode 的平均收益解读为全面领先。

## 实验原则

保留教学版本和优化版本，比较完整算子，分别记录编译、GPU 时间和接口时间。SOTA 是需要验证的实验结论。本项目的验收目标是 prefill、decode 相对逐项最快合法基线分别达到 1.10× 几何平均加速，最差单项不低于 0.95×；以结果章节记录的实测状态为准。

原始参考为 MLC Community 的 [Modern GPU Programming For MLSys](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys)，参考提交 `a78b4e6e9f2be5697705c174d49bbe4974e0c4a5`。本书用中文独立撰写、重绘图示，并将原书的 TIRx/SM100 实践改为 Triton/SM120 实践。来源及架构适用边界见附录。
