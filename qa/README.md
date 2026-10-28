# 软件验收记录

最终验收以 `acceptance.json` 和最新 `integration_report.json` 为准。

- `kernel_equivalence.json`：与作者旧代码的小规模同种子初始化、前向、损失、梯度对照。
- `gui_test.json`：真实GUI“检查数据”按钮、后台进程、状态轮询和结束后控件恢复测试。
- `integration_report.json`：三个模型的一轮缩小规模训练，以及原权重在case 16末端区间两帧的全粒子预测、指标核对、NPZ/VTU/PVD和云图导出。
- `vtk_validation.json`：独立VTK/PyVista读取与NPZ字段对应检查。
- `gui_setup.png`、`gui_prediction.png`：界面截图。

这些是软件与数据流验证，不是三个模型重新完成了500轮训练，也不是重新评估了全部三个测试算例的全部74个状态。原参考值始终保存在 `pretrained/` 中。

开发过程中产生的旧运行目录保留用于追溯；其中可能有已修复问题的失败记录。请不要把历史失败目录或smoke模式低精度结果当作最终模型结果。
