这些脚本记录本次恢复过程，不是训练入口。脚本中的 2public/dataset 在整理时指替换前输入；该输入现归档于 F:/HGL_WORK_PASSED/posefusion/dataset_reorganization_20260929/archive/dataset_before_background_20260929。重现恢复时需将相应输入路径改为归档目录，并使用独立输出目录。源 Blender 文件未覆盖。

GT_source_manifest.json 逐份记录 33 个新版 GT mesh 的来源和哈希；各 background_recovery.json 记录真实数据逐帧核对；两份 render_report 记录常规虚拟数据按原相机补渲染。海豹来源和完整核对见 seal/gt。
