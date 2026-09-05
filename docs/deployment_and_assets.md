# 部署与资源文件策略

## 当前策略

Streamlit 应用采用“克隆仓库即可运行”的部署方式，因此线上推理必需的模型产物继续由 Git 跟踪。`scripts/check_runtime_assets.py` 会在 CI 中验证：

- 九个必需模型文件存在且已被 Git 跟踪；
- 任一文件均未达到 GitHub 的 100MB 单文件限制；
- 模型产物总大小可被持续观察。

`data/models/nn_neighbors.npz` 约 84MB，只供离线评估使用，不是应用运行依赖，已在 `.gitignore` 中排除。线上推荐使用 `nn_neighbors.pkl`。

## 封面现状

当前 `app/covers/` 有 5,362 个文件，约 113MB；`verified_covers.json` 标记其中 447 张为已核验封面，约 30MB。页面只直接展示已核验封面，其他文件仍保留为爬取和后续人工核验素材。

当前 9 个线上必需模型产物合计约 43.45MB；本地检查以脚本输出为准，不在文档中维护容易过期的逐文件精确大小。

不在普通功能分支中批量删除封面，原因是：

1. 首页候选池会因封面集合变化而变化；
2. 从当前版本删除文件不会缩小既有 Git 历史；
3. 真正缩小仓库需要历史重写，所有协作者都要重新克隆。

## 后续大规模瘦身方案

若需要显著缩短克隆或 Streamlit 部署时间，应单独执行资源迁移：

1. 将未验证封面移到 GitHub Release、对象存储或可再生成的数据包；
2. 为资源包生成版本号与 SHA-256 校验值；
3. 应用启动时只下载固定版本的资源包，并提供缺失封面的占位图；
4. 在所有协作者确认后，使用 `git filter-repo` 重写历史并强制更新远端；
5. 重新运行九页面冒烟测试，再切换生产部署分支。

历史重写和资源删除均属于破坏性操作，必须单独审批，不与模型或页面功能修改混在同一提交中。

## 本地检查

```bash
python scripts/check_runtime_assets.py
python -m ruff check app src tests crawler scripts
python -m pytest
```
