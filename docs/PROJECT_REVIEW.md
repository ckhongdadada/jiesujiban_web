# 项目审视

以下结论只基于当前目录中的现有文件和运行验证，不涉及删除或修改你的原始文件。

## 主要不足

### 1. 前端传了 `_force_unit`，后端却没有消费，导致“切换候选单位重新生成回复”实际不生效

- 前端在 [index.html](C:/Users/28414/PycharmProjects/接诉即办项目/templates/index.html:584) 到 [index.html](C:/Users/28414/PycharmProjects/接诉即办项目/templates/index.html:591) 会把 `_force_unit` 发到 `/api/analyze`
- 但后端 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py:222) 到 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py:235) 只读取了 `tag/title/body`，没有读取 `_force_unit`

影响：
- 用户点击第二、第三候选单位时，界面看起来像“按该单位重生了”，实际仍然按 top1 单位生成，属于功能性偏差

### 2. 健康检查和懒加载逻辑互相打架，页面可能一直显示“模型加载中”

- 后端健康接口 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py:244) 到 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py:250) 只检查模型对象是否已加载
- 但模型是懒加载的，要等首次分析请求才会初始化，见 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py:47) 和 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py:87)
- 前端在 [index.html](C:/Users/28414/PycharmProjects/接诉即办项目/templates/index.html:670) 到 [index.html](C:/Users/28414/PycharmProjects/接诉即办项目/templates/index.html:683) 把“未加载”当成“服务还没就绪”

影响：
- 服务其实已经启动，但页面会持续轮询并提示“模型加载中”，容易让使用者误判为系统异常

### 3. 模型路径是本机绝对路径，项目不可迁移，也不利于部署

- 生成模型路径写死在 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py:20)
- LoRA 路径写死在 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py:21)

影响：
- 换电脑、换目录、交付给别人或部署到服务器，都会直接失效

### 4. 分类模型目录当前没有实际模型文件，按现状无法正常跑通分类

- 代码预期从 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py:17) 和 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py:18) 读取分类模型与 `label_map.json`
- 但当前目录 [empty.gitkeep](C:/Users/28414/PycharmProjects/接诉即办项目/final_model_fgm/empty.gitkeep) 之外没有任何权重或标签映射文件

影响：
- 即使 Flask 服务能启动，第一次调用分类也会失败，属于运行时阻塞问题

### 5. 安居客与链家已经存在反爬/登录拦截，单纯 `requests` 方案不稳定

- 本地连通性验证时，链家返回了登录内容，安居客返回了反爬验证内容

影响：
- 如果后续直接把线上词典更新完全建立在这两个站点的静态抓取上，成功率会波动很大，需要准备 cookie、Playwright 或 API 兜底

## 三个后续选项

### 选项 1
先把增强版跑通。

适合现在就要演示流程。
动作：
- 继续用新增的 `app_enhanced.py`
- 补最小量测试数据
- 验证 `NER + RAG + 回复生成` 链路

### 选项 2
先把词典建设做扎实。

适合你接下来重点做“北京地址归区”的准确率。
动作：
- 用我新增的爬虫脚本产出扩展词典
- 再接入高德 API 做站点/街道的区级归属补全
- 形成可持续更新的别名字典

### 选项 3
先补项目稳定性。

适合你准备把这个系统长期保留或交给别人运行。
动作：
- 把绝对路径改成配置文件或环境变量
- 分离健康检查和模型预热
- 补模型存在性检查、错误提示和启动自检
