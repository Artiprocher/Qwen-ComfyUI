# Qwen-ComfyUI

这是一个 Qwen 模型的 ComfyUI 节点库，可以用来做复杂的文本处理工作，包括以下节点：

* Qwen Model Loader: 加载模型，可以用本地路径（从[这里](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)下载，支持其他 Qwen 模型），也可以用百炼 API（从[这里](https://bailian.console.aliyun.com/)注册并获取 API KEY）
* Qwen Agent: 模型代理。其中的文本框用来填 System prompt
* TextReader: 从文件读取文本
* TextWriter: 把文本写入到文件
* TextFormater: 把输出内容前后的 ``` 删除
* TextMerger: 拼接文本
* TextInput: 以文本框的形式输入文本

使用方法：

* 安装 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* 把 `QwenAgent` 文件夹放到 `custom_nodes` 里
* 启动 ComfyUI
* 把 `example_workflow.json` 拖进去

