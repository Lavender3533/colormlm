# Polaris S14 官方聊天编码

DeepSeek-V4-Flash-0731 不提供 Jinja chat template。当前目录固定并随仓库保存官方 revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062` 的 `encoding` 参考实现，使 S14 后续的
system、user、assistant、reasoning effort、工具定义、工具调用和工具结果使用 donor 自己的协议。

入口：

```python
from fast16.research.polaris_meridian_v1.s14_chat_encoding import encode_messages

prompt = encode_messages(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
    ],
    thinking_mode="thinking",
    reasoning_effort="max",
)
```

运行离线自检：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.s14_chat_encoding.selftest
```

自检覆盖官方4组 encode/parse fixtures、固定源文件指纹，并在本机 tokenizer 存在时确认首 token
确实为官方 BOS ID 0。它不运行模型，也不证明 S14 质量。

## Forced-prefill token 合同

`forced_prefill.py` 将严格的 `messages + reasoning_effort + tools` 输入交给上述官方编码，
再用显式指定的本地 `tokenizer.json` 产生确定的 token ID 数组。正式 `s14` profile 会硬校验
词表大小 129,280、BOS ID 0 和 BOS 单 token 编码；输出声明从 position 0 开始按数组顺序
逐 token prefill，且明确记录没有执行模型、没有生成 token。

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.s14_chat_encoding.forced_prefill `
  --input fast16/research/polaris_meridian_v1/s14_chat_encoding/fixtures/forced_prefill_input.json `
  --tokenizer D:/models/Polaris-S14/tokenizer.json `
  --output forced_prefill.json
```

输入必须匹配 `forced_prefill_contract.schema.json`；实现还会硬拒绝重复 JSON key、非 UTF-8、
空消息、非法角色/工具序列和协议保留标记注入。仓内小 tokenizer 只供 `fixture` profile
离线测试，输出会标为 `polaris_s14_compatible=false`，不能冒充正式 S14 token。

`encoding_dsv4.py`、`test_encoding_dsv4.py`、`OFFICIAL_README.md` 和 `tests/` 来自官方
MIT 模型仓固定 revision；本地快照统一为 LF 和单个结尾换行，测试脚本只剥离该结尾换行，
编码运行语义未修改。来源和本地 SHA-256
记录在 `source_contract.json`。
