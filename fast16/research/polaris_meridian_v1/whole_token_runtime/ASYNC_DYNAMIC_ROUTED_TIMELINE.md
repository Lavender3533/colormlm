# S14 dynamic routed upload 接入 whole-token timeline

日期：2026-08-03

## 证据边界

- 权威N=8日志为`.tmp-polaris-tests/n8-production-20260803-proxy.stdout.log`。
- position 0--6热态为`13.081878--16.692493s/token`，均值`14.638320s/token`，对应
  `0.068314 token/s`。position 7的`119.549109s`包含11次冷缺页下载，不计入热态。
- 每个热token仍有43次router probe host wait、43次dynamic routed transfer fence wait、
  23次streamed-static transfer fence wait和1次final wait，共140次host API wait。
- 日志没有给每类wait的exclusive毫秒，因此只能把这条逐层串行边界判为当前最强结构瓶颈，
  不能把43次计数伪装成已量出的最大耗时。既有前代profile中attention exclusive约9s/token，
  只作定位先验，不冒充本次N=8同口径计时。

## 本次实现

production两个入口不再调用每层`copy_and_wait`的dynamic routed uploader。online top-6的36个
Range仍逐项通过既有身份、布局和SHA合同，随后写入原双bank host staging；唯一一次
staging→device copy录入已有transfer command，由whole-token timeline提交。matching MoE
continuation通过transfer ticket等待数据可见，错误路径仍由现有drain/rollback收敛。

结构目标是把`dynamic_routed_transfer_fence_waits`从每token `43`降为`0`；43次必须读回48B
top-6的router probe wait继续保留，因为当前host分页计划仍依赖真实专家身份。

## 已跑门

- `cargo check --offline -p ssd_inference --lib`
- `cargo check --offline -p ssd_inference --example s14_position0_paged_43_layers_real`
- `cargo check --offline --release -p ssd_inference --example s14_position0_paged_43_layers_real`
- hybrid staging定向单测：`1/1`
- repeated ratio4 timeline定向单测：`1/1`
- production源码结构门：异步调用点`2`，旧同步调用点`0`

本轮按合同没有启动模型、没有重跑N=8、没有跑旧榜，因此没有端到端提速数字。局部kernel、
结构wait消除和端到端token/s必须分开报告。

## 物理上限

权威日志每token报告约`3.614GB static + 3.449GB routed + 1.059GB head`的逻辑搬运合同。
本次只删除逐层host fence，不减少这些字节。只靠这一步不可能从`0.068 token/s`跳到
`20--50 token/s`；后续必须把GPU router、分页预取和K=4/8 union块结合起来，跨token/块复用
权重扫描并压低实际PCIe字节，最后再以完整端到端门验收。
