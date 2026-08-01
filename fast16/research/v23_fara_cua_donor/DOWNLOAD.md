# v23 Fara1.5-27B 下载清单

状态：等待用户下载。当前 D 盘约余 69.48 GiB；两文件合计约 19.97 GiB，下载后约余49.5 GiB。

固定社区GGUF revision：`dd7cba968d1a9c8feab0c2b85d93b117e6cc16fe`。

## 1. 文本主干（Q5_K_M，研究供体优先保精度）

- 文件：`Fara1.5-27B-Q5_K_M.gguf`
- 大小：20,513,802,048字节（19.105 GiB）
- SHA-256：`91bb8785f51df1f175d8811f9b275228c566ca4b92249d8a851b57dfb42e86a8`
- 放到：
  `D:\project\大模型ssd化\fast16\models\donor\fara15_27b\Fara1.5-27B-Q5_K_M.gguf`
- 官方CDN：
  <https://huggingface.co/bartowski/Fara1.5-27B-GGUF/resolve/dd7cba968d1a9c8feab0c2b85d93b117e6cc16fe/Fara1.5-27B-Q5_K_M.gguf?download=true>
- 国内镜像：
  <https://hf-mirror.com/bartowski/Fara1.5-27B-GGUF/resolve/dd7cba968d1a9c8feab0c2b85d93b117e6cc16fe/Fara1.5-27B-Q5_K_M.gguf>

## 2. 视觉投影（F16）

- 文件：`mmproj-Fara1.5-27B-f16.gguf`
- 大小：927,607,456字节（0.864 GiB）
- SHA-256：`6358165b66c42f375c69da8a990b4fa60ffd2a6b0175c90c48f7d15737ce059d`
- 放到：
  `D:\project\大模型ssd化\fast16\models\donor\fara15_27b\mmproj-Fara1.5-27B-f16.gguf`
- 官方CDN：
  <https://huggingface.co/bartowski/Fara1.5-27B-GGUF/resolve/dd7cba968d1a9c8feab0c2b85d93b117e6cc16fe/mmproj-Fara1.5-27B-f16.gguf?download=true>
- 国内镜像：
  <https://hf-mirror.com/bartowski/Fara1.5-27B-GGUF/resolve/dd7cba968d1a9c8feab0c2b85d93b117e6cc16fe/mmproj-Fara1.5-27B-f16.gguf>

不要同时下载Q4/Q6/Q8副本。Q5只作为高精度供体与活体采集模型，不会整模塞进最终ColorLM；
通过能力和坐标门后才提取连续岛或视觉栈。
