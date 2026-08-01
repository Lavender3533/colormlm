# Design Genome 解码提示（v1）

只输出一行 JSON，不要 Markdown、解释或空白。固定字段顺序与槽数：2 个 copy 引用、4 个视觉基因、3 个布局基因、5 个角色组件、4 个角色动作、2 个角色响应式变换、`a=255`、`z=inline`。标题和导语分别引用运行时给出的 copy slot 0/1；禁止自由生成用户文案。

槽位顺序不可交换：组件为 `primary/controls/content/detail/support`，动作为 `data/view/commit/state`，响应式为 `main/overlay`。某个动作不适用时选 `none`，不要用重复项填槽。

按需求语义选择闭集组合：

- 运维/数据控制台：`metrics.ops / filters.status / table.ops / drawer.detail / dialog.confirm`；动作 `filter/open/confirm/sort`；响应 `table>cards/drawer>full`。
- 编辑式商店：`hero.editorial / filters.catalog / products.magazine / dialog.preview / bag.counter`；动作 `filter/open/count/reset`；响应 `grid>stack/drawer>full`。不要选 docs 侧栏。
- 案例过程：`hero.plain / filters.project / timeline.case-study / compare.before-after / metrics.generic`；动作 `filter/compare/none/announce`。
- 预约表单：`hero.plain / filters.generic / form.booking / dialog.success / metrics.generic`；动作 `filter/open/validate/reset`。
- API 文档：`sidebar.docs / tabs.language / code.request / tabs.response / drawer.navigation`；动作 `navigate/tab/copy/announce`。不要选商品或状态筛选。
- 隐私设置：`sidebar.settings / toggles.privacy / table.devices / dialog.danger / note.generic`；动作 `sort/open/confirm/toggle`。
- 多舞台日程：`hero.plain / filters.date-stage / schedule.stages / dialog.confirm / metrics.generic`；动作 `filter/accordion/favorite/announce`。
- 年份数据故事：`hero.story / filters.year / chart.year-series / table.data / note.source`；动作 `year/inspect/none/sort`；响应 `chart>scroll/none`。

编译前会执行角色、布局、动作和响应式依赖校验；语法合法但语义错配会被拒绝，不会静默换成另一种页面。最终运行路径是 terminal hidden 上的并行 Genome Head；本 GBNF 只用于教师制作与冷启动。
