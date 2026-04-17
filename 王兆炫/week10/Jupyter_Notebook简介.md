因为首次使用Jupyter Notebook, 所以此处先进行了一个整体的了解, 以便后续工作的进行

### 基本知识

> 发音 Jupyter 发音应为 /ˈdʒuːpɪtər/

Jupyter Notebook 是一个开源的Web应用程序，用于创建和共享包含实时代码、公式、可视化图表和解释性文本的交互式文档。它本质上是一个在浏览器中运行的交互式笔记本，支持 Python、R、Julia 等多种语言，是数据分析、机器学习和科学计算的常用工具。 

核心特点与应用场景：
- 交互式编程： 可以在浏览器中直接运行代码，并在单元格下方即时查看运行结果（文字、图表、HTML等），非常适合探索性分析和快速原型设计。
- 富文本与文档化： 支持Markdown语法编写文本，以及LaTeX公式，使代码与说明文档能完美结合，非常适合撰写数据报告、教程和可重复的研究论文。
- 模块化开发： 代码按单元格（Cell）组织，可独立执行，无需每次从头运行整个脚本，调试效率高。

> 生成的文件后缀为.ipynb , 因为其原名为ipython notebook

| 项目 | .py（Python 脚本） | .ipynb（Jupyter Notebook） |
|---|---|---|
| 本质 | 纯文本，一整份从上到下执行 | JSON 文件，里面是很多单元格（cell） |
| 内容 | 几乎全是代码 | 可混用 Markdown 说明和代码 |
| 执行方式 | 整文件或从 IDE 指定入口跑 | 通常按单元格运行（Shift+Enter），顺序和次数都可以灵活 |
| 结果 | 多在终端打印或写文件 | 图表、表格、输出可以留在格子下面，适合教学与可视化 |
| 适用 | 项目、脚本、部署 | 学习、实验、展示、带图的分析等富文本需求 |


打开对应conda环境后,执行以下代码安装Jupyter notebook
```bash
pip install jupyter notebook
```

可以直接在本地ide中使用jupyter扩展打开,

也可以在终端中
使用如下命令运行jupyter notebook (会在默认浏览器里打开)
```bash
jupyter notebook
# 运行特定文件
jupyter notebook my_notebook.ipynb
```

### 使用介绍

内容来自官方[user documentation](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html#introduction)

#### 页面布局

<img width="827" height="242" alt="2cbe4ec8-6202-4c4f-9934-95e34d40b6d4" src="https://github.com/user-attachments/assets/33a5efe3-1e0d-4430-9e12-a7357cf225d0" />

notebook由一系列单元格组成。单元格是一个多行文本输入框，其内容可以通过按下 Shift+Enter 键，或点击工具栏中的“运行”按钮，或通过菜单栏中的“单元格”>“运行”来执行。单元格的执行行为由其类型决定。单元格分为三种类型：代码单元格、Markdown 单元格和原始单元格。每个单元格初始均为代码单元格，但可以通过工具栏上的下拉菜单（初始选项为“代码”）或键盘快捷键更改其类型。

#### 快捷键

+ Shift-Enter: run cell : 执行当前单元格，显示任何输出结果，并跳转到下方的下一个单元格。如果在最后一个单元格上按下 Shift+Enter，则会在下方创建一个新单元格。这相当于单击“单元格”菜单中的“运行”选项，或单击工具栏中的“播放”按钮。

+ Esc: Command mode : 在命令模式下，您可以使用键盘快捷键在笔记本中进行导航(navigate)。

+ Enter: Edit mode : In edit mode, you can edit text in cells.

+ Ctrl+Alt+Enter: 修改完毕当前单元格,确认修改







