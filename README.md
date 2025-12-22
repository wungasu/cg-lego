# Lego 3D Viewer & Builder

一个基于 Python、OpenGL 和 LDraw 格式的 3D 乐高积木查看与搭建工具。

## 依赖环境

推荐使用 Conda 创建独立的运行环境：

```bash
# 1. 创建环境 (Python 3.12.12)
conda create -n lego-viewer python=3.12.12

# 2. 激活环境
conda activate lego-viewer

# 3. 安装依赖
pip install -r requirements.txt
```

### 手动安装

如果你不使用 Conda，请确保已安装 Python 3.12+，然后运行：

```bash
pip install -r requirements.txt
```

## 运行方式

1.  确保已下载 LDraw 零件库（放置于src同级目录下/samples）。
2.  运行主程序：
    ```bash
    python src/main.py
    ```

## 快捷键与操作

*   **鼠标左键**：选中零件 / 旋转视角（点击空白处）/ 拖拽零件。
*   **鼠标右键**：平移视角。
*   **滚轮**：缩放视角。
*   **Delete / Backspace**：删除选中零件。
*   **Ctrl + D**：复制选中零件。

## 项目结构

*   `src/main.py`: 程序入口，负责窗口管理、输入处理和主循环。
*   `src/renderer.py`: OpenGL 渲染器，处理 Shader、VBO/VAO 和光照。
*   `src/ldraw.py`: LDraw 文件解析器，处理几何体构建和颜色解析。
*   `src/scene.py`: 场景图管理，处理对象层级和矩阵变换。
*   `src/camera.py`: 摄像机逻辑。

## 许可证

MIT License
