# 附录：windows系统中编程环境配置示例

这个文档以windows系统为例，介绍如何配置基于conda的程序环境，以完成课程第四次作业。

1.	下载miniconda3(anaconda的精简版)，链接 <https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Windows-x86_64.exe>, 双击后安装。
2.	在开始菜单->程序中找到`anaconda powershell prompt(miniconda3)`，双击后打开conda 命令行，输入`conda create -n myenv python=3.8.* ` 创建名为`myenv`的conda环境。
3.	输入`conda activate myenv`后，进入`myenv`环境。输入`pip install torch==1.8.0 torchtext===0.9.0` 在conda中安装pytorch/torchtext的package. 如果下载速度太慢，可以更换pip的安装源为清华镜像源，具体方法可参考<https://mirrors.tuna.tsinghua.edu.cn/help/pypi/>
4. 进入作业程序所在的文件夹，输入`python SampleCode.py`来运行或调试你写好的程序。


注：若conda中进入Python程序报错`UnicodeDecodeError: 'gbk' codec can't decode byte 0x84 in position 68: illegal multibyte sequence`，根据提示修改相应的`history.py`文件`C:\Users\UserName\AppData\Roaming\Python\Python38\site-packages\pyreadline\lineeditor\history.py` (路径名因人而异，以报错提示的路径为准)，将文件中的`Line82: for line in open(filename, 'r'):`改为`for line in open(filename, 'r', encoding='UTF-8'):` 即可。这个报错不影响程序运行。
