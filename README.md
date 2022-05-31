# deeplearning
深度学习课程编程作业

1. 将本仓库克隆到本地
2. 在https://github.com/DengPingFan/FS2K 下载数据集和代码
下载后应该有FS2K.zip和FS2K-main.zip两个文件
3. 在主目录（和src同级的目录）解压出FS2K和FS2K-main两个目录
此时仓库应该是这样的
```
DEEPLEARNING
├── FS2K
│       ├── photo
│       │       ...
│       ├── sketch
│       │       ...
│       ├── anno_test.json
│       ├── anno_train.json
│       └── README.pdf
├── FS2K-main
│       ├── tools
│       │       ├──split_train_test.py
│       │       ...
│       ...
├── src
│       ├── homework.py
│       ├── run.ipynb
│       └── run.py
└── README.md
```
4. `cd FS2K-main`进入FS2K-main目录
5. `python tools/split_train_test.py`分割数据集
6. `cd ../src`进入src目录
7. `python run.py`运行run.py即可