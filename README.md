# YOLOv5 Graduation Design

> 基于 YOLOv5 和 Flask 实现的简单的目标检测 Web 应用（pytorch yolov5n），权重已经训练好
> 实现图片和本地摄像头的视频检测有无口罩佩戴，并且支持多目标
> Written by CZY SDU

## 运行方法

- 创建虚拟环境以供隔离其他项目环境（可选）例如`conda create -n your_env_name python=x.x.`
- 激活虚拟环境`conda activate your_env_name`
- 安装 Python 环境依赖 `pip install -r requirement.txt` 这一步注意检查本地是否配好 cuda 环境，否则可能安装的是 cpu 版本，对速度有影响
- 进入根目录后 `python web_main.py` 运行 Flask 后端
- 访问本地端口`5000`即可进行目标检测

## 补充说明

`yolov5_config.json`是对模型的一些配置，自己可以更换，例如可以更换模型实现不同场景下的目标检测，或者不用本地的摄像头，更换其中的视频流，具体见`camera.py`，支持添加额外的视频流，需要自己在`video.html`里修改

## 日志和更新计划

- 3 月晚些时候 后端设计
- 4.9 系统界面和路由跳转 开源
- 4.15 左右 完成图片检测/视频检测的界面设计
