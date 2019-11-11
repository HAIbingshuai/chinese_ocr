# chinese_ocr
中文自然场景文字检测及识别
解决方法为：文本检测：YOLOv3进行；文本序列识别方案使用华科白翔老师团队2015年的结合了CNN, RNN 与 CTC loss 的CRNN。

一 各个单一功能点(独立成包，可单独实用)： 
    1，目标检测文本行
    2，文本行倾斜角度检测及纠正
    2，目标识别文本行
    3，证件及车票结构化数据识别
二 总功能（合并各包，整体使用）：
    自然场景下、文本环境下中文（包括英文）检测和识别
    
    
项目感谢：
chineseocr https://github.com/chineseocr/chineseocr
yolo3 https://github.com/pjreddie/darknet.git
crnn https://github.com/meijieru/crnn.pytorch.git
ctpn https://github.com/eragonruan/text-detection-ctpn
CTPN https://github.com/tianzhi0549/CTPN
keras yolo3 https://github.com/qqwweee/keras-yolo3.git
darknet keras 模型转换参考 参考：https://www.cnblogs.com/shouhuxianjian/p/10567201.html
语言模型实现 https://github.com/lukhy/masr
