**LabVIEW 2017 + ONNX Runtime** (x64-1.24.3)
+ YOLOv8n for for coin detection
+ custom neural network for predicting rotation angle
+ EfficientNet-B4 for for coin classification

Tested:
- Ubuntu 24.04lts - 64bit
- Windows 10 - 64bit

Ubuntu example (Test YOLO):
![FP_Linux](https://github.com/IvanLisRus/LabVIEW-ONNX_Runtime/blob/main/img_LabVIEW/FP_lin.png)
![BD_Linux](https://github.com/IvanLisRus/LabVIEW-ONNX_Runtime/blob/main/img_LabVIEW/BD_lin.png)

Windows example (Test EfficientNet-B4):
![FP_Linux](https://github.com/IvanLisRus/LabVIEW-ONNX_Runtime/blob/main/img_LabVIEW/FP_win.png)
![BD_Linux](https://github.com/IvanLisRus/LabVIEW-ONNX_Runtime/blob/main/img_LabVIEW/BD_win.png)

**Логика работы следующая:**

**1 этап** детекция (ИНС YOLOv8n):
- преобразование изображения [1024, 1024] в тензор [1, 3, 1024, 1024] и подача его на вход ИНС
- преобразование полученных от ИНС данных в читаемый вид, фильтрация по порогу и NMS
- отображение результата детектирования

**2 этап** компенсация угла поворота (ИНС самодельная):
- сегментация изображения (по результатам 1 этапа)
- изменение размера изображения [х.з] -> [512, 512]
- преобразование изображения [512, 512] в тензор [1, 3, 512, 512] и подача его на вход ИНС
- преобразование полученных от ИНС данных [sin, cos] в угол и его компенсация
- отображение результата (сегментация, масштабирование, поворот)

**3 этап** идентификация (ИНС EfficientNet-B4):
- изменение размера [512, 512] -> [380, 380]
- преобразование изображения [380, 380] в тензор [1, 3, 380, 380] и подача его на вход ИНС
- преобразование полученных от ИНС данных в вероятности для каждого из двух классов [Avers, Revers]
- отображение результата


