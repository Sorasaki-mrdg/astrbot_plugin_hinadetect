import os

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
import numpy as np
import onnxruntime as ort
import cv2

from astrbot.core.message.components import Plain, Reply
from astrbot.core.star.filter.event_message_type import EventMessageType

device = 'cpu'
global ort_session


def load_model():
    # 加载 ONNX 模型
    global ort_session
    onnx_model_path = './data/plugins/astrbot_plugin_nailongdetect/nailong01.onnx'
    ort_session = ort.InferenceSession(onnx_model_path)


# 图像预处理（使用 OpenCV）
def preprocess_image(image_path):
    # 使用 OpenCV 读取图像并转换为 RGB 格式
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 读取的是 BGR，转换为 RGB
    image = cv2.resize(image, (224, 224))  # 根据模型要求调整图像大小
    image = image.astype(np.float32) / 255.0  # 将像素值归一化到 [0, 1] 范围
    image = np.transpose(image, (2, 0, 1))  # 转换为 (C, H, W) 格式
    image = np.expand_dims(image, axis=0)  # 增加 batch 维度
    return image


# 处理用户输入的图像路径并进行推理
def predict_image(image_path):
    # 预处理图像
    image = preprocess_image(image_path)

    # 获取模型输入名称
    input_name = ort_session.get_inputs()[0].name

    # 推理：获取预测结果
    ort_inputs = {input_name: image}
    ort_outs = ort_session.run(None, ort_inputs)

    # 使用 argmax 获取最大值所在的位置（即预测的类别）
    pred = np.argmax(ort_outs[0], axis=1)
    return pred.item() == 1


@register("nailondDetect", "orchidsziyou", "检测发送的图片是不是奶龙", "1.0.0")
class MyPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        load_model()

    @filter.event_message_type(EventMessageType.ALL)
    async def detect_nailong(self, event: AstrMessageEvent):
        # obj=event.message_obj
        message_chain = event.get_messages()
        if len(message_chain) == 1:
            # print(message_chain[0])
            if message_chain[0].type == "Image":
                image_filename = message_chain[0].file
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import \
                    AiocqhttpMessageEvent
                assert isinstance(event, AiocqhttpMessageEvent)
                client = event.bot
                payloads2 = {
                    "file_id": image_filename
                }
                response = await client.api.call_action('get_image', **payloads2)
                # print(response)
                image_path = response['file']
                str_path= str(image_path)
                if str_path.endswith('.gif'):
                    return
                if "Emoji" in str_path:
                    return
                if os.path.exists(image_path):
                    result = predict_image(image_path)
                    if result:
                        message_chain = [
                            Reply(id=event.message_obj.message_id),
                            Plain("不准发奶龙！")
                        ]
                        yield event.chain_result(message_chain)
                    else:
                        pass
                else:
                    pass
