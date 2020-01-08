import os
import uuid
import numpy as np
import paddle.fluid as fluid
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 上传文件
@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['img']
    # 设置保存路径
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)
    return 'success, save path: ' + img_path

# 预处理图片
def load_image(file):
    img = Image.open(file)
    # 统一图像大小
    img = img.resize((224, 224), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if __name__ == '__main__':
    # 启动服务，并指定端口号
    app.run(port=80)