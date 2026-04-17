# -*- coding: utf-8 -*-
import cv2
import os
import re
import numpy as np
from paddleocr import PaddleOCR


class OCRScanner:
    """
    针对面板信息（GlassID, Panel, REVCnt）的 OCR 识别类
    已由 EasyOCR 切换为 PaddleOCR
    """

    def __init__(self, gpu=True):
        """
        初始化 PaddleOCR 识别器
        """
        print("正在加载 PaddleOCR 模型...")
        # 使用 paddle_ocr.py 中的初始化参数
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=gpu, show_log=False)

    def _cv_imread(self, file_path):
        """支持中文路径读取"""
        data = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def smart_fix(self, text, mode='general'):
        if not text: return ""
        # 统一转大写并移除空格
        text = text.upper().replace(" ", "")
        garbage_tails = ["L", "I", "1", "[", "|"]

        if mode == 'general':
            # 匹配 PAN + 2位任意字符 + [IL|1]，替换为 PANEL
            found = True
            while found:
                found = False
                for tail in garbage_tails:
                    target = "PANE" + tail
                    if target in text:
                        text = text.replace(target, "PANE")
                        found = True
            # 匹配 GLA + 4位任意字符 + [IL|1]，替换为 GLASSID
            text = re.sub(r"GLA.{4}[\[IL|1]", "GLASSID", text)
        elif mode == 'rev':
            # 匹配 REV + 3位任意字符 + [IL|]，替换为 REVCNT
            text = re.sub(r"REV.{3}[\[IL|]", "REVCNT", text)
            # 针对 REVCNT 的常见错误字符替换
            text = text.replace('O', '0').replace('G', '9').replace('S', '5').replace('B', '8')
            text = text.replace('L', '1').replace('A', '4').replace('Z', '2')

        return text

    def extract_precise(self, t1, t2):
        """
        从清洗后的文本中提取 GlassID, Panel, REVCnt
        """
        t1 = self.smart_fix(t1, 'general')
        t2 = self.smart_fix(t2, 'rev')

        # 提取 GlassID: GLASSID 后面 11 位
        g_id = "N/A"
        g_match = re.search(r"GLA.{4}(.{11})", t1)
        if g_match: g_id = g_match.group(1)

        # 提取 Panel: PANEL 后面 3 位
        p_id = "N/A"
        p_match = re.search(r"PANE(.{3})", t1)
        if p_match: p_id = p_match.group(1)

        # 提取 REVCnt: REVCNT 后面 3 位
        r_cnt = "N/A"
        r_match = re.search(r"REV.{3}(.{3})", t2)
        if r_match: r_cnt = r_match.group(1)

        return g_id, p_id, r_cnt

    def process_image(self, image_path):
        """
        处理单张图像并返回识别结果
        """
        img = self._cv_imread(image_path)
        if img is None:
            return None

        h, w = img.shape[:2]

        # 按照 paddle_ocr.py 的坐标比例进行裁剪
        # 区域 1: 左下角 GlassID/Panel 区域
        crop_img1 = img[h - 110:h - 60, 0:w - 1700]
        # 区域 2: 右下角 REVCnt 区域
        crop_img2 = img[h - 155:h - 100, 1750:w - 400]

        # 使用 PaddleOCR 识别
        res1_list = self.ocr_engine.ocr(crop_img1, cls=True)
        res2_list = self.ocr_engine.ocr(crop_img2, cls=True)

        # 解析 PaddleOCR 的嵌套列表结果
        res1_text = ""
        if res1_list and res1_list[0]:
            res1_text = "".join([line[1][0] for line in res1_list[0]])

        res2_text = ""
        if res2_list and res2_list[0]:
            res2_text = "".join([line[1][0] for line in res2_list[0]])

        # 提取精确字段
        g_id, p_id, r_cnt = self.extract_precise(res1_text, res2_text)

        return {
            "FileName": os.path.basename(image_path),
            "GlassID": g_id,
            "Panel": p_id,
            "REVCnt": r_cnt,
            "Raw": f"{res1_text} | {res2_text}"
        }