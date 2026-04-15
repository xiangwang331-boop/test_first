import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

class SlitsRecognizer:
    """
    狭缝（Slits）识别器类
    用于自动识别图像中的各种结构（Slit1-6, Dam2, Crack Stop等）
    """

    def __init__(self, manual_offsets=None, output_folder=None):
        """
        初始化识别器

        Parameters:
        -----------
        manual_offsets : dict, optional
            手动偏移量字典，默认使用预设值
        output_folder : str, optional
            输出文件夹路径
        """
        # 默认手动偏移量
        self.default_offsets = {
            "Slit1": -940, "Slit2": -830, "Slit3": -560, "Slit4": -445,
            "Slit5": -330, "Slit6": -190, "Dam2": 0,
            "Crack Stop": 230, "D-UC1": 430, "D-UC2": 680
        }

        self.manual_offsets = manual_offsets if manual_offsets else self.default_offsets
        self.output_folder = output_folder

        # 子结构搜索配置
        self.sub_structure_configs = [
            {"name": "10", "offset1": "D-UC1", "offset2": "D-UC2", "margin1": 0, "margin2": 0},
            {"name": "9", "offset1": "Crack Stop", "offset2": "D-UC1", "margin1": 0, "margin2": 0},
            {"name": "8", "offset1": "Dam2", "offset2": "Crack Stop", "margin1": 0, "margin2": 0},
            {"name": "7", "offset1": "Slit6", "offset2": "Dam2", "margin1": 0, "margin2": 0},
            {"name": "6", "offset1": "Slit5", "offset2": "Slit6", "margin1": 20, "margin2": -20},
            {"name": "5", "offset1": "Slit4", "offset2": "Slit5", "margin1": 20, "margin2": -20},
            {"name": "4", "offset1": "Slit3", "offset2": "Slit4", "margin1": 20, "margin2": -30},
            {"name": "3", "offset1": "Slit1", "offset2": "Slit2", "margin1": 0, "margin2": 0}
        ]

        # 峰值检测参数
        self.peak_height_threshold = 100
        self.peak_width_threshold = 5

    def standardizing_image_orientation(self, img_bgr):
        """
        标准化图像方向：
        1. 旋转：确保线条是垂直的（纵向）。
        2. 翻转：确保核心结构（Slit/Dam）位于右侧。
        """
        if img_bgr is None:
            return None

        # --- 步骤 1: 旋转检测（横竖判断） ---
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # 取中间区域进行波动测试
        mid_h, mid_w = h // 2, w // 2
        h_slice = np.mean(gray[mid_h - 20:mid_h + 20, :], axis=0)
        v_slice = np.mean(gray[:, mid_w - 20:mid_w + 20], axis=1)

        # 垂直波动大说明是横向线条
        if np.std(v_slice) > np.std(h_slice):
            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
            print("已执行顺时针旋转 90 度（由横变竖）")
            # 旋转后重新获取灰度和尺寸
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

        # --- 步骤 2: 左右翻转检测 ---
        # 原理：核心结构（递增线、Slit等）所在区域的灰度标准差显著高于空白背景区
        probe_w = int(w * 0.5)  # 取左右各 20% 的区域进行探测
        roi_mid_y = h // 2
        roi_data = gray[roi_mid_y - 50:roi_mid_y + 50, :]  # 采样中间 100 行

        left_zone = roi_data[:, :probe_w]
        right_zone = roi_data[:, w - probe_w:]

        left_std = np.std(left_zone)
        right_std = np.std(right_zone)

        # 如果左侧波动大，说明核心结构在左边，执行水平翻转
        if left_std < right_std:
            img_bgr = cv2.flip(img_bgr, 1)
            print(f"检测到核心结构在左侧 (L_Std:{left_std:.2f} > R_Std:{right_std:.2f})，已执行水平翻转。")
        else:
            print(f"核心结构已在右侧 (R_Std:{right_std:.2f} > L_Std:{left_std:.2f})，无需翻转。")

        return img_bgr

    def find_dam2_anchor(self, smooth_intensities):
        """
        识别 Dam2 锚点位置

        Returns:
        --------
        dam2_pos : int or None
            Dam2的位置，如果未找到返回None
        """
        all_p, props = find_peaks(smooth_intensities, height=200, distance=30, width=1)
        if len(all_p) == 0:
            return None
        widths = props.get('widths', props.get('right_ips', 0) - props.get('left_ips', 0))
        dam2_pos = all_p[np.argmax(widths)]
        return dam2_pos

    def find_sub_structure(self, smooth_intensities, dam2_pos, config, gray_image=None, roi_y=None):
        """
        在指定区间内查找子结构
        """
        # 计算搜索区间
        pos1 = dam2_pos + self.manual_offsets[config["offset1"]] + config.get("margin1", 0)
        pos2 = dam2_pos + self.manual_offsets[config["offset2"]] + config.get("margin2", 0)
        search_start = min(pos1, pos2)
        search_end = max(pos1, pos2)

        if search_start < 0 or search_end >= len(smooth_intensities):
            return None

        roi_wave = smooth_intensities[search_start:search_end]
        mid_p, mid_props = find_peaks(roi_wave, height=self.peak_height_threshold,
                                      width=self.peak_width_threshold)

        if len(mid_p) == 0:
            return None

        target_idx = np.argmax(mid_props.get('widths', [0]))
        peak_left = int(mid_props['left_ips'][target_idx] + search_start)
        peak_right = int(mid_props['right_ips'][target_idx] + search_start)

        pos = peak_left + (peak_right - peak_left) // 2
        result = {
            "name": config["name"],
            "center": pos,
            "left": peak_left,
            "right": peak_right,
            "width": peak_right - peak_left
        }

        # ========== 新增：计算矩形框区域的平均灰度值 ==========
        if gray_image is not None and roi_y is not None:
            h, w = gray_image.shape
            # 将位移坐标转换为图像坐标
            left_x = w - 1 - peak_right
            right_x = w - 1 - peak_left

            # 矩形框的Y范围（上下各35像素，与绘制时的参数一致）
            y_start = max(0, roi_y - 35)
            y_end = min(h, roi_y + 35)
            x_start = max(0, left_x)
            x_end = min(w, right_x)

            # 提取矩形框区域并计算平均灰度值
            box_region = gray_image[y_start:y_end, x_start:x_end]
            if box_region.size > 0:
                result["gray_mean"] = np.mean(box_region)
        # ====================================================
        return result
    def get_basic_structures(self, dam2_pos, w):
        """
        获取基本结构的位置信息

        Returns:
        --------
        basic_labels : dict
            基本结构标签字典
        results_to_save : list
            待保存的结果列表
        """
        basic_labels = {}
        results_to_save = []

        for name, offset in self.manual_offsets.items():
            p_pos = dam2_pos + offset
            if p_pos < 0 or p_pos >= w:  # 注意这里用w而不是smooth长度
                continue
            original_x_left = (w - 1) - p_pos

            basic_labels[p_pos] = {
                "name": name,
                "color": "red" if "Slit" in name else "blue",
                "bgr": (0, 0, 255) if "Slit" in name else (255, 0, 0),
                "is_sub": False
            }
            results_to_save.append({
                "Label": name,
                "Distance_From_Right": p_pos,
                "Original_X_Left": original_x_left
            })

        return basic_labels, results_to_save

    def draw_annotations(self, img_bgr, final_labels, roi_y):
        """
        在图像上绘制标注

        Parameters:
        -----------
        img_bgr : np.array
            原始图像
        final_labels : dict
            所有标签信息
        roi_y : int
            ROI区域的Y坐标

        Returns:
        --------
        img_draw : np.array
            标注后的图像
        """
        h, w = img_bgr.shape[:2]
        img_draw = img_bgr.copy()

        for p_pos, info in final_labels.items():
            orig_x = w - 1 - p_pos

            # 子结构：画矩形框
            if info.get("is_sub"):
                x_start_img = int(w - 1 - info["right"])
                x_end_img = int(w - 1 - info["left"])

                cv2.rectangle(img_draw, (x_start_img, roi_y - 350),
                              (x_end_img, roi_y-300), info["bgr"], 2)
                cv2.putText(img_draw, info["name"], (x_start_img, roi_y - 370),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, info["bgr"], 4)
                # ========== 新增：在矩形框下方显示平均灰度值 ==========
                if "gray_mean" in info:
                    text = f"{info['gray_mean']:.1f}"
                    x_center_img=x_start_img+(x_end_img-x_start_img)//2
                    # 获取文字实际尺寸
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 4
                    )

                    # 计算文字左上角坐标（使文字中心对齐到 x_center_img）
                    text_x = x_center_img - text_width // 2
                    text_y = roi_y-200  # 文字底部Y坐标

                    cv2.putText(img_draw, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, info["bgr"], 4)
            else:
                # 基本结构：画线标注
                cv2.line(img_draw, (orig_x, roi_y - 60), (orig_x, roi_y + 60),
                         info["bgr"], 4)
                cv2.putText(img_draw, info["name"], (orig_x - 40, roi_y - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, info["bgr"], 4)

        return img_draw

    def plot_results(self, img_draw, smooth_intensities, final_labels):
        """
        显示结果图表（图像+波形图）
        """
        plt.figure(figsize=(16, 10))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 上图：标注后的图像
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
        plt.title("识别结果：子结构为矩形框，其余为线标注")
        plt.axis('off')

        # 下图：剖面图
        plt.subplot(2, 1, 2)
        plt.plot(smooth_intensities, color='blue', linewidth=1.5, label='亮度曲线')
        plt.axhline(y=200, color='red', linestyle='--', alpha=0.4)

        for p_pos, info in final_labels.items():
            if p_pos < len(smooth_intensities):
                val = smooth_intensities[p_pos]
                plt.plot(p_pos, val, "o", color=info["color"], markersize=6)
                label_text = f"{info['name']}\n({val:.1f})"
                plt.annotate(label_text, xy=(p_pos, val),
                             xytext=(0, 10), textcoords='offset points',
                             ha='center', fontsize=10, color=info["color"], weight='bold')

        plt.xlabel("从右向左的像素位移")
        plt.ylabel("灰度值")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()
        plt.close()

    def recognize(self, image_path, output_path=None, show_plot=True):
        """
        识别图像中的结构

        Parameters:
        -----------
        image_path : str
            输入图像路径
        output_path : str, optional
            输出图像路径，如果不指定则自动生成
        show_plot : bool, default=True
            是否显示图表

        Returns:
        --------
        results_to_save : list
            识别结果列表
        img_width : int
            图像宽度
        """
        # 读取图像
        data = np.fromfile(image_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"无法读取图像: {image_path}")
            return None, 0

        # 标准化方向
        img_bgr = self.standardizing_image_orientation(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_y = h // 2

        # 计算强度曲线
        raw_intensities = np.mean(gray[roi_y - 20:roi_y + 20, :], axis=0)[::-1]
        smooth_intensities = savgol_filter(raw_intensities, 11, 3)

        # 识别 Dam2 锚点
        dam2_pos = self.find_dam2_anchor(smooth_intensities)

        if dam2_pos is None:
            print("未能识别到 Dam2 锚点")
            return None, w

        # 获取基本结构
        basic_labels, results_to_save = self.get_basic_structures(dam2_pos, w)
        final_labels = basic_labels.copy()

        # 识别子结构
        for config in self.sub_structure_configs:
            sub_struct = self.find_sub_structure(smooth_intensities, dam2_pos, config, gray, roi_y)
            if sub_struct:
                sub_info = {
                    "name": sub_struct["name"],
                    "bgr": (0, 255, 0),
                    "color": "green",
                    "style": "box",
                    "is_sub": True,
                    "left": sub_struct["left"],
                    "right": sub_struct["right"]
                }

                # ========== 新增：添加平均灰度值 ==========
                if "gray_mean" in sub_struct:
                    sub_info["gray_mean"] = sub_struct["gray_mean"]
                    print(f"{sub_struct['name']}号结构平均灰度值: {sub_struct['gray_mean']:.2f}")
                # ========================================

                final_labels[sub_struct["center"]] = sub_info
                print(f"{sub_struct['name']}号结构：中心={sub_struct['center']}, "
                      f"起点={sub_struct['left']}, 终点={sub_struct['right']}, "
                      f"宽度={sub_struct['width']}")
        # --- 新增：提取所有子结构的灰度数据 ---
        # --- 1. 提取所有子结构的灰度数据 ---
            # 在 recognize 方法中替换排序和计算逻辑
            # 1. 提取所有子结构并建立映射 (确保包含 3, 4, 5, 6, 7 号结构)
            # 这里的 key 是 config 里的 name，即 "3", "4", "5" 等
        structs = {}
        for p_pos, info in final_labels.items():
            if info.get("is_sub"):
                structs[info["name"]] = info["gray_mean"]

        # 定义计算顺序映射：表格标签 -> (后一个结构ID, 前一个结构ID)
        calc_map = {
            "Slit1-2": ("4", "3"),
            "Slit3-4": ("5", "4"),
            "Slit4-5": ("6", "5"),
            "Slit5-6": ("7", "6"),
            "Slit5-Dam2": ("8", "7"),
            "Dam2-Crack Stop": ("9", "8")
        }

        step_results = []
        for label, (post_id, pre_id) in calc_map.items():
            if post_id in structs and pre_id in structs:
                gray_post = structs[post_id]
                gray_pre = structs[pre_id]
                # 计算差值：后减前
                diff = gray_post - gray_pre

                step_results.append({
                    "label": label,
                    "gray_val": gray_pre,  # 显示当前主体的灰度
                    "diff": diff
                })

        if not step_results:
            return None

        # 2. 根据差值大小分配得分 (A-E)
        # 差值越小（波动越小）得分越高（A），差值越大（波动越大）得分越低（E）
        # 注意：这里按 diff 从小到大排，第一名是 A
        ranked_by_abs = sorted(step_results, key=lambda x: abs(x["diff"]))
        grades = ["A", "B", "C", "D", "E", "F"]
        for i, res in enumerate(ranked_by_abs):
            res["grade"] = grades[i] if i < len(grades) else "F"


        # 绘制标注
        img_draw = self.draw_annotations(img_bgr, final_labels, roi_y)

        # 显示图表
        if show_plot:
            self.plot_results(img_draw, smooth_intensities, final_labels)

        return results_to_save, w,step_results,img_draw


