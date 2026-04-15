# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtGui import QPixmap, QImage
from demo import Ui_Form
from process_slit import SlitsRecognizer
from paddle_ocr import OCRScanner

class MainWindow(QtWidgets.QWidget, Ui_Form):
    """主窗口类 - 集成狭缝识别功能"""

    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 设置UI

        # 初始化狭缝识别器
        self.slits_recognizer = SlitsRecognizer()
        # 初始化OCR识别器
        self.ocr_scanner = OCRScanner()
        # 图像数据
        self.current_image = None  # 当前图像
        self.annotated_image = None  # 标注后的图像
        self.recognition_results = None  # 识别结果

        # 初始化UI
        self.init_ui()

        # 连接信号槽
        self.connect_signals()

        # 在 MainWindow 类中添加这个方法
    def resizeEvent(self, event):
        """原生事件：窗口大小改变时自动触发"""
        super().resizeEvent(event)  # 调用父类默认逻辑

        # 只要当前有图片，就根据新的窗口大小重新按比例绘制
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.display_image(self.current_image, self.original_image)

        if hasattr(self, 'annotated_image') and self.annotated_image is not None:
            self.display_image(self.annotated_image, self.compare_image)

    def init_ui(self):
        """初始化UI界面"""
        # 修改窗口标题
        self.setWindowTitle("IJP检测系统 - 狭缝识别")

        self.label_path.setMinimumWidth(0)
        self.label_path.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.label_path.setWordWrap(False)
        self.original_image.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.compare_image.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        # 设置标签的初始提示
        self.original_image.setText("原始图像")
        self.original_image.setAlignment(QtCore.Qt.AlignCenter)
        self.compare_image.setText("识别结果")
        self.compare_image.setAlignment(QtCore.Qt.AlignCenter)
        self.res_label.setText("输出数据")


        # 启用HTML渲染并设置文本格式
        self.res_label.setTextFormat(QtCore.Qt.RichText)  # 富文本格式


    def connect_signals(self):
        """连接信号与槽"""
        # 选择图片按钮
        self.open_image_Button.clicked.connect(self.load_image)
        # 识别按钮
        self.detect_Button.clicked.connect(self.recognize_slits)
        self.save_Button.clicked.connect(self.save_annotated_image)

    def load_image(self,file_name=None):
        """加载图像"""
        try:
            # 打开文件对话框
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "选择图像",
                "",
                "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
            )

            if file_name:
                # 更新路径显示
                self.res_label.setText("输出数据")
                self.file_name = file_name
                self.label_path.setText(f"文件路径: {file_name}")
                self.label_path.setToolTip(file_name)

                # 读取图像
                data = np.fromfile(file_name, dtype=np.uint8)
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    QtWidgets.QMessageBox.warning(self, "错误", "无法读取图像文件！")
                    return

                # 保存当前图像
                self.current_image = img_bgr

                # 显示原始图像
                self.display_image(img_bgr, self.original_image)

                # 清空识别结果
                self.annotated_image = None
                self.recognition_results = None
                self.compare_image.clear()
                self.compare_image.setText("识别结果")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")

    def display_image(self, frame, target_label):
        """显示图像，保持比例自适应"""
        if frame is None: return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        step = channel * width
        qImg = QImage(frame_rgb.data, width, height, step, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        # 核心：使用 KeepAspectRatio 保持原始比例
        scaled_pixmap = pixmap.scaled(
            target_label.size(),
            QtCore.Qt.KeepAspectRatio,  # 保持长宽比
            QtCore.Qt.SmoothTransformation  # 平滑抗锯齿
        )
        target_label.setPixmap(scaled_pixmap)

    def recognize_slits(self):
        """执行狭缝识别"""
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先加载图像！")
            return

        try:
            image_path = self.file_name
            # 执行识别
            results, img_width,step_results,img_draw = self.slits_recognizer.recognize(
                image_path,
                output_path=None,
                show_plot=False
            )
            # ========== 1. 获取OCR结果 ==========
            ocr_res = None
            try:
                ocr_res = self.ocr_scanner.process_image(self.file_name)
            except Exception as e:
                print(f"OCR 识别出错: {e}")

            glass_id = ocr_res['GlassID'] if ocr_res else "N/A"
            panel = ocr_res['Panel'] if ocr_res else "N/A"
            rev_cnt = ocr_res['REVCnt'] if ocr_res else "N/A"

            # 溢流结果
            overflow_item = max(step_results, key=lambda x: abs(x["diff"]))["label"] if step_results else "N/A"
            # 【新增】将核心计算结果存为实例属性，供保存 CSV 时直接调用，实现数据与 UI 解耦
            self.step_results = step_results
            self.glass_id = glass_id
            self.panel = panel
            self.rev_cnt = rev_cnt
            self.overflow_item = overflow_item
            # =============================================================
            # # ========== 2. Label显示（新格式：一行表头一行数据）==========
            # display_text = ""
            #
            # # 表头行
            # display_text += f"{'Glass ID':<12} {'Panel':<8} {'REVCnt':<8}"
            # for res in step_results:
            #     display_text += f" {res['label']:<14}"
            # display_text += f" {'溢流结果':<10}\n"
            #
            # # 数据行
            # display_text += f"{glass_id:<12} {panel:<8} {rev_cnt:<8}"
            # for res in step_results:
            #     display_text += f" {res['grade']:<14}"
            # display_text += f" {overflow_item:<10}"


            # ========== 使用HTML表格输出 ==========
            html = """
               <style>
                   .data-table {
                       width: 100%;
                       border-collapse: collapse;
                       font-family: 'Microsoft YaHei', Arial, sans-serif;
                       font-size: 12px;
                   }
                   .data-table th {
                       background-color: #2980b9;
                       color: white;
                       padding: 8px;
                       border: 1px solid #ddd;
                       text-align: center;
                       font-weight: bold;
                   }
                   .data-table td {
                       padding: 6px;
                       border: 1px solid #ddd;
                       text-align: center;
                   }
                   .data-table tr:hover {
                       background-color: #f5f5f5;
                   }
                   
               </style>
               <table class="data-table">
                   <tr>
                       <th>Glass ID</th>
                       <th>Panel</th>
                       <th>REVCnt</th>
               """

            # 动态添加结构名称列
            for res in step_results:
                html += f'<th>{res["label"]}</th>'
            html += '<th>溢流结果</th></tr><tr>'

            # 数据行
            html += f'<td>{glass_id}</td>'
            html += f'<td>{panel}</td>'
            html += f'<td>{rev_cnt}</td>'

            for res in step_results:
                html += f'<td>{res["grade"]}</td>'

            # 溢流结果高亮显示
            html += f'<td>{overflow_item}</td>'
            html += '</tr></table>'

            # 设置标签支持HTML并填满
            self.res_label.setText(html)

            # 输出到界面
            # self.res_label.setText(display_text)

            # ========== 3. 控制台输出（包含详细信息：灰度值、差值波动）==========
            print("\n" + "=" * 100)
            print(f"Glass ID: {glass_id}    Panel: {panel}    REVCnt: {rev_cnt}")
            print("=" * 100)

            if step_results:
                # 表头
                header = "| 结构名称     "
                gray_row = "| 灰度值       "
                diff_row = "| 差值波动     "
                grade_row = "| 得分         "

                for res in step_results:
                    header += f"| {res['label']:^12} "
                    gray_row += f"| {res['gray_val']:^12.2f} "
                    diff_row += f"| {res['diff']:^12.2f} "
                    grade_row += f"| {res['grade']:^12} "

                print(header + "|")
                print("-" * (len(header) + 5))
                print(gray_row + "|")
                print(diff_row + "|")
                print(grade_row + "|")
                print("-" * (len(header) + 5))
                print(f"溢流结果(波动最大): {overflow_item}")

            print("=" * 100 + "\n")
            if results and len(results) > 0:
                # 保存识别结果
                self.recognition_results = results

                self.annotated_image = img_draw
                if self.annotated_image is not None:
                # 显示标注结果
                    self.display_image(self.annotated_image, self.compare_image)

                    QtWidgets.QMessageBox.information(
                        self,
                        "识别完成",
                        f"成功识别到 {len(results)} 个结构！"
                    )
                else:
                    QtWidgets.QMessageBox.warning(self, "警告", "无法生成标注图像。")
            else:
                QtWidgets.QMessageBox.warning(self, "警告", "未能识别到任何结构，请检查图像质量。")


        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"狭缝识别失败: {str(e)}")

    def save_annotated_image(self):
        """保存标注图像和识别结果"""
        if self.annotated_image is None and self.recognition_results is None:
            QtWidgets.QMessageBox.warning(self, "警告", "没有可保存的内容！请先识别图像。")
            return

        # ========== 1. 保存标注图片 ==========
        if self.annotated_image is not None:
            reply = QtWidgets.QMessageBox.question(
                self,
                "保存标注图片",
                "是否保存标注图片？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )

            if reply == QtWidgets.QMessageBox.Yes:
                # 生成默认文件名
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_name = f"slits_annotation_{timestamp}.jpg"

                # 弹出保存对话框
                file_name, _ = QFileDialog.getSaveFileName(
                    self,
                    "保存标注图像",
                    default_name,
                    "JPEG Files (*.jpg);;PNG Files (*.png);;All Files (*)"
                )

                if file_name:
                    cv2.imwrite(file_name, self.annotated_image)
                    QtWidgets.QMessageBox.information(self, "保存成功", f"标注图像已保存至:\n{file_name}")
                    print(f"标注图像已保存: {file_name}")

        # ========== 2. 保存CSV结果 ==========
        if self.recognition_results is not None:
            reply = QtWidgets.QMessageBox.question(
                self,
                "保存CSV结果",
                "是否保存CSV结果？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )

            if reply == QtWidgets.QMessageBox.Yes:
                # 生成默认文件名
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_name = f"slits_results_{timestamp}.csv"

                # 弹出保存对话框
                file_name, _ = QFileDialog.getSaveFileName(
                    self,
                    "保存CSV结果",
                    default_name,
                    "CSV Files (*.csv);;All Files (*)"
                )

                if file_name:
                    try:
                        # === 【修改】直接读取实例属性，不再从 UI 解析文本 ===
                        struct_names = [res["label"] for res in getattr(self, 'step_results', [])]
                        grades = [res["grade"] for res in getattr(self, 'step_results', [])]

                        glass_id = getattr(self, 'glass_id', 'N/A')
                        panel = getattr(self, 'panel', 'N/A')
                        rev_cnt = getattr(self, 'rev_cnt', 'N/A')
                        overflow_result = getattr(self, 'overflow_item', 'N/A')

                        # 构建表头和数据
                        headers = ['Glass ID', 'Panel', 'REVCnt'] + struct_names + ['溢流结果']
                        datas = [glass_id, panel, rev_cnt] + grades + [overflow_result]

                        # 写入CSV
                        import csv
                        with open(file_name, 'w', newline='', encoding='utf-8-sig') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(headers)
                            writer.writerow(datas)

                        QtWidgets.QMessageBox.information(
                            self,
                            "保存成功",
                            f"CSV结果已保存至:\n{file_name}"
                        )
                        print(f"识别结果已保存至: {file_name}")

                    except Exception as e:
                        QtWidgets.QMessageBox.critical(self, "错误", f"保存CSV失败: {str(e)}")

    def save_image(self):
        """保存标注图像"""
        if self.annotated_image is None:
            QtWidgets.QMessageBox.warning(self, "警告", "没有标注图像可保存！")
            return

        # 生成默认文件名
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"slits_annotation_{timestamp}.jpg"

        # 弹出保存对话框
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "保存标注图像",
            default_name,
            "JPEG Files (*.jpg);;PNG Files (*.png);;All Files (*)"
        )

        if file_name:
            cv2.imwrite(file_name, self.annotated_image)
            QtWidgets.QMessageBox.information(self, "保存成功", f"标注图像已保存至:\n{file_name}")
            print(f"标注图像已保存: {file_name}")

    def save_csv(self):
        """保存识别结果到CSV文件"""
        if self.recognition_results is None:
            QtWidgets.QMessageBox.warning(self, "警告", "没有识别结果可保存！")
            return

        # 生成默认文件名
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"slits_results_{timestamp}.csv"

        # 弹出保存对话框
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "保存CSV结果",
            default_name,
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_name:
            try:
                # 获取属性数据
                struct_names = [res["label"] for res in getattr(self, 'step_results', [])]
                grades = [res["grade"] for res in getattr(self, 'step_results', [])]

                glass_id = getattr(self, 'glass_id', 'N/A')
                panel = getattr(self, 'panel', 'N/A')
                rev_cnt = getattr(self, 'rev_cnt', 'N/A')
                overflow_result = getattr(self, 'overflow_item', 'N/A')

                # 写入CSV
                import csv
                with open(file_name, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)

                    # 写入标题行
                    writer.writerow(['Glass ID', 'Panel', 'REVCnt'])
                    writer.writerow([glass_id, panel, rev_cnt])
                    writer.writerow([])  # 空行

                    # 写入结构名称行
                    writer.writerow(['结构名称'] + struct_names)
                    # 写入得分行
                    writer.writerow(['得分'] + grades)
                    writer.writerow([])  # 空行

                    # 写入溢流结果
                    writer.writerow(['溢流结果', overflow_result])

                QtWidgets.QMessageBox.information(
                    self,
                    "保存成功",
                    f"CSV结果已保存至:\n{file_name}"
                )
                print(f"识别结果已保存至: {file_name}")

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def print_results_to_console(self, results):
        """在控制台打印识别结果"""
        print("\n" + "=" * 50)
        print("狭缝识别结果")
        print("=" * 50)

        for item in results:
            print(f"\n结构名称: {item['Label']}")
            print(f"  距右侧距离: {item['Distance_From_Right']} 像素")
            print(f"  左侧X坐标: {item['Original_X_Left']:.1f} 像素")

        print(f"\n共识别到 {len(results)} 个结构")
        print("=" * 50 + "\n")

    def get_recognition_results(self):
        """
        获取识别结果的接口
        返回识别结果列表
        """
        return self.recognition_results

    def get_annotated_image(self):
        """
        获取标注图像的接口
        返回标注后的图像
        """
        return self.annotated_image

    def get_current_image(self):
        """
        获取当前加载的图像
        返回原始图像
        """
        return self.current_image


class SlitsRecognitionApp:
    """独立的狭缝识别应用类（用于批量处理）"""

    def __init__(self):
        self.recognizer = SlitsRecognizer()

    def process_single_image(self, image_path, output_path=None):
        """处理单张图像"""
        return self.recognizer.recognize(image_path, output_path, show_plot=True)


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 创建主窗口
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())



if __name__ == "__main__":
    # GUI模式（默认）
    main()
    print("验证Git")
