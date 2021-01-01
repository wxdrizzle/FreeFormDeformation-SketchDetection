# -*- coding: utf-8 -*-

"""
Free Form Deformation (including video for visualizing naive_transformation function) & Edge Detection

Authors: Xin Wang, Xiaoyu Zhou, and Danran Chen
All rights are reserved.
"""

import sys
from moviepy.editor import VideoFileClip
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ffd import compute_warped_img, resize_image
from generate_video import generate_video
from image2sketch import *


class FreeFormDeformationWidget(QWidget):
    """
    widget for free form deformation
    """

    def __init__(self, mesh_range):
        super().__init__()
        # initialize widget parameters
        self.margin = mesh_range * 0.2
        self.mesh_range = mesh_range
        self.mesh_size = 5
        self.mesh_spacing = self.mesh_range / (self.mesh_size + 1.)

        # initialize images
        self.win_img_size = np.int(np.floor(self.mesh_range - 2 * self.mesh_spacing))
        self.win_img = 255 * np.ones((self.win_img_size, self.win_img_size, 3))
        self.original_img_size = self.win_img_size
        self.original_img = self.win_img
        self.ffd_img_size = 128
        self.ffd_input_img = 255 * np.ones((self.ffd_img_size, self.ffd_img_size, 3))
        self.sketch_input_img = self.ffd_input_img

        # initialize meshes
        self.original_mesh = np.ones((2, self.mesh_size + 3, self.mesh_size + 3))
        for i in range(self.mesh_size + 3):
            for j in range(self.mesh_size + 3):
                self.original_mesh[:, i, j] = [i * self.mesh_spacing, j * self.mesh_spacing]
        self.current_mesh = self.original_mesh.copy()

        # initialize flags
        self.are_videos_updated = [False, False, False, False]
        self.is_current_mesh_original = True
        self.is_deformation_activated = False
        self.activated_ctl_point_col_idx = 0
        self.activated_ctl_point_row_idx = 0
        self.speed_option = 3

    def get_img(self, img_path):
        """
        get image for FFD
        """
        self.original_img = np.array(Image.open(img_path).convert("RGB"))
        self.original_img_size = self.original_img.shape[0]
        self.ffd_input_img = resize_image(self.original_img, (self.ffd_img_size, self.ffd_img_size))
        self.win_img = resize_image(self.original_img, (self.win_img_size, self.win_img_size))
        self.sketch_input_img = self.ffd_input_img
        self.reset_img()

    def draw_image(self, qp, mesh, is_ffd_input_win_img=False, iter_num=5, lr=40):
        """
        draw current control point original_mesh and warped image
        Args:
            qp: QPainter
            mesh: current original_mesh
            is_ffd_input_win_img: if win image is being used as input of FFD
            iter_num: number of iterations for computing warped image
            lr: learning rate for computing warped image
        """
        point_pen = QPen(Qt.red, 3)
        line_pen = QPen(Qt.darkBlue, 1, Qt.DotLine)

        if not self.is_current_mesh_original:
            # If current original_mesh is the original original_mesh, there is no need to perform FFD.
            if is_ffd_input_win_img:
                # using win img as the input of FFD
                new_ffd_output = compute_warped_img((self.current_mesh - self.mesh_spacing),
                                                    resize_image(self.original_img, (self.win_img_size, self.win_img_size)),
                                                    self.mesh_spacing,
                                                    iter_num, lr)
            else:
                new_ffd_output = compute_warped_img(
                    (self.current_mesh - self.mesh_spacing) / self.win_img_size * self.ffd_img_size,
                    self.ffd_input_img,
                    self.mesh_spacing / self.win_img_size * self.ffd_img_size,
                    iter_num, lr)
            self.sketch_input_img = np.uint8(new_ffd_output)
            self.win_img = resize_image(new_ffd_output, (self.win_img_size, self.win_img_size))

        img_pil = Image.fromarray(np.uint8(self.win_img))
        img_pix = img_pil.toqpixmap()

        # draw warped image and current original_mesh
        qp.drawPixmap(self.mesh_spacing + self.margin * 0.5, self.mesh_spacing + self.margin * 0.5, img_pix)
        for i in range(self.mesh_size + 2):
            for j in range(self.mesh_size + 2):
                qp.setPen(point_pen)
                qp.drawPoint(mesh[:, i, j][0] + self.margin * 0.5, mesh[:, i, j][1] + self.margin * 0.5)
                if i != self.mesh_size + 1 and j != self.mesh_size + 1:
                    qp.setPen(line_pen)
                    qp.drawLine(mesh[:, i, j][0] + self.margin * 0.5, mesh[:, i, j][1] + self.margin * 0.5,
                                mesh[:, i + 1, j][0] + self.margin * 0.5, mesh[:, i + 1, j][1] + self.margin * 0.5)
                    qp.drawLine(mesh[:, i, j][0] + self.margin * 0.5, mesh[:, i, j][1] + self.margin * 0.5,
                                mesh[:, i, j + 1][0] + self.margin * 0.5, mesh[:, i, j + 1][1] + self.margin * 0.5)
                if i == self.mesh_size + 1 and j != self.mesh_size + 1:
                    qp.setPen(line_pen)
                    qp.drawLine(mesh[:, i, j][0] + self.margin * 0.5, mesh[:, i, j][1] + self.margin * 0.5,
                                mesh[:, i, j + 1][0] + self.margin * 0.5, mesh[:, i, j + 1][1] + self.margin * 0.5)
                if i != self.mesh_size + 1 and j == self.mesh_size + 1:
                    qp.setPen(line_pen)
                    qp.drawLine(mesh[:, i, j][0] + self.margin * 0.5, mesh[:, i, j][1] + self.margin * 0.5,
                                mesh[:, i + 1, j][0] + self.margin * 0.5, mesh[:, i + 1, j][1] + self.margin * 0.5)

    def paintEvent(self, event):
        """
        decide painting options using flags
        """
        qp = QPainter()
        qp.begin(self)

        if self.is_current_mesh_original:
            self.draw_image(qp, self.current_mesh)
            qp.end()
            return

        if self.is_deformation_activated:
            # If speed is chosen to be high, use win image as FFD's input.
            self.draw_image(qp, self.current_mesh, is_ffd_input_win_img=(self.speed_option == 1))
            qp.end()
            return

        # If deformation is not activated, which means the mouse was released,
        # use larger iteration number to get better warped result.
        self.draw_image(qp, self.current_mesh, is_ffd_input_win_img=(self.speed_option < 3), iter_num=40, lr=5)
        qp.end()

    def _neighbor(self, x, y):
        mouse_position = np.array([[[x - self.margin * 0.5]], [[y - self.margin * 0.5]]])
        dist = np.linalg.norm(self.current_mesh - mouse_position, axis=0, ord=2)
        a, b = np.unravel_index(np.argmin(dist), dist.shape)

        return a, b, dist[a, b]

    def mousePressEvent(self, event):
        """
        determine if any control point is activated by mouse cursor; if so, activate deformation
        """
        dist_threshold = 10
        if event.button() != Qt.LeftButton:
            return

        neighbor_info = self._neighbor(event.x(), event.y())
        if neighbor_info[2] <= dist_threshold:
            self.is_deformation_activated = True
            self.activated_ctl_point_col_idx, self.activated_ctl_point_row_idx = neighbor_info[:2]
            self.are_videos_updated = [False, False, False, False]
            self.is_current_mesh_original = False
            self.update()

    def mouseMoveEvent(self, event):
        """
        update control point original_mesh and warped image in real time
        """
        if event.buttons() == Qt.LeftButton:
            if self.is_deformation_activated:
                self.current_mesh[:, self.activated_ctl_point_col_idx, self.activated_ctl_point_row_idx] = \
                    [event.x() - self.margin * 0.5, event.y() - self.margin * 0.5]
                self.update()

    def mouseReleaseEvent(self, event):
        """
        update control point original_mesh and warped image for the last time, and deactivate deformation
        """
        if event.button() != Qt.LeftButton:
            return

        if self.is_deformation_activated:
            self.current_mesh[:, self.activated_ctl_point_col_idx, self.activated_ctl_point_row_idx] = \
                [event.x() - self.margin * 0.5, event.y() - self.margin * 0.5]
            self.is_deformation_activated = False
            self.update()

    def reset_img(self, mesh_size=None):
        """
        reset control point original_mesh and image to their original states
        Args:
            mesh_size: size of control point original_mesh
        """
        if mesh_size is None:
            # need not to change original original_mesh
            self.current_mesh = self.original_mesh.copy()
        else:
            # need to re-initialize meshes
            self.mesh_size = mesh_size
            self.mesh_spacing = self.mesh_range / (self.mesh_size + 1.)
            self.win_img_size = np.int(np.floor(self.mesh_range - 2 * self.mesh_spacing))
            self.original_mesh = np.ones((2, self.mesh_size + 3, self.mesh_size + 3))
            for i in range(self.mesh_size + 3):
                for j in range(self.mesh_size + 3):
                    self.original_mesh[:, i, j] = [i * self.mesh_spacing, j * self.mesh_spacing]
            self.current_mesh = self.original_mesh.copy()

        self.sketch_input_img = self.ffd_input_img
        self.win_img = resize_image(self.original_img, (self.win_img_size, self.win_img_size))

        self.is_current_mesh_original = True
        self.are_videos_updated = [False, False, False, False]
        self.update()


class MainWidget(QWidget):
    """
    main widget for GUI
    """

    def __init__(self):
        super().__init__()
        self.sub_win_width = 350
        main_layout = QGridLayout()

        # module of FFD
        self.ffd_widget = FreeFormDeformationWidget(mesh_range=self.sub_win_width)
        self.ffd_widget.setStyleSheet("QLabel{background:white;}")
        self.ffd_widget.setFixedSize(self.ffd_widget.mesh_range + self.ffd_widget.margin,
                                     self.ffd_widget.mesh_range + self.ffd_widget.margin)

        # module of sketch
        self.sketch_label = QLabel()
        self.sketch_label_size = self.ffd_widget.win_img_size
        self.sketch_label.setFixedSize(self.sketch_label_size, self.sketch_label_size)
        self.sketch_label.setStyleSheet("QLabel{background:white;}")
        self.sketch_img_with_original_size = Image.fromarray(
            255 * np.ones((self.ffd_widget.original_img_size, self.ffd_widget.original_img_size))).convert("L")
        sketch_layout = QHBoxLayout()
        sketch_layout.addWidget(self.sketch_label)
        sketch_layout.setContentsMargins(self.ffd_widget.margin, 0, self.ffd_widget.margin, 0)

        # module of video for naive_transformation visualization
        self.video_label = QLabel()
        self.video_label.setFixedSize(self.sub_win_width, self.sub_win_width)
        self.video_label.setStyleSheet("QLabel{background:white;}")
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.setSpacing(self.ffd_widget.margin)

        # button of video generation
        self.generate_video_btn = QPushButton("Play Video")
        self.generate_video_btn.clicked.connect(self.generate_video)
        self.video_path = ["", "", "", ""]

        # ComboBox for selecting video type
        self.video_selection = QComboBox()
        self.video_selection.addItems(["Square", "Dots", "Vectors", "Vectors with Grid"])

        # button of generating sketch
        self.img_to_sketch_btn = QPushButton("Generate Sketch")
        self.img_to_sketch_btn.clicked.connect(self.img_to_sketch)

        # button of saving sketch
        self.save_sketch_btn = QPushButton("Save Sketch")
        self.save_sketch_btn.clicked.connect(self.save_sketch_trigger_process)

        # ComboBox for selecting edge detection method
        self.method_selection_box = QComboBox()
        self.method_selection_box.addItems(["Sobel", "Canny", "Laplacian", "LoG", "DoG", "XDoG"])
        self.method_selection_box.activated.connect(self.change_arg_text)

        # argments for edge detection methods
        self.arg1 = QLineEdit()
        self.arg2 = QLineEdit()
        self.arg3 = QLineEdit()
        self.arg4 = QLineEdit()
        self.arg5 = QLineEdit()
        self.arg6 = QLineEdit()

        self.arg1.setPlaceholderText("100")
        self.arg2.setPlaceholderText("")
        self.arg3.setPlaceholderText("")
        self.arg4.setPlaceholderText("")
        self.arg5.setPlaceholderText("")
        self.arg6.setPlaceholderText("")

        self._set_opacity(self.arg1, 1)
        self._set_opacity(self.arg2, 0)
        self._set_opacity(self.arg3, 0)
        self._set_opacity(self.arg4, 0)
        self._set_opacity(self.arg5, 0)
        self._set_opacity(self.arg6, 0)

        self.arg1_text = QLabel()
        self.arg2_text = QLabel()
        self.arg3_text = QLabel()
        self.arg4_text = QLabel()
        self.arg5_text = QLabel()
        self.arg6_text = QLabel()

        self.arg1_text.setText("Threshold")
        self.arg1_text.setToolTip("二值化阈值，负数表示不进行阈值化")
        self.arg2_text.setText("")
        self.arg3_text.setText("")
        self.arg4_text.setText("")
        self.arg5_text.setText("")
        self.arg6_text.setText("")

        sub_grid_layout = QGridLayout()
        sub_grid_layout.addWidget(self.save_sketch_btn, 0, 0)
        sub_grid_layout.addWidget(self.method_selection_box, 0, 1)
        btn_grid = QGridLayout()
        btn_grid.setSpacing(10)
        btn_grid.addWidget(self.generate_video_btn, 0, 0)
        btn_grid.addWidget(self.video_selection, 0, 1)
        btn_grid.addWidget(self.img_to_sketch_btn, 1, 0)
        btn_grid.addLayout(sub_grid_layout, 1, 1)

        btn_grid.addWidget(self.arg1, 2, 1)
        btn_grid.addWidget(self.arg2, 3, 1)
        btn_grid.addWidget(self.arg3, 4, 1)
        btn_grid.addWidget(self.arg4, 5, 1)
        btn_grid.addWidget(self.arg5, 6, 1)
        btn_grid.addWidget(self.arg6, 7, 1)
        btn_grid.addWidget(self.arg1_text, 2, 0)
        btn_grid.addWidget(self.arg2_text, 3, 0)
        btn_grid.addWidget(self.arg3_text, 4, 0)
        btn_grid.addWidget(self.arg4_text, 5, 0)
        btn_grid.addWidget(self.arg5_text, 6, 0)
        btn_grid.addWidget(self.arg6_text, 7, 0)

        btn_layout = QHBoxLayout()
        btn_layout.addLayout(btn_grid)
        btn_layout.setContentsMargins(30, 40, 30, 10)
        main_layout.addWidget(self.ffd_widget, 0, 0)
        main_layout.addLayout(video_layout, 0, 1)
        main_layout.addLayout(sketch_layout, 1, 0)
        main_layout.addLayout(btn_layout, 1, 1)

        self.setLayout(main_layout)

    @staticmethod
    def _set_opacity(widget, opacity):
        tmp = QGraphicsOpacityEffect()
        tmp.setOpacity(opacity)
        widget.setGraphicsEffect(tmp)

    def save_sketch_trigger_process(self):
        """
        save sketch
        """
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Sketch', 'DeformedSketch', '*.png')
        if savePath[0] == "":
            return
        self.sketch_img_with_original_size.save(savePath[0])

    def menu_trigger_process(self, q):
        """
        open image or reset original_mesh
        """
        if q.text() == "Open Image":
            imgName, imgType = QFileDialog.getOpenFileName(self, "open image", "", "All Files(*)")
            if imgName == "":
                return
            self.ffd_widget.get_img(imgName)
        elif q.text() == "Reset Mesh":
            self.ffd_widget.reset_img()
            self.sketch_label.setPixmap(
                Image.fromarray(255 * np.ones((self.sketch_label_size, self.sketch_label_size))).convert(
                    "L").toqpixmap())

    def speed_trigger_process(self, q):
        """
        set speed
        """
        self.ffd_widget.reset_img()
        if q.text() == "High":
            self.ffd_widget.speed_option = 3
            self.high.setChecked(True)
            self.medium.setChecked(False)
            self.low.setChecked(False)
        elif q.text() == "Medium":
            self.ffd_widget.speed_option = 2
            self.high.setChecked(False)
            self.medium.setChecked(True)
            self.low.setChecked(False)
        elif q.text() == "Low":
            self.ffd_widget.speed_option = 1
            self.high.setChecked(False)
            self.medium.setChecked(False)
            self.low.setChecked(True)

    def mesh_size_trigger_process(self, q):
        """
        set original_mesh size
        """
        for i in range(8):
            if i + 2 == int(q.text()):
                self.mesh_size_opts[i].setChecked(True)
            else:
                self.mesh_size_opts[i].setChecked(False)
        self.ffd_widget.reset_img(int(q.text()))

        img_pil = Image.fromarray(255 * np.ones((self.ffd_widget.win_img_size, self.ffd_widget.win_img_size))).convert(
            "L")
        img_pix = img_pil.toqpixmap()
        self.sketch_label.setPixmap(img_pix)

    def change_arg_text(self):
        """
        change names of argments based on selected edge detection method
        """
        selected_method = self.method_selection_box.currentText()
        self.arg1.setText("")
        self.arg2.setText("")
        self.arg3.setText("")
        self.arg4.setText("")
        self.arg5.setText("")
        self.arg6.setText("")

        if selected_method == "Sobel":
            self.arg1_text.setText("Threshold")
            self.arg1_text.setToolTip("二值化阈值，负数表示不进行阈值化")
            self.arg2_text.setText("")
            self.arg3_text.setText("")
            self.arg4_text.setText("")
            self.arg5_text.setText("")
            self.arg6_text.setText("")

            self.arg1.setPlaceholderText("100")
            self.arg2.setPlaceholderText("")
            self.arg3.setPlaceholderText("")
            self.arg4.setPlaceholderText("")
            self.arg5.setPlaceholderText("")
            self.arg6.setPlaceholderText("")

            self._set_opacity(self.arg1, 1)
            self._set_opacity(self.arg2, 0)
            self._set_opacity(self.arg3, 0)
            self._set_opacity(self.arg4, 0)
            self._set_opacity(self.arg5, 0)
            self._set_opacity(self.arg6, 0)
        elif selected_method == "Canny":
            self.arg1_text.setText("Sigma")
            self.arg1_text.setToolTip("高斯核函数中的参数sigma")
            self.arg2_text.setText("Kernel Size")
            self.arg2_text.setToolTip("高斯核的大小，长宽相等")
            self.arg3_text.setText("Ratio Min")
            self.arg3_text.setToolTip("低阈值比例")
            self.arg4_text.setText("Ratio Max")
            self.arg4_text.setToolTip("高阈值比例")
            self.arg5_text.setText("")
            self.arg6_text.setText("")

            self.arg1.setPlaceholderText("1.3")
            self.arg2.setPlaceholderText("7")
            self.arg3.setPlaceholderText("0.08")
            self.arg4.setPlaceholderText("0.5")
            self.arg5.setPlaceholderText("")
            self.arg6.setPlaceholderText("")

            self._set_opacity(self.arg1, 1)
            self._set_opacity(self.arg2, 1)
            self._set_opacity(self.arg3, 1)
            self._set_opacity(self.arg4, 1)
            self._set_opacity(self.arg5, 0)
            self._set_opacity(self.arg6, 0)
        elif selected_method == "Laplacian":
            self.arg1_text.setText("Operator")
            self.arg1_text.setToolTip("1：四邻域拉普拉斯算子，2：八邻域拉普拉斯算子")
            self.arg2_text.setText("Threshold")
            self.arg2_text.setToolTip("二值化阈值，负数表示不进行阈值化")
            self.arg3_text.setText("")
            self.arg4_text.setText("")
            self.arg5_text.setText("")
            self.arg6_text.setText("")

            self.arg1.setPlaceholderText("1")
            self.arg2.setPlaceholderText("225")
            self.arg3.setPlaceholderText("")
            self.arg4.setPlaceholderText("")
            self.arg5.setPlaceholderText("")
            self.arg6.setPlaceholderText("")

            self._set_opacity(self.arg1, 1)
            self._set_opacity(self.arg2, 1)
            self._set_opacity(self.arg3, 0)
            self._set_opacity(self.arg4, 0)
            self._set_opacity(self.arg5, 0)
            self._set_opacity(self.arg6, 0)
        elif selected_method == "LoG":
            self.arg1_text.setText("Sigma")
            self.arg1_text.setToolTip("高斯核函数中的参数sigma")
            self.arg2_text.setText("Kernel Size")
            self.arg2_text.setToolTip("高斯核的大小，长宽相等")
            self.arg3_text.setText("Threshold")
            self.arg3_text.setToolTip("二值化阈值，负数表示不进行阈值化")
            self.arg4_text.setText("")
            self.arg5_text.setText("")
            self.arg6_text.setText("")

            self.arg1.setPlaceholderText("1.3")
            self.arg2.setPlaceholderText("7")
            self.arg3.setPlaceholderText("100")
            self.arg4.setPlaceholderText("")
            self.arg5.setPlaceholderText("")
            self.arg6.setPlaceholderText("")

            self._set_opacity(self.arg1, 1)
            self._set_opacity(self.arg2, 1)
            self._set_opacity(self.arg3, 1)
            self._set_opacity(self.arg4, 0)
            self._set_opacity(self.arg5, 0)
            self._set_opacity(self.arg6, 0)
        elif selected_method == "DoG":
            self.arg1_text.setText("K")
            self.arg1_text.setToolTip("两个高斯核函数标准差分别为sigma和k*sigma")
            self.arg2_text.setText("Sigma")
            self.arg2_text.setToolTip("两个高斯核函数标准差分别为sigma和k*sigma")
            self.arg3_text.setText("Kernel Size")
            self.arg3_text.setToolTip("高斯核的大小，长宽相等")
            self.arg4_text.setText("Threshold")
            self.arg4_text.setToolTip("二值化阈值，负数表示不进行阈值化")
            self.arg5_text.setText("")
            self.arg6_text.setText("")

            self.arg1.setPlaceholderText("1.6")
            self.arg2.setPlaceholderText("1")
            self.arg3.setPlaceholderText("5")
            self.arg4.setPlaceholderText("-1")
            self.arg5.setPlaceholderText("")
            self.arg6.setPlaceholderText("")

            self._set_opacity(self.arg1, 1)
            self._set_opacity(self.arg2, 1)
            self._set_opacity(self.arg3, 1)
            self._set_opacity(self.arg4, 1)
            self._set_opacity(self.arg5, 0)
            self._set_opacity(self.arg6, 0)
        elif selected_method == "XDoG":
            self.arg1_text.setText("P")
            self.arg1_text.setToolTip("调整两个高斯滤波器的权重")
            self.arg2_text.setText("K")
            self.arg2_text.setToolTip("两个高斯核函数标准差分别为sigma和k*sigma")
            self.arg3_text.setText("Sigma")
            self.arg3_text.setToolTip("两个高斯核函数标准差分别为sigma和k*sigma")
            self.arg4_text.setText("Kernel Size")
            self.arg4_text.setToolTip("高斯核的大小，长宽相等")
            self.arg5_text.setText("Threshold")
            self.arg5_text.setToolTip("灰度值变为255的阈值")
            self.arg6_text.setText("Phi")
            self.arg6_text.setToolTip("控制输出图像中黑白区域交界处渐变的锐化程度")

            self.arg1.setPlaceholderText("45")
            self.arg2.setPlaceholderText("1.6")
            self.arg3.setPlaceholderText("1")
            self.arg4.setPlaceholderText("5")
            self.arg5.setPlaceholderText("100")
            self.arg6.setPlaceholderText("0.025")

            self._set_opacity(self.arg1, 1)
            self._set_opacity(self.arg2, 1)
            self._set_opacity(self.arg3, 1)
            self._set_opacity(self.arg4, 1)
            self._set_opacity(self.arg5, 1)
            self._set_opacity(self.arg6, 1)

    def img_to_sketch(self):
        """
        generate sketch from warped image using selected edge detection method, and draw result
        """
        img = np.array(Image.fromarray(np.uint8(self.ffd_widget.sketch_input_img)).convert("L"))
        arg_list = [self.arg1.text(), self.arg2.text(), self.arg3.text(), self.arg4.text(), self.arg5.text(),
                    self.arg6.text()]

        selected_method = self.method_selection_box.currentText()
        if selected_method == "Sobel":
            threshold = 100
            arg_value_list = [threshold]
            for i, arg_text in enumerate(arg_list):
                if arg_text and i == 0:
                    arg_value_list[i] = float(arg_text)
            sketch = sobel(img, *arg_value_list)
        elif selected_method == "Canny":
            sigma = 1.3
            kernel_size = 7
            ratiomin = 0.08
            ratiomax = 0.5
            arg_value_list = [sigma, (kernel_size, kernel_size), ratiomin, ratiomax]
            for i, arg_text in enumerate(arg_list):
                if arg_text and i <= 3:
                    if i == 1:
                        arg_value_list[i] = (int(arg_text), int(arg_text))
                    else:
                        arg_value_list[i] = float(arg_text)
            sketch = canny(img, *arg_value_list)
        elif selected_method == "Laplacian":
            operator = 1
            threshold = 225
            arg_value_list = [operator, threshold]
            for i, arg_text in enumerate(arg_list):
                if arg_text and i <= 1:
                    arg_value_list[i] = float(arg_text)
            sketch = laplacian(img, *arg_value_list)
        elif selected_method == "LoG":
            sigma = 1.3
            kernel_size = 7
            threshold = 100
            arg_value_list = [sigma, (kernel_size, kernel_size), threshold]
            for i, arg_text in enumerate(arg_list):
                if arg_text and i <= 2:
                    if i == 1:
                        arg_value_list[i] = (int(arg_text), int(arg_text))
                    else:
                        arg_value_list[i] = float(arg_text)
            sketch = LoG(img, *arg_value_list)
        elif selected_method == "DoG":
            k = 1.6
            sigma = 1
            kernel_size = 5
            threshold = -1
            arg_value_list = [k, sigma, (kernel_size, kernel_size), threshold]
            for i, arg_text in enumerate(arg_list):
                if arg_text and i <= 3:
                    if i == 2:
                        arg_value_list[i] = (int(arg_text), int(arg_text))
                    else:
                        arg_value_list[i] = float(arg_text)
            sketch = DoG(img, *arg_value_list)
        elif selected_method == "XDoG":
            p = 45
            k = 1.6
            sigma = 1
            kernel_size = 5
            threshold = 100
            phi = 0.025
            arg_value_list = [p, k, sigma, (kernel_size, kernel_size), threshold, phi]
            for i, arg_text in enumerate(arg_list):
                if arg_text:
                    if i == 3:
                        arg_value_list[i] = (int(arg_text), int(arg_text))
                    else:
                        arg_value_list[i] = float(arg_text)
            sketch = XDoG(img, *arg_value_list)
        else:
            return

        self.sketch_img_with_original_size = Image.fromarray(np.uint8(sketch)).resize(
            (self.ffd_widget.original_img_size, self.ffd_widget.original_img_size), resample=Image.NEAREST)
        img_pil = Image.fromarray(np.uint8(sketch)).resize((self.sketch_label_size, self.sketch_label_size),
                                                           resample=Image.NEAREST)
        img_pix = img_pil.toqpixmap()
        self.sketch_label.setPixmap(img_pix)

    def generate_video(self):
        """
        generate video for naive_transformation visualization based on selected video type
        """
        video_name = self.video_selection.currentText()

        if "Dots" in video_name:
            if not self.ffd_widget.are_videos_updated[1]:
                self.video_path[1] = generate_video(scene_names=["FFDDots"],
                                                    original_mesh=self.ffd_widget.original_mesh - self.ffd_widget.mesh_spacing,
                                                    current_mesh=self.ffd_widget.current_mesh - self.ffd_widget.mesh_spacing,
                                                    mesh_spacing=self.ffd_widget.mesh_spacing)
                FFD_video = (VideoFileClip(self.video_path[1]))
                FFD_video.write_gif(self.video_path[1][:-3] + "gif")
                self.ffd_widget.are_videos_updated[1] = True

            self.deformation_video = QMovie(self.video_path[1][:-3] + "gif")
            self.video_label.setMovie(self.deformation_video)
            self.deformation_video.start()
            self.deformation_video.setScaledSize(QSize(self.sub_win_width, self.sub_win_width))
        elif "Square" in video_name:
            if not self.ffd_widget.are_videos_updated[0]:
                self.video_path[0] = generate_video(scene_names=["FFDSquare"],
                                                    original_mesh=self.ffd_widget.original_mesh - self.ffd_widget.mesh_spacing,
                                                    current_mesh=self.ffd_widget.current_mesh - self.ffd_widget.mesh_spacing,
                                                    mesh_spacing=self.ffd_widget.mesh_spacing)
                FFD_video = (VideoFileClip(self.video_path[0]))
                FFD_video.write_gif(self.video_path[0][:-3] + "gif")
                self.ffd_widget.are_videos_updated[0] = True

            self.deformation_video = QMovie(self.video_path[0][:-3] + "gif")
            self.video_label.setMovie(self.deformation_video)
            self.deformation_video.start()
            self.deformation_video.setScaledSize(QSize(self.sub_win_width, self.sub_win_width))
        elif "with" in video_name:
            if not self.ffd_widget.are_videos_updated[3]:
                self.video_path[3] = generate_video(scene_names=["FFDVectorsWithGrid"],
                                                    original_mesh=self.ffd_widget.original_mesh - self.ffd_widget.mesh_spacing,
                                                    current_mesh=self.ffd_widget.current_mesh - self.ffd_widget.mesh_spacing,
                                                    mesh_spacing=self.ffd_widget.mesh_spacing)
                FFD_video = (VideoFileClip(self.video_path[3]))
                FFD_video.write_gif(self.video_path[3][:-3] + "gif")
                self.ffd_widget.are_videos_updated[3] = True

            self.deformation_video = QMovie(self.video_path[3][:-3] + "gif")
            self.video_label.setMovie(self.deformation_video)
            self.deformation_video.start()
            self.deformation_video.setScaledSize(QSize(self.sub_win_width, self.sub_win_width))
        elif "Vectors" in video_name:
            if not self.ffd_widget.are_videos_updated[2]:
                self.video_path[2] = generate_video(scene_names=["FFDVectors"],
                                                    original_mesh=self.ffd_widget.original_mesh - self.ffd_widget.mesh_spacing,
                                                    current_mesh=self.ffd_widget.current_mesh - self.ffd_widget.mesh_spacing,
                                                    mesh_spacing=self.ffd_widget.mesh_spacing)
                FFD_video = (VideoFileClip(self.video_path[2]))
                FFD_video.write_gif(self.video_path[2][:-3] + "gif")
                self.ffd_widget.are_videos_updated[2] = True

            self.deformation_video = QMovie(self.video_path[2][:-3] + "gif")
            self.video_label.setMovie(self.deformation_video)
            self.deformation_video.start()
            self.deformation_video.setScaledSize(QSize(self.sub_win_width, self.sub_win_width))


class MainWindow(QMainWindow):
    """
    main window for GUI
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Free Form Deformation & Sketch Detection")

        self.menu_bar = self.menuBar()
        self.status_bar = self.statusBar()
        main_widget = MainWidget()

        # menu bar
        ffd = self.menu_bar.addMenu("FFD")
        open_image = QAction("Open Image", self)
        reset_mesh = QAction("Reset Mesh", self)
        ffd.addAction(open_image)
        ffd.addAction(reset_mesh)
        ffd.triggered[QAction].connect(main_widget.menu_trigger_process)

        # speed control
        speed_ctl = ffd.addMenu("Speed")
        speed_ctl.triggered[QAction].connect(main_widget.speed_trigger_process)
        main_widget.high = speed_ctl.addAction("High")
        main_widget.high.setCheckable(True)
        main_widget.high.setChecked(True)
        main_widget.medium = speed_ctl.addAction("Medium")
        main_widget.medium.setCheckable(True)
        main_widget.low = speed_ctl.addAction("Low")
        main_widget.low.setCheckable(True)

        # original_mesh size control
        mesh_size_ctl = ffd.addMenu("Mesh Size")
        mesh_size_ctl.triggered[QAction].connect(main_widget.mesh_size_trigger_process)
        main_widget.mesh_size_opts = ["", "", "", "", "", "", "", ""]
        for i in range(8):
            main_widget.mesh_size_opts[i] = mesh_size_ctl.addAction(str(i + 2))
            main_widget.mesh_size_opts[i].setCheckable(True)
            if i == 3:
                main_widget.mesh_size_opts[i].setChecked(True)

        self.setCentralWidget(main_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MainWindow()
    demo.show()
    sys.exit(app.exec_())
