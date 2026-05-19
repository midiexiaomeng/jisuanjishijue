#!/usr/bin/env python3
"""
测试GUI启动
"""

import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer

def test_gui_launch():
    """测试GUI启动"""
    print("测试GUI启动...")
    
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置应用程序信息
    app.setApplicationName("水下目标检测系统测试")
    app.setApplicationVersion("1.0")
    
    # 显示测试消息
    msg_box = QMessageBox()
    msg_box.setWindowTitle("GUI测试")
    msg_box.setText("GUI启动测试成功！")
    msg_box.setInformativeText("水下目标检测系统GUI已成功启动。")
    msg_box.setIcon(QMessageBox.Information)
    
    # 设置定时器，3秒后关闭消息框并退出
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(msg_box.accept)
    timer.start(3000)
    
    # 显示消息框
    msg_box.exec_()
    
    print("GUI启动测试完成！")
    return 0

if __name__ == "__main__":
    sys.exit(test_gui_launch())
