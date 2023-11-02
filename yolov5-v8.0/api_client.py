#!/usr/bin/env python
# coding=utf-8

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRegExp, QFileInfo, QUrl
from PyQt5.QtGui import QDoubleValidator, QDesktopServices
from contextlib import contextmanager
import re
import time
import sys
import requests
import json
import os


class MultiInputDialog(QDialog):
    class RowInserter:
        def __init__(self, grid_layout):
            self.row = 0
            self.layout = grid_layout

        def __call__(self, label, widget):
            label = QLabel(label)
            label.setAlignment(Qt.AlignRight)
            self.layout.addWidget(label, self.row, 0)
            if isinstance(widget, QWidget):
                self.layout.addWidget(widget, self.row, 1)
            elif isinstance(widget, QLayout):
                self.layout.addLayout(widget, self.row, 1)
            self.row += 1

    def __init__(self, ):
        super(MultiInputDialog, self).__init__()

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.layout = QGridLayout()
        layout.addLayout(self.layout)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 5)

        layout.addItem(QSpacerItem(0, 20, QSizePolicy.Expanding))

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        buttonBox = QDialogButtonBox(QBtn)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

    def row_inserter(self):
        return self.RowInserter(self.layout)


class IpAddressDialog(MultiInputDialog):
    def __init__(self, base_ip, base_port, pan_tilt_ip, pan_tilt_port):
        super(IpAddressDialog, self).__init__()

        self.setWindowTitle('Input Address')

        insert_row = self.row_inserter()

        self.base_ip = QLineEdit()
        self.base_ip.setText(base_ip)
        insert_row('base IP:', self.base_ip)

        self.base_port = QSpinBox()
        self.base_port.setMaximum(65535)
        self.base_port.setMinimum(1)
        self.base_port.setValue(base_port)
        insert_row('base port:', self.base_port)

        self.pan_tilt_ip = QLineEdit()
        self.pan_tilt_ip.setText(pan_tilt_ip)
        insert_row('pan-tilt IP:', self.pan_tilt_ip)

        self.pan_tilt_port = QSpinBox()
        self.pan_tilt_port.setMaximum(65535)
        self.pan_tilt_port.setMinimum(1)
        self.pan_tilt_port.setValue(pan_tilt_port)
        insert_row('pan-tilt port:', self.pan_tilt_port)


class MoveDialog(MultiInputDialog):
    def __init__(self, open_doors):
        super(MoveDialog, self).__init__()

        self.setWindowTitle('Move to location')

        insert_row = self.row_inserter()

        self.goal = QSpinBox()
        self.goal.setMaximum(65535)
        self.goal.setMinimum(-1)
        self.goal.setValue(0)
        insert_row('goal:', self.goal)

        self.open_doors = QLineEdit(str(open_doors))
        insert_row('open doors:', self.open_doors)


class CheckBoxDialog(MultiInputDialog):
    def __init__(self, title, name, default_state=False):
        super(CheckBoxDialog, self).__init__()

        self.setWindowTitle(title)

        insert_row = self.row_inserter()

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(default_state)
        insert_row(name, self.checkbox)


class CtrlPanTiltDialog(MultiInputDialog):
    def __init__(self):
        super(CtrlPanTiltDialog, self).__init__()

        self.setWindowTitle('Control Pan-Tilt')

        insert_row = self.row_inserter()

        qval = QDoubleValidator(-180, 180, 6)

        self.pan = QLineEdit('0')
        self.pan.setValidator(qval)
        insert_row('pan', self.pan)

        self.tilt = QLineEdit('0')
        self.tilt.setValidator(qval)
        insert_row('tilt', self.tilt)


@contextmanager
def new_hlayout(parent_layout, left_aligned=True):
    hlayout = QHBoxLayout()
    parent_layout.addLayout(hlayout)

    yield hlayout

    if left_aligned:
        spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        hlayout.addSpacerItem(spacer)


class ApiClient(QWidget):
    text_received = pyqtSignal(str, name='text_received')
    html_received = pyqtSignal(str, name='html_received')

    def __init__(self, base_ip, base_port, pan_tilt_ip, pan_tilt_port):
        super(ApiClient, self).__init__()

        self.base_ip = base_ip
        self.base_port = base_port
        self.pan_tilt_ip = pan_tilt_ip
        self.pan_tilt_port = pan_tilt_port
        self.base_url = 'http://{}:{}'.format(self.base_ip, self.base_port)
        self.pan_tilt_url = 'http://{}:{}'.format(self.pan_tilt_ip, self.pan_tilt_port)

        vlayout = QVBoxLayout(self)

        self.btn_addr = QPushButton('address', self)
        self.btn_addr.clicked.connect(self._input_address)
        vlayout.addWidget(self.btn_addr)

        self.label_address = QLabel('    [base]    {}\n[pan-tilt]    {}'.format(self.base_url, self.pan_tilt_url), self)
        vlayout.addWidget(self.label_address)

        self.view_response = QTextBrowser(self)
        self.view_response.setReadOnly(True)
        self.view_response.setOpenLinks(False)
        self.view_response.setOpenExternalLinks(False)
        self.view_response.anchorClicked.connect(self._jump_link)

        self.clear_response = QPushButton('clear', self)
        self.clear_response.clicked.connect(self.view_response.clear)

        self.text_received.connect(self.view_response.setPlainText)
        self.html_received.connect(self.view_response.setHtml)

        self.splitter = QSplitter(Qt.Vertical)

        vlayout.addWidget(self.btn_addr)
        vlayout.addWidget(self.label_address)
        vlayout.addWidget(self.clear_response)
        vlayout.addWidget(self.splitter)

        self.splitter.addWidget(self.view_response)
        btn_frame = QFrame()
        self.splitter.addWidget(btn_frame)
        self.splitter.setStretchFactor(0, 10)
        self.splitter.setStretchFactor(1, 1)

        self.vlayout = QVBoxLayout(btn_frame)
        btn_frame.setLayout(self.vlayout)
        self.vlayout.setAlignment(Qt.AlignTop)

    def _jump_link(self, link):
        path = link.path()
        if os.path.isdir(path):
            QDesktopServices.openUrl(link)
        elif os.path.isfile(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _input_address(self):
        dlg = IpAddressDialog(self.base_ip, self.base_port, self.pan_tilt_ip, self.pan_tilt_port)
        if dlg.exec_():
            base_ip = dlg.base_ip.text()
            base_port = int(dlg.base_port.text())
            pan_tilt_ip = dlg.pan_tilt_ip.text()
            pan_tilt_port = int(dlg.pan_tilt_port.text())
            if re.match('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$', base_ip) and \
                    re.match('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$', pan_tilt_ip):
                self.base_ip = base_ip
                self.base_port = base_port
                self.pan_tilt_ip = pan_tilt_ip
                self.pan_tilt_port = pan_tilt_port
                self.base_url = 'http://{}:{}'.format(self.base_ip, self.base_port)
                self.pan_tilt_url = 'http://{}:{}'.format(self.pan_tilt_ip, self.pan_tilt_port)

                self.label_address.setText('base: {}\npan-tilt: {}'.format(self.base_url, self.pan_tilt_url))
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText('Error')
                msg.setInformativeText('Wrong format of address')
                msg.setWindowTitle('Error')
                msg.exec_()

    def _update_view_resonse(self, txt, html=False):
        t = time.time()
        t_str = str(time.strftime('[%H:%M:%S-{}]'.format(t), time.localtime(t)))
        if html:
            self.html_received.emit('<html>{}<br/>{}</html>'.format(t_str, txt))
        else:
            self.text_received.emit(t_str + '\n' + txt)

    def api_get(self, api, base_url=True, params=None):
        try:
            ret = requests.get((self.base_url if base_url else self.pan_tilt_url) + api, params=params,
                               headers={
                                   'content-length': '0',
                                   'content-type': 'application/json',
                               })
            jdata = ret.json()
            self._update_view_resonse('{} {}\n{}'.format(
                ret.status_code, ret.reason,
                json.dumps(jdata, sort_keys=False, indent=4)))
        except requests.exceptions.RequestException as e:
            self._update_view_resonse(str(e))

    def api_get_file(self, api, output, base_url=True, params=None):
        try:
            ret = requests.get((self.base_url if base_url else self.pan_tilt_url) + api, params=params,
                               headers={
                                   'content-length': '0',
                               },
                               stream=True)
            with open(output, 'wb') as ofile:
                for chuck in ret.iter_content(chunk_size=None):
                    ofile.write(chuck)
            self._update_view_resonse('{} {}<br/>{}'.format(
                ret.status_code, ret.reason,
                'file saved at: <a href="file://{1}/{0}">{0}</a>'.format(output, os.getcwd())),
                html=True)
        except requests.exceptions.RequestException as e:
            self._update_view_resonse(str(e))

    def api_post(self, api, base_url=True, jdata=None):
        if jdata is None:
            jdata = {}
        try:
            ret = requests.post((self.base_url if base_url else self.pan_tilt_url) + api, json=jdata)

            try:
                jdata = ret.json()
                self._update_view_resonse('{} {}\n{}'.format(
                    ret.status_code, ret.reason,
                    json.dumps(jdata, sort_keys=False, indent=4)))
            except ValueError:
                self._update_view_resonse('{} {}'.format(
                    ret.status_code, ret.reason))
        except requests.exceptions.RequestException as e:
            self._update_view_resonse(str(e))


class Gui(ApiClient):
    def __init__(self):
        super(Gui, self).__init__(base_ip='127.0.0.1', base_port=35181,
                                  pan_tilt_ip='127.0.0.1', pan_tilt_port=35281)

        def add_btn(_label, _callable):
            btn = QPushButton(_label)
            btn.clicked.connect(_callable)
            hlayout.addWidget(btn)

        def add_split():
            with new_hlayout(self.vlayout, left_aligned=False) as hlayout:
                sep = QFrame()
                sep.setFrameStyle(QFrame.HLine | QFrame.Sunken)
                sep.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                hlayout.addWidget(sep)

        with new_hlayout(self.vlayout) as hlayout:
            add_btn('status', lambda: self.api_get('/beepatrol/status', base_url=True))
            add_btn('pan-tilt', lambda: self.api_get('/beepatrol/pan_tilt', base_url=False))

        with new_hlayout(self.vlayout) as hlayout:
            add_btn('move', self._input_move)
            add_btn('stop', lambda: self.api_post('/beepatrol/stop', base_url=True))

        with new_hlayout(self.vlayout) as hlayout:
            add_btn('ctrl lifter', self._input_ctrl_lifter)
            add_btn('ctrl pan-tilt', self._input_ctrl_pan_tilt)
            add_btn('set pan-tilt posture', self._input_set_pan_tilt_posture)
            add_btn('switch fill light', self._input_switch_fill_light)

        with new_hlayout(self.vlayout) as hlayout:
            add_btn('rgb shot', lambda: self.api_get_file('/beepatrol/rgb_shot',
                                                          output='rgb.png',
                                                          base_url=False))
            add_btn('heatmap shot', self._input_heatmap_shot)

        add_split()

        with new_hlayout(self.vlayout) as hlayout:
            add_btn('start taking over', lambda: self.api_post('/beepatrol/start_taking_over', base_url=True))
            add_btn('stop taking over', self._input_stop_taking_over)
            add_btn('clear error', lambda: self.api_post('/beepatrol/clear_error', base_url=True))
            add_btn('enable base', lambda: self.api_post('/beepatrol/enable_base', base_url=True, jdata=dict(enable=True)))
            add_btn('disable base', lambda: self.api_post('/beepatrol/enable_base', base_url=True, jdata=dict(enable=False)))
        
        with new_hlayout(self.vlayout) as hlayout:
            add_btn('open door', lambda: self.api_post('/v1/ac/door', base_url=True, jdata=dict(
                method='open',
                param=dict(
                    moid=0,
                    name='C2-4F-冷通道01-东-门禁'
                )
            )))

        self._last_open_doors = []

    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Q, Qt.Key_W] and \
                QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.close()

    def _input_move(self):
        dlg = MoveDialog(self._last_open_doors)
        if dlg.exec_():
            open_doors = eval(dlg.open_doors.text())
            if type(open_doors) not in [tuple, list]:
                open_doors = [open_doors, ]
            self._last_open_doors = map(lambda a: int(a), open_doors)
            self.api_post('/beepatrol/move',
                          jdata=dict(
                              goal=int(dlg.goal.text()),
                              open_doors=self._last_open_doors,
                          ),
                          base_url=True)

    def _input_stop_taking_over(self):
        dlg = CheckBoxDialog('Stop Taking Over', 'manual docked')
        if dlg.exec_():
            self.api_post('/beepatrol/stop_taking_over',
                          jdata=dict(manual_docked=dlg.checkbox.isChecked()),
                          base_url=True)

    def _input_ctrl_lifter(self):
        height, ok = QInputDialog().getDouble(self, 'Control Lifter', '', 0, 0.1, 2.2, 3)
        if ok:
            self.api_post('/beepatrol/ctrl_lifter',
                          jdata=dict(height=height),
                          base_url=True)

    def _input_ctrl_pan_tilt(self):
        dlg = CtrlPanTiltDialog()
        if dlg.exec_():
            self.api_post('/beepatrol/ctrl_pan_tilt',
                          jdata=dict(pan=float(dlg.pan.text()), tilt=float(dlg.tilt.text())),
                          base_url=False)

    def _input_set_pan_tilt_posture(self):
        item, ok = QInputDialog().getItem(self, 'Set Pan-Tilt Posture', '',
                                          ['init', 'left', 'right'], 0, False)
        if ok:
            self.api_post('/beepatrol/set_pan_tilt_posture',
                          jdata=dict(preset=item),
                          base_url=False)

    def _input_heatmap_shot(self):
        dlg = CheckBoxDialog('Heatmap Shot', 'image')
        if dlg.exec_():
            save_as_image = dlg.checkbox.isChecked()
            self.api_get_file('/beepatrol/heatmap_shot',
                              output='heatmap.png' if save_as_image else 'heatmap.csv',
                              params=dict(image=1 if save_as_image else 0),
                              base_url=False)

    def _input_switch_fill_light(self):
        dlg = CheckBoxDialog('Switch Fill Light', 'turn on')
        if dlg.exec_():
            self.api_post('/beepatrol/switch_fill_light',
                          jdata=dict(turn_on=dlg.checkbox.isChecked()),
                          base_url=False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    wnd = Gui()
    wnd.setWindowTitle('API client')
    wnd.setMinimumWidth(720)
    wnd.show()

    sys.exit(app.exec_())
