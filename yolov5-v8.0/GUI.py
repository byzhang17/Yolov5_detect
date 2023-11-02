from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import detectFlow
import cv2
import threading

PATH = Path(__file__).parent
'''
cap = cv2.VideoCapture(0)  # 创建摄像头对象
def tkImage(image_width, image_height):
    ref, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 摄像头翻转
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    pilImage = Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage
'''


class Cleaner(ttk.Frame):

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.pack(fill=BOTH, expand=YES)
        # self.after(100, self.refresh_data)

        # application images

        self.images = [
            ttk.PhotoImage(
                name='logo',
                file=PATH / 'icon' / 'robot.png'),
            ttk.PhotoImage(
                name='cleaner',
                file=PATH / 'icon' / 'root.png'),
            ttk.PhotoImage(
                name='registry',
                file=PATH / 'icon' / 'inspect.png'),
            ttk.PhotoImage(
                name='tools',
                file=PATH / 'icon' / 'icon_map.png'),
            ttk.PhotoImage(
                name='options',
                file=PATH / 'icon' / 'user.png'),
            ttk.PhotoImage(
                name='privacy',
                file=PATH / 'icon' / 'init_pic.png'),
            ttk.PhotoImage(
                name='junk',
                file=PATH / 'icon' / 'map.png'),
            ttk.PhotoImage(
                name='protect',
                file=PATH / 'icon' / 'setting.png')
        ]

        # header
        hdr_frame = ttk.Frame(self, padding=20, bootstyle=SECONDARY)
        hdr_frame.grid(row=0, column=0, columnspan=3, sticky=EW)

        hdr_label = ttk.Label(
            master=hdr_frame,
            image='logo',
            bootstyle=(INVERSE, SECONDARY)
        )
        hdr_label.pack(side=LEFT)

        logo_text = ttk.Label(
            master=hdr_frame,
            text='智能巡检系统',
            font=('TkDefaultFixed', 30),
            bootstyle=(INVERSE, SECONDARY)
        )
        logo_text.pack(side=LEFT, padx=10)

        # action buttons
        action_frame = ttk.Frame(self)
        action_frame.grid(row=1, column=0, sticky=NSEW)

        cleaner_btn = ttk.Button(
            master=action_frame,
            image='cleaner',
            # text='首页',
            compound=TOP,
            bootstyle=INFO
        )
        cleaner_btn.pack(side=TOP, fill=BOTH, ipadx=10, ipady=10)

        registry_btn = ttk.Button(
            master=action_frame,
            image='registry',
            # text='巡检',
            compound=TOP,
            bootstyle=INFO
        )
        registry_btn.pack(side=TOP, fill=BOTH, ipadx=10, ipady=10)

        tools_btn = ttk.Button(
            master=action_frame,
            image='tools',
            # text='地图',
            compound=TOP,
            bootstyle=INFO
        )
        tools_btn.pack(side=TOP, fill=BOTH, ipadx=10, ipady=10)

        options_btn = ttk.Button(
            master=action_frame,
            image='options',
            # text='设置',
            compound=TOP,

            bootstyle=INFO
        )
        options_btn.pack(side=TOP, fill=BOTH, ipadx=10, ipady=10)

        # option notebook
        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=1, sticky=NSEW, pady=(25, 25))

        # windows tab
        windows_tab = ttk.Frame(notebook, padding=10)
        wt_scrollbar = ttk.Scrollbar(windows_tab)
        wt_scrollbar.pack(side=RIGHT, fill=Y)
        wt_scrollbar.set(0, 1)

        wt_canvas = ttk.Canvas(
            master=windows_tab,
            relief=FLAT,
            borderwidth=10,
            selectborderwidth=0,
            highlightthickness=0,
            yscrollcommand=wt_scrollbar.set
        )
        wt_canvas.pack(side=LEFT, fill=BOTH)

        # adjust the scrollregion when the size of the canvas changes
        wt_canvas.bind(
            sequence='<Configure>',
            func=lambda e: wt_canvas.configure(
                scrollregion=wt_canvas.bbox(ALL))
        )
        wt_scrollbar.configure(command=wt_canvas.yview)
        scroll_frame = ttk.Frame(wt_canvas)
        wt_canvas.create_window((0, 0), window=scroll_frame, anchor=NW)

        values1 = ["C2机房4楼2号线", "C2机房4楼4号线", "C2机房4楼6号线"]
        self.combobox1 = ttk.Combobox(master=scroll_frame, font=('', 20), values=values1)
        self.combobox1.place(relx=0.2, rely=0.1, relwidth=1, relheight=1)
        self.combobox1.pack(fill=BOTH, expand=YES, padx=20, pady=10)

        btn1 = ttk.Button(master=scroll_frame, text='开始巡检', command=self.DetectStart, bootstyle=INFO)
        btn1.pack(side='left', fill=BOTH, padx=20, ipadx=30, ipady=10, anchor='sw')

        btn2 = ttk.Button(master=scroll_frame, text='返回充电', command=self.callCharge, bootstyle=INFO)
        btn2.pack(side='right', fill=BOTH, padx=20, ipadx=30, ipady=10, anchor='se')

        btn3 = ttk.Button(master=scroll_frame, text='返回充电', command=self.callCharge, bootstyle=INFO)
        btn3.pack(side='right', fill=BOTH, padx=20, ipadx=30, ipady=10, anchor='se')

        #btn3 = ttk.Button(master=scroll_frame, text='停止移动', command=self.callStop, bootstyle=INFO)
        #btn3.pack(side='left', fill=BOTH,ipadx=10, padx=15, pady=25, anchor='nw')

        #btn1.place(relwidth=3, relheight=1)
        #btn2.place(relx=0.2, rely=0.7, relwidth=0.1, relheight=0.1)
        #btn3.place(relx=0.4, rely=0.7, relwidth=0.1, relheight=0.1)



        '''
        radio_options = []

        edge = ttk.Labelframe(
            master=scroll_frame,
            text='Microsoft Edge',
            padding=(20, 5)
        )
        edge.pack(fill=BOTH, expand=YES, padx=20, pady=10)

        explorer = ttk.Labelframe(
            master=scroll_frame,
            text='Internet Explorer',
            padding=(20, 5)
        )
        explorer.pack(fill=BOTH, padx=20, pady=10, expand=YES)

        # add radio buttons to each label frame section
        for section in [edge, explorer]:
            for opt in radio_options:
                cb = ttk.Checkbutton(section, text=opt, state=NORMAL)
                cb.invoke()
                cb.pack(side=TOP, pady=2, fill=X)
        '''
        notebook.add(windows_tab, text='巡检管理')

        # empty tab for looks
        # notebook.add(ttk.Frame(notebook), text='applications')

        # results frame
        results_frame = ttk.Frame(self)
        results_frame.grid(row=1, column=2, sticky=NSEW)

        # progressbar with text indicator
        pb_frame = ttk.Frame(results_frame, padding=(0, 10, 10, 10))
        pb_frame.pack(side=TOP, fill=X, expand=YES)

        pb = ttk.Progressbar(
            master=pb_frame,
            bootstyle=(INFO, STRIPED),
            variable='progress'
        )
        pb.pack(side=LEFT, fill=X, expand=YES, padx=(15, 10))

        ttk.Label(pb_frame, text='%').pack(side=RIGHT)
        ttk.Label(pb_frame, textvariable='progress').pack(side=RIGHT)
        self.setvar('progress', 100)

        # result cards
        cards_frame = ttk.Frame(
            master=results_frame,
            name='cards-frame',
            bootstyle=SECONDARY,
            height=300
        )
        cards_frame.pack(fill=BOTH, expand=YES)

        # privacy card
        priv_card = ttk.Frame(
            master=cards_frame,
            padding=1,
        )
        priv_card.pack(side=LEFT, fill=BOTH, padx=(10, 5), pady=10)

        priv_container = ttk.Frame(
            master=priv_card,
            padding=40,
        )
        priv_container.pack(fill=BOTH, expand=YES)

        priv_lbl = ttk.Label(
            master=priv_container,
            image='privacy',
            text='实时监控',
            compound=TOP,
            anchor=CENTER
        )
        priv_lbl.pack(fill=BOTH, padx=20, pady=(40, 0))

        ttk.Label(
            master=priv_container,
            textvariable='priv_lbl',
            bootstyle=PRIMARY
        ).pack(pady=(0, 20))
        # self.setvar('priv_lbl', '6025 tracking file(s) removed')

        # junk card
        junk_card = ttk.Frame(
            master=cards_frame,
            padding=1,
        )
        junk_card.pack(side=LEFT, fill=BOTH, padx=(5, 10), pady=10)

        junk_container = ttk.Frame(junk_card, padding=40)
        junk_container.pack(fill=BOTH, expand=YES)

        junk_lbl = ttk.Label(
            master=junk_container,
            image='junk',
            text='地图',
            compound=TOP,
            anchor=CENTER,
        )
        junk_lbl.pack(fill=BOTH, padx=20, pady=(40, 0))

        ttk.Label(
            master=junk_container,
            textvariable='junk_lbl',
            bootstyle=PRIMARY,
            justify=CENTER
        ).pack(pady=(0, 20))
        # self.setvar('junk_lbl', '1,150 MB of unneccesary file(s)\nremoved')

        # user notification
        note_frame = ttk.Frame(
            master=results_frame,
            bootstyle=SECONDARY,
            padding=40
        )
        note_frame.pack(fill=BOTH)

        note_msg = ttk.Label(
            master=note_frame,
            # text='',
            anchor=CENTER,
            font=('Helvetica', 12, 'italic'),
            bootstyle=(INVERSE, SECONDARY)
        )
        note_msg.pack(fill=BOTH)

    def DetectStart(self):
        route = self.combobox1.get()
        try:
            e = threading.Event()
            t1 = threading.Thread(target=detectFlow.inspectRoute(route), args=(e,))
            t1.start()
            # detectFlow.inspectRoute(route)
        except:
            print("inspect exception")

    def callCharge(self):
        try:
            detectFlow.backToCharge()
        except:
            print("charge exception")

    def callExample(self):
        try:
            detectFlow.backToCharge()
        except:
            print("charge exception")

    def callStop(self):
        try:
            detectFlow.stopMove()
        except:
            print("stop move exception")

    def refresh_data(self):

        print("refresh data...")
        self.setvar('progress', 50)
        self.after(100, self.refresh_data)


if __name__ == '__main__':
    app = ttk.Window("智能巡检系统", "pulse", )
    Cleaner(app)
    app.geometry('%dx%d+%d+%d' % (2100, 790, 10, 10))
    e = threading.Event()
    t1 = threading.Thread(target=detectFlow.openDoorThread, args=(e,))
    t1.start()
    app.mainloop()
    e.set()

