class Params:
    def __init__(self):
        self.C0 = 0.0193
        self.lbda = 530e-09
        self.f0 = 50e-03
        self.fe = 24e-3 # 20e-03
        self.SLMpitch = 3.74e-06
        self.W = 4.0 # 10mm Doublet Lens Version macro photography

        self.oledWidth = 2560
        self.oledHeight = 2560
        self.slmWidth = 4000 #3840
        self.slmHeight = 2464 #2160
        self.camWidth = 1440
        self.camHeight = 1080
        # self.maindisplayWidth = 1512
        # self.maindisplayWidth = 1920
        # self.maindisplayHeight = 1200
        # self.nominal_a = -self.slmHeight/self.slmWidth

        self.xoffset = 168
        self.yoffset = 52
        self.xbegin = 1210 # 1160
        self.xend = self.xbegin + (5020-1160) # 5020
        self.ybegin = 900 # 942
        self.yend = 3040 # 3082
        self.ybottomcroppx = 40 #130
        self.xcrop1 = 30 #60 #30 #60 for sailor
        self.xcrop2 = 60 #280 #60 #280 for sailor

        self.patch_size = 31 #31 #31 21
        self.window_size = 101 #101 #101 51
        self.constant_factor = 1.8

    def set_W(self, W):
        self.W = W

    def set_max_depth_correspondence(self, maxScaleX, Dmax):
        self.Dmax = Dmax
        self.maxScaleX = maxScaleX
