from method.dc_base import DC_BASE

class Random(DC_BASE):
    def __init__(self, args):
        super(Random, self).__init__(args)

    def condense(self):
        self.save_img(self.path)
        self.visualize(self.path,0)