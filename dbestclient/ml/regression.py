# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from qregpy import qreg

from dbestclient.ml import mdn


class DBEstReg:
    def __init__(self, config):
        self.reg = None
        self.config = config

    def fit(self, x, y, runtime_config):
        reg_type = self.config.config["reg_type"]

        if reg_type == 'qreg':
            self.reg = qreg.QReg(
                base_models=["linear", "polynomial"], verbose=False
            ).fit(x, y)
        if reg_type == 'mdn':
            dim_input = int(x.shape[0] / y.shape[0])
            self.reg = mdn.RegMdn(
                self.config, dim_input=dim_input
            ).fit(x, y, runtime_config)
        return self.reg
