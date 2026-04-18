from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import os
from datetime import datetime
import numpy as np

class ModelSnapshot(SummaryWriter):
    def __init__(self, model_file):
        self.hyperparameters = []
        self.has_writer = False
        self.model_file = model_file

    def hyper(self, key, value):
        assert not self.has_writer, "cannot add hyperparameter after SummaryWriter has already been created"
        self.hyperparameters.append([key, value])
        return value
    
    def writer(self):
        assert not self.has_writer, "cannot create second SummaryWriter for model snapshot"
        self.has_writer = True

        suffix = "-".join(list(map(lambda p: str(p[0]) + "=" + str(p[1]), self.hyperparameters)))
        
        log_dir = None
        timestamp = datetime.now().strftime("%d-%m-%Y-%H_%M_%S")
        if "EXP_ID" in os.environ.keys():
            exp_id = os.environ["EXP_ID"]
            rand = str(np.random.randint(10000,20000))
            log_dir = f"runs/{exp_id}-{rand}"

        writer = SummaryWriter(log_dir=log_dir, comment=suffix)
        shutil.copyfile(self.model_file, writer.log_dir + "/model.py")

        return writer