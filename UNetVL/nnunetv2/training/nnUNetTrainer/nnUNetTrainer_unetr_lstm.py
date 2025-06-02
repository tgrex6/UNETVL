from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import torch
from torch.optim import AdamW
from torch import nn
from nnunetv2.nets.unetr import UNETR
from nnunetv2.nets.unetr_lstm import UNETR_LSTM
from nnunetv2.paths import nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from datetime import datetime

class nnUNetTrainer_unetr_lstm(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), lstm: bool = False, no_kan: bool = True):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.no_kan = no_kan
        self.lstm = lstm
        self.initial_lr = 1e-4
        self.grad_scaler = None
        self.weight_decay = 1e-3
        # print image size, patch size, batch size
        print(f"Dataset: {self.dataset_json['name']}")
        print(f"Batch size: {self.configuration_manager.batch_size}")
        print(f"Patch size: {self.configuration_manager.patch_size}")
        if self.lstm:
            print(f"Using UNETR_LSTM Architecture with no_kan={self.no_kan}")
        else:
            self.no_kan = True
            print(f"Using UNETR Architecture")
            print(f"KAN is not supported for UNETR")

        if self.lstm:
            self.name = 'UNETR_LSTM'
            if self.no_kan:
                self.name += '_NO_KAN'

        else:
            self.name = 'UNETR'
        
        self.update_output_folders()
    
    def update_output_folders(self):
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                    self.__class__.__name__ + '__' + self.name + '__' + self.plans_manager.plans_name + "__" + self.configuration_name) \
        if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{self.fold}')
        maybe_mkdir_p(self.output_folder)
        timestamp = datetime.now()
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        
        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)


    def build_network_architecture(self,
                                   plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        
        label_manager = plans_manager.get_label_manager(dataset_json)
        
        assert len(configuration_manager.patch_size) == 3, "only 3D supported"

        if self.lstm:
            model = UNETR_LSTM(img_shape=configuration_manager.patch_size, 
                               input_dim=num_input_channels, 
                               output_dim=label_manager.num_segmentation_heads, 
                               embed_dim=384, 
                               patch_size=16, 
                               depth=12,
                               dropout=0.0, 
                               no_kan=self.no_kan)
        elif not self.lstm:
            model = UNETR(img_shape=configuration_manager.patch_size, 
                          input_dim=num_input_channels, 
                          output_dim=label_manager.num_segmentation_heads, 
                          embed_dim=768, 
                          patch_size=16, 
                          num_heads=12, 
                          dropout=0.0)
        else:
            raise NotImplementedError("Only 3D models are supported")
        
        return model

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        del data
        l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}



    def configure_optimizers(self):

        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-6)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass