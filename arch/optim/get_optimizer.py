from arch.optim.filter_optimizer import create_filtered_optimizer, create_2D_filtered_optimizer

def get_optimizer(model, nconfig, raw_model):
    match nconfig.optim:
        case 'adamw':
            from torch.optim import AdamW
            optimizer = AdamW(model.parameters(), lr=nconfig.learning_rate, betas=(0.9, 0.95), weight_decay=nconfig.weight_decay)
            optimizers = [optimizer]
        case 'spam':
            from arch.optim.spam import SPAMAdamW
            optimizer = SPAMAdamW(model.parameters(), lr=nconfig.learning_rate, betas=(0.9, 0.95), weight_decay=nconfig.weight_decay)
            optimizers = [optimizer]
        case 'muon':
            from torch.optim import AdamW
            from arch.optim.muon import Muon
            optimizer1 = AdamW([raw_model.transformer.wte.weight], lr=nconfig.learning_rate * 10, betas=(0.9, 0.95), fused=True)
            optimizer2 = AdamW([raw_model.lm_head.weight], lr=nconfig.learning_rate, betas=(0.9, 0.95), weight_decay=nconfig.weight_decay, fused=True)
            optimizer3 = create_2D_filtered_optimizer(Muon, raw_model.transformer.h.parameters(), lr=nconfig.learning_rate, momentum=0.95, weight_decay=nconfig.weight_decay)
            optimizers = [optimizer1, optimizer2, optimizer3]
            if optimizer4 := create_filtered_optimizer(AdamW, raw_model.transformer.h.parameters(), lr=nconfig.learning_rate, betas=(0.9, 0.95), weight_decay=nconfig.weight_decay, fused=True):
                optimizers.append(optimizer4)
                
        case 'upgraded-muon':
            from arch.optim.spam import SPAMAdamW
            from arch.optim.muon import Muon
            optimizer1 = SPAMAdamW([raw_model.transformer.wte.weight], lr=0.3, betas=(0.9, 0.95), weight_decay=0.01)
            optimizer2 = SPAMAdamW([raw_model.lm_head.weight], lr=0.002, betas=(0.9, 0.95), weight_decay=0.01)
            optimizer3 = create_2D_filtered_optimizer(Muon, raw_model.transformer.h.parameters(), lr=nconfig.learning_rate, momentum=0.95)
            optimizers = [optimizer1, optimizer2, optimizer3]
            if optimizer4 := create_filtered_optimizer(SPAMAdamW, raw_model.transformer.h.parameters(), lr=1e-3, betas=(0.9, 0.95), fused=True):
                optimizers.append(optimizer4)        
        case 'stable-spam':
            from arch.optim.stableSPAM import StableSPAM
            optimizer = StableSPAM(model.parameters(), lr=nconfig.learning_rate, weight_decay=nconfig.weight_decay)
            optimizers = [optimizer]
        case _:
            raise ValueError(f"Optimizer {nconfig.optim} not supported")
