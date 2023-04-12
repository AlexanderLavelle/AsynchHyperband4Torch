import torch, wandb
import data, modeling, training
from tqdm.auto import tqdm
import numpy as np

from ray import tune, air
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import ScalingConfig
import ray.train.torch as rt
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.train.torch import TorchTrainer



def training_loop(config, batch_size, train_dict=None, val_dict=None): 
        
    with wandb.init(project="torch_stock_transformer", job_type="initialize", config=config) as run:
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = modeling.SimpleTransformer(config).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'])
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.2, patience=7, min_lr=0.00000001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.005, epochs=2_000, steps_per_epoch=541, pct_start=.1)
        

        
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            checkpoint_dict = loaded_checkpoint.to_dict()
            last_step = checkpoint_dict["step"]
            step = last_step + 1
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            reduce_lr.load_state_dict(checkpoint_dict['plateau_scheduler_dict'])
            model.load_state_dict(checkpoint_dict['model_state_dict'])
            
        else:
            step = 0
            
        # model.compile()            
            
        model = rt.prepare_model(model, wrap_ddp=True, ddp_kwargs={'find_unused_parameters':True})
        
        criterion = torch.nn.CrossEntropyLoss()

        wandb.watch(model, log="all")
        torch.set_grad_enabled(True)
        model.train()
    
        while True:
            
            train_dl = data.loader(train_dict, config, batch_size) 
            val_dl = data.loader(val_dict, config, batch_size)

            train_losses = []
            for i in tqdm(train_dl, total=len(train_dl)):
                optimizer.zero_grad()
                
                x, idx, y = [*i]

                preds = model(x.squeeze().to(device), idx.to(device))
                loss = criterion(preds.squeeze().to(device), y.to(device))

                loss.backward()
                optimizer.step()
                scheduler.step()
                reduce_lr.step(metrics=loss)

                train_losses.append(loss.item())

            val_losses = [] 
            for i in val_dl:
                model.eval()
                with torch.no_grad():
                    
                    x, idx, y = [*i]

                    preds = model(x.squeeze().to(device), idx.to(device))
                    loss = criterion(preds.squeeze().to(device), y.to(device))

                    val_losses.append(loss.item())
            
            
            wandb.log({
                "Train Loss": np.mean(train_losses),
                "Test Loss": np.mean(val_losses),
            })
            
            
            checkpoint = Checkpoint.from_dict(
                {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), 
                        "scheduler_state_dict": scheduler.state_dict(),
                        "plateau_scheduler_dict": reduce_lr.state_dict(),
                        "step": step
                    },
            )
            session.report({
                "train_loss":np.mean(train_losses),
                "val_loss":np.mean(val_losses)
            }, checkpoint=checkpoint)
            step += 1


def setup_ray(train_dict, val_dict, batch_size, param_space, n_tries):
    trainer = TorchTrainer(
        train_loop_per_worker=tune.with_parameters(training_loop, batch_size=batch_size, train_dict=train_dict, val_dict=val_dict),
        scaling_config=ScalingConfig(
            num_workers=1, # Number of workers to use for data parallelism.
            use_gpu=True,
        )
    )

    bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=250,
        reduction_factor=4,
        stop_last_trials=False,
        metric='val_loss',
        mode='min', 
        #  grace_period=10    # One may want to set this
        )

    algo = TuneBOHB(
        param_space,
        metric='val_loss', 
        mode='min'
        )    

    tune_config = tune.TuneConfig(
        search_alg=algo,
        num_samples=n_tries,
        scheduler=bohb,
        max_concurrent_trials=2,
        reuse_actors=False
        )
    
    return trainer, tune_config
            