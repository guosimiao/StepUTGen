import math
import sys,datetime
from tqdm import tqdm
import torch
from utils.utils import setup_seed

class DefaultRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None, **kwargs,
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.kwargs = kwargs
        self.accelerator = accelerator
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        features,labels = batch 
        
        #loss
        with self.accelerator.autocast():
            preds = self.net(features)
            loss = self.loss_fn(preds,labels)

        #backward()
        if self.stage=="train" and self.optimizer is not None:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        all_loss = self.accelerator.gather(loss).sum()
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        
        #losses (or plain metrics that can be averaged)
        step_losses = {self.stage+"_loss":all_loss.item()}
        
        #metrics (stateful metrics)
        step_metrics = {self.stage+"_"+name:metric_fn(all_preds, all_labels).item() 
                        for name,metric_fn in self.metrics_dict.items()}
        
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics

class EpochRunner:
    def __init__(self,steprunner,quiet=False):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.accelerator = steprunner.accelerator
        self.net = steprunner.net
        self.quiet = quiet
        
    def __call__(self,dataloader, step_callback=None):
        n = dataloader.size  if hasattr(dataloader,'size') else len(dataloader)
        loop = tqdm(enumerate(dataloader,start=1), 
                    total=n,
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process or self.quiet,
                    ncols=100
                   )
        epoch_losses = {}
        
        for step, batch in loop: 
            with self.accelerator.accumulate(self.net):
                step_losses,step_metrics = self.steprunner(batch)   
            step_log = dict(step_losses,**step_metrics)
            for k,v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k,0.0)+v

            if step_callback is not None:
                step_callback(step, step_log)
                    
            if step<n:
                loop.set_postfix(**step_log)

            elif step==n:
                epoch_metrics = step_metrics
                epoch_losses = {k:v/step for k,v in epoch_losses.items()}
                epoch_log = dict(epoch_losses,**epoch_metrics)
                loop.set_postfix(**epoch_log)

            else:
                break
        return epoch_log

class Trainer:
    
    StepRunner, EpochRunner = DefaultRunner, EpochRunner
    
    def __init__(self,net,accelerator,tokenizer,optimizer,config=None,lr_scheduler = None, init_batch = 0,**kwargs):
        super().__init__()
        self.net,self.accelerator,self.optimizer = net, accelerator, optimizer
        self.lr_scheduler, self.init_batch = lr_scheduler, init_batch
        self.config,self.tokenizer = config,tokenizer
        self.kwargs = kwargs
        
    def save_ckpt(self, ckpt_path=None):
        accelerator = self.accelerator
        net_dict = accelerator.get_state_dict(self.net)
        accelerator.save(net_dict,ckpt_path if ckpt_path is not None else self.ckpt_path)

    def evaluate(self, eval_dataloader, epoch=None, global_step=None, quiet=True):
        was_training = self.net.training
        valid_step_runner = self.StepRunner(
                net = self.net,
                accelerator = self.accelerator,
                stage="valid",
                optimizer = None,
                lr_scheduler = None,
                config = self.config,
                tokenizer = self.tokenizer
        )

        valid_epoch_runner = self.EpochRunner(valid_step_runner, quiet=quiet)
        valid_metrics = {}
        if epoch is not None:
            valid_metrics["epoch"] = epoch
        if global_step is not None:
            valid_metrics["global_step"] = global_step

        with torch.no_grad():
            valid_metrics.update(valid_epoch_runner(eval_dataloader))

        if was_training:
            self.net.train()
        else:
            self.net.eval()
        return valid_metrics
    
    def fit(self, dataloader, ckpt_path = "checkpoint.pt", epochs=10, eval_dataloader=None):

        setup_seed(self.config.seed)
        self.accelerator.print(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preparing model, optimizer, scheduler, and dataloader..."
        )
    
        self.net, self.optimizer, self.lr_scheduler, train_dataloader = self.accelerator.prepare(
            self.net, self.optimizer, self.lr_scheduler, dataloader
        )
        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        total_local_batches = len(train_dataloader)
        if self.init_batch:
            train_dataloader = self.accelerator.skip_first_batches(
                train_dataloader,
                self.init_batch,
            )
        local_batches = len(train_dataloader)
        grad_accumulation = max(int(self.config.gradient_accumulation), 1)
        eval_steps = int(self.config.get("eval_steps", 0) or 0)
        best_valid_loss = None
        best_ckpt_path = str(self.config.get("best_ckpt_path", f"{ckpt_path}_best_valid"))
        last_eval_step = None
        self.accelerator.print(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training plan | "
            f"total_local_micro_batches={total_local_batches} | "
            f"initial_batch={self.init_batch} | "
            f"remaining_local_micro_batches={local_batches} | "
            f"gradient_accumulation={grad_accumulation} | "
            f"approx_optimizer_steps_per_epoch={math.ceil(local_batches / grad_accumulation)}"
        )
        if eval_dataloader is not None:
            self.accelerator.print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation plan | "
                f"local_valid_batches={len(eval_dataloader)} | "
                f"eval_steps={eval_steps if eval_steps > 0 else 'epoch_end_only'} | "
                f"best_ckpt_path={best_ckpt_path}"
            )

        for key in self.kwargs:
            self.kwargs[key] = self.accelerator.prepare(self.kwargs[key])
        
        self.history = {"train": [], "valid": []}
        global_step = int(self.init_batch)

        def run_validation(epoch, step):
            nonlocal best_valid_loss, last_eval_step
            valid_metrics = self.evaluate(eval_dataloader, epoch=epoch, global_step=step, quiet=True)
            self.history["valid"].append(valid_metrics)
            last_eval_step = step

            valid_loss = valid_metrics.get("valid_loss")
            self.accelerator.print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Validation summary | {valid_metrics}"
            )
            if (
                valid_loss is not None
                and math.isfinite(float(valid_loss))
                and bool(self.config.get("save_best_valid_checkpoint", True))
                and (best_valid_loss is None or float(valid_loss) < best_valid_loss)
            ):
                best_valid_loss = float(valid_loss)
                self.save_ckpt(best_ckpt_path, accelerator = self.accelerator)
                self.accelerator.print(
                    f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Saved best validation checkpoint to {best_ckpt_path} "
                    f"(valid_loss={best_valid_loss:.6f})"
                )
            return valid_metrics
                
        for epoch in range(0,epochs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.accelerator.print("\n"+"=========="*8 + "%s"%nowtime)
            self.accelerator.print("Epoch {0} / {1}".format(epoch, epochs)+"\n")

            # 1. Train -------------------------------------------------
            train_step_runner = self.StepRunner(
                    net = self.net,
                    accelerator = self.accelerator,
                    stage="train",
                    optimizer = self.optimizer,
                    lr_scheduler = self.lr_scheduler,
                    config = self.config,
                    tokenizer = self.tokenizer,
                    batch_count=global_step,
            )

            train_epoch_runner = self.EpochRunner(train_step_runner)
            train_metrics = {'epoch':epoch}
            def after_train_step(step, step_log):
                nonlocal global_step
                global_step += 1
                if eval_dataloader is not None and eval_steps > 0 and global_step % eval_steps == 0:
                    run_validation(epoch, global_step)

            train_metrics.update(train_epoch_runner(train_dataloader, step_callback=after_train_step))
            train_metrics["global_step"] = global_step
            self.history["train"].append(train_metrics)
            self.save_ckpt(ckpt_path,accelerator = self.accelerator)
            if eval_dataloader is not None and last_eval_step != global_step:
                run_validation(epoch, global_step)
            self.accelerator.print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch} summary | {train_metrics}"
            )
            
        self.net = self.accelerator.unwrap_model(self.net)
        self.net.cpu()
        self.accelerator.print(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training loop completed."
        )
        return train_metrics
