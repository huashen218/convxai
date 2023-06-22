import os
import math
import json
import torch
import logging
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)

best_valid_ppl = float("inf")

class Trainer(object):

    def __init__(self, configs, model, tokenizer, accelerator, writer, optimizer, lr_scheduler):
        super(Trainer, self).__init__()
        self.configs = configs
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.writer = writer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.completed_steps = 0
        self.total_loss = 0.



    def train(self, epoch, train_dataloader, starting_epoch, resume_step):

        ### Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.configs["train_configs"]["max_train_steps"]), disable=not self.accelerator.is_local_main_process)
        

        ### Figure out how many steps we should save the Accelerator states
        if hasattr(self.configs["save_configs"]["checkpointing_steps"], "isdigit"):
            checkpointing_steps = self.configs["save_configs"]["checkpointing_steps"]
            if self.configs["save_configs"]["checkpointing_steps"].isdigit():
                checkpointing_steps = int(self.configs["save_configs"]["checkpointing_steps"])
        else:
            checkpointing_steps = None



        self.model.train()

        if self.configs["save_configs"]["with_tracking"]:
            self.total_loss = 0

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if self.configs["save_configs"]["resume_from_checkpoint"] and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    self.completed_steps += 1
                    continue
            outputs = self.model(**batch)
            loss = outputs.loss
            self.writer.add_scalar("Loss/train", loss.mean(), epoch)

            if self.configs["save_configs"]["with_tracking"]:
                self.total_loss += loss.detach().float()
            loss = loss / self.configs["train_configs"]["gradient_accumulation_steps"]
            self.accelerator.backward(loss.mean())   # accelerator.backward(loss)
            if step % self.configs["train_configs"]["gradient_accumulation_steps"] == 0 or step == len(train_dataloader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                self.completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if self.completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{self.completed_steps }"
                    if self.configs["save_configs"]["output_dir"] is not None:
                        output_dir = os.path.join(self.configs["save_configs"]["output_dir"], output_dir)
                    self.accelerator.save_state(output_dir)
            if self.completed_steps >= self.configs["train_configs"]["max_train_steps"]:
                break

        return 



    def evaluate(self, epoch, dev_dataloader, dev_dataset, tf_writer):
        global best_valid_ppl
        best_valid_ppl = float("inf") if epoch == 0 else best_valid_ppl


        self.model.eval()
        losses = []
        for step, batch in enumerate(dev_dataloader):
            batch = {k: v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            loss = outputs.loss
            self.writer.add_scalar(f"Loss/{tf_writer}", loss.mean(), epoch)
            losses.append(self.accelerator.gather(loss.repeat(self.configs["data_configs"]["per_device_eval_batch_size"])))

        losses = torch.cat(losses)
        losses = losses[: len(dev_dataset)]

        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        self.writer.add_scalar(f"Perplexity/{tf_writer}", perplexity, epoch)
        self.writer.flush()

        logger.info(f"epoch {epoch}: perplexity: {perplexity}")

        if self.configs["save_configs"]["with_tracking"]:
            self.accelerator.log(
                {"perplexity": perplexity, "train_loss": self.total_loss, "epoch": epoch, "step": self.completed_steps},
            )

        with open(os.path.join(self.configs["save_configs"]["output_dir"], "all_results.json"), "w") as f:
            json.dump({f"epoch={epoch}": f"perplexity={perplexity}"}, f)


        """Save model and tokenizer at best validation ppl"""
        if tf_writer == "validation":
            if perplexity < best_valid_ppl:
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(
                    self.configs["save_configs"]["output_dir"], is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
                )
                if self.accelerator.is_main_process:
                    self.tokenizer.save_pretrained(self.configs["save_configs"]["output_dir"])

                best_valid_ppl = perplexity
                logger.info(f"Saving best model at epock={epoch} with perplexity={perplexity}")


        """Save model and tokenizer at best validation ppl"""
        if self.configs["save_configs"]["checkpointing_steps"] == "epoch":
            output_dir = f"epoch_{epoch}"
            output_dir = os.path.join(self.configs["save_configs"]["output_dir"], output_dir)
            self.accelerator.save_state(output_dir)


        return perplexity
