from dataclasses import dataclass
from torchvision.models import resnet18, ResNet18_Weights
import torch
import einops
from typing import Optional, Tuple
import torch.nn.functional as F
import models.bet.libraries.mingpt.model as option_model
from models.bet.latent_generators.mingpt import MinGPT
from models.bet.libraries.loss_fn import FocalLoss, soft_cross_entropy
import numpy as np

class LiberoModel(MinGPT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agentview_model = resnet18(num_classes = 64)
        self.eye_in_hand_model = resnet18(num_classes = 64)
        self.gpt_config.input_size = 9
        num_options = self.option_model.num_options
        self.option_model = option_model.GPT(self.gpt_config, num_options=num_options)

        weights = ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()

    def encode_images(self, seq_agentview, seq_eyeinhand):
        """list seq_agentview numpy array B, C, H, W, returns (B, 128)"""
        seq_agentview = self.preprocess(torch.from_numpy(np.stack(seq_agentview)))
        seq_eyeinhand = self.preprocess(torch.from_numpy(np.stack(seq_eyeinhand)))
        if seq_eyeinhand.dim()==3:
            seq_eyeinhand = seq_eyeinhand.unsqueeze(0)
            seq_agentview = seq_agentview.unsqueeze(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.concat((self.agentview_model(seq_agentview.to(device)),self.eye_in_hand_model(seq_eyeinhand.to(device))), axis=1)
    
    def get_latent_and_loss(
        self,
        obs_rep: torch.Tensor,
        target_latents: torch.Tensor,
        seq_masks: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
        idx = None,
        dataset = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx is not None and dataset is not None:
            # Append images to observation.
            # For every sequence in batch, calculate output
            # after all outputs are calculated, concat to enc_obs
            outputs_agentview = []
            outputs_eye_in_hand = []
            counter = 0
            for sequence in idx:
                counter+=1
                seq_agentview= []
                seq_eyeinhand = []
                for timestep in sequence:
                    img1, img2 = dataset.get_images(timestep.item())
                    seq_agentview.append(img1)
                    seq_eyeinhand.append(img2)
                
                seq_agentview = self.preprocess(torch.from_numpy(np.stack(seq_agentview)))
                seq_eyeinhand = self.preprocess(torch.from_numpy(np.stack(seq_eyeinhand)))
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                outputs_agentview.append(self.agentview_model(seq_agentview.to(device)))
                outputs_eye_in_hand.append(self.eye_in_hand_model(seq_eyeinhand.to(device)))

            enc_obs, option = obs_rep
            res = torch.concat((enc_obs, torch.stack(outputs_agentview), torch.stack(outputs_eye_in_hand)), axis=2)
            obs_rep = res, option

        if self.predict_offsets:
            target_latents, target_offsets = target_latents

        is_soft_target = (target_latents.shape[-1] == self.vocab_size) and (
            self.vocab_size != 1
        )

        if is_soft_target:
            target_latents = target_latents.view(-1, target_latents.size(-1))
            criterion = soft_cross_entropy
        else:
            target_latents = target_latents.view(-1)
            if self.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                target_latents = torch.zeros_like(target_latents)
            criterion = FocalLoss(gamma=self.focal_loss_gamma)
        
        if self.predict_offsets:
            output, _ = self.model(obs_rep)
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size :]
            batch = logits.shape[0]
            seq = logits.shape[1]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.action_dim,
            )
            class_loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
            selected_offsets = offsets[
                torch.arange(offsets.size(0)),
                target_latents.argmax(dim=-1).view(-1)
                if is_soft_target
                else target_latents.view(-1),
            ]
            offset_loss = self.offset_loss_scale * F.mse_loss(
                selected_offsets, target_offsets.view(-1, self.action_dim)
            )
            loss = offset_loss + class_loss
            logits = einops.rearrange(logits, "batch seq classes -> seq batch classes")
            offsets = einops.rearrange(
                offsets,
                "(N T) V A -> T N V A",  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
                N=batch,
                T=seq,
            )
            if return_loss_components:
                return (
                    (logits, offsets),
                    loss,
                    {"offset": offset_loss, "class": class_loss, "total": loss},
                )
            else:
                return (logits, offsets), loss
        else:
            logits, _ = self.model(obs_rep)
            loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
            logits = einops.rearrange(
                logits, "batch seq classes -> seq batch classes"
            )  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
            if return_loss_components:
                return logits, loss, {"class": loss, "total": loss}
            else:
                return logits, loss
            
    def generate_latents(
        self, seq_obses: torch.Tensor, seq_masks: torch.Tensor, option=None
    ) -> torch.Tensor:
        seq, batch, embed = seq_obses.size()
        obs_rep = einops.rearrange(seq_obses, "seq batch embed -> batch seq embed")
        output, _ = self.model((obs_rep, option), None)
        if self.predict_offsets:
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size :]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.action_dim,
            )
        else:
            logits = output
        next_option = F.softmax(self.option_model((obs_rep, option))[0], -1)
        next_option_logs = next_option.view(-1, next_option.shape[-1])
        next_option = torch.multinomial(next_option_logs, num_samples=1)
        probs = F.softmax(logits, dim=-1)
        batch, seq, choices = probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
        sampled_data = einops.rearrange(
            sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
        )
        next_option = next_option[-1].unsqueeze(0)
        if self.predict_offsets:
            sampled_offsets = offsets[
                torch.arange(offsets.shape[0]), sampled_data.flatten()
            ].view(batch, seq, self.action_dim)

            return (sampled_data, sampled_offsets), (next_option, next_option_logs[-1][next_option].item())
        else:
            return sampled_data, (next_option, next_option_logs[-1][next_option].item())
