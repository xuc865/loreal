import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import math
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import os
_tokenizer = _Tokenizer()
from collections import OrderedDict


def _get_attr_cfg(cfg):
    # Paper Sec. 3.2 uses K resolution-robust attributes. In this repo the
    # default K is 5, and each attribute can own M learnable prompt tokens.
    attr_cfg = cfg.TRAINER.ATPROMPT
    attr_num = int(attr_cfg.ATT_NUM)
    if attr_num < 1:
        raise ValueError("TRAINER.ATPROMPT.ATT_NUM must be positive")

    attr_specs = []
    for idx in range(1, attr_num + 1):
        n_key = f"N_ATT{idx}"
        text_key = f"ATT{idx}_TEXT"
        if not hasattr(attr_cfg, n_key) or not hasattr(attr_cfg, text_key):
            raise ValueError(
                f"Missing TRAINER.ATPROMPT.{n_key} or {text_key} for ATT_NUM={attr_num}"
            )

        n_tokens = int(getattr(attr_cfg, n_key))
        text = str(getattr(attr_cfg, text_key)).strip()
        if n_tokens < 1:
            raise ValueError(f"TRAINER.ATPROMPT.{n_key} must be positive")
        if not text:
            raise ValueError(f"TRAINER.ATPROMPT.{text_key} must be non-empty")
        attr_specs.append((n_tokens, text))

    return attr_specs

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x 

class PromptLearner(nn.Module):
    """CoOp prompt learner augmented with LOREAL attribute slots.

    Paper mapping:
    - The normal CoOp context is P0 in Sec. 3.1.
    - Each attribute slot is "Si [Ai]" in Sec. 3.2.
    - The meta-nets below implement Sk = Mk(fv) in Eq. (5).
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.text_projection.shape[1]

        if cfg.TRAINER.COOP.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        self.use_atp = cfg.TRAINER.ATPROMPT.USE_ATPROMPT
        self.attr_specs = _get_attr_cfg(cfg) if self.use_atp else []
        self.attr_num = len(self.attr_specs)
        self.attr_token_counts = [spec[0] for spec in self.attr_specs]
        self.attr_text_lens = [len(_tokenizer.encode(spec[1])) for spec in self.attr_specs]
        print(f"Use attribute prompts: {self.use_atp}")
        print(f"Number of attributes: {self.attr_num}")

        if self.use_atp:
            prompts = []
            for name in classnames:
                # Template from the paper:
                # "A photo of a [CLS] with S1 [A1] ... SK [AK]".
                # CoOp keeps the class name near the suffix; the learnable
                # attribute tokens are materialized by meta-nets at runtime.
                parts = []
                for n_att, attr_text in self.attr_specs:
                    parts.append(" ".join(["X"] * n_att))
                    parts.append(attr_text)
                parts.extend([prompt_prefix, f"{name}."])
                prompts.append(" ".join(parts))
        else:
            prompts = [prompt_prefix + " " + name + "." for name in classnames]

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            print(f"embedding size is {embedding.size()}")

        self.register_buffer("token_prefix", embedding[:, :1, :])

        if self.use_atp:
            offset = 1
            for idx, ((n_att, _), attr_text_len) in enumerate(
                zip(self.attr_specs, self.attr_text_lens), start=1
            ):
                offset += n_att
                middle = embedding[:, offset : offset + attr_text_len, :]
                self.register_buffer(f"token_middle{idx}", middle)
                offset += attr_text_len

            self.register_buffer("token_suffix", embedding[:, offset + n_ctx :, :])
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

        hidden_dim = cfg.LOREAL.DIM
        self.metanets = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(visual_dim, hidden_dim)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(hidden_dim, ctx_dim)),
            ]))
            for _ in range(max(self.attr_num, 1))
        ]).to(dtype)

    def attribute_contexts(self, image_features):
        """Return the K generated attribute contexts used by LLD.

        Each item has shape [batch, M, Dt]. The same generated token is
        repeated M times because the paper uses M learnable tokens per
        attribute, and CoOp's text encoder expects token-level embeddings.
        """
        if not self.use_atp:
            return []
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        contexts = []
        for idx, n_att in enumerate(self.attr_token_counts):
            ctx = self.metanets[idx](image_features)
            ctx = ctx.unsqueeze(1).expand(-1, n_att, -1)
            contexts.append(ctx)
        return contexts

    def forward(self, image_features=None):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if not self.use_atp:
            return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

        if image_features is None:
            raise ValueError("Attribute prompts require image features")

        attr_contexts = self.attribute_contexts(image_features)
        batch_size = image_features.shape[0]

        # The prompt becomes image-conditioned: one prompt bank per image,
        # then one prompt per class inside that bank. CustomCLIP flattens this
        # [B, C, T, D] tensor before sending it through CLIP's text encoder.
        prefix = self.token_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        ctx = ctx.unsqueeze(0).expand(batch_size, -1, -1, -1)

        parts = [prefix]
        for idx, attr_ctx in enumerate(attr_contexts, start=1):
            attr_ctx = attr_ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
            middle = getattr(self, f"token_middle{idx}")
            middle = middle.unsqueeze(0).expand(batch_size, -1, -1, -1)
            parts.extend([attr_ctx, middle])
        parts.extend([ctx, suffix])

        return torch.cat(parts, dim=2)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def only_image_outputs(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features

    def forward(self, image, student_visual=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # LOREAL bridges the students across resolutions:
        # the visual feature from the other student fills this student's
        # attribute prompt slots. If no bridge feature is provided, inference
        # uses the current image feature, matching Fig. 4(c).
        prompts = self.prompt_learner(student_visual if student_visual is not None else image_features)
        logit_scale = self.logit_scale.exp()

        if prompts.dim() == 4:
            batch_size, n_cls, n_tokens, dim = prompts.shape
            prompts = prompts.reshape(batch_size * n_cls, n_tokens, dim)
            tokenized_prompts = self.tokenized_prompts.unsqueeze(0).expand(batch_size, -1, -1)
            tokenized_prompts = tokenized_prompts.reshape(batch_size * n_cls, -1)
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.reshape(batch_size, n_cls, -1)
            logits = logit_scale * torch.einsum("bd,bcd->bc", image_features, text_features)
        else:
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ text_features.t()

        return logits 
  
@TRAINER_REGISTRY.register()
class CoOp_LOREAL(TrainerX):
    """LOREAL self-distillation implemented on top of CoOp.

    This trainer demonstrates the CoOp example: load two pretrained CoOp
    prompt learners, share attribute meta-nets, and train the meta-nets with
    CE + HLD + LLD under LR inputs.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def _stage_dir(self, stage_name, config_name, to_size, seed):
        """Resolve the two CoOp pretraining checkpoints used by stage 3.

        LOREAL first pretrains two CoOp students separately:
        stage 1 uses standard-resolution inputs, stage 2 uses LR inputs.
        Stage 3 loads both prompt learners and trains only meta-nets.
        """
        cfg = self.cfg
        override = cfg.LOREAL.STAGE1_DIR if stage_name == "stage1" else cfg.LOREAL.STAGE2_DIR
        if override:
            return override

        method = cfg.TRAINER.NAME
        output_dir = osp.normpath(cfg.OUTPUT_DIR)
        stage3_tail = osp.join(
            f"{method}_stage3_students_sd",
            str(to_size),
            config_name,
            f"seed{seed}",
        )

        if output_dir.endswith(stage3_tail):
            base_dir = output_dir[: -len(stage3_tail)].rstrip(os.sep)
        else:
            dataset_name = self._dataset_key()
            base_dir = osp.join(
                "output",
                method,
                "base2new",
                "train_base",
                dataset_name,
            )

        if stage_name == "stage1":
            return osp.join(
                base_dir,
                f"{method}_stage1_students_pretraining_first",
                config_name,
                f"seed{seed}",
            )

        if stage_name == "stage2":
            return osp.join(
                base_dir,
                f"{method}_stage2_students_pretraining_second",
                str(to_size),
                config_name,
                f"seed{seed}",
            )

        raise ValueError(f"Unsupported stage name: {stage_name}")

    def _dataset_key(self):
        cfg = self.cfg
        name_map = {
            "FGVCAircraft": "fgvc_aircraft",
            "DescribableTextures": "dtd",
            "Caltech101": "caltech101",
            "EuroSAT": "eurosat",
            "Food101": "food101",
            "OxfordFlowers": "oxford_flowers",
            "StanfordCars": "stanford_cars",
            "UCF101": "ucf101",
            "SUN397": "sun397",
            "OxfordPets": "oxford_pets",
            "ImageNet": "imagenet",
            "ImageNetV2": "imagenetv2",
            "ImageNetSketch": "imagenet_sketch",
            "ImageNetA": "imagenet_a",
            "ImageNetR": "imagenet_r",
        }
        return cfg.LOREAL.SYUME if cfg.LOREAL.SYUME else name_map[cfg.DATASET.NAME]

    def _load_prompt_checkpoint(self, prompt_learner, directory, epoch):
        # Stage 3 starts from two independently trained CoOp prompt learners.
        # Token buffers depend on class names and are regenerated by the
        # current PromptLearner, so only learnable weights are loaded.
        model_path = osp.join(directory, "prompt_learner", f"model.pth.tar-{epoch}")
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if key.startswith("token_"):
                del state_dict[key]
        prompt_learner.load_state_dict(state_dict, strict=False)

    def _set_loreal_trainable(self, model):
        # Sec. 3.4 states that the VLM backbone is frozen and only meta-nets
        # are learnable during attribute-driven self-distillation.
        for name, param in model.named_parameters():
            param.requires_grad_("prompt_learner.metanets" in name)

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # --------------------------------------------------
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model) 
        TOSI = cfg.LOREAL.TOSIZE
        SEED = cfg.SEED 
        output_dir = osp.normpath(cfg.OUTPUT_DIR)
        CONFIG = (
            osp.basename(osp.dirname(output_dir))
            if osp.basename(output_dir) == f"seed{SEED}"
            else "vit_b16_ep50.yaml"
        )
        stage2_dir = self._stage_dir("stage2", CONFIG, TOSI, SEED)
        # Low-resolution student beta in the paper.
        self._load_prompt_checkpoint(self.model.prompt_learner, stage2_dir, cfg.OPTIM.MAX_EPOCH)
        self.model.to(self.device) 
        self._set_loreal_trainable(self.model)

        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS) 
        # ----------------------------------------------------




        # -------------------------------------------------- 
        clip_model_teacher = load_clip_to_cpu(cfg)
        self.model_teacher = CustomCLIP(cfg, classnames, clip_model_teacher) 
        self.train_modal = cfg.TRAINER.MODAL 
        stage1_dir = self._stage_dir("stage1", CONFIG, TOSI, SEED)
        # Standard-resolution student alpha in the paper.
        self._load_prompt_checkpoint(self.model_teacher.prompt_learner, stage1_dir, cfg.OPTIM.MAX_EPOCH)
        # Fig. 4 marks the meta-net as shared. Keep separate CoOp contexts for
        # the two pretrained students, but bind both prompt learners to the
        # exact same meta-net module so stage 3 optimizes one shared set.
        self.model_teacher.prompt_learner.metanets = self.model.prompt_learner.metanets
        self.model_teacher.to(self.device) 
        self._set_loreal_trainable(self.model_teacher)
        # ----------------------------------------------------
        
         

        # if "prompt_learner.token_prefix2" in state_dict:
        #     del state_dict["prompt_learner.token_prefix2"]  
        # if "prompt_learner.token_suffix" in state_dict:
        #     del state_dict["prompt_learner.token_suffix"]
        # if "prompt_learner.token_suffix2" in state_dict:
        #     del state_dict["prompt_learner.token_suffix2"] 

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        
        params = []
        seen_params = set()
        for prompt_learner in (self.model.prompt_learner, self.model_teacher.prompt_learner):
            for p in prompt_learner.parameters():
                if p.requires_grad and id(p) not in seen_params:
                    params.append(p)
                    seen_params.add(id(p))
        # The optimizer sees one deduplicated shared meta-net set. The frozen
        # CLIP encoders and pretrained CoOp contexts stay unchanged in stage 3.
        self.optim = build_optimizer(params, cfg.OPTIM) 
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched) 
        self.register_model("prompt_learner2", self.model_teacher.prompt_learner, None, None) 
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None 
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count() 
        self.temperature = cfg.LOREAL.TEMP # TRAINER.PROMPTKD.TEMPERATURE
  
        # Double check
        num_trainable_params = 0
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Parameters to be updated: {enabled}") 
        print(f"Number of trainable parameters: {num_trainable_params}")
  
    def low_level_distillation(self, contexts_a, contexts_b):
        """Eq. (7): contrastively align generated attribute contexts.

        Positives are the same attribute index across the standard-resolution
        and LR students. Negatives are the other attributes in the same sample.
        """
        if not contexts_a or not contexts_b:
            return torch.zeros([], device=self.device)

        ctx_a = torch.stack([ctx.mean(dim=1) for ctx in contexts_a], dim=1)
        ctx_b = torch.stack([ctx.mean(dim=1) for ctx in contexts_b], dim=1)
        ctx_a = F.normalize(ctx_a, dim=-1)
        ctx_b = F.normalize(ctx_b, dim=-1)

        logits = torch.einsum("bkd,bqd->bkq", ctx_a, ctx_b) / self.temperature
        batch_size, attr_num, _ = logits.shape
        labels = torch.arange(attr_num, device=logits.device).unsqueeze(0).expand(batch_size, -1)
        return F.cross_entropy(logits.reshape(batch_size * attr_num, attr_num), labels.reshape(-1))

    def aligned_mix_weight(self):
        """Schedule student conditioning toward the LR inference path.

        With the default start=0 and max=1, this is immediately equivalent to
        using the LR inference feature from the first epoch.
        """
        max_weight = float(self.cfg.LOREAL.ALIGN_MIX_MAX)
        if max_weight <= 0:
            return 0.0

        start = float(self.cfg.LOREAL.ALIGN_MIX_START)
        start = min(max(start, 0.0), 1.0)
        if start <= 0.0:
            return max_weight

        progress = (self.epoch + 1) / max(float(self.max_epoch), 1.0)
        if progress <= start:
            return 0.0

        denom = max(1.0 - start, 1e-6)
        return max_weight * min((progress - start) / denom, 1.0)

    def forward_backward(self, batch): 
        image, niimage, label = self.parse_batch_train(batch) 
        stu2 = self.model.only_image_outputs(niimage)
        stu1 = self.model_teacher.only_image_outputs(image)
        
        # Main path stays aligned with inference: teacher uses HR visual
        # semantics, student uses LR visual semantics.
        tea_logits = self.model_teacher(image, stu1)
        mix_w = self.aligned_mix_weight()
        student_cond = F.normalize((1.0 - mix_w) * stu1 + mix_w * stu2, dim=-1)
        output = self.model(niimage, student_cond)

        # Keep the original LOREAL cross-resolution cycle as a low-weight
        # regularizer, without letting it drive the supervised CE path.
        loss_bpd = torch.zeros([], device=self.device)
        if float(self.cfg.LOREAL.COEF_BPD) > 0:
            tea_bridge = self.model_teacher(image, stu2)
            output_bridge = self.model(niimage, stu1)
            loss_bpd = F.kl_div(
                F.log_softmax(output_bridge / self.temperature, dim=1),
                F.softmax(tea_bridge.detach() / self.temperature, dim=1),
                reduction="batchmean",
            ) * (self.temperature * self.temperature)

        # Final objective from Sec. 3.4:
        # L = LCE + lambda1 * LHLD + lambda2 * (1/K) * LLLD.
        # The implemented LLD is averaged by cross_entropy over B*K entries,
        # so LOREAL.COEF2 directly corresponds to lambda2.
        loss_ce = F.cross_entropy(output, label)
        loss_hld = self.cfg.TRAINER.PROMPTKD.KD_WEIGHT * F.kl_div(
            F.log_softmax(output / self.temperature, dim=1),
            F.softmax(tea_logits.detach() / self.temperature, dim=1),
            reduction="batchmean",
        ) * (self.temperature * self.temperature)   
        contexts_hr = self.model_teacher.prompt_learner.attribute_contexts(stu1)
        contexts_lr = self.model.prompt_learner.attribute_contexts(stu2)
        loss_lld = self.low_level_distillation(contexts_hr, contexts_lr)
        loss = (
            loss_ce
            + self.cfg.LOREAL.COEF1 * loss_hld
            + self.cfg.LOREAL.COEF2 * loss_lld
            + self.cfg.LOREAL.COEF_BPD * loss_bpd
        )
        
        self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_hld": loss_hld.item(),
            "loss_lld": loss_lld.item(),
            "loss_bpd": loss_bpd.item(),
            "mix_w": mix_w,
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
             
        return loss_summary

    def federated_avg(self): # self.distribute(idx)
        # Kept for backward compatibility with old scripts. The current
        # implementation follows Fig. 4 directly: both students reference the
        # same meta-net module, so there is nothing to average.
        return

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        niinput = batch["niimg"]
        input = input.to(self.device)
        niinput = niinput.to(self.device)
        label = label.to(self.device)
        return input, niinput, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            if epoch is not None and epoch < 0:
                all_model_files = os.listdir(osp.join(directory, name))
                all_model_files = [file_ for file_ in all_model_files if file_ != 'checkpoint']
                model_epochs = [int(file_.split('-')[-1]) for file_ in all_model_files]
                last_epoch = max(model_epochs)
                model_file = 'model.pth.tar-' + str(last_epoch)

            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors recomputed from the current class names.
            for key in list(state_dict.keys()):
                if key.startswith("token_"):
                    del state_dict[key]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False 
            self._models[name].load_state_dict(state_dict, strict=False)
