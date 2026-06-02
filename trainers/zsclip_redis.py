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
import pickle
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import os
import pdb
_tokenizer = _Tokenizer()
from collections import OrderedDict

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
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX 
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})" 
        # random initialization
        if cfg.TRAINER.COOP.CSC:  # usually false
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02) 
        # use given words to initialize context vectors
        ctx_init = "A_photo_of_a".replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefix = ctx_init 
  
        self.use_atp = cfg.TRAINER.ATPROMPT.USE_ATPROMPT
        self.atp_num = cfg.TRAINER.ATPROMPT.ATT_NUM 
        print(f'self.use_atp is {self.use_atp}')
        print(f'self.atp_num is {self.atp_num}')

        prompts = [prompt_prefix + " " + name + "." for name in classnames] 
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}") 
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        if self.use_atp:
            print("USE ATPROPMT-ING 1")
            n_att1 = cfg.TRAINER.ATPROMPT.N_ATT1
            att1_text = cfg.TRAINER.ATPROMPT.ATT1_TEXT

            n_att2 = cfg.TRAINER.ATPROMPT.N_ATT2
            att2_text = cfg.TRAINER.ATPROMPT.ATT2_TEXT
            
            n_att3 = cfg.TRAINER.ATPROMPT.N_ATT3
            att3_text = cfg.TRAINER.ATPROMPT.ATT3_TEXT

            att_vectors_1 = torch.empty(n_att1, ctx_dim, dtype=dtype)
            att_vectors_2 = torch.empty(n_att2, ctx_dim, dtype=dtype)
            att_vectors_3 = torch.empty(n_att3, ctx_dim, dtype=dtype)

            nn.init.normal_(att_vectors_1, std=0.01)
            prefix1 = " ".join(["X"] * n_att1)
            nn.init.normal_(att_vectors_2, std=0.01)
            prefix2 = " ".join(["X"] * n_att2)
            nn.init.normal_(att_vectors_3, std=0.01)
            prefix3 = " ".join(["X"] * n_att3)
            
            self.ctx_att1 = nn.Parameter(att_vectors_1)
            self.ctx_att2 = nn.Parameter(att_vectors_2)
            self.ctx_att3 = nn.Parameter(att_vectors_3)

            # print(f'Attribute Num is {self.atp_num}')
            if self.atp_num == 1:
                prompts = [prefix1 + " " + att1_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            elif self.atp_num == 2:
                prompts = [prefix1 + " " + att1_text + " " + prefix2 + " " + att2_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            elif self.atp_num == 3:
                prompts = [prefix1 + " " + att1_text + " " + prefix2 + " " + att2_text + " " + prefix3 + " " + att3_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            else:
                print("wrong parameter.")
                raise ValueError 
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            print(f'embedding size is {embedding.size()}')
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS [102, 0, 512]
        self.anx = cfg.TRAINER.ATPROMPT.N_ATT1

        if self.use_atp:
            print("USE ATPROPMT-ING 2")
            if self.atp_num == 1:
                self.register_buffer("token_middle1", embedding[:, 1+n_att1 : 1+n_att1+1, :])
                self.register_buffer("token_suffix", embedding[:, 1+n_att1+1+n_ctx :, :])

            elif self.atp_num == 2:
                self.register_buffer("token_middle1", embedding[:, 1+n_att1 : n_att1+1+1, :])
                self.register_buffer("token_middle2", embedding[:, 1+n_att1+1+n_att2 : 1+n_att1+1+n_att2+1, :])
                self.register_buffer("token_suffix", embedding[:, 1+n_att1+1+n_att2+1+n_ctx :, :])

            elif self.atp_num == 3:
                self.register_buffer("token_middle1", embedding[:, 1+n_att1 : n_att1+1+1, :])
                self.register_buffer("token_middle2", embedding[:, 1+n_att1+1+n_att2 : 1+n_att1+1+n_att2+1, :])
                self.register_buffer("token_middle3", embedding[:, 1+n_att1+1+n_att2+1+n_att3 : 1+n_att1+1+n_att2+1+n_att3+1, :])
                self.register_buffer("token_suffix", embedding[:, 1+n_att1+1+n_att2+1+n_att3+1:, :])
            else:
                raise ValueError 
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
            
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        # three-layer metanet
        self.metanets = nn.ModuleList([
            nn.Sequential(OrderedDict([
                        ("linear1", nn.Linear(512, 512 // 16)),
                        ("relu", nn.ReLU(inplace=True)),
                        ("linear2", nn.Linear(512 // 16, ctx_dim))
                    ])) for i in range(3)]).to(dtype)

    def forward(self, image_features): 
        if self.use_atp:
            ctx_att1 = self.metanets[0](image_features).mean(dim=0,keepdim=True).repeat(self.anx, 1) # self.ctx_att1
            ctx_att2 = self.metanets[1](image_features).mean(dim=0,keepdim=True).repeat(self.anx, 1) # self.ctx_att2
            ctx_att3 = self.metanets[2](image_features).mean(dim=0,keepdim=True).repeat(self.anx, 1) # self.ctx_att3
  
        if self.use_atp: 
            ctx_att1 = ctx_att1.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_att2 = ctx_att2.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_att3 = ctx_att3.unsqueeze(0).expand(self.n_cls, -1, -1)
             
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if self.use_atp:
            if self.atp_num == 1:
                middle_attribute1 = self.token_middle1
                prompts = torch.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1, 
                        suffix,
                    ],
                    dim=1,
                )
            elif self.atp_num == 2:
                middle_attribute1 = self.token_middle1
                middle_attribute2 = self.token_middle2
                prompts = torch.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1,
                        ctx_att2,
                        middle_attribute2, 
                        suffix,
                    ],
                    dim=1,
                )
            elif self.atp_num == 3:
                middle_attribute1 = self.token_middle1
                middle_attribute2 = self.token_middle2
                middle_attribute3 = self.token_middle3
                prompts = torch.cat(
                    [
                        prefix, 
                        ctx_att1,
                        middle_attribute1, 
                        ctx_att2,
                        middle_attribute2,
                        ctx_att3,
                        middle_attribute3, 
                        suffix, 
                    ],
                    dim=1,
                )
            else:
                raise ValueError
        else:
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim) 
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        with open("PATH-main/wars.pkl","wb") as f:
            aa = prompts
            pickle.dump([aa.detach(), self.tokenized_prompts], f)
        f.close()
        pdb.set_trace()

        return prompts


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

        prompts = self.prompt_learner(student_visual if student_visual is not None else image_features)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts) 
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits 
  
@TRAINER_REGISTRY.register()
class ZeroshotCLIP_REDIS(TrainerX): 
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

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
        METHOD = cfg.TRAINER.NAME
        map = {"FGVCAircraft":"fgvc_aircraft","DescribableTextures":"dtd","Caltech101":"caltech101","EuroSAT":"eurosat", 
            "Food101":"food101","OxfordFlowers":"oxford_flowers","StanfordCars":"stanford_cars","UCF101":"ucf101","SUN397":"sun397",
            "OxfordPets":"oxford_pets","ImageNet":"imagenet"}
        DATASET = map[cfg.DATASET.NAME] # 
        CONFIG = "vit_b16_ep50.yaml"
        TOSI = cfg.POW.TOSIZE
        SEED = cfg.SEED 
        # model_path = f"PATH/output/{METHOD}/base2new/train_base/{DATASET}/{METHOD}_stage2_students_pretraining_second/{TOSI}/{CONFIG}/seed{SEED}/prompt_learner/model.pth.tar-{cfg.OPTIM.MAX_EPOCH}"
        # checkpoint = load_checkpoint(model_path)
        # state_dict = checkpoint["state_dict"]  
        # if "token_prefix" in state_dict: # coop does not need this
        #     del state_dict["token_prefix"]
        # if "token_suffix" in state_dict: # coop does not need this
        #     del state_dict["token_suffix"]
            
        # self.model.prompt_learner.load_state_dict(state_dict, strict=False)
        self.model.to(self.device) 
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS) 
        # ----------------------------------------------------




        # -------------------------------------------------- 
        clip_model_teacher = load_clip_to_cpu(cfg)
        self.model_teacher = CustomCLIP(cfg, classnames, clip_model_teacher) 
        # model_path = f"PATH/output/{METHOD}/base2new/train_base/{DATASET}/{METHOD}_stage1_students_pretraining_first/{CONFIG}/seed{SEED}/prompt_learner/model.pth.tar-{cfg.OPTIM.MAX_EPOCH}"
        # self.train_modal = cfg.TRAINER.MODAL 
        # checkpoint = load_checkpoint(model_path)
        # state_dict = checkpoint["state_dict"]  
        # if "token_prefix" in state_dict: # coop does not need this
        #     del state_dict["token_prefix"]
        # if "token_suffix" in state_dict: # coop does not need this
        #     del state_dict["token_suffix"]
            
        # self.model_teacher.prompt_learner.load_state_dict(state_dict, strict=False)
        self.model_teacher.to(self.device) 
        for name, param in self.model_teacher.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        # ----------------------------------------------------
        
         

        # if "prompt_learner.token_prefix2" in state_dict:
        #     del state_dict["prompt_learner.token_prefix2"]  
        # if "prompt_learner.token_suffix" in state_dict:
        #     del state_dict["prompt_learner.token_suffix"]
        # if "prompt_learner.token_suffix2" in state_dict:
        #     del state_dict["prompt_learner.token_suffix2"] 

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        
        # list(model1.parameters()) + list(model2.parameters())
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim = build_optimizer(list(self.model.prompt_learner.parameters())+list(self.model_teacher.prompt_learner.parameters()), cfg.OPTIM) 
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched) 
        self.register_model("prompt_learner2", self.model_teacher.prompt_learner, self.optim, self.sched) 
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None 
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count() 
        self.temperature = cfg.TRAINER.PROMPTKD.TEMPERATURE
  
    def forward_backward(self, batch): 
        image, niimage, label = self.parse_batch_train(batch) 
        stu2 = self.model.only_image_outputs(niimage)
        stu1 = self.model_teacher.only_image_outputs(image)
        
        tea_logits = self.model_teacher(image, stu2)  
        output = self.model(image, stu1)
        loss = F.cross_entropy(output, label) 
        loss += F.pairwise_distance(stu1, stu2).mean()
        
        # lossu = self.cfg.TRAINER.PROMPTKD.KD_WEIGHT * F.kl_div(
        #     F.log_softmax(output / self.temperature, dim=1),
        #     F.softmax(tea_logits.detach() / self.temperature, dim=1),
        #     reduction='sum',
        # ) * (self.temperature * self.temperature)  
        # loss += lossu
        
        self.model_backward_and_update(loss)
        self.federated_avg()
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
             
        return loss_summary

    def federated_avg(self): # self.distribute(idx)
        import copy
        w_glob = None 
        w_local1 = self.model.prompt_learner.state_dict()
        w_local2 = self.model_teacher.prompt_learner.state_dict()
        w_glob = copy.deepcopy(w_local1)
        for k in w_glob.keys():
            w_glob[k] += w_local2[k] 
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], 2) 
        self.model.prompt_learner.load_state_dict(w_glob, strict=False)
        self.model_teacher.prompt_learner.load_state_dict(w_glob, strict=False)

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
            if epoch < 0:
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

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_middle1" in state_dict:
                del state_dict["token_middle1"]
            if "token_middle2" in state_dict:
                del state_dict["token_middle2"]
            if "token_middle3" in state_dict:
                del state_dict["token_middle3"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False 
            self._models[name].load_state_dict(state_dict, strict=False)