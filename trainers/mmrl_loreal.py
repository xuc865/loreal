import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss
import time
from tqdm import tqdm
import copy
from collections import OrderedDict
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from mmrlclip import clip
from mmrlclip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}


def load_clip_to_cpu(cfg, model_name="CLIP"):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"model": model_name,
                      "rep_tokens_layers": cfg.TRAINER.MMRL.REP_LAYERS,
                      "n_rep_tokens": cfg.TRAINER.MMRL.N_REP_TOKENS}
    model = clip.build_model_MMRL(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder_MMRL(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_rep_tokens_text):

        n_rep_tokens = compound_rep_tokens_text[0].shape[0]
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        eot_index = tokenized_prompts.argmax(dim=-1)
        combined = [x, compound_rep_tokens_text, 0, eot_index]  # third argument is the counter which denotes depth of representation tokens
        outputs = self.transformer(combined)

        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), eot_index + n_rep_tokens] @ self.text_projection
 
        
        return x


class TextEncoder_CLIP(nn.Module):
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
        outputs = self.transformer(x)

        x = outputs
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  
        return x


def _get_text_base_features_zero_shot(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device

    text_encoder = text_encoder.cuda()
    dataset = cfg.DATASET.NAME
    template = CUSTOM_TEMPLATES[dataset]

    with torch.no_grad():
        tokenized_prompts = []
        for text in tqdm(classnames, desc="Extracting text features"):
            tokens = clip.tokenize(template.format(text.replace('_', ' ')))  #(n_tokens)
            tokens = tokens.to(device)
            tokenized_prompts.append(tokens) 
        tokenized_prompts = torch.cat(tokenized_prompts) # (n_classes, n_tokens)  

        embeddings = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype) # (n_classes, n_tokens, embed_dim)
        outputs = text_encoder(embeddings.cuda(), tokenized_prompts.cuda()) 

        text_embeddings = outputs

    text_encoder = text_encoder.to(device)
    return text_embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiModalRepresentationLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__() 
        n_rep_tokens = cfg.TRAINER.MMRL.N_REP_TOKENS
        self.dtype = clip_model.dtype 
        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0] 
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE
        rep_dim = cfg.TRAINER.MMRL.REP_DIM 
        self.rep_layers_length = len(cfg.TRAINER.MMRL.REP_LAYERS)  # max=12
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
 
        self.compound_rep_tokens = nn.Parameter(torch.empty(n_rep_tokens, rep_dim))
        nn.init.normal_(self.compound_rep_tokens, std=0.02)

        single_layer_r2v = nn.Linear(rep_dim, visual_dim)
        single_layer_r2t = nn.Linear(rep_dim, text_dim)

        self.compound_rep_tokens_r2vproj = _get_clones(single_layer_r2v, self.rep_layers_length)
        self.compound_rep_tokens_r2tproj = _get_clones(single_layer_r2t, self.rep_layers_length)
      
        # dataset = cfg.DATASET.NAME
        # template = CUSTOM_TEMPLATES[dataset]
        # tokenized_prompts = []
        # for text in classnames:
        #     tokens = clip.tokenize(template.format(text.replace('_', ' ')))  # (n_tokens)
        #     tokenized_prompts.append(tokens)
        # self.tokenized_prompts = torch.cat(tokenized_prompts)  # (n_classes, n_tokens)
        # with torch.no_grad():
        #     self.prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype) # (n_classes, n_tokens, embed_dim)  
 
 
 
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
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
        prompt_prefix = " ".join(["X"] * n_ctx) 
        self.ctx = nn.Parameter(ctx_vectors) 
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
                self.register_buffer("token_suffix", embedding[:, 1+n_att1+1+n_att2+1+n_att3+1+n_ctx:, :])
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

    def contruct_prompts(self, image_features):
        ctx = self.ctx
        if self.use_atp:
            ctx_att1 = self.metanets[0](image_features).mean(dim=0,keepdim=True).repeat(self.anx, 1) # self.ctx_att1
            ctx_att2 = self.metanets[1](image_features).mean(dim=0,keepdim=True).repeat(self.anx, 1) # self.ctx_att2
            ctx_att3 = self.metanets[2](image_features).mean(dim=0,keepdim=True).repeat(self.anx, 1) # self.ctx_att3

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
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
                        ctx,
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
                        ctx,
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
                        ctx,     
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
                    ctx,     # (n_cls, n_ctx, dim) 
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        return prompts
 
    def forward(self, image_features=None):
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []

        prom = self.contruct_prompts(image_features)
        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            rep_mapped_to_text = self.compound_rep_tokens_r2tproj[index](rep_tokens)
            rep_mapped_to_visual = self.compound_rep_tokens_r2vproj[index](rep_tokens)                        
            compound_rep_tokens_text.append(rep_mapped_to_text.type(self.dtype))
            compound_rep_tokens_visual.append(rep_mapped_to_visual.type(self.dtype))      

        return compound_rep_tokens_text, compound_rep_tokens_visual, prom


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.alpha = cfg.TRAINER.MMRL.ALPHA
        self.classnames = classnames
        self.representation_learner = MultiModalRepresentationLearner(cfg, classnames, clip_model).type(clip_model.dtype)
        self.tokenized_prompts = self.representation_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_MMRL(clip_model)
        self.dtype = clip_model.dtype
        self.text_features_for_inference = None
        self.compound_rep_tokens_text_for_inference = None
        self.compound_rep_tokens_visual_for_inference = None

    def only_image_outputs(self, image):
        with torch.no_grad():
            if self.representation_learner.training:
                compound_rep_tokens_text, compound_rep_tokens_visual, prompt_embeddings = self.representation_learner()
                text_features = self.text_encoder(prompt_embeddings, self.tokenized_prompts, compound_rep_tokens_text)
            else:
                if self.text_features_for_inference is None:
                    self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference, prompt_embeddings = self.representation_learner()
                    self.text_features_for_inference = self.text_encoder(prompt_embeddings, self.tokenized_prompts, self.compound_rep_tokens_text_for_inference)

                compound_rep_tokens_text, compound_rep_tokens_visual = self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference
                text_features = self.text_features_for_inference

            image_features, image_features_rep = self.image_encoder([image.type(self.dtype), compound_rep_tokens_visual])
        
            alpha = self.alpha
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features_rep = image_features_rep / image_features_rep.norm(dim=-1, keepdim=True)  
            image_featuresa = alpha * image_features + (1 - alpha) * image_features_rep
            return image_featuresa 

    def forward(self, image, stu=None):
        
        if self.representation_learner.training:
            compound_rep_tokens_text, compound_rep_tokens_visual, prompt_embeddings = self.representation_learner(stu)
            text_features = self.text_encoder(prompt_embeddings, self.tokenized_prompts, compound_rep_tokens_text)
        else:
            if self.text_features_for_inference is None:
                self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference, prompt_embeddings = self.representation_learner(stu)
                self.text_features_for_inference = self.text_encoder(prompt_embeddings, self.tokenized_prompts, self.compound_rep_tokens_text_for_inference)

            compound_rep_tokens_text, compound_rep_tokens_visual = self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference
            text_features = self.text_features_for_inference

        image_features, image_features_rep = self.image_encoder([image.type(self.dtype), compound_rep_tokens_visual])
    
        alpha = self.alpha
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_rep = image_features_rep / image_features_rep.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100. * image_features @ text_features.t()
        logits_rep = 100. * image_features_rep @ text_features.t()
        logits_fusion = alpha * logits + (1 - alpha) * logits_rep

        return logits, logits_rep, logits_fusion, image_features, text_features


class MMRL_Loss(_Loss):
    def __init__(self, reg_weight=1.0, alpha=0.7):
        super(MMRL_Loss, self).__init__()
        self.reg_weight = reg_weight
        self.alpha = alpha 

    def forward(self, logits, logits_rep,
                image_features, text_features, 
                image_features_clip, text_features_clip, 
                label):
    
        xe_loss1 = F.cross_entropy(logits, label)
        xe_loss2 = F.cross_entropy(logits_rep, label)

        cossim_reg_img = 1 - torch.mean(F.cosine_similarity(image_features, image_features_clip, dim=1))
        cossim_reg_text = 1 - torch.mean(F.cosine_similarity(text_features, text_features_clip, dim=1))

        return self.alpha * xe_loss1 + (1-self.alpha) * xe_loss2 +  + self.reg_weight * cossim_reg_img + self.reg_weight * cossim_reg_text


import torch, gc
device = torch.device('cuda')
  
@TRAINER_REGISTRY.register()
class MMRL_LOREAL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMRL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)
        
        self.image_encoder_clip = clip_model_zero_shot.visual  
        self.image_encoder_clip.to(self.device)    

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, "MMRL")
        clip_model_zero_shot = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MMRL.PREC == "fp32" or cfg.TRAINER.MMRL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            clip_model_zero_shot.float()

        self.dtype = clip_model.dtype 
        with torch.no_grad():
            self.text_encoder_clip = TextEncoder_CLIP(clip_model_zero_shot)
            text_features_clip = _get_text_base_features_zero_shot(cfg, classnames, clip_model_zero_shot, self.text_encoder_clip)
            self.text_features_clip = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)
        



        # print("Building custom CLIP")
        # self.model = CustomCLIP(cfg, classnames, clip_model) 
        # print("Turning off gradients in both the image and the text encoder")
        # names_to_update = ["representation_learner", "image_encoder.proj_rep"] 
        # for name, param in self.model.named_parameters():
        #     update = False 
        #     for name_to_update in names_to_update:
        #         if name_to_update in name:
        #             update = True
        #             break
        #     param.requires_grad_(update) 
        # # Double check
        # enabled = set()
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         enabled.add(name)
        # print(f"Parameters to be updated: {enabled}") 
        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS) 
        # self.model.to(self.device)
        
        
        # --------------------------------------------------
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model) 
        METHOD = cfg.TRAINER.NAME
        map = {"FGVCAircraft":"fgvc_aircraft","DescribableTextures":"dtd","Caltech101":"caltech101","EuroSAT":"eurosat", 
            "Food101":"food101","OxfordFlowers":"oxford_flowers","StanfordCars":"stanford_cars","UCF101":"ucf101","SUN397":"sun397",
            "OxfordPets":"oxford_pets"}
        DATASET = map[cfg.DATASET.NAME] # 
        CONFIG = "vit_b16.yaml"
        TOSI = cfg.LOREAL.TOSIZE
        SEED = cfg.SEED 
        model_path = f"PATH/output/{METHOD}/base2new/train_base/{DATASET}/{METHOD}_stage2_students_pretraining_second/{TOSI}/{CONFIG}/seed{SEED}/prompt_learner/model.pth.tar-{cfg.OPTIM.MAX_EPOCH}"
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]  
        if "token_prefix" in state_dict: # coop does not need this
            del state_dict["token_prefix"]
        if "token_suffix" in state_dict: # coop does not need this
            del state_dict["token_suffix"]
            
        self.model.prompt_learner.load_state_dict(state_dict, strict=False)
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
        model_path = f"PATH/output/{METHOD}/base2new/train_base/{DATASET}/{METHOD}_stage1_students_pretraining_first/{CONFIG}/seed{SEED}/prompt_learner/model.pth.tar-{cfg.OPTIM.MAX_EPOCH}"
        self.train_modal = cfg.TRAINER.MODAL 
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]  
        if "token_prefix" in state_dict: # coop does not need this
            del state_dict["token_prefix"]
        if "token_suffix" in state_dict: # coop does not need this
            del state_dict["token_suffix"]
            
        self.model_teacher.prompt_learner.load_state_dict(state_dict, strict=False)
        self.model_teacher.to(self.device) 
        for name, param in self.model_teacher.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        # ----------------------------------------------------

         
        reg_weight = cfg.TRAINER.MMRL.REG_WEIGHT
        alpha = cfg.TRAINER.MMRL.ALPHA
        self.criterion = MMRL_Loss(reg_weight=reg_weight, alpha=alpha) 
        # NOTE: only give representation_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched) 
        self.scaler = GradScaler() if cfg.TRAINER.MMRL.PREC == "amp" else None 

    def forward_backward(self, batch):
        image, niimage, label = self.parse_batch_train(batch) 
        stu2 = self.model.only_image_outputs(niimage)
        stu1 = self.model_teacher.only_image_outputs(image)
        
        tea_logits = self.model_teacher(image, stu2)  
        output = self.model(niimage, stu1)
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

 
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        niinput = batch["niimg"] 
        niinput = niinput.to(self.device)
        input = input.to(self.device)
        label = label.to(self.device)
        return input, niinput, label

    def parse_batch_sas(self, batch):
        input = batch["img"]
        label = batch["label"]
        niinput = batch["niimg"]
        input = input.to(self.device)
        niinput = niinput.to(self.device)
        label = label.to(self.device)
        return input, niinput, label

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        sub_cls = self.cfg.DATASET.SUBSAMPLE_CLASSES
        dataset = self.cfg.DATASET.NAME
        task = self.cfg.TASK

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")  
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            # input, label = self.parse_batch_test(batch)
            ta = time.time()
            input, nis, label = self.parse_batch_sas(batch)
            logits, _, logits_fusion, _, _, = self.model(nis)
 
            if task == "B2N":
                output = logits_fusion if sub_cls == "base" else logits
            elif task == "FS":
                output = logits_fusion
            elif task == "CD":
                output = logits_fusion if dataset == "ImageNet" else logits
            else:
                raise ValueError("The TASK must be either B2N, CD, or FS.")

            self.evaluator.process(output, label) 
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        # model_file = 'model-best.pth.tar'

        # if epoch is not None:
        #     model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            #model_path = osp.join(directory, name, model_file)
            model_path_prefix = osp.join(directory, name)
            if not osp.exists(model_path_prefix):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path_prefix)
                )
            for file in os.listdir(model_path_prefix):
                if "model-best.pth" in file:
                    model_path = osp.join(model_path_prefix, file)
                    break
                if "model.pth" in file:
                    model_path = osp.join(model_path_prefix, file)
 
            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )            


            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            state_dict = {k: v for k, v in state_dict.items() if "prompt_embeddings" not in k}

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict: 
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_middle1" in state_dict:
                del state_dict["prompt_learner.token_middle1"]
            if "prompt_learner.token_middle2" in state_dict:
                del state_dict["prompt_learner.token_middle2"]
            if "prompt_learner.token_middle3" in state_dict:
                del state_dict["prompt_learner.token_middle3"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)