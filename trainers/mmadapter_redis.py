from collections import OrderedDict
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from mmaclip import clip

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, retrun_adapater_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if retrun_adapater_func == None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, retrun_adapater_func])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
    
class AdapterLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.n_cls = len(classnames)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self._build_text_embedding(cfg, classnames, clip_model)

        # build multi-modal adapter
        self.text_adapter_func = lambda x: self.return_text_adapter(index=x)
        self.text_adapter = self._build_adapter(
            clip_model.ln_final.weight.shape[0], 
            len(clip_model.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype
        )
        
        self.visual_adapter_func = lambda x: self.return_visual_adapter(index=x)
        self.visual_adapter = self._build_adapter(
            clip_model.visual.ln_post.weight.shape[0],
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype
        )

        self.shared_adapter = self._build_adapter(
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype
        )
        self.adapter_scale = float(cfg.TRAINER.MMADAPTER.ADAPTER_SCALE)

    def return_text_adapter(self, index):
        return self.text_adapter[index], self.shared_adapter[index], self.adapter_scale

    def return_visual_adapter(self, index):
        return self.visual_adapter[index], self.shared_adapter[index], self.adapter_scale


    def _build_text_embedding(self, cfg, classnames, clip_model):
        # dtype = clip_model.dtype
        # text_ctx_init = cfg.TRAINER.MMADAPTER.TEXT_CTX_INIT 
        # classnames = [name.replace("_", " ") for name in classnames]
        # prompts = [text_ctx_init + " " + name + "." for name in classnames]
        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        # self.register_buffer("token_embedding", embedding)
        # self.register_buffer("tokenized_prompts", tokenized_prompts)

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
         
    def contruct_prompts(self, image_features=None):
        ctx = self.ctx
        if self.use_atp:
            if image_features is not None:
                ctx_att1 = self.metanets[0](image_features).mean(dim=0,keepdim=True).repeat(self.anx, 1) # self.ctx_att1
                ctx_att2 = self.metanets[1](image_features).mean(dim=0,keepdim=True).repeat(self.anx, 1) # self.ctx_att2
                ctx_att3 = self.metanets[2](image_features).mean(dim=0,keepdim=True).repeat(self.anx, 1) # self.ctx_att3
            else:
                ctx_att1 = self.ctx_att1
                ctx_att2 = self.ctx_att2
                ctx_att3 = self.ctx_att3

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

         
    def _build_adapter(self, d_model, n_layers, l_start, l_end, mid_dim, dtype):

        adapter = [None] * (n_layers + 1)
        for i in range(l_start, l_end+1):
            if mid_dim == d_model:
                adapter[i] = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU()
                )
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(nn.Linear(d_model, mid_dim), nn.ReLU())),
                    ("up", nn.Linear(mid_dim, d_model))
                ]))
        adapter = nn.ModuleList([a for a in adapter])
        for m in adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        if dtype == torch.float16:
            for m in adapter.modules():
                m.half()
    
        return adapter
    
    def forward(self, image_features=None):
        embedding = self.contruct_prompts(image_features)
        if self.text_adapter[0] is not None:
            token_embedding = self.text_adapter[0].down(embedding)
            shared_adapter = self.shared_adapter[0]
            token_embedding = shared_adapter(token_embedding)
            token_embedding = self.text_adapter[0].up(token_embedding)
            embedding = embedding + self.adapter_scale * token_embedding
        return embedding, self.text_adapter_func, self.visual_adapter_func

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.prompt_learner = AdapterLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.text_features_for_inference = None

    def encode_text(self, prompts, tokenized_prompts, text_adapter_func=None):
        if text_adapter_func is not None:
            text_features = self.text_encoder(
                prompts, tokenized_prompts, text_adapter_func
            )
        else:
            text_features = self.text_encoder(
                prompts, tokenized_prompts
            )
        return text_features
    
    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            image_features = self.image_encoder(
                [image.type(self.dtype), visual_adapter_func]
            )
        else:
            image_features = self.image_encoder(
                image.type(self.dtype)
            )
        return image_features


    def only_image_outputs(self, image):
        with torch.no_grad():
            token_embedding, text_adapter_func, visual_adapter_func = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts 
            if self.prompt_learner.training:
                text_features = self.encode_text(
                    token_embedding, tokenized_prompts, text_adapter_func
                )
            else:
                if self.text_features_for_inference is None:
                    self.text_features_for_inference = self.encode_text(
                        token_embedding, tokenized_prompts, text_adapter_func
                    )   
                text_features = self.text_features_for_inference 
            image_features = self.encode_image(image, visual_adapter_func) 
            image_features = F.normalize(image_features, dim=-1) 
            return image_features

    def forward(self, image, stus=None):
        token_embedding, text_adapter_func, visual_adapter_func = self.prompt_learner(stus)
        tokenized_prompts = self.tokenized_prompts

        if self.prompt_learner.training:
            text_features = self.encode_text(
                token_embedding, tokenized_prompts, text_adapter_func
            )
        else:
            if self.text_features_for_inference is None:
                self.text_features_for_inference = self.encode_text(
                    token_embedding, tokenized_prompts, text_adapter_func
                )   
            text_features = self.text_features_for_inference

        image_features = self.encode_image(image, visual_adapter_func)

        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class MultiModalAdapter_REDIS(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMADAPTER.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MMADAPTER.PREC == "fp32" or cfg.TRAINER.MMADAPTER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()


        # print("Building custom CLIP")
        # self.model = CustomCLIP(cfg, classnames, clip_model) 
        # print("Turning off gradients in both the image and the text encoder") 
        # for name, param in self.model.named_parameters():
        #     if "text_adapter" not in name and "visual_adapter" not in name and "shared_adapter" not in name:
        #         param.requires_grad_(False) 
        # # Double check
        # num_trainable_params = 0
        # enabled = set()
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         enabled.add(name)
        #         num_trainable_params += param.data.nelement()
        # print(f"Parameters to be updated: {enabled}") 
        # print(f"Number of trainable parameters: {num_trainable_params}") 


        # --------------------------------------------------
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model) 
        METHOD = cfg.TRAINER.NAME
        map = {"FGVCAircraft":"fgvc_aircraft","DescribableTextures":"dtd","Caltech101":"caltech101","EuroSAT":"eurosat", 
            "Food101":"food101","OxfordFlowers":"oxford_flowers","StanfordCars":"stanford_cars","UCF101":"ucf101","SUN397":"sun397",
            "OxfordPets":"oxford_pets"}
        DATASET = map[cfg.DATASET.NAME]  
        CONFIG = "vit_b16_ep5.yaml"
        TOSI = cfg.POW.TOSIZE
        SEED = cfg.SEED 
        model_path = f"PATH/output/{METHOD}/base2new/train_base/{DATASET}/{METHOD}_stage2_students_pretraining_second/{TOSI}/{CONFIG}/seed{SEED}/prompt_learner/model.pth.tar-{cfg.OPTIM.MAX_EPOCH}"
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]  
        if "token_prefix" in state_dict: 
            del state_dict["token_prefix"]
        if "token_suffix" in state_dict: 
            del state_dict["token_suffix"]
            
        self.model.prompt_learner.load_state_dict(state_dict, strict=False)
        self.model.to(self.device) 
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False) 
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
 
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched) 
        self.scaler = GradScaler() if cfg.TRAINER.MMADAPTER.PREC == "amp" else None
 

    def forward_backward(self, batch):
        image, niimage, label = self.parse_batch_train(batch) 
        stu2 = self.model.only_image_outputs(niimage)
        stu1 = self.model_teacher.only_image_outputs(image)
        
        tea_logits = self.model_teacher(image, stu2)  
        output = self.model(niimage, stu1)
        loss = F.cross_entropy(output, label) 
        loss += F.pairwise_distance(stu1, stu2).mean() 
        
        self.model_backward_and_update(loss)
        self.federated_avg()
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
             
        return loss_summary

    # def parse_batch_train(self, batch):
    #     input = batch["img"]
    #     label = batch["label"]
    #     input = input.to(self.device)
    #     label = label.to(self.device)
    #     return input, label

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        niinput = batch["niimg"] 
        niinput = niinput.to(self.device)
        input = input.to(self.device)
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