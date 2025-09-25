import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from einops import rearrange
#from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from peft import LoraConfig, TaskType, get_peft_model

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x) # x（4,1，128，64），（4,1，128*64） ，
        x = self.linear(x) # （8192，96），（4,1，96）
        x = self.dropout(x)
        return x

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class MFFN(nn.Module):
    
    def __init__(self, configs, device):
        super(MFFN, self).__init__()
        self.is_gpt = configs.is_gpt
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.top_k = 5
        self.d_ff = configs.d_ff
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_nums = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_nums += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))

        peft_config = LoraConfig(
            # task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
        self.gpt2 = get_peft_model(self.gpt2, peft_config)

        print_trainable_parameters(self.gpt2)

        #self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        #self.description = 'Weather is recorded every 10 minutes for the 2020 whole year, which contains 21 meteorological indicators, such as air temperature, humidity, etc..'
        self.description = 'The QBO dataset, sourced from ERA5 reanalysis, documents equatorial stratospheric zonal wind speeds and geopotential heights since 1959. With hourly global observations, it preserves the ~26-month quasi-biennial oscillation cycle—a critical feature for studying stratospheric dynamics and their influence on tropospheric weather systems.'


        self.mlp = nn.Sequential(nn.Linear(configs.patch_size, configs.d_model),
                                 nn.LayerNorm(configs.d_model),
                                 nn.ReLU(),
                                 nn.Linear(configs.d_model, configs.d_model),
                                 nn.LayerNorm(configs.d_model),
                                 nn.ReLU(),
                                 nn.Linear(configs.d_model, configs.d_model),
                                 nn.LayerNorm(configs.d_model),
                                 nn.ReLU(),)

        #self.final_projection = nn.Sequential(nn.Linear(configs.d_model, configs.d_model),
                                #nn.LayerNorm(configs.d_model))

        # Position-wise Feed-Forward
        self.in_layer = nn.Sequential(nn.Linear(configs.patch_size, configs.d_model),
                                      nn.GELU(),
                                      nn.Linear(configs.d_model, configs.d_model))


        #self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_nums, configs.pred_len) #(768*64,96)

        self.head_nf = self.d_ff * self.patch_nums


        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)


        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name or 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.output_projection, self.mlp):
        #for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        #if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec) # x_enc（4, 512, 1）
        return dec_out[:, -self.pred_len:, :] #（4,96，1）
        #return None


    def forecast(self, x, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x.shape #(4,512,1)

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        B, T, N = x.size()  # （4, 512, 1）
        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  # （4, 512, 1）

        min_values = torch.min(x, dim=1)[0]
        max_values = torch.max(x, dim=1)[0]
        medians = torch.median(x, dim=1).values
        lags = self.calcute_lags(x)
        trends = x.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x = x.reshape(B, N, T).permute(0, 2, 1).contiguous()  # （4, 512, 1）
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,max_length=2048).input_ids  # （4, 126）
        prompt_embeddings = self.gpt2.get_input_embeddings()(prompt.to(x.device))  # (batch, prompt_token, dim) # （4, 126，768）

        x = rearrange(x, 'b l m -> b m l') #(4,1,512)
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #4,1, 64, 16
        x = rearrange(x, 'b m n p -> (b m) n p') #(4,64,16)


        outputs = self.mlp(x)#(4,64,768)
        #x = torch.max(x, -1)[0]
        #outputs = self.final_projection(x)
        #outputs = self.in_layer(x)
        outputs = torch.cat([prompt_embeddings, outputs], dim=1)  # （4, 190，768）

        dec_out = self.gpt2(inputs_embeds=outputs).last_hidden_state # （4, 190，768）
        #outputs = outputs[:, :-self.patch_num, :] # （4, 126，768）

        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))  # （4,1， 197，128）
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()  # （4,1， 128，197）

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])  # （4,1， 128，64），（4,1，96）
        dec_out = dec_out.permute(0, 2, 1).contiguous()  # （4,96，1）

        dec_out = dec_out * stdev
        outputs = dec_out + means

        return outputs
