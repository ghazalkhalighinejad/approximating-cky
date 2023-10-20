import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys

from transformers import BertConfig, BertModel, AutoConfig, AutoModel, BertForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast
from models.ndr import NDREncoder


logger = logging.getLogger(__name__)

class PretrainedThing(nn.Module):
    def __init__(self, nnt, args, pretrained_with_mlm = None):
        super().__init__()
        self.nnt = nnt
        self.args = args
        self.pretrained_with_mlm = pretrained_with_mlm
        self.mlr = args.mlr # if mlr, add a final NT which is a dummy one
        self.enclosed = not args.no_enclose
        self.type = args.model_type
        conf = AutoConfig.from_pretrained(args.pretrained_model)
        if pretrained_with_mlm:
            conf.pad_token_id = 1759
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained("saved_models/checkpoint-307500/tokenizer.json")
            model = BertForMaskedLM(conf)
            model.resize_token_embeddings(len(self.tokenizer))
            sstate = torch.load(args.pretrained_model_file+'/pytorch_model.bin')
            model.load_state_dict(sstate)
            self.encoder = model.bert
            hidden_size , layer_norm_eps = conf.hidden_size, conf.layer_norm_eps

        elif args.from_scratch:
            if self.type == 'transformer':
                for key in ['position_embedding_type', 'hidden_size', 'intermediate_size',
                            'num_hidden_layers', 'hidden_dropout_prob', 'attention_probs_dropout_prob',
                            'num_attention_heads']:
                    if hasattr(args, key) and args.__dict__[key] is not None:
                        conf.__dict__[key] = args.__dict__[key]
                self.encoder = BertModel(conf)   
                hidden_size , layer_norm_eps = conf.hidden_size, conf.layer_norm_eps
                
            elif self.type == 'ndr':
                self.encoder = NDREncoder(d_model = args.hidden_size if args.hidden_size else 768, nhead = args.num_attention_heads if args.num_attention_heads else 12, 
                num_encoder_layers =  args.num_hidden_layers if args.num_hidden_layers else 12, dim_feedforward = 4*args.hidden_size if args.hidden_size else 3072, dropout = args.hidden_dropout_prob or 0.1,
                activation = nn.GELU(), attention_dropout = args.attention_probs_dropout_prob or 0.1, attention_type = args.attention_type, n_input_tokens=args.ntypes)
                hidden_size , layer_norm_eps = args.hidden_size or 768,  1e-12


        else:
            self.encoder = AutoModel.from_pretrained(args.pretrained_model)
            conf = AutoConfig.from_pretrained(args.pretrained_model)

        if args.share_layers:
            for i in range(1, len(self.encoder.encoder.layer)):
                self.encoder.encoder.layer[i] = self.encoder.encoder.layer[0]

        self.cls = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                nn.GELU(),
                                nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                                nn.Linear(hidden_size, nnt + int(self.mlr)))

    def get_chart_scores(self, inp, T, firsts):
        device = inp.device
        bsz = inp.size(0)
        attn_mask = (inp != 0)
        if self.type == 'transformer':
            states = self.encoder(
                input_ids=inp, output_hidden_states=True, attention_mask=attn_mask).hidden_states
            bertoutput = states[-1]
            output = bertoutput.gather(1, firsts.unsqueeze(2).expand(bsz, T, bertoutput.size(2)))
        elif self.type == 'ndr':
            src_len = torch.count_nonzero(inp, dim=1)
            states = self.encoder(
                input_ids=inp, src_len=src_len)
            output = states.gather(1, firsts.unsqueeze(2).expand(bsz, T, states.size(2)))


        halfsz = output.size(2) // 2

        # make bsz x T x T x dim, where chartrep[b][j,i] is concatenation of first half
        # of i-th token's rep, and second half of j-th token's rep
        chartrep = torch.cat(
            [output[:, :, :halfsz].unsqueeze(1).expand(bsz, T, T, halfsz),
             output[:, :, halfsz:].unsqueeze(2).expand(bsz, T, T, halfsz)], 3)
        chartscores = self.cls(chartrep) # bsz x T x T x K
        # the scores we want are either on the lower or upper triangle (depending on whether
        # want ij pairs or ji pairs, resp.): [00 10 20 30; 01 11 21 31; 02 12 22 32; 03 13 23 33].
        # we align them w/ what torch_struct viterbi does (1st dim is length, 2nd left idx)
        # by shifting things up: [01 12 23 30; 02 13 20 31; 03 10 21 32; 00 11 22 33]

        nuidxs = (torch.arange(T, device=device).view(-1, 1) # T x T
                  + torch.arange(1, T+1, device=device).view(1, -1)) % T
        chartscores = chartscores.gather( # bsz x T x T x K
            1, nuidxs.view(1, T, T, 1).expand(bsz, T, T, chartscores.size(-1)))
        return chartscores

    def forward(self, return_chart=False, **kwargs):
        pad_token = self.encoder.config.pad_token_id
        inp = kwargs["ids"]   
        device = inp.device  
        # convert inp to its mapping in the tokenizer
        if self.pretrained_with_mlm:
            inp_to_tokenize = [' '.join(map(str, inp[i].tolist())).replace('.0', '') for i in range(inp.size(0))]
            inp2 = self.tokenizer(inp_to_tokenize).input_ids
            # convert inp into tensor
            inp = torch.tensor(inp2, dtype=torch.long, device=device)
            # pad_token = 1014

        parses = kwargs["parse"]
        firsts = kwargs["firsts"]

        lengths = (inp != pad_token).sum(-1)

        if self.enclosed:
            lengths.add_(-2)


        bsz, T = firsts.size()
        if self.mlr:
            target = torch.full((bsz, T, T), self.nnt, device=device)
            for b in range(bsz):
                
                target[b, :lengths[b], :lengths[b]][parses[b][0], parses[b][1]] = parses[b][2]
        else:
            target = torch.zeros(bsz, T, T, self.nnt, device=device) # can really ignore last row
            for b in range(bsz):
                target[b, :lengths[b], :lengths[b]][parses[b]] = 1

        chartscores = self.get_chart_scores(inp, T, firsts) # bsz x T x T x K, in torch_struct fmt
        if self.mlr:
            losses = F.cross_entropy(
                chartscores.view(-1, chartscores.size(-1)), target.view(-1), reduction='none').view(
                    bsz, T, T, 1) # unsqueeze last dim to be compatible w/ mask
        else:
            losses = F.binary_cross_entropy_with_logits(chartscores, target, reduction='none')

        # mask out illegal predictions: get lower triangle then flip
        mask = torch.zeros(bsz, T, T, device=losses.device)
        for b in range(bsz):
            lenb = lengths[b].item()
            mask[b, :lenb, :lenb].copy_(
                torch.ones(lenb, lenb, device=mask.device).tril(diagonal=-1).flip(0))
        loss = (losses*mask.unsqueeze(-1)).sum() / mask.sum()
        # if not self.mlr:
        #     loss = loss / self.nnt
            
        ret_dict = {'loss': loss}

        if return_chart: # set all illegal scores to something very low
            mask.add_(-1)
            mask.mul_(1e5)
            chartscores.add_(mask.unsqueeze(-1))
            ret_dict["chart"] = chartscores

        return ret_dict

    def configure_optimizers(self, args):
        #no_bias_correction = args.no_bias_correct
        no_decay = ["bias", "LayerNorm.weight"]
        lr = args.learning_rate
        #beta1, beta2 = args.adam_beta1, args.adam_beta2
        beta1, beta2 = (0.9, 0.999)
        #eps = args.adam_epsilon
        eps = 1e-06
        awd = args.weight_decay

        grouped_parameters = [
            {"params": [p for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)], "weight_decay": awd,},
            {"params": [p for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},]
        optim1st = torch.optim.AdamW(grouped_parameters, lr=lr, betas=(beta1, beta2), eps=eps,)
                      #correct_bias=(not no_bias_correction))
        return optim1st

def chartscores_from_2hids(hids1, hids2, decoder, cat=False):
    """
    hids1 - bsz x T x dim
    hids2 - bsz x T x dim
    """
    device = hids1.device
    bsz, T, dim = hids1.size()
    # make bsz x T x T x dim, where chartrep[b][j,i] is concatenation of i-th token's h1 rep,
    # and j-th token's h2 rep
    if cat:
        chartrep = torch.cat([hids1.unsqueeze(1).expand(bsz, T, T, dim),
                              hids2.unsqueeze(2).expand(bsz, T, T, dim)], 3)
    else:
        chartrep = (hids1.unsqueeze(1).expand(bsz, T, T, dim)
                    + hids2.unsqueeze(2).expand(bsz, T, T, dim))
    chartscores = decoder(chartrep) # bsz x T x T x K
    # the scores we want are either on the lower or upper triangle (depending on whether
    # want ij pairs or ji pairs, resp.): [00 10 20 30; 01 11 21 31; 02 12 22 32; 03 13 23 33].
    # we align them w/ what torch_struct viterbi does (1st dim is length, 2nd left idx)
    # by shifting things up: [01 12 23 30; 02 13 20 31; 03 10 21 32; 00 11 22 33]
    nuidxs = (torch.arange(T, device=device).view(-1, 1) # T x T
              + torch.arange(1, T+1, device=device).view(1, -1)) % T
    chartscores = chartscores.gather( # bsz x T x T x K
        1, nuidxs.view(1, T, T, 1).expand(bsz, T, T, chartscores.size(-1)))
    return chartscores


class PretrainedThingHO(PretrainedThing):
    def __init__(self, nnt, args):
        super().__init__(nnt, args)
        conf = self.encoder.config
        #clsin = 2*conf.hidden_size if self.catrep else conf.hidden_size
        clsin = conf.hidden_size
        # same as above but adds dropout
        self.cls = nn.Sequential(nn.Dropout(p=conf.hidden_dropout_prob),
                                 nn.Linear(clsin, conf.hidden_size), nn.GELU(),
                                 nn.LayerNorm(conf.hidden_size, eps=conf.layer_norm_eps),
                                 nn.Linear(conf.hidden_size, nnt + int(self.mlr)))

        self.icls = nn.Sequential(nn.Linear(conf.hidden_size, conf.hidden_size), nn.GELU(),
                                  nn.LayerNorm(conf.hidden_size, eps=conf.layer_norm_eps),
                                  nn.Linear(conf.hidden_size, 1, bias=False))
        #self.inner_mean_pool = args.inner_mean_pool
        self.ho_stuff = args.ho_stuff
        self.no_nt_norm = args.no_nt_norm

    def inner_score(model, inp, inputs_embeds, firsts, lengths, mean_pool=False):
        bsz, T = firsts.size()
        attn_mask = (inp != model.encoder.config.pad_token_id)
        states = model.encoder(
            inputs_embeds=inputs_embeds, output_hidden_states=True,
            attention_mask=attn_mask).hidden_states
        bertoutput = states[-1]    
        """
        bertoutput = bertoutput + ((attn_mask.float() - 1).unsqueeze(2)*1e6) # mask pads
        #iscores, _ = bertoutput.view(bsz, -1).max(-1)
        iscore = model.icls(bertoutput.max(1)[0]).sum()
        """
        if mean_pool:
            pmask = attn_mask.float()
            pmask.div_(pmask.sum(1).view(-1, 1))
            iscore = model.icls((bertoutput * pmask.unsqueeze(2)).sum(1)).sum()
        else:
            iscore = model.icls(bertoutput[:,0]).sum()
        return iscore, states
        
    def get_chart_scores(self, inp, firsts, lengths):
        bsz, T = firsts.size()
        inputs_embeds = self.encoder.embeddings.word_embeddings(inp)
        attn_mask = (inp != self.encoder.config.pad_token_id)
        

        with torch.inference_mode(False):

            states = self.encoder(
                inputs_embeds=inputs_embeds, output_hidden_states=True,
                attention_mask=attn_mask).hidden_states
            bertoutput = states[-1]

            if "amlp" in self.ho_stuff:
                pmask = attn_mask.float()
                pmask.div_(pmask.sum(1).view(-1, 1))
                iscore = self.icls((bertoutput * pmask.unsqueeze(2)).sum(1)).sum()
            elif "afeats" in self.ho_stuff:
                pmask = attn_mask.float()
                pmask.div_(pmask.sum(1).view(-1, 1))            
                iscore = (bertoutput * pmask.unsqueeze(2)).sum(1).sum()
            elif "mfeats" in self.ho_stuff:
                iscore = bertoutput.view(bertoutput.size(0), -1).max(1)[0].sum()
            elif "mmlp" in self.ho_stuff:
                iscore = self.icls(bertoutput.max(1)[0]).sum()
            else:
                iscore = self.icls(bertoutput[:, 0]).sum()

            #iscore, states = inner_score(
            #    self, inp, inputs_embeds, firsts, lengths, mean_pool=self.inner_mean_pool)
            """
            gradargs = [inputs_embeds, states[-1]]
            di_de1, di_de2 = torch.autograd.grad(iscore, gradargs, create_graph=True)
            di_de1 = di_de1.gather(
                1, firsts.unsqueeze(2).expand(bsz, T, di_de1.size(2)))
            di_de2 = di_de2.gather(
                1, firsts.unsqueeze(2).expand(bsz, T, di_de2.size(2)))
            decscores = chartscores_from_2hids( # bsz x T x T x K
                di_de1, di_de2, self.cls, cat=self.catrep)
            """
            di_des = torch.autograd.grad(iscore, states, create_graph=True)
        
        if "apoolall" in self.ho_stuff:
            pooled = torch.stack(di_des).mean(0)
        elif "apool0" in self.ho_stuff:
            pooled = di_des[0]
        else:
            pooled, _ = torch.stack(di_des).max(0) # bsz x T x dim

        pooled = pooled.gather(
            1, firsts.unsqueeze(2).expand(bsz, T, pooled.size(2)))
        decscores = chartscores_from_hids(pooled, self.cls)
        return decscores

    def get_outer_loss_stuff(self, **kwargs):
        inp = kwargs["ids"]
        parses = kwargs["parse"]
        device = parses[0][0].device
        firsts = kwargs["firsts"]
        lengths = (inp != self.encoder.config.pad_token_id).sum(-1)
        if self.enclosed:
            lengths.add_(-2)
        bsz, T = firsts.size()

        decscores = self.get_chart_scores(inp, firsts, lengths)

        if self.mlr:
            target = torch.full((bsz, T, T), self.nnt, device=device)
            for b in range(bsz):
                target[b, :lengths[b], :lengths[b]][parses[b][0], parses[b][1]] = parses[b][2]
            losses = F.cross_entropy(
                decscores.view(-1, decscores.size(-1)), target.view(-1), reduction='none').view(
                    bsz, T, T, 1) # unsqueeze last dim to be compatible w/ mask
        else:
            target = torch.zeros(bsz, T, T, self.nnt, device=device) # can really ignore last row
            for b in range(bsz):
                target[b, :lengths[b], :lengths[b]][parses[b]] = 1
            losses = F.binary_cross_entropy_with_logits(decscores, target, reduction='none')

        # mask out illegal predictions: get lower triangle then flip
        mask = torch.zeros(bsz, T, T, device=losses.device)
        for b in range(bsz):
            lenb = lengths[b].item()
            mask[b, :lenb, :lenb].copy_(
                torch.ones(lenb, lenb, device=mask.device).tril(diagonal=-1).flip(0))
        ntfac = 1 if (self.mlr or self.no_nt_norm) else self.nnt
        loss = (losses*mask.unsqueeze(-1)).sum() / (mask.sum()*ntfac)
        return decscores, loss, mask

    def forward(self, **kwargs):
        _, loss, _ = self.get_outer_loss_stuff(**kwargs)
        return loss

    def get_loss_and_chart(self, **kwargs):
        decscores, loss, mask = self.get_outer_loss_stuff(**kwargs)
        mask.add_(-1)
        mask.mul_(1e5)
        decscores.add_(mask.unsqueeze(-1))
        return {'loss': loss, 'chart': decscores}


def chartscores_from_hids(hids, decoder):
    """
    hids - bsz x T x dim
    """
    device = hids.device
    bsz, T, _ = hids.size()
    halfsz = hids.size(2) // 2
    # make bsz x T x T x dim, where chartrep[b][j,i] is concatenation of first half
    # of i-th token's rep, and second half of j-th token's rep
    chartrep = torch.cat(
        [hids[:, :, :halfsz].unsqueeze(1).expand(bsz, T, T, halfsz),
         hids[:, :, halfsz:].unsqueeze(2).expand(bsz, T, T, halfsz)], 3)
    chartscores = decoder(chartrep) # bsz x T x T x K
    # the scores we want are either on the lower or upper triangle (depending on whether
    # want ij pairs or ji pairs, resp.): [00 10 20 30; 01 11 21 31; 02 12 22 32; 03 13 23 33].
    # we align them w/ what torch_struct viterbi does (1st dim is length, 2nd left idx)
    # by shifting things up: [01 12 23 30; 02 13 20 31; 03 10 21 32; 00 11 22 33]
    nuidxs = (torch.arange(T, device=device).view(-1, 1) # T x T
              + torch.arange(1, T+1, device=device).view(1, -1)) % T
    chartscores = chartscores.gather( # bsz x T x T x K
        1, nuidxs.view(1, T, T, 1).expand(bsz, T, T, chartscores.size(-1)))
    return chartscores


def inner_score2(model, inp, inputs_embeds, firsts, lengths):
    """
    for use w/ functorch; a bit less flexible...
    """
    bsz, T = firsts.size()
    attn_mask = (inp != model.encoder.config.pad_token_id)
    states = model.encoder(
        inputs_embeds=inputs_embeds, output_hidden_states=True,
        attention_mask=attn_mask).hidden_states
    bertoutput = states[-1]
    iscore = model.icls(bertoutput[:,0]).sum() # bsz
    return iscore


class PretrainedThingFT(PretrainedThingHO):
    def __init__(self, nnt, args):
        super().__init__(nnt, args)
        self.di_de = functorch.grad(inner_score2, argnums=2)

    def get_chart_scores(self, inp, firsts, lengths):
        bsz, T = firsts.size()
        inputs_embeds = self.encoder.embeddings.word_embeddings(inp)
        #dinner_dembs = functorch.grad(inner_score2, argnums=2)(
        #    self, inp, inputs_embeds, firsts, lengths) # bsz x T x demb
        dinner_dembs = self.di_de(self, inp, inputs_embeds, firsts, lengths)
        dinner_dembs = dinner_dembs.gather(
            1, firsts.unsqueeze(2).expand(bsz, T, dinner_dembs.size(2)))
        decscores = chartscores_from_hids(dinner_dembs, self.cls) # bsz x T x T x K
        return decscores