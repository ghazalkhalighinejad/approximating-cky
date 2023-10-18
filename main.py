import sys
import numpy as np
import tqdm
import time
import argparse
import logging
import torch
import wandb
torch.backends.cuda.matmul.allow_tf32 = True
import math
#import functorch
from transformers import (set_seed, get_linear_schedule_with_warmup,
                          get_constant_schedule, get_constant_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup, AutoConfig, AutoTokenizer)
from accelerate import Accelerator, DistributedType
import os
from my_utils import ByLengthSampler
from models.all_models import PretrainedThing, PretrainedThingHO
import my_utils 
sys.path.insert(0, "syntheticpcfg/")
from syntheticpcfg import utility
from my_utils import ParseExample
# sys.path.append("getdata/") # stupid hack to avoid unpickling issues

TRUE_PCFG = None
IDX2NT = None



parser = argparse.ArgumentParser()
# model arguments
parser.add_argument("--model-type", type=str, default="transformer", choices=["transformer", "ndr"])
parser.add_argument("--attention-type", type = str, default = None) #use for geom. attention experiments
parser.add_argument("--hidden-dropout-prob", type=float, default=None, help="")
parser.add_argument("--attention-probs-dropout-prob", type=float, default=None, help="")
parser.add_argument("--pretrained-model", type=str, default="bert-base-uncased")
parser.add_argument("--train-from", type=str, default="")
parser.add_argument("--from-scratch", action="store_true", help="")
parser.add_argument("--position-embedding-type", type=str, default=None)
parser.add_argument("--hidden-size", type=int, default=None)
parser.add_argument("--intermediate-size", type=int, default=None)
parser.add_argument("--num-hidden-layers", type=int, default=None)
parser.add_argument("--num-attention-heads", type=int, default=None)
parser.add_argument("--mlr", action="store_true", help="")
parser.add_argument("--share-layers", action="store_true", help="")
# training stuff
parser.add_argument("--grad-accum-steps", type=int, default=1)
parser.add_argument("--clip", type=float, default=1.0)
parser.add_argument("--train-batch-size", type=int, default=32)
parser.add_argument("--eval-batch-size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--learning-rate", type=float, default=0.00001)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--warmup-steps", type=int, default=4000)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--lr-schedule", type=str, default="constant+warmup",
                    choices=["constant+warmup", "constant", "linear", "invsqrt"])
# dataset stuff
parser.add_argument("--data-path", type=str, default="data/treebanks/all_LDC99T42-1-0.pt")
parser.add_argument("--save", type=str, default="")
parser.add_argument("--multiple-files", action="store_true", help="")
parser.add_argument("--maxlen", type=int, default=256)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--trperc", type=float, default=0.99, help="")
parser.add_argument("--no-enclose", action="store_true", help="")
parser.add_argument("--eval2", action="store_true", help="")
# parser.add_argument("--inner-mean-pool", action="store_true", help="")

# second order stuff
parser.add_argument("--ho-stuff", type=str, default=None, help="")
parser.add_argument("--no-nt-norm", action="store_true", help="")
parser.add_argument("--higher-order", action="store_true", help="")

parser.add_argument("--just-eval", action="store_true", help="")
parser.add_argument("--check-compat", action="store_true", help="")
parser.add_argument("--log-interval", type=int, default=1000)
parser.add_argument("--seed", type=int, default=3636)
parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
parser.add_argument("--nruns", type=int, default=1)
parser.add_argument("--gseed", type=int, default=0)
parser.add_argument("--load-pretrained", action="store_true", help="If passed, will load a model pretrained on mlm.")
parser.add_argument("--pretrained-model-file", type=str, default="pretrained_model.pt")
args = parser.parse_args()


class MyTupDataset(torch.utils.data.Dataset):
    def __init__(self, tuplist, cls, sep, offset=1, add_special=True):
        """
        tuplist - a list of (tensor of idxs, parse tuple) tuples
        """
        super().__init__()
        self.tuplist = tuplist
        self.offset = offset # vocab offset so we can still pad....
        self.add_special = add_special
        self.cls_id, self.sep_id = cls, sep

    def __len__(self):
        return len(self.tuplist)

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]

        
        if args.pretrained_model_file:
            tokenizer = AutoTokenizer.from_pretrained("mlm_stuff/birul800-500k_tokenizer")

            ids = self.tuplist[index][0]
            inp_to_tokenize = ' '.join(map(str, ids.tolist())).replace('.0', '')
            tokout = tokenizer(inp_to_tokenize)
            wids = tokout.word_ids()
            firsttokenids = [i for i in range(1, len(wids)-1) if wids[i] is not None and wids[i] != wids[i-1]]
            firsts = torch.LongTensor(firsttokenids)

            if len(self.tuplist[index]) > 3:
    
                return {"input_ids": ids, "firsts": firsts, "parses": self.tuplist[index][1], "raw_pos": self.tuplist[index][2], "raw_sent": self.tuplist[index][3], "true_tree": self.tuplist[index][4]}
            else:
                return {"input_ids": ids, "firsts": firsts, "parses": self.tuplist[index][1]}


        ids = self.tuplist[index][0] + self.offset

        if self.add_special:
            inpids = ids.new(ids.size(0) + 2)
            inpids[0] = self.cls_id
            inpids[-1] = self.sep_id
            inpids[1:-1].copy_(ids)
            firsts = torch.arange(1, ids.size(0)+1)
        else:
            inpids, firsts = ids, torch.arange(ids.size(0))

        if len(self.tuplist[index]) > 3:
    
            return {"input_ids": inpids, "firsts": firsts, "parses": self.tuplist[index][1], "raw_pos": self.tuplist[index][2], "raw_sent": self.tuplist[index][3], "true_tree": self.tuplist[index][4]}
        else:
            return {"input_ids": inpids, "firsts": firsts, "parses": self.tuplist[index][1]}


def configure_scheduler(optim, nsteps, args):
    warmup_init_lr = 1e-7
    if hasattr(args, "warmup_ratio"):
        warmup_steps = int(args.warmup_ratio * nsteps)
    else:
        warmup_steps = args.warmup_steps
    lr = optim.param_groups[0]['lr']
    if args.lr_schedule == "constant":
        scheduler = get_constant_schedule(optim)
    elif args.lr_schedule == "constant+warmup":
        scheduler = get_constant_schedule_with_warmup(optim, warmup_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optim, warmup_steps, nsteps)
    return scheduler

def span_eval(predchart, truparse, mlr=False, check_compatibility=False):
    """
    predchart - bsz x T x T x K
    truparse - bsz-length list of 3-tuples
    """
    T = predchart.size(1)
    nspans, npredspans = 0, 0
    loverlap, uoverlap = 0, 0
    sz = predchart.size(0)
    total_time = 0
    for b in range(predchart.size(0)):
        predspans, ex_time = my_utils.stupid_extract_parse(
            predchart[b])
        total_time += ex_time
        goldspans, _ = my_utils.extract_parse(truparse[b], T)
        # assuming all spans are unique...
        npredspans += len(predspans)
        nspans += len(goldspans)
        loverlap += len(set(predspans) & set(goldspans))
        predspans = [(l, r) for (l, r, _) in predspans]
        goldspans = [(l, r) for (l, r, _) in goldspans]
        uoverlap += len(set(predspans) & set(goldspans))
    return loverlap, uoverlap, nspans, npredspans, sz, total_time

def span_eval2(predchart, true_tree, mlr=False, check_compatibility=False, raw_pos = None, raw_sent = None, idx2nt =None):
    """
    predchart - bsz x T x T x K
    truparse - bsz-length list of 3-tuples
    """
    T = predchart.size(1)
    bsz = predchart.size(0)
    difflogprob = 0
    total_diff_mean_logprob = 0
    total_valid_parses = 0
    global TRUE_PCFG
    for b in range(predchart.size(0)):
        pred_tree = my_utils.stupid_extract_parse2(
            predchart[b], T, mlr=mlr, check_compatibility=False, raw_pos=raw_pos[b], raw_sent=raw_sent[b], idx2nt=idx2nt)
        pred_mean_logprob, pred_nrules, pred_flag = TRUE_PCFG.mean_log_prob(pred_tree, flag = False)  
        true_mean_logprob, true_nrules, true_flag = TRUE_PCFG.mean_log_prob(true_tree[b], flag = False)

        pred_mean_logprob = pred_mean_logprob / pred_nrules
        true_mean_logprob = true_mean_logprob / true_nrules

        total_diff_mean_logprob += abs(true_mean_logprob - pred_mean_logprob)

    return total_diff_mean_logprob/bsz, total_valid_parses


def eval_function2(model, val_dataloader, check_compatibility=False,
                  higher_order=False):
    total_valloss, nval_batches = 0, 0
    total_diff = 0
    total_valid_parses = 0
    sz = 0
    global IDX2NT
    for step, batch in enumerate(val_dataloader):
        if higher_order:
            outputs = model.get_loss_and_chart(**batch)
        else:
            with torch.no_grad():
                outputs = model(return_chart=True, **batch)
        total_valloss += outputs["loss"].item()
        nval_batches += 1
        sz += outputs["chart"].size(0)
        diff, valid_parses = span_eval2(outputs["chart"], batch["true_tree"], mlr=model.mlr,
                                check_compatibility=check_compatibility, raw_pos = batch["raw_pos"], raw_sent = batch["raw_sent"], idx2nt = IDX2NT)
            
        total_diff+= diff
        total_valid_parses += valid_parses
    
    return total_diff/nval_batches, total_valid_parses/sz

def eval_function(model, val_dataloader, calc_f=True, check_compatibility=False,
                  higher_order=False):
    total_valloss, nval_batches = 0, 0
    total_lo, total_uo, total_spans, total_predspans, nsents = 0, 0, 0, 0, 0
    total_time = 0
    for step, batch in enumerate(val_dataloader):
        if higher_order:
            with torch.inference_mode():
                start_time = time.time()
                outputs = model.get_loss_and_chart(**batch)
                end_time = time.time()
        else:
            with torch.inference_mode():
                start_time = time.time()
                outputs = model(return_chart=True, **batch)
                end_time = time.time()
        total_valloss += outputs["loss"].item()
        nval_batches += 1
        if calc_f:
            # print(f'parse: {batch["parse"]}')
            lo, uo, nspans, npredspans, sz, ex_time = span_eval(
                outputs["chart"], batch["parse"], mlr=model.mlr,
                check_compatibility=check_compatibility)
            
            nsents+=sz
            total_lo += lo
            total_uo += uo
            total_spans += nspans
            total_predspans += npredspans
            total_time += ex_time + (end_time - start_time)
    if calc_f:
        lprec, lrec = total_lo/total_predspans, total_lo/total_spans
        uprec, urec = total_uo/total_predspans, total_uo/total_spans
        try:
            lf1 = 2*lprec*lrec/(lprec + lrec)
        except ZeroDivisionError:
            lf1 = 0
        try:
            uf1 = 2*uprec*urec/(uprec + urec)
        except ZeroDivisionError:
            uf1 = 0
    else:
        lf1, uf1 = -1, -1
    

    return total_valloss/nval_batches, lf1, uf1, nsents/total_time


def training_function(train_dataloader, val_dataloader, logger, args, higher_order=False):

    accelerator = Accelerator(cpu=args.cpu)
    ntrbatches = len(train_dataloader)

    if args.load_pretrained:
        print(f"loading mlm-pretrained model from {args.pretrained_model_file}...")
        sstate, m_args = None, args
        model = PretrainedThingHO(args.nnt, args, pretrained_with_mlm = args.pretrained_model_file) if args.higher_order else PretrainedThing(args.nnt, args, pretrained_with_mlm = args.pretrained_model_file)
        

    elif args.train_from:
        print(f"loading model from {args.train_from}...")
        sstate = torch.load(args.train_from)
        m_args, sd, osd, ssd = sstate["opt"], sstate["sd"], sstate["osd"], sstate["ssd"]
        model = PretrainedThingHO(args.nnt, m_args) if args.higher_order else PretrainedThing(args.nnt, m_args)
    else:
        sstate, m_args = None, args
        model = PretrainedThingHO(args.nnt, args) if args.higher_order else PretrainedThing(args.nnt, args)

    if args.model_type == 'transformer' and not args.load_pretrained:
        model.encoder.resize_token_embeddings(args.ntypes+3) # 0=pad, ntypes+1=cls, ntypes+2=sep
        if args.from_scratch:
            model.encoder.post_init()  

    if sstate is not None:
        model.load_state_dict(sd)

    model = model.to(accelerator.device)
    optimizer = model.configure_optimizers(args)

    if sstate is not None:
        optimizer.load_state_dict(osd)

    scheduler = configure_scheduler(
        optimizer, ntrbatches*args.epochs//args.grad_accum_steps, args)
    if sstate is not None:
        scheduler.load_state_dict(ssd)

    # model = torch.compile(model)
    # print("got the compiled model")
    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader)

    if args.just_eval:

        model.eval()
        epvalloss, lf1, uf1, timepersent = eval_function(
            model, val_dataloader, calc_f=True, check_compatibility=args.check_compat,
            higher_order=higher_order)
        print(f"loss {epvalloss:.5f} | LF {lf1:.5f} | UF {uf1:.5f}")
        print(f"time per sentence: {timepersent}")
        delta_score, mean_valid_parses = eval_function2(
        model, val_dataloader, check_compatibility=args.check_compat,
        higher_order=higher_order)
        
        delta_score, mean_valid_parses = eval_function2(
        model, val_dataloader, check_compatibility=args.check_compat,
        higher_order=higher_order)
        logger.info(
            f"Delta {delta_score} | Mean valid parses {mean_valid_parses}")
        # else:

        sys.exit(0)

    best_loss, bestlf1 = 1e38, 0
    no_improvement_cntr = 0
    optimizer.zero_grad()
    niter = 0
    run = wandb.init(project="synthetic", reinit=True)
    wandb.config.update(args)

    for epoch in range(args.epochs):
        model.train()
        total_loss, nbatches = 0, 0
        for step, batch in enumerate(train_dataloader):
            #batch.to(accelerator.device)
            if higher_order:
                loss = model(**batch)
            else:
                loss = model(**batch)["loss"]
            total_loss += loss.item()
            nbatches += 1
            niter += 1
            loss = loss / args.grad_accum_steps
            accelerator.backward(loss)


            if step % args.grad_accum_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step > 0 and step % args.log_interval == 0:
                logger.info(
                    f"batch {step}/{ntrbatches} | lr {scheduler.get_last_lr()[0]:.5f} "
                    f"| loss {total_loss/nbatches:.5f}")

            if args.max_steps is not None and niter / args.grad_accum_steps >= args.max_steps:
                break

        model.eval()
        epvalloss, lf1, uf1, _ = eval_function(
            model, val_dataloader, check_compatibility=args.check_compat, calc_f=True,
            higher_order=higher_order)
        # if eval2:
        delta_score, mean_valid_parses = eval_function2(
        model, val_dataloader, check_compatibility=args.check_compat,
        higher_order=higher_order)
        logger.info(
            f"Val epoch {epoch} | Delta {delta_score} | Mean valid parses {mean_valid_parses}")
        # else:
        logger.info(
            f"Val epoch {epoch} | LF {lf1:.5f}")
        wandb.log({"LF": lf1})
        
        # sys.exit(0)
        no_improvement_cntr += 1
        if epoch == 0 or lf1 > bestlf1 or epvalloss < best_loss:
            no_improvement_cntr = 0
            bestlf1 = max(bestlf1, lf1)
            wandb.log({"best LF": bestlf1})
            best_loss = min(best_loss, epvalloss)
            if args.save:
                logger.info(f"saving model to {args.save}")
                torch.save({"opt": args, "sd": model.state_dict(), "osd": optimizer.state_dict(),
                            "ssd": scheduler.state_dict(), "best_f": bestlf1}, args.save)

   
        if no_improvement_cntr > args.patience:
            run.finish()
            break
        if args.max_steps is not None and niter / args.grad_accum_steps >= args.max_steps:
            run.finish()
            break
    run.finish()


def read_pt_files(base_path):
    data = {"trdata": [], "ntidx": None, "posidxr": None,"widxr": None, "truepcfg": None, "idx2nt": None}
    path = os.path.join(base_path)
    for root, dirs, files in os.walk(path):
        for name in files:
            path = os.path.join(root, name)
            exdata = torch.load(path)
            data["ntidx"], data["widxr"], data["posidxr"],data["truepcfg"], data["idx2nt"] = \
                exdata["ntidx"], exdata["widxr"], exdata["posidxr"], exdata["truepcfg"], exdata["idx2nt"]
            data["trdata"] += exdata["trdata"]
    return data

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"{args}")

    # set_seed(args.seed)
    if args.multiple_files:
        data = read_pt_files(args.data_path)
    else:    
        data = torch.load(args.data_path+".pt")
    # data = torch.load(args.data_path+".pt")

    nnt, ntypes = len(data["ntidx"]), len(data["widxr"])
    args.nnt, args.ntypes = nnt, ntypes
    conf = AutoConfig.from_pretrained(args.pretrained_model)

    # if "valdata" not in data:
    torch.manual_seed(1)
    alldata = data["trdata"]
    perm = torch.randperm(len(alldata))
    ntr = int(len(alldata) * args.trperc)
    data["trdata"] = [alldata[idx] for idx in perm[:ntr]]
    data["valdata"] = [alldata[idx] for idx in perm[ntr:]]

    global TRUE_PCFG 
    global IDX2NT 

    TRUE_PCFG = data["truepcfg"]
    TRUE_PCFG.smooth_normalize()
    # sys.exit()
    IDX2NT = data["idx2nt"]
    # data["ntidx"], data["widxr"], data["posidxr"],data["truepcfg"], data["idx2nt"] = \
    # exdata["ntidx"], exdata["widxr"], exdata["posidxr"], exdata["truepcfg"], exdata["idx2nt"]

    norig = len(data["trdata"])
    lenthresh = min(args.maxlen, conf.max_position_embeddings)
    trdata = [thing.tup[:5] for thing in data["trdata"] if len(thing) <= lenthresh]
    # check if any of the two trdata are the same

    logger.info(f"ignoring {norig - len(trdata)} of {norig} training examples")
    norig = len(data["valdata"])

    valdata = [thing.tup[:5] for thing in data["valdata"] if len(thing) <= lenthresh]

    logger.info(f"ignoring {norig - len(valdata)} of {norig} val examples")

    valdata = MyTupDataset(valdata, ntypes+1, ntypes+2, add_special=(not args.no_enclose))

    trdata = MyTupDataset(trdata, ntypes+1, ntypes+2, add_special=(not args.no_enclose))

    def collate_fn(exlist):
        if args.pretrained_model_file:
            tokenizer = AutoTokenizer.from_pretrained("mlm_stuff/birul800-500k_tokenizer")

            # get the first token ids of each word in the sentence with the tokenizer
            ids = []
            for ex in exlist:
                inp_to_tokenize = ' '.join(map(str, ex['input_ids'].tolist())).replace('.0', '')
                ids.append(inp_to_tokenize)

            return {'ids': tokenizer(ids, padding=True, return_tensors="pt")['input_ids'],
                    'parse': [ex['parses'] for ex in exlist],
                    'firsts': torch.nn.utils.rnn.pad_sequence(
                    [ex['firsts'] for ex in exlist], batch_first=True,
                    padding_value=conf.pad_token_id),}
        
        return {'ids': torch.nn.utils.rnn.pad_sequence(
                    [ex['input_ids'] for ex in exlist], batch_first=True,
                    padding_value=conf.pad_token_id),
                'parse': [ex['parses'] for ex in exlist],
                'firsts': torch.nn.utils.rnn.pad_sequence(
                    [ex['firsts'] for ex in exlist], batch_first=True,
                    padding_value=conf.pad_token_id),}

    def bls_collate_fn(exlist): # for ByLengthSampler

        if args.pretrained_model_file:

            tokenizer = AutoTokenizer.from_pretrained("mlm_stuff/birul800-500k_tokenizer")
            ids = []
            for ex in exlist[0]:
                inp_to_tokenize = ' '.join(map(str, ex['input_ids'].tolist())).replace('.0', '')
                ids.append(inp_to_tokenize)
                
            return {'ids': tokenizer(ids, padding=True, return_tensors="pt")['input_ids'],
                    'parse': [ex['parses'] for ex in exlist[0]],
                    'firsts': torch.stack([ex['firsts'] for ex in exlist[0]]),
                    'true_tree': [ex['true_tree'] for ex in exlist[0]],
                    'raw_pos': [ex['raw_pos'] for ex in exlist[0]],
                    'raw_sent': [ex['raw_sent'] for ex in exlist[0]]}
                 
        return {'ids': torch.nn.utils.rnn.pad_sequence(
                    [ex['input_ids'] for ex in exlist[0]], batch_first=True,
                    padding_value=conf.pad_token_id),
                'parse': [ex['parses'] for ex in exlist[0]],
                'firsts': torch.stack([ex['firsts'] for ex in exlist[0]]),
                'true_tree': [ex['true_tree'] for ex in exlist[0]],
                'raw_pos': [ex['raw_pos'] for ex in exlist[0]],
                'raw_sent': [ex['raw_sent'] for ex in exlist[0]]}

    
    set_seed(args.seed)

    train_dataloader = torch.utils.data.DataLoader(
        trdata, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataloader = torch.utils.data.DataLoader(
            valdata, batch_size=1, # weird hack for dataloaders
            sampler=ByLengthSampler(valdata, args.eval_batch_size, shuffle=False),
            collate_fn=bls_collate_fn)

    training_function(
        train_dataloader,val_dataloader, logger, args, higher_order=args.higher_order,)       

if __name__ == "__main__":
    main()