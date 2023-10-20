import argparse
import sys
import numpy as np
import torch
from tqdm import tqdm
sys.path.append("../syntheticpcfg")
import pcfg
from numpy.random import RandomState
from torch.nn.utils.rnn import pad_sequence
import utility
import time
try:
    sys.path.insert(0, "../pytorch-struct")
    from torch_struct import SentCFG
except ImportError:
    print("can't import torch_struct; ignoring...")
parser = argparse.ArgumentParser()
parser.add_argument("--nsamples", type=str, default=50000)
parser.add_argument("--seed", type=str, default=1)
parser.add_argument("--data-path", type=str, default='')
parser.add_argument("--save", type=str, default='save')
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--pidx", type=int, default=1)
parser.add_argument("--nprocs", type=int, default=40)
parser.add_argument("--chunk-size", type=int, default=1)
torch.backends.cuda.matmul.allow_tf32 = True


args = parser.parse_args()
my_pcfg = pcfg.load_pcfg_from_file(args.data_path)
nonterminals = dict.fromkeys([])
poses = dict.fromkeys([])
words = dict.fromkeys([])


for prod in my_pcfg.productions:
    if len(prod) == 3:
        # binary
        
        nt,_,_ = prod
        nonterminals[nt] = None
    else:
        # lexical
        pos,wrd = prod
        poses[pos] = None
        words[wrd] = None
def make_voc(types):
    
    i2type=[thing for thing in types]
    type2i = {thing: i for i, thing in enumerate(i2type)}
    return i2type, type2i

i2nt, nt2i = make_voc(nonterminals)
i2pos, pos2i = make_voc(poses)
i2wrd, wrd2i = make_voc(words)
print(f'i2nt: {i2nt}')
class ParseExample:
    def __init__(self, tup):
        self.tup = tup

    def __len__(self):
        return len(self.tup[0])  # source length

    def __getitem__(self, index):
        return self.tup[index]

def sampleK(K, seed = 1, maxlength = 30):
    samples = []
    i = 0
    prng = RandomState(seed)
    mysampler = pcfg.Sampler(my_pcfg, random=prng)
     
    while i < K:
        tree = mysampler.sample_tree()
        # defatul is string.
        s = utility.collect_yield(tree)
        if len(s) <= maxlength:
            samples.append({"str": s, "raw_pos":utility.tree_to_preterminals(tree), "true_tree": tree})
            i += 1

    return samples

def extract_params(device):
    roots = torch.zeros(len(nonterminals)).to(device)
    S = len(nonterminals)+len(poses)
    rules = torch.zeros(len(nonterminals), S, S).to(device)
    emiss = torch.zeros(len(my_pcfg.terminals), len(poses)).to(device)
    roots[nt2i['S']] += 100

    for prod in my_pcfg.productions:
        if len(prod) == 3:
            l, r1, r2 = prod       
            if r1 in pos2i and r2 in pos2i:
                rules[nt2i[l], pos2i[r1] + len(i2nt), pos2i[r2]+len(i2nt)] = my_pcfg.parameters[prod]
            if r1 in pos2i and r2 in nt2i:
                rules[nt2i[l], pos2i[r1]+len(i2nt), nt2i[r2]] = my_pcfg.parameters[prod]
            if r1 in nt2i and r2 in pos2i:
                rules[nt2i[l], nt2i[r1], pos2i[r2]+len(i2nt)] = my_pcfg.parameters[prod]
            if r1 in nt2i and r2 in nt2i:
                rules[nt2i[l], nt2i[r1], nt2i[r2]] = my_pcfg.parameters[prod]       
        else:
            pos, wrd = prod
            emiss[wrd2i[wrd], pos2i[pos]] = my_pcfg.parameters[prod]
    roots.div_(roots.sum()).log_()
    rules.log_()
    emiss.log_() 
    
    return roots, rules, emiss  


def get_spans(termlist, lengths, rules, roots, device):
    bsz = len(termlist)
    terms = pad_sequence(termlist, batch_first=True).to(device) # bsz x maxlen x num_pos, 0 padded
    lengths = torch.tensor(lengths).to(device)
    params = (terms, rules.unsqueeze(0).expand(bsz, -1, -1, -1), roots.unsqueeze(0).expand(bsz, -1))
    dist = SentCFG(params, lengths=lengths)
    _, _, _, spans = dist.argmax
    spans = spans.cpu()
    
    return spans


def tree_from_span(span, length, raw_pos, raw_sent):
    tree_with_idx = tuple(([(i, (raw_pos[i], raw_sent[i])) for i in range(length)]))
    tree_with_idx = dict(tree_with_idx)
    spans = []

    if isinstance(span, tuple):  # if we've saved these before:
        cover = torch.stack(span).t()
    else:
        cover = (span > 0).float().nonzero()
    
    for i in range(cover.shape[0]):
        w, r, A = cover[i].tolist()
        w = w + 1
        r = r + w
        l = r - w
        spans.append((l, r, A))
        assert r < length
        span = tuple((str(i2nt[A]), tree_with_idx[l], tree_with_idx[r]))
        # else:
            # r = length - 1
            # span = tuple((str(idx2nt[A]), tree_with_idx[l], tree_with_idx[r]))
        tree_with_idx[r] = tree_with_idx[l] = span
    return tree_with_idx[0]

def get_parseexamples(nsamples, rules, emiss, roots, chunk_size=1, nprocs = 1, pidx = 1):
    device = roots.device
    total_time=0
    sents, all_sents, all_raw_pos, all_raw_sent, all_spans, true_trees = sampleK(K = int(nsamples)),[], [], [], [], []
    nperproc = len(sents) // nprocs
    sidx = (pidx - 1)*nperproc
    eidx = pidx*nperproc if pidx < nprocs else len(sents)
    sents = sents[sidx:eidx]
    for i in tqdm(range(0, len(sents), chunk_size)):
        chunk, lengths = [], []
        for j in range(i, min(i + chunk_size, len(sents))):
            sent = [wrd2i[wrd] for wrd in sents[j]["str"]]
            raw_pos = sents[j]["raw_pos"]      
            raw_sent = sents[j]["str"]
            true_tree = sents[j]["true_tree"]
            if len(sent) >= 2:
                lengths.append(len(sent))
                x = torch.LongTensor(sent).to(device)
                chunk.append(emiss[x])  # L x num_pos     
        if not chunk:
            continue
        
        spans = get_spans(chunk, lengths, rules, roots, device)

        if spans is not None:
            all_spans.extend([span.nonzero(as_tuple=True) for span in spans])
            all_sents.append(x)
            all_raw_sent.append(raw_sent)
            all_raw_pos.append(raw_pos)
            true_trees.append(true_tree)

    assert len(all_sents) == len(all_raw_pos)
    assert len(all_sents) == len(all_spans)
    return [ParseExample((all_sents[i], all_spans[i], all_raw_pos[i], all_raw_sent[i], true_trees[i]))
            for i in range(len(all_sents))]

if __name__ == "__main__":

    device = torch.device("cpu") if args.cpu else torch.device("cuda")
    roots, rules, emiss = extract_params(device)
    parse_examples = get_parseexamples(nsamples = args.nsamples, rules = rules, emiss = emiss, roots = roots, chunk_size = args.chunk_size, nprocs = args.nprocs, pidx= args.pidx)
    torch.save({"trdata": parse_examples, "truepcfg": my_pcfg,
                "ntidx": nt2i, "posidxr": pos2i,"widxr": wrd2i, "idx2nt":i2nt}, f"{args.save}.pt")
