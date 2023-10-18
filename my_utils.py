from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
import sys
import time
try:
    sys.path.insert(0, "../pytorch-struct")
    from torch_struct import SentCFG
except ImportError:
    print("can't import torch_struct; ignoring...")
import numpy

class ByLengthSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, batchsize, shuffle=False):
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.seqlens = torch.LongTensor([len(example['firsts']) for example in dataset])
        self.nbatches = len(self._generate_batches())

    def _generate_batches(self):
        # shuffle examples
        seqlens = self.seqlens
        perm = torch.randperm(seqlens.size(0)) if self.shuffle else torch.arange(seqlens.size(0))
        batches = []
        len2batch = defaultdict(list)
        for i, seqidx in enumerate(perm):

            seqlen, seqidx = seqlens[seqidx].item(), seqidx.item()

            len2batch[seqlen].append(seqidx)
            if len(len2batch[seqlen]) >= self.batchsize:
                batches.append(len2batch[seqlen][:])
                del len2batch[seqlen]
        # add any remaining batches
        for length, batchlist in len2batch.items():
            if len(batchlist) > 0:
                batches.append(batchlist)

        # shuffle again so we don't always start w/ the most common sizes
        batchperm = torch.randperm(len(batches)) if self.shuffle else torch.arange(len(batches))
        return [batches[idx] for idx in batchperm]

    def batch_count(self):
        return self.nbatches

    def __len__(self):
        return len(self.seqlens)

    def __iter__(self):
        batches = self._generate_batches()
        for batch in batches:
            yield batch

def tree_from_span(span, length, raw_pos, raw_sent, idx2nt):
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
        span = tuple((str(idx2nt[A]), tree_with_idx[l], tree_with_idx[r]))
        # else:
            # r = length - 1
            # span = tuple((str(idx2nt[A]), tree_with_idx[l], tree_with_idx[r]))
        tree_with_idx[r] = tree_with_idx[l] = span
    return tree_with_idx[0]

def extract_parse(span, length):
    # span is N x N x NT
    # and basically first dim is 0-indexed window length (INCLUSIVE, so 0 is for 1, which
    # is really a span of length 2!), and second is left index (0-indexed), and final dim
    # is the nonterminal.
    tree = [(i, str(i)) for i in range(length)]
    tree = dict(tree)
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
        span = "({} {})".format(tree[l], tree[r])
        tree[r] = tree[l] = span
        
    return spans, tree[0]


def stupid_extract_parse2(span, length = None, mlr=False, check_compatibility=False, raw_pos=None, raw_sent=None, idx2nt=None):
    """
    span - N x N x NT
    """
    length = span.size(0)
    assert length == span.size(0)
    spans = []
    # get max NT for each possible span
    ntmaxes, ntargmaxes = span.max(-1)  # N x N
    ntmaxes, ntargmaxes = ntmaxes.view(-1), ntargmaxes.view(-1)
    if mlr: # ignore anything we predicted a dummy label for
        ntmaxes[ntargmaxes == span.size(2)-1] = -float("inf")
    
    srtscores, srtidxs = ntmaxes.sort(descending=True)  # N*N, N*N
    srtscores, srtidxs, ntargmaxes = numpy.array(srtscores.tolist()), numpy.array(srtidxs.tolist()), numpy.array(ntargmaxes.tolist()) 

    pred_tree = None
    chart = torch.zeros(length, length, span.size(2))

    for i in range(len(srtidxs)):
        flat_idx = srtidxs[i]
        w = (flat_idx // length) + 1
        l = flat_idx % length
        r = l + w
        chart[flat_idx // length, flat_idx % length, ntargmaxes[flat_idx]] = 1
        spans.append((l, r, ntargmaxes[flat_idx]))
        if len(spans) >= length - 1:
            break
    if raw_pos and idx2nt:
        pred_tree = tree_from_span(chart, length, raw_pos, raw_sent, idx2nt)
    return pred_tree


def stupid_extract_parse(span, length = None, mlr=False, check_compatibility=False):
    """
    span - N x N x NT
    """
    length = span.size(0)
    assert length == span.size(0)
    spans = []
    first = time.time()
    # get max NT for each possible span
    ntmaxes, ntargmaxes = span.max(-1)  # N x N
    ntmaxes, ntargmaxes = ntmaxes.view(-1), ntargmaxes.view(-1)
    if mlr: # ignore anything we predicted a dummy label for
        ntmaxes[ntargmaxes == span.size(2)-1] = -float("inf")
    
    srtscores, srtidxs = ntmaxes.sort(descending=True)  # N*N, N*N

    srtscores, srtidxs, ntargmaxes = numpy.array(srtscores.tolist()), numpy.array(srtidxs.tolist()), numpy.array(ntargmaxes.tolist())

    
    for i in range(len(srtidxs)):
        # counter += 1
        flat_idx = srtidxs[i]
        w = (flat_idx // length) + 1
        l = flat_idx % length
        r = l + w
        # if check_compatibility:
        #     # check if compatible
        #     if any((lp < l <= rp < r) or (l < lp <= r < rp) for (lp, rp, _) in spans):
        #         continue
        #     #if any((lp < l < rp < r) or (l < lp < r < rp) for (lp, rp, _) in spans):
        #     #    continue
        # chart[flat_idx // length, flat_idx % length, ntargmaxes[flat_idx].item()] = 1
        spans.append((l, r, ntargmaxes[flat_idx]))
        if len(spans) >= length - 1:
            break

    return spans, time.time() - first

def get_spans(termlist, lengths, rules, roots, device):
    bsz = len(termlist)
    terms = pad_sequence(termlist, batch_first=True).to(device) # bsz x maxlen x num_pos, 0 padded
    lengths = torch.tensor(lengths).to(device)
    params = (terms, rules.unsqueeze(0).expand(bsz, -1, -1, -1), roots.unsqueeze(0).expand(bsz, -1))
    dist = SentCFG(params, lengths=lengths)
    _, _, _, spans = dist.argmax
    spans = spans.cpu()
    return spans

class ParseExample:
    def __init__(self, tup):
        self.tup = tup

    def __len__(self):
        return len(self.tup[0])  # source length

    def __getitem__(self, index):
        return self.tup[index]
