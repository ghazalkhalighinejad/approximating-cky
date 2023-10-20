
import argparse
import sys
import time
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from collections import Counter, defaultdict
import nltk
import torch
import tqdm

NT_THRESH, POS_THRESH, WRD_THRESH = 300, 300, 5

def make_voc(cntr, thresh=50, add_pad=False):
    specials = ["[UNK]"]
    if add_pad:
        specials.insert(0, "[PAD]")
    i2type = specials[:]
    i2type.extend(
        [
            thing
            for (thing, count) in cntr.items()
            if count >= thresh and thing not in specials
        ]
    )
    type2i = {thing: i for i, thing in enumerate(i2type)}
    return i2type, type2i

def get_trees_and_vocs(train_path, maxlen=1000000):
    trees = BracketParseCorpusReader("", [train_path]).parsed_sents()
    # make into CNF and remove POS
    cnf_trees = []

    for tree in tqdm.tqdm(trees):
        if len(tree.leaves()) > maxlen:
            continue
        tree.collapse_unary(collapsePOS=True, joinChar="::")
        # adapted from https://github.com/nikitakit/self-attentive-parser/blob/master/src/benepar/decode_chart.py#L31
        if tree.label() in ("TOP", "ROOT", "S1", "VROOT"):
            if len(tree) == 1:
                tree = tree[0]
            else:
                tree.set_label("")
        tree.chomsky_normal_form()
        # no_pos_tree = collapse_unary_strip_pos(tree)
        if not isinstance(tree, str):  # if a single word just ends up being a string
            assert all(
                len(prod) == 2 or prod.is_lexical() for prod in tree.productions()
            )
            cnf_trees.append(tree)
    

    # count nonterminals, pos-tags, and terminals
    ntcntr = Counter(
        prod.lhs()
        for tree in cnf_trees
        for prod in tree.productions()
        if not prod.is_lexical()
    )


    i2nt, nt2i = make_voc(ntcntr, thresh=NT_THRESH)

    poscntr = Counter(
        prod.lhs()
        for tree in cnf_trees
        for prod in tree.productions()
        if prod.is_lexical()
    )
    all_pos = set(poscntr.keys())  # want to separate these from nts even if we unk them
    i2pos, pos2i = make_voc(poscntr, thresh=POS_THRESH)

    wcntr = Counter(
        wrd
        for tree in cnf_trees
        for prod in tree.productions()
        for wrd in prod.rhs()
        if isinstance(wrd, str)
    )
    i2w, w2i = make_voc(wcntr, thresh=WRD_THRESH, add_pad=True)
    
    return cnf_trees, i2nt, nt2i, i2pos, pos2i, all_pos, i2w, w2i
    
def get_potentials_and_trees(train_path, maxlen=100):
    cnf_trees, i2nt, nt2i, i2pos, pos2i, all_pos, i2w, w2i = get_trees_and_vocs(
        train_path, maxlen=maxlen
    )
    print("got trees and vocs")
    # aggregate compound nonterminals by dominating one
    head2rest = defaultdict(list)
    pos2rest = defaultdict(list)
    nnt_specials = 1
    for nt in i2nt[nnt_specials:]:
        if "|" in nt.symbol():
            pieces = nt.symbol().split("|")
            assert len(pieces) == 2
            head2rest[pieces[0]].append(pieces[1])
    
    for pos in i2pos[1:]:
        if "::" in pos.symbol():
            pieces = pos.symbol().split("::")
            poss , nts = pieces[-1], "::".join(pieces[:-1])
            pos2rest[poss].append(nts)
    
    
    def smooth_ntidx(nt, find_longest_match=True):

        if nt in nt2i:
            return nt2i[nt]
        if nt in pos2i:
            return pos2i[nt]
        if "|" in nt.symbol():  # try to find a similar nonterminal
            pieces = nt.symbol().split("|")
            assert len(pieces) == 2
            if pieces[0] in head2rest and find_longest_match:  # find longest match
                head, rest = pieces[0], pieces[1][1:-1]
                maxoverlap, argmax = 0, -1
                for i, thing in enumerate(head2rest[head]):
                    # this will catch pfx or suffix...may only want one
                    if thing[1:-1] in rest and len(thing) > maxoverlap:
                        maxoverlap, argmax = len(thing), i
                if maxoverlap > 0:
                    return nt2i[
                        nltk.Nonterminal("|".join([head, head2rest[head][argmax]]))
                    ]

            if nltk.Nonterminal(pieces[0]) in nt2i:
                return nt2i[nltk.Nonterminal(pieces[0])]
        return nt2i["[UNK]"]

    def smooth_posidx(pos, find_longest_match=True):

        if pos in pos2i:
            return pos2i[pos]
        if "::" in pos.symbol(): 
            pieces = pos.symbol().split("::")
            nts, poss = "::"+"::".join(pieces[:-1])+"::", pieces[-1] 
        
            if poss in pos2rest and find_longest_match:  
                maxoverlap, argmax = 0, -1
                for i, thing in enumerate(pos2rest[poss]):
                    if "::"+thing+"::" in nts and len(thing) > maxoverlap:
                        maxoverlap, argmax = len(thing), i            
                if maxoverlap > 0:
                    return pos2i[
                        nltk.Nonterminal("::".join([pos2rest[poss][argmax], poss]))
                    ]

            if nltk.Nonterminal(poss) in pos2i:
                return pos2i[nltk.Nonterminal(poss)]

        return pos2i["[UNK]"]

    def get_rule_idx(nt):
        if nt in all_pos:
            idx = pos2i[nt] if nt in pos2i else pos2i["[UNK]"]
            idx += len(i2nt)  # increment because goes after true nonterminals
            return idx
        return smooth_ntidx(nt)

    # make grammar stuff; using POS tags as Ghazal suggested.
    roots = torch.ones(len(i2nt))
    S = len(i2nt) + len(i2pos)
    rules = torch.zeros(len(i2nt), S, S).add_(1e-5)  # smooth slightly
    emiss = torch.zeros(len(i2w), len(i2pos)).add_(1e-5)

    for tree in cnf_trees:
        for prod in tree.productions():

            if prod.is_lexical():
                pos, wrd = prod.lhs(), prod.rhs()[0]
                # could smooth poses too
                # pidx = pos2i[pos] if pos in pos2i else pos2i["[UNK]"]
                pidx = smooth_posidx(pos)
                widx = w2i[wrd] if wrd in w2i else w2i["[UNK]"]
                emiss[widx, pidx] += 1
            else:
                lidx = smooth_ntidx(prod.lhs())
                r1, r2 = prod.rhs()
                r1idx, r2idx = get_rule_idx(r1), get_rule_idx(r2)
                rules[lidx, r1idx, r2idx] += 1

    roots.div_(roots.sum())
    rules.div_(rules.view(len(i2nt), -1).sum(1).view(len(i2nt), 1, 1))
    emiss.div_(emiss.sum(1).view(-1, 1))
    return cnf_trees, roots, rules, emiss, i2nt, nt2i, i2pos, pos2i, i2w, w2i


def convert_to_pcfg_file(roots, rules, emiss, i2nt, i2pos, i2w, filename):
    with open(filename, 'w') as f:
        for i in range(len(i2nt)):
            for j in range(len(i2nt)):
                f.write(f'{roots[i]} S -> NT{i} NT{j}\n')
        
        for i in range(len(i2nt)):
            for j in range(len(i2pos)+len(i2nt)):
                for k in range(len(i2pos)+len(i2nt)):
                        if j < len(i2nt) and k < len(i2nt):
                            f.write(f'{rules[i][j][k]} NT{i} -> NT{j} NT{k}\n')
                        elif j < len(i2nt) and k >= len(i2nt):
                            f.write(f'{rules[i][j][k]} NT{i} -> NT{j} NT{k+len(i2nt)}\n')
                        elif j >= len(i2nt) and k < len(i2nt):
                            f.write(f'{rules[i][j][k]} NT{i} -> NT{j+len(i2nt)} NT{k}\n')
                        else:
                            f.write(f'{rules[i][j][k]} NT{i} -> NT{j+len(i2nt)} NT{k+len(i2nt)}\n')

        for i in range(len(i2w)):
            for j in range(len(i2pos)):
                    f.write(f'{emiss[i][j]} NT{j+len(i2nt)} -> {i2w[i]}\n')
    f.close()

if __name__ == "__main__":
    cnf_trees, roots, rules, emiss, i2nt, nt2i, i2pos, pos2i, i2w, w2i = get_potentials_and_trees(train_path = 'train_02-21.LDC99T42' )
    convert_to_pcfg_file(roots, rules, emiss, i2nt, i2pos, i2w, 'PCFG.txt')