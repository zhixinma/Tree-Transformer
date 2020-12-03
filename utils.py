from nltk.tree import Tree
from tabulate import tabulate
import torch


def sort_tree_by_height(tree):
    pos_dfs = tree.treepositions()
    tag_dfs = [tree[i].label() if isinstance(tree[i], Tree) else tree[i] for i in pos_dfs]
    h_dfs = [tree[i].height() if isinstance(tree[i], Tree) else 1 for i in pos_dfs]

    ind = list(range(len(pos_dfs)))
    ind = sorted(ind, key=lambda x: h_dfs[x], reverse=True)

    return map(list, zip(*[[pos_dfs[i], tag_dfs[i], h_dfs[i]] for i in ind]))


def format_fa(tree_position):
    pos_to_idx = {tree_position[idx]: idx for idx in range(len(tree_position))}
    return [pos_to_idx[p[:-1]] for p in tree_position]  # p[:-1] is it's father


def remove_edge(father, height, layer):
    return [father[i] if height[father[i]] <= layer+1 else i for i in range(len(father))]


def remove_edge_focus_verb(father, height, layer):
    return [father[i] if height[i] <= layer else i for i in range(len(father))]


def union(father):
    def find(idx):
        while True:
            if father[idx] == idx:
                return idx
            idx = father[idx]
    node_num = len(father)
    root = [i for i in range(node_num)]
    for i in range(node_num):
        root[i] = find(i)
    return root


def get_attention_mask(node_cls):
    node_num = len(node_cls)
    cats = set(node_cls)
    mask = torch.zeros(node_num, node_num, dtype=torch.long)
    for cat in cats:
        nodes = [i for i in range(node_num) if node_cls[i] == cat]
        row = torch.tensor(nodes).unsqueeze(1)
        col = torch.tensor(nodes).unsqueeze(0)
        mask[row, col] = 1
    return mask


def tree_to_mask(tree):
    pos, tag, h = sort_tree_by_height(tree)
    fa = format_fa(pos)
    fa = remove_edge(fa, h, layer=2)
    node_cat = union(fa)
    attention_mask = get_attention_mask(node_cat)

    # tree.pretty_print()
    # ind = list(range(len(pos)))
    # print(tabulate([["tag"]+tag,
    #                 ["height"]+h,
    #                 ["index"]+ind,
    #                 ["father"]+fa,
    #                 ["class"]+node_cat], tablefmt="pretty"))
    #
    # mask_tab = [[tag[i]] + attention_mask.tolist()[i] for i in range(attention_mask.shape[0])]
    # print(tabulate(mask_tab, tag, tablefmt="pretty"))

    return attention_mask


if __name__ == '__main__':
    _t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
    mask = tree_to_mask(_t)
