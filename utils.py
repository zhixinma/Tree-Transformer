from nltk.tree import Tree
import torch

MAX_TREE_HEIGHT = 12


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
    """ remove the invisible edges in different layers
    """
    return [father[i] if height[father[i]] <= layer+1 else i for i in range(len(father))]


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
    attention_masks = []
    layer_num = max(max(h), MAX_TREE_HEIGHT)
    for i in range(layer_num):
        fa_i = remove_edge(fa, h, layer=i)
        node_cat = union(fa_i)
        attention_masks.append(get_attention_mask(node_cat))
    return tag, torch.stack(attention_masks).bool()


if __name__ == '__main__':
    t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
    t.pretty_print()

    ids, masks = tree_to_mask(t)
    print(ids)
    print("mask:", masks.shape)

