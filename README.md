# Tree-Transformer
Transformer integrating parsed tree-structre knowledge.


---
### Pipeline
We modified the ```torch.nn.TransformerEncoder``` class and feed different masks for different layers. For example, in the following case, the words are only visible to the nodes in the same subtree with height 2 in the 1st layer. 
```
              S               
      ________|_____           
     |              VP        
     |         _____|___       
     NP       |         NP    
  ___|___     |      ___|___   
 D       N    V     D       N 
 |       |    |     |       |  
the     dog chased the     cat
```

### Usage
To use this package, you need to conert your parsed tree into ```nltk.tree.Tree```. Here is an example that converts ```CoreNLP_pb2.ParseTree``` into ```nltk.tree.Tree```:

```python
from nltk.tree import Tree
def convert_parse_tree_to_nltk_tree(parse_tree):
    return Tree(parse_tree.value, [get_nltk_tree(child) for child in parse_tree.child]) if parse_tree.child else parse_tree.value

tree = convert_parse_tree_to_nltk_tree(constituency_parse)
```

Then you can get the mask with the converted tree object and feed the inputs into tree transformer.

```python
def tok_list_to_id(x):
    """ customize
    """
    return torch.arange(0, 14, 1).long()

from nltk.tree import Tree
from utils import tree_to_mask

vocab_size, d_word, d_model = 100, 500, 500
embed = nn.Embedding(vocab_size, d_word)
ttf_encoder = TreeTransformer(d_model)

t = Tree.fromstring('(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))')
t.pretty_print()

cap_toks, att_mask = tree_to_mask(t)
cap_id = tok_list_to_id(cap_toks)
att_mask = att_mask.unsqueeze(0)  # add batch dim
cap_id = cap_id.unsqueeze(0)  # add batch dim

output, hidden = ttf_encoder(embed(cap_id), att_mask)
print("out:", output.shape)
```

### Example

This is the constitute tree of the sentence: "the dog chased the cat".  
```
              S               
      ________|_____           
     |              VP        
     |         _____|___       
     NP       |         NP    
  ___|___     |      ___|___   
 D       N    V     D       N 
 |       |    |     |       |  
the     dog chased the     cat
```

In the first layer, a word is only visible to itself, the words relationship, class and corresponding mask are like this:
```
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
|  tag   | S | VP | NP | NP | D | N | V | D | N | the | dog | chased | the | cat |
| height | 5 | 4  | 3  | 3  | 2 | 2 | 2 | 2 | 2 |  1  |  1  |   1    |  1  |  1  |
| index  | 0 | 1  | 2  | 3  | 4 | 5 | 6 | 7 | 8 |  9  | 10  |   11   | 12  | 13  |
| father | 0 | 1  | 2  | 3  | 4 | 5 | 6 | 7 | 8 |  9  | 10  |   11   | 12  | 13  |
| class  | 0 | 1  | 2  | 3  | 4 | 5 | 6 | 7 | 8 |  9  | 10  |   11   | 12  | 13  |
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+

+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
|        | S | VP | NP | NP | D | N | V | D | N | the | dog | chased | the | cat |
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
|   S    | 1 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   VP   | 0 | 1  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   NP   | 0 | 0  | 1  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   NP   | 0 | 0  | 0  | 1  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   D    | 0 | 0  | 0  | 0  | 1 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   N    | 0 | 0  | 0  | 0  | 0 | 1 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   V    | 0 | 0  | 0  | 0  | 0 | 0 | 1 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   D    | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 1 | 0 |  0  |  0  |   0    |  0  |  0  |
|   N    | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 1 |  0  |  0  |   0    |  0  |  0  |
|  the   | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  1  |  0  |   0    |  0  |  0  |
|  dog   | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  1  |   0    |  0  |  0  |
| chased | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   1    |  0  |  0  |
|  the   | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  1  |  0  |
|  cat   | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  1  |
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
```

In the second layer, a word is visible to the words in its height 1 subtree:
```
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
|  tag   | S | VP | NP | NP | D | N | V | D | N | the | dog | chased | the | cat |
| height | 5 | 4  | 3  | 3  | 2 | 2 | 2 | 2 | 2 |  1  |  1  |   1    |  1  |  1  |
| index  | 0 | 1  | 2  | 3  | 4 | 5 | 6 | 7 | 8 |  9  | 10  |   11   | 12  | 13  |
| father | 0 | 1  | 2  | 3  | 4 | 5 | 6 | 7 | 8 |  4  |  5  |   6    |  7  |  8  |
| class  | 0 | 1  | 2  | 3  | 4 | 5 | 6 | 7 | 8 |  4  |  5  |   6    |  7  |  8  |
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+

+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
|        | S | VP | NP | NP | D | N | V | D | N | the | dog | chased | the | cat |
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
|   S    | 1 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   VP   | 0 | 1  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   NP   | 0 | 0  | 1  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   NP   | 0 | 0  | 0  | 1  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   D    | 0 | 0  | 0  | 0  | 1 | 0 | 0 | 0 | 0 |  1  |  0  |   0    |  0  |  0  |
|   N    | 0 | 0  | 0  | 0  | 0 | 1 | 0 | 0 | 0 |  0  |  1  |   0    |  0  |  0  |
|   V    | 0 | 0  | 0  | 0  | 0 | 0 | 1 | 0 | 0 |  0  |  0  |   1    |  0  |  0  |
|   D    | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 1 | 0 |  0  |  0  |   0    |  1  |  0  |
|   N    | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 1 |  0  |  0  |   0    |  0  |  1  |
|  the   | 0 | 0  | 0  | 0  | 1 | 0 | 0 | 0 | 0 |  1  |  0  |   0    |  0  |  0  |
|  dog   | 0 | 0  | 0  | 0  | 0 | 1 | 0 | 0 | 0 |  0  |  1  |   0    |  0  |  0  |
| chased | 0 | 0  | 0  | 0  | 0 | 0 | 1 | 0 | 0 |  0  |  0  |   1    |  0  |  0  |
|  the   | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 1 | 0 |  0  |  0  |   0    |  1  |  0  |
|  cat   | 0 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 1 |  0  |  0  |   0    |  0  |  1  |
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
```

Similarly, a word is visible to the words in its height 2 subtree in the third layer:
```
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
|  tag   | S | VP | NP | NP | D | N | V | D | N | the | dog | chased | the | cat |
| height | 5 | 4  | 3  | 3  | 2 | 2 | 2 | 2 | 2 |  1  |  1  |   1    |  1  |  1  |
| index  | 0 | 1  | 2  | 3  | 4 | 5 | 6 | 7 | 8 |  9  | 10  |   11   | 12  | 13  |
| father | 0 | 1  | 2  | 3  | 2 | 2 | 6 | 3 | 3 |  4  |  5  |   6    |  7  |  8  |
| class  | 0 | 1  | 2  | 3  | 2 | 2 | 6 | 3 | 3 |  2  |  2  |   6    |  3  |  3  |
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+

+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
|        | S | VP | NP | NP | D | N | V | D | N | the | dog | chased | the | cat |
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
|   S    | 1 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   VP   | 0 | 1  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |  0  |  0  |   0    |  0  |  0  |
|   NP   | 0 | 0  | 1  | 0  | 1 | 1 | 0 | 0 | 0 |  1  |  1  |   0    |  0  |  0  |
|   NP   | 0 | 0  | 0  | 1  | 0 | 0 | 0 | 1 | 1 |  0  |  0  |   0    |  1  |  1  |
|   D    | 0 | 0  | 1  | 0  | 1 | 1 | 0 | 0 | 0 |  1  |  1  |   0    |  0  |  0  |
|   N    | 0 | 0  | 1  | 0  | 1 | 1 | 0 | 0 | 0 |  1  |  1  |   0    |  0  |  0  |
|   V    | 0 | 0  | 0  | 0  | 0 | 0 | 1 | 0 | 0 |  0  |  0  |   1    |  0  |  0  |
|   D    | 0 | 0  | 0  | 1  | 0 | 0 | 0 | 1 | 1 |  0  |  0  |   0    |  1  |  1  |
|   N    | 0 | 0  | 0  | 1  | 0 | 0 | 0 | 1 | 1 |  0  |  0  |   0    |  1  |  1  |
|  the   | 0 | 0  | 1  | 0  | 1 | 1 | 0 | 0 | 0 |  1  |  1  |   0    |  0  |  0  |
|  dog   | 0 | 0  | 1  | 0  | 1 | 1 | 0 | 0 | 0 |  1  |  1  |   0    |  0  |  0  |
| chased | 0 | 0  | 0  | 0  | 0 | 0 | 1 | 0 | 0 |  0  |  0  |   1    |  0  |  0  |
|  the   | 0 | 0  | 0  | 1  | 0 | 0 | 0 | 1 | 1 |  0  |  0  |   0    |  1  |  1  |
|  cat   | 0 | 0  | 0  | 1  | 0 | 0 | 0 | 1 | 1 |  0  |  0  |   0    |  1  |  1  |
+--------+---+----+----+----+---+---+---+---+---+-----+-----+--------+-----+-----+
```
