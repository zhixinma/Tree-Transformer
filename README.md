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
import torch
import torch.nn as nn
from tree_transformer import tree_to_mask
from tree_transformer_mask import TreeTransformer

embed = nn.Embedding(vocab_size, word_dim)
ttf_encoder = TreeTransformer(d_model)

cap_toks, att_mask = tree_to_mask()
cap_id = tok_list_to_id(cap_toks)

output, hidden = ttf_encoder(embed(cap_id), att_mask)
```

### Example

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

------  -  --  --  --  -  -  -  -  -  ---  ---  ------  ---  ---
tag     S  VP  NP  NP  D  N  V  D  N  the  dog  chased  the  cat
height  5  4   3   3   2  2  2  2  2  1    1    1       1    1
index   0  1   2   3   4  5  6  7  8  9    10   11      12   13
father  0  1   2   3   4  5  6  7  8  4    5    6       7    8
class   0  1   2   3   4  5  6  7  8  4    5    6       7    8
------  -  --  --  --  -  -  -  -  -  ---  ---  ------  ---  ---
```
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

------  -  --  --  --  -  -  -  -  -  ---  ---  ------  ---  ---
tag     S  VP  NP  NP  D  N  V  D  N  the  dog  chased  the  cat
height  5  4   3   3   2  2  2  2  2  1    1    1       1    1
index   0  1   2   3   4  5  6  7  8  9    10   11      12   13
father  0  1   2   3   2  2  1  3  3  4    5    6       7    8
class   0  1   2   3   2  2  1  3  3  2    2    1       3    3
------  -  --  --  --  -  -  -  -  -  ---  ---  ------  ---  ---
```

The mask:
```
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
