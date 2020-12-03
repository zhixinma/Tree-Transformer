# Tree-Transformer
Transformer with pre-defined tree structre

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
