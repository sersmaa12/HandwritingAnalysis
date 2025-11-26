from strsimpy.levenshtein import Levenshtein

lev = Levenshtein()
word1 = "нэг үйл"
word2 = "нэг үйл"


dist = lev.distance(word1, word2)
print("distance:", dist)  
