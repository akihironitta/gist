import torch
from torch_geometric.utils import scatter

done = torch.tensor([False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False,  True, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,  True,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False,  True, False, False, False, False, False,
    False, False, False,  True, False, False, False, False, False, False,
    False, False, False, False, False, False, False,  True, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False,  True, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False,  True, False, False, False, False,
    False, False, False, False, False,  True, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False,  True, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False], dtype=torch.bool)

index = torch.tensor(
    [  0,   0,   0,   1,   1,   1,   1,   2,   2,   2,   3,   3,   3,   3,
      4,   4,   4,   5,   5,   5,   6,   6,   7,   7,   7,   8,   9,   9,
      9,   9,  10,  10,  11,  11,  12,  12,  13,  13,  13,  14,  14,  15,
     15,  15,  16,  16,  16,  16,  17,  17,  18,  18,  18,  19,  19,  19,
     19,  20,  20,  20,  21,  21,  21,  22,  22,  23,  23,  23,  23,  24,
     25,  25,  25,  25,  26,  27,  27,  28,  28,  28,  29,  30,  30,  30,
     30,  31,  31,  31,  32,  32,  32,  33,  33,  34,  34,  34,  34,  35,
     35,  35,  35,  36,  36,  37,  37,  38,  38,  38,  39,  39,  39,  40,
     40,  40,  41,  42,  42,  43,  43,  43,  44,  44,  44,  45,  46,  46,
     46,  47,  47,  47,  48,  48,  48,  48,  49,  49,  49,  50,  51,  51,
     51,  52,  52,  52,  52,  53,  53,  53,  54,  54,  55,  55,  56,  56,
     56,  57,  57,  57,  58,  58,  59,  59,  60,  60,  61,  61,  61,  62,
     63,  63,  63,  64,  64,  64,  65,  65,  66,  66,  67,  67,  67,  68,
     68,  68,  69,  69,  70,  71,  71,  72,  72,  73,  73,  73,  74,  74,
     75,  75,  75,  76,  76,  76,  77,  78,  79,  80,  80,  81,  81,  81,
     82,  82,  82,  82,  83,  83,  84,  85,  85,  85,  86,  86,  87,  87,
     87,  88,  88,  88,  88,  89,  89,  90,  90,  90,  91,  92,  92,  92,
     92,  92,  93,  94,  94,  94,  95,  95,  95,  95,  96,  96,  96,  97,
     97,  98,  98,  99, 100, 100, 101, 102, 102, 102, 103, 103, 103, 104,
    105, 105, 105, 106, 106, 106, 106, 107, 107, 107, 107, 108, 108, 108,
    109, 109, 109, 110, 110, 110, 110, 111, 111, 111, 112, 112, 113, 114,
    114, 115, 115, 115, 115, 116, 116, 116, 116, 117, 118, 118, 118, 119,
    120, 121, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125,
    125, 126, 126, 127, 127, 127], dtype=torch.long)

assert len(done) == len(index)

def doit():
    print("Short - Long")
    print(torch.where(
        scatter(done.short(), index=index.long(), dim=0, reduce="mul") == 1))

    print("Long - Long")
    print(torch.where(
        scatter(done.long(), index=index.long(), dim=0, reduce="mul") == 1))

print("-----------------")
print("On CPU")
print("-----------------")
done = done.cpu()
index = index.cpu()
doit()

print("-----------------")
print("On GPU")
print("-----------------")
done = done.cuda()
index = index.cuda()
doit()