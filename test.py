import torch

test = [[[1,2,3],[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3],[1,2,4]]]
test = torch.tensor(test).float()
test_mult = torch.tensor([[0,0,1],[1,0,0]]).float()
print(test.size())
print(test_mult.size())
answer_size = 1+test_mult.size()[0]
# print(test)
test = test[:,-answer_size:-1, :]
# print(test)
# print(test.size())
softmax = torch.nn.Softmax(dim=2)
test = softmax(test)
# print(test)
print(test)
answer = torch.log(test)
print(answer)
answer = torch.mul(answer, test_mult)

print(answer)
answer = torch.sum(answer, (1,2))
print(answer)