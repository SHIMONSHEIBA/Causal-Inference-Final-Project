import torch as tr

a = tr.rand(3,4,5)

print(tr.__version__)

rnn = tr.nn.LSTMCell(10, 20)
print("rnn is", rnn)
input = tr.randn(6, 3, 10)
print("input is", input)
hx = tr.randn(3, 20)
print("hx is", hx)
cx = tr.randn(3, 20)
print("cx is", cx)
output = []
for i in range(6):
    hx, cx = rnn(input[i], (hx, cx))
    print("hx is", hx)
    print("cx is", cx)
    output.append(hx)
    print("output is", output)

