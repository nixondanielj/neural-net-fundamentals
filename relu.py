class RELU:
    def forward(self, x):
        return min(x, 1)
    
    def backward(self, x):
        return 5