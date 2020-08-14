class FATHER():
    def __init__(self):
        self.x = 123
        # self.y = 903


class SON(FATHER):
    def __init__(self):
        self.y = 456
        # super(SON, self).__init__()



son = SON()
father = FATHER()
print(son.y)
print(son.father.x)
# print(father.y)
