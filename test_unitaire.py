import unittest

characters = ["a",
              "\sum ",
              "\\forall ",
              "\exists ",
              "\int ",
              "\mathbb{R}",
              "\in ",
              ",",
              "x",
              "\geq ",
              "\leq ",
              "=",
              "i",
              "n"]

def list_str(list):
    return [str(e) for e in list]

class Symbol():
    def __init__(self, i, height,rect, parent = None):
        miny, maxy, minx, maxx = rect
        self.parent = parent
        self.height = height         #-1 : indice, 0 : normal, 1 : exposant
        self.y = (miny+maxy)//2
        self.base_character = characters[i]
        self.indices = []
        self.exposants = []
        self.rect = rect
        self.last_addition = None
    
    def __str__(self):
        if len(self.indices) > 0:
            if len(self.exposants) > 0:
                return self.base_character + "_{" + "".join(list_str(self.indices))+ "}^{" + "".join(list_str(self.exposants)) + "}"
            else:
                return self.base_character + "_{" + "".join(list_str(self.indices))+ "}"
        else:
            if len(self.exposants) > 0:
                return self.base_character + "^{" + "".join(list_str(self.exposants)) + "}"
            else:
                return self.base_character

class TestSymbol(unittest.TestCase):
    def test_exposant(self):
        symb = Symbol(0,0,[0,0,0,0], None)
        symb.exposants.append(Symbol(1,0,[0,0,0,0], None))
        out = str(symb)
        self.assertEqual(out,str(characters[0]) + "^{" + str(characters[1]) + "}")
    def test_indice(self):
        symb = Symbol(0,0,[0,0,0,0], None)
        symb.indices.append(Symbol(1,0,[0,0,0,0], None))
        out = str(symb)
        self.assertEqual(out,str(characters[0]) + "_{" + str(characters[1]) + "}")
    def test_both(self):
        symb = Symbol(0,0,[0,0,0,0], None)
        symb.indices.append(Symbol(1,0,[0,0,0,0], None))
        symb.exposants.append(Symbol(2,0,[0,0,0,0], None))
        out = str(symb)
        self.assertEqual(out,str(characters[0]) + "_{" + str(characters[1]) +"}^{" + str(characters[2]) + "}")
    def test_neither(self):
        symb = Symbol(0,0,[0,0,0,0], None)
        out = str(symb)
        self.assertEqual(out,str(characters[0]))


if __name__ == '__main__':
    unittest.main()