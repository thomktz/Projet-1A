"""
Pseudo code utilisé pour les screenshots du Powerpoint
Ce programme ne fonctionnerait pas car Symbol attend 3 entrées et prend un indice, pas un string
"""

from main import Symbol, list_str


symbol = Symbol("\sum")

exposant = Symbol("n")
exposant.exposants += [Symbol("2")]
symbol.exposants += [exposant]

ind = [Symbol("i"), Symbol("="), Symbol("0")]
symbol.indices += ind

symbols = [symbol, Symbol("i")]


latex_to_img("".join(list_str(symbols)))

str(symbol) + str(Symbol("i"))

str(symbol) + "i"

"\sum" + "_{" + "".join(list_str(symbol.indices)) + "}^{" + "".join(list_str(symbol.exposants)) + "}" + "i"

"\sum_{i=0}^{" + str(exposant) + "} i"

"\sum_{i=0}^{" + "n" + "^{" + str(Symbol("2")) + "}" + "} i"

"\sum_{i=0}^{n^{2}} i"