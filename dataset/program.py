# bridging file to enable unpickling of datasets in which the 
# FactBase class was still known as Program
from factbase import FactBase, PredicateDeclaration, Constant
Program = FactBase