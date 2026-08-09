import sympy
from sympy.parsing.sympy_parser import parse_expr

gd = {"__builtins__": {}}
gd.update(sympy.__dict__)

gp = parse_expr("2+2", global_dict=gd, evaluate=False)
pp = parse_expr("4", global_dict=gd, evaluate=False)
print("gp:", gp)
print("pp:", pp)
print("sympy.simplify(gp - pp) == 0 :", sympy.simplify(gp - pp) == 0)

gp2 = parse_expr("sqrt(4)", global_dict=gd, evaluate=False)
pp2 = parse_expr("2", global_dict=gd, evaluate=False)
print("sympy.simplify(gp2 - pp2) == 0 :", sympy.simplify(gp2 - pp2) == 0)
