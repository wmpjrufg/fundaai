"""Engineering layer — pure analytical checks compliant with NBR 6118 / NBR 6122.

This subpackage will host the stateless functions that compute
admissible soil pressure, applied stresses (composite bending),
punching shear at section C, geometric overhang and AABB overlap.

Each function takes plain numerical arguments and returns plain numbers
or simple dataclasses — no DataFrames in the signature. DataFrame
adapters live one layer up (``core.api`` or ``core.io``).

Resumo em português:
    Verificações analíticas puras (NBR 6118/6122). Funções sem estado:
    tensão admissível do solo, σ_max e σ_min, punção (seção C),
    geometria mínima e sobreposição AABB. Sem DataFrames na interface.
"""
