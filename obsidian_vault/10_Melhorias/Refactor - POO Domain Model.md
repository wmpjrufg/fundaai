---
tags: [melhorias, refactor, poo, sugestao]
---

# Refactor — POO Domain Model

> [!note] Sugestão
> Hoje todo o projeto trabalha com `pandas.DataFrame` "wide" (uma coluna por combinação). Isso é prático para Excel mas frágil para testar e para evoluir o domínio. Modelar com **classes** explícitas remove a "magia" de strings em colunas.

## Entidades propostas

### `Solo`
```python
@dataclass(frozen=True)
class Solo:
    tipo: Literal["pedregulho", "areia", "silte", "argila"]
    spt: float
    @property
    def sigma_adm_kpa(self) -> float: ...   # delega para engineering.tensao
```

### `Pilar`
```python
@dataclass(frozen=True)
class Pilar:
    rotulo: str
    a_p: float          # m
    b_p: float          # m
    xg: float           # m
    yg: float           # m
```

### `Combinacao`
```python
@dataclass(frozen=True)
class Combinacao:
    rotulo: str         # "c1"
    f_z: float          # kN
    m_x: float          # kN·m
    m_y: float          # kN·m
```

### `Sapata` (variável de projeto)
```python
@dataclass
class Sapata:
    h_x: float
    h_y: float
    h_z: float
    pilar: Pilar
    @property
    def volume(self) -> float: return self.h_x * self.h_y * self.h_z
    @property
    def vertices(self) -> list[tuple[float,float]]: ...
```

### `FundacaoProjeto` (raiz agregadora)
```python
@dataclass
class FundacaoProjeto:
    pilares: list[Pilar]
    solo_por_pilar: dict[str, Solo]
    combinacoes: dict[str, list[Combinacao]]   # por pilar
    f_ck: float
    cobrimento: float
    def avaliar(self, x: np.ndarray) -> ResultadoAvaliacao: ...
```

### `ResultadoAvaliacao`
```python
@dataclass(frozen=True)
class ResultadoAvaliacao:
    sapatas: list[Sapata]
    g_tensao: np.ndarray         # shape (N_fund,)
    g_puncao: np.ndarray
    g_geometria: np.ndarray
    g_sobreposicao: np.ndarray
    volume_bruto: float
    penalty_total: float
    @property
    def of(self) -> float: return self.volume_bruto + self.penalty_total
```

## Vantagens

- Tipagem estática (mypy/pyright pega bug antes de rodar).
- Imutabilidade dos dados de entrada (`frozen=True`) evita efeito colateral.
- Cada classe tem testes unitários simples.
- O `FundacaoProjeto.avaliar(x)` substitui `obj_felipe_lucas` e retorna **estrutura tipada** em vez de DataFrame mutado.
- Adapter `to_dataframe(resultado)` para a UI/Excel quando necessário.

## Cuidado

- Não migrar tudo de uma vez. Comece por `Solo` (menor dependência).
- Manter o DataFrame como camada de **I/O**, não de domínio.

## Vínculos

- [[10_Melhorias/Refactor - Plano Geral]]
- [[10_Melhorias/Refactor - Configuração com Pydantic]]
- [[10_Melhorias/Refactor - Vetorização da FO]]
