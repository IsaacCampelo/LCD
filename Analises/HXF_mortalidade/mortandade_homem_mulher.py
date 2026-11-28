import pandas as pd
import matplotlib.pyplot as plt

COLS_2 = ["TIPOBITO", "DTOBITO", "SEXO"]
INICIO_PANDEMIA = pd.Timestamp("2020-03-01")


def carrega_dados_analise_2(path):
    '''Carrega o CSV de um ano do SIM, só com as colunas da análise de óbitos por sexo.'''
    df = pd.read_csv(path, sep=';', encoding='latin-1', dtype=str,
                     usecols=lambda c: c in COLS_2, low_memory=False)
    return df.reindex(columns=COLS_2)  # garante todas as colunas, mesmo que o layout do ano varie


def carrega_dados_range_anos(start, stop, loader):
    '''Retorna uma tupla com os dados do SIM de cada ano do intervalo, em ordem crescente.'''
    datasets = []
    for year in range(start, stop + 1):
        url = f"https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/csv/Mortalidade_Geral_{year}_csv.zip"
        datasets.append(loader(url))
    return tuple(datasets)


SIM_2018, SIM_2019, SIM_2020, SIM_2021, SIM_2022 = carrega_dados_range_anos(2018, 2022, carrega_dados_analise_2)
anos = [SIM_2018, SIM_2019, SIM_2020, SIM_2021, SIM_2022]


def prepara_mensal_sexo(df):
    '''Mantém óbitos não fetais (exclui mortes intrauterinas, TIPOBITO == 2), padroniza o sexo
    (M/F) e cria a coluna mes_ref (primeiro dia do mês do óbito). Sexo ignorado é descartado.'''
    df = df[df["TIPOBITO"] == "2"].copy()
    df["sexo"] = df["SEXO"].map({"1": "M", "M": "M", "2": "F", "F": "F"})
    df = df.dropna(subset=["sexo"])
    d = df["DTOBITO"].fillna("").str.zfill(8)  # ddmmaaaa
    df["mes_ref"] = pd.to_datetime(d.str[4:8] + "-" + d.str[2:4] + "-01", errors="coerce")
    return df.dropna(subset=["mes_ref"])


# 1) Base única, não fetal, com sexo e mês
sexo_m = prepara_mensal_sexo(pd.concat(anos, ignore_index=True))

# 2) Contagem mensal por sexo
tabela = pd.DataFrame({
    "homens": sexo_m[sexo_m["sexo"] == "M"].groupby("mes_ref").size(),
    "mulheres": sexo_m[sexo_m["sexo"] == "F"].groupby("mes_ref").size(),
})
tabela["razao_homem_mulher"] = tabela["homens"] / tabela["mulheres"]

# 3) Índice: média mensal de 2019 = 100, para comparar a evolução relativa
base = tabela.loc["2019", ["homens", "mulheres"]].mean()
indice = tabela[["homens", "mulheres"]] / base * 100

# 4) Gráficos
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

axes[0].plot(tabela.index, tabela["homens"], marker="o", ms=3, color="tab:blue")
axes[0].set_title("Óbitos por mês: homens")

axes[1].plot(tabela.index, tabela["mulheres"], marker="o", ms=3, color="tab:red")
axes[1].set_title("Óbitos por mês: mulheres")

axes[2].plot(indice.index, indice["homens"], label="Homens", color="tab:blue")
axes[2].plot(indice.index, indice["mulheres"], label="Mulheres", color="tab:red")
axes[2].axhline(100, color="grey", ls="--", lw=1)
axes[2].set_title("Evolução relativa (média mensal de 2019 = 100)")
axes[2].legend()

for ax in axes:
    ax.axvline(INICIO_PANDEMIA, color="black", ls=":", lw=1)  # início da pandemia (mar/2020)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("comparacao_mensal_sexo.png", dpi=150)

tabela.to_csv("tabela_mensal_sexo.csv")
print(tabela.round(2).to_string())