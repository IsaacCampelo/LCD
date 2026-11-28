import pandas as pd
import matplotlib.pyplot as plt

COLS = ["TIPOBITO", "DTOBITO", "SEXO", "IDADE"]


def carrega_dados_analise_1(path):
    '''Carrega o CSV de um ano do SIM, só com as colunas da análise de mortalidade e gravidez.'''
    df = pd.read_csv(path, sep=';', encoding='latin-1', dtype=str,
                     usecols=lambda c: c in COLS, low_memory=False)
    return df.reindex(columns=COLS)  # garante todas as colunas, mesmo que o layout do ano varie


def carrega_dados_range_anos(start, stop):
    '''Retorna uma tupla com os dados do SIM de cada ano do intervalo, em ordem crescente.'''
    datasets = []
    for year in range(start, stop + 1):
        url = f"https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/csv/Mortalidade_Geral_{year}_csv.zip"
        datasets.append(carrega_dados_analise_1(url))
    return tuple(datasets)


def decodifica_idade(serie):
    '''Converte o campo IDADE (1 dígito de unidade + 2 de quantidade) para anos.'''
    s = serie.fillna("").str.zfill(3)
    unidade = s.str[0]
    qtd = pd.to_numeric(s.str[1:], errors="coerce")
    anos = pd.Series(float("nan"), index=serie.index)
    anos.loc[unidade.isin(["1", "2", "3"])] = 0                      # menor de 1 ano
    anos.loc[unidade == "4"] = qtd[unidade == "4"]                   # anos
    anos.loc[unidade == "5"] = 100 + qtd[unidade == "5"]             # 100 anos ou mais
    return anos


SIM_2018,SIM_2019,SIM_2020,SIM_2021,SIM_2022 = carrega_dados_range_anos(2018, 2022)

def seleciona_idade_fertil(df, idade_min=10, idade_max=49):
    '''Mantém apenas as observações de mulheres em idade fértil (10 a 49 anos por padrão).
    Usa as colunas SEXO e IDADE do SIM e acrescenta a coluna idade_anos.'''
    df = df.copy()
    df["idade_anos"] = decodifica_idade(df["IDADE"])
    mulher = df["SEXO"].isin(["2", "F"])
    fertil = df["idade_anos"].between(idade_min, idade_max).fillna(False)
    return df[mulher & fertil].reset_index(drop=True)

anos = [SIM_2018, SIM_2019, SIM_2020, SIM_2021, SIM_2022]
fertil = pd.concat([seleciona_idade_fertil(d) for d in anos], ignore_index=True)

def prepara_mensal(df):
    '''Mantém óbitos não fetais e cria a coluna mes_ref (primeiro dia do mês do óbito).'''
    df = df[df["TIPOBITO"] == "2"].copy()
    d = df["DTOBITO"].fillna("").str.zfill(8)  # ddmmaaaa
    df["mes_ref"] = pd.to_datetime(d.str[4:8] + "-" + d.str[2:4] + "-01", errors="coerce")
    return df.dropna(subset=["mes_ref"])

geral = prepara_mensal(pd.concat(anos, ignore_index=True))
fertil_m = prepara_mensal(fertil)

# 2) Contagem mensal dos dois grupos
tabela = pd.DataFrame({
    "geral": geral.groupby("mes_ref").size(),
    "mulheres_10_49": fertil_m.groupby("mes_ref").size(),
})
tabela["pct_mulheres_10_49"] = tabela["mulheres_10_49"] / tabela["geral"] * 100

# 3) Índice: média mensal de 2019 = 100, para comparar a evolução relativa
base = tabela.loc["2019", ["geral", "mulheres_10_49"]].mean()
indice = tabela[["geral", "mulheres_10_49"]] / base * 100

# 4) Gráficos
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

axes[0].plot(tabela.index, tabela["geral"], marker="o", ms=3)
axes[0].set_title("Óbitos por mês: geral")

axes[1].plot(tabela.index, tabela["mulheres_10_49"], marker="o", ms=3, color="tab:red")
axes[1].set_title("Óbitos por mês: mulheres de 10 a 49 anos")

axes[2].plot(indice.index, indice["geral"], label="Geral")
axes[2].plot(indice.index, indice["mulheres_10_49"], label="Mulheres 10-49", color="tab:red")
axes[2].axhline(100, color="grey", ls="--", lw=1)
axes[2].set_title("Evolução relativa (média mensal de 2019 = 100)")
axes[2].legend()

for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("comparacao_mensal.png", dpi=150)

tabela.to_csv("tabela_mensal.csv")
print(tabela.round(1).to_string())

