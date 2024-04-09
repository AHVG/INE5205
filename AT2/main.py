import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


def filter_excel(df, product, packaging, seg, brand):
    return df[(df['Marca'] == brand) &
              (df[packaging].notna()) &
              (df["Seg"] == seg) &
              (df["Produto"] == product)]

class Grouping:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def midpoint(self):
        return (self.lower + self.upper) / 2.0

    def __contains__(self, item):
        return self.lower <= item < self.upper

    def __repr__(self):
        return f"{self.lower:5.2f} |- {self.upper:5.2f}"
    

class BaseStatsCalculator(ABC):
    def __init__(self, df, column_name):
        self.df = df
        self.column_name = column_name

        self.N = len(df)
        self.min = df[column_name].min()
        self.max = df[column_name].max()
        self.R = self.max - self.min

        self.calculate_num_class()
        self.C = round((self.R / self.num_class) * 1.1, 2)

        self.spec = pd.DataFrame({
            "N": [self.N],
            "min": [self.min],
            "max": [self.max],
            "R": [self.R],
            "num classes": [self.num_class],
            "C": [self.C],
        })

        self.model = {
            "k": [],
            "CLASSES": [],
            "Freq": [],
            "pi": [],
            "Pi": [],
            "Xi": [],
            "Xi*pi": [],
            "di^2*pi": [],
        }

        Pi = 0

        for i in range(self.num_class):
            k = i + 1
            classe = Grouping(self.min + self.C * i, self.min + self.C * (i + 1))
            freq = len([value for value in self.df[self.column_name] if value in classe])
            pi = freq / self.N
            Pi += pi
            Xi = classe.midpoint()
            Xi_pi = pi * Xi
            di_2_pi = 0

            self.model["k"].append(k)
            self.model["CLASSES"].append(classe)
            self.model["Freq"].append(freq)
            self.model["pi"].append(pi)
            self.model["Pi"].append(Pi)
            self.model["Xi"].append(Xi)
            self.model["Xi*pi"].append(Xi_pi)
        
        self.weighted_average = sum(self.model["Xi*pi"])
        self.simple_average = self.df[self.column_name].mean()

        for i in range(self.num_class):
            di_2_pi = (self.model["Xi"][i] - self.weighted_average)**2 * self.model["pi"][i]
            self.model["di^2*pi"].append(di_2_pi)

        self.model = pd.DataFrame(self.model)

        self.var = sum(self.model["di^2*pi"])
        self.desvpad = math.sqrt(self.var)
        self.erro_relativo_agrupamento = 100.0 * abs(self.simple_average - self.weighted_average) / self.simple_average
        self.cv = self.desvpad / self.weighted_average
        self.moda = self.model['Xi'][self.model['Freq'].idxmax()]
        self.assimetria = (self.weighted_average - self.moda) / self.desvpad

        self.descriptive_measures = pd.DataFrame({
            "var": [self.var],
            "desvpad": [self.desvpad],
            "erro relativo agrupamento": [self.erro_relativo_agrupamento],
            "cv": [self.cv],
            "moda": [self.moda],
            "assimetria": [self.assimetria],
        })

    @abstractmethod
    def calculate_num_class(self):
        pass

class RaizNCalculator(BaseStatsCalculator):
    def calculate_num_class(self):
        self.num_class = int(round(math.sqrt(self.N), 0))

class SturgesCalculator(BaseStatsCalculator):
    def calculate_num_class(self):
        self.num_class = int(round(1 + 3.32 * math.log10(self.N), 0))

def make_histogram(df):
    infos = SturgesCalculator(df, "C03 - VIDRO 600ML RET")

    print(pd.concat([infos.spec, infos.model, infos.descriptive_measures], axis=1))

    plt.hist(df["C03 - VIDRO 600ML RET"], bins=infos.num_class, edgecolor='black')
    plt.xlabel('Preço ($)')
    plt.ylabel('Frequência')
    plt.title('Histograma dos Preços das Bebidas')
    plt.show()

    df["C03 - VIDRO 600ML RET"].plot(kind='box')
    plt.text(x=1, y=df["C03 - VIDRO 600ML RET"].min(), s='X1', ha='center', va='top')
    plt.text(x=1, y=df["C03 - VIDRO 600ML RET"].quantile(0.25), s='Q1', ha='center', va='center')
    plt.text(x=1, y=df["C03 - VIDRO 600ML RET"].median(), s='Q2', ha='center', va='center')
    plt.text(x=1, y=df["C03 - VIDRO 600ML RET"].quantile(0.75), s='Q3', ha='center', va='center')
    plt.text(x=1, y=df["C03 - VIDRO 600ML RET"].max(), s='Xn', ha='center', va='bottom')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Lê um arquivo Excel e imprime na tela.")
    parser.add_argument("--excel_path", help="Caminho do arquivo Excel para ser lido.", default="./doc/BASE DADOS_DESAFIO INDIVIDUAL.xlsx")
    args = parser.parse_args()

    excel_path = args.excel_path
    df = pd.read_excel(excel_path)

    amstel_lager = filter_excel(df, "CERVEJA", "C03 - VIDRO 600ML RET", 2, "AMSTEL LAGER")
    antarctica_pilsen = filter_excel(df, "CERVEJA", "C03 - VIDRO 600ML RET", 2, "ANTARCTICA PILSEN")

    make_histogram(amstel_lager)
    make_histogram(antarctica_pilsen)

main()
