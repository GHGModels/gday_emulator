#!/usr/bin/env python

import os, re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation
from cStringIO import StringIO
from sklearn.grid_search import GridSearchCV


def main(met_fname, gday_outfname, var):

    # Load met data
    s = remove_comments_from_header(met_fname)
    df_met = pd.read_csv(s, parse_dates=[[0,1]], skiprows=4, index_col=0,
                         sep=",", keep_date_col=True,
                         date_parser=date_converter)

    # Need to build numpy array, so drop year, doy cols
    met_data = df_met.ix[:,2:].values

    # Load GDAY outputs
    df = pd.read_csv(gday_outfname, skiprows=3, sep=",", skipinitialspace=True)
    df['date'] = make_data_index(df)
    df = df.set_index('date')
    target = df[var].values

    # build emulator
    regmod = KNeighborsRegressor(n_neighbors=20, weights="distance")
    regmod.fit(met_data, df[var])
    predict = regmod.predict(met_data)

    df = pd.DataFrame({'DT': df.index, 'emu': predict, 'gday': df[var]})

    plt.plot_date(df.index, df['emu'], 'o', label='Emulator')
    plt.plot_date(df.index, df['gday'], '.', label='GDAY')
    plt.ylabel('GPP (g C m$^{-2}$ s$^{-1}$)')
    plt.legend()
    plt.show()

def make_data_index(df):
    dates = []
    for index, row in df.iterrows():
        s = str(int(float(row['YEAR']))) + " " + str(int(float(row['DOY'])))
        dates.append(dt.datetime.strptime(s, '%Y %j'))
    return dates

def date_converter(*args):
    return dt.datetime.strptime(str(int(float(args[0]))) + " " +\
                                str(int(float(args[1]))), '%Y %j')

def remove_comments_from_header(fname):
    """ I have made files with comments which means the headings can't be
    parsed to get dictionary headers for pandas! Solution is to remove these
    comments first """
    s = StringIO()
    with open(fname) as f:
        for line in f:
            if '#' in line:
                line = line.replace("#", "").lstrip(' ')
            s.write(line)
    s.seek(0) # "rewind" to the beginning of the StringIO object

    return s


if __name__ == "__main__":

    fdir = ("/Users/%s/research/FACE/gday_simulations/"
            "DUKE/step_change/met_data" % (os.getlogin()))
    met_fname = os.path.join(fdir, "DUKE_met_data_amb_co2.csv")

    fdir = ("/Users/%s/research/FACE/gday_simulations/"
            "DUKE/step_change/outputs" % (os.getlogin()))
    gday_outfname = os.path.join(fdir, "D1GDAYDUKEAMB.csv")

    var = "GPP"
    main(met_fname, gday_outfname, var)
