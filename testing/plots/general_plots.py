import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from psychoacoustics.psychometrics import psychometric_func, get_psychometric_point, fit_psychometric_func


def plot_thresholds(df, x_label, output_path, include_plotlyjs='directory'):
    fig = px.line(df,
                  x='experiment_parameter',
                  y='gap_threshold',
                  color='impairment',
                  facet_col='method',
                  title='Gap Threshold',
                  labels=dict(experiment_parameter=x_label,
                              gap_threshold="Gap Threshold [ms]"))
    fig.write_html(output_path,
                   include_plotlyjs=include_plotlyjs)


def scatter_and_psychometric_fit_plot(df, y, outputpath, title, with_scale, with_loc, ylimit=None, fit_offset=0):
    if ylimit is None:
        ylimit = [0, 1]
    fig = px.strip(df, x='gap_length', y=y, color='experiment_parameter', title=title)
    plot_x = np.linspace(0, 40, 100)
    for trace in fig['data']:
        trace['legendgroup'] = trace['name']
        df_at_lvl = df[df['experiment_parameter'] == float(trace['name'])]

        xdata = df_at_lvl['gap_length'].to_numpy()
        ydata = df_at_lvl[y].to_numpy()

        if len(xdata) <= fit_offset or len(ydata) <= fit_offset:
            print(f'Couldn\'t fit psychometric function to {trace["name"]}: not enough data')
            continue

        p = fit_psychometric_func(xdata=xdata[fit_offset:],
                                  ydata=ydata[fit_offset:],
                                  with_scale=with_scale,
                                  with_loc=with_loc)
        if p is None:
            print(f'Couldn\'t fit psychometric function to {trace["name"]}: RuntimeError')
            continue
        threshold = get_psychometric_point(0.707, p[0], p[1])
        fig.add_trace(go.Scatter(x=plot_x,
                                 y=psychometric_func(plot_x,
                                                     *p),
                                 mode='lines',
                                 name=f"{trace['name']} {threshold}",
                                 line={'color': trace['marker']['color']},
                                 legendgroup=trace['legendgroup'],
                                 showlegend=True))
    fig.update_yaxes(range=ylimit)
    if outputpath is None:
        fig.show()
    else:
        fig.write_html(outputpath,
                       include_plotlyjs='directory')


def plot_dprime(df, path, include_plotlyjs='directory'):
    fig = px.line(df,
                  x='gap_length',
                  y='dprime',
                  color='experiment_parameter',
                  title='Gap D\'',
                  facet_col='dprime_type')
    fig.write_html(path,
                   include_plotlyjs=include_plotlyjs)
