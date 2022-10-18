import os
from typing import Annotated

import plotly.graph_objects as go

from dnn_modelling.helpers import load_history
from typing_tools.annotations import check_annotations
from typing_tools.annotation_checkers import PathExists


@check_annotations
def plot_history(model_path: os.PathLike, output_folder: Annotated[os.PathLike, PathExists]) -> None:
    """
    Plots the training history of the given model

    @param model_path:
        path to the model (used for loading the training-history)
    @param output_folder:
        output folder to save the plot in
    """
    # Load history
    history_df = load_history(model_path)

    # Plot history
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['train_loss'],
                             mode='lines+markers',
                             name='Training Loss'))
    fig.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['valid_loss'],
                             mode='lines+markers',
                             name='Validation Loss'))
    fig.write_html(os.path.join(output_folder,
                                'history.html'),
                   include_plotlyjs='directory')
