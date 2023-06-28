import plotly.express as px


def get_color_with_alpha(index, alpha):
    c = px.colors.hex_to_rgb(px.colors.qualitative.Alphabet[index])
    return f'rgba({c[0]},{c[1]},{c[2]},{alpha})'
