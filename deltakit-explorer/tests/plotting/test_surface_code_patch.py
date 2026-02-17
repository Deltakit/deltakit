# (c) Copyright Riverlane 2020-2025.

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from deltakit_explorer.codes._planar_code import RotatedPlanarCode, UnrotatedToricCode
from deltakit_explorer.plotting._surface_code_patch import draw_surface_code_patch


def test_planar_draw_patch_saves_png(tmp_path):
    code = RotatedPlanarCode(width=2, height=2, use_ancilla_qubits=True)
    out = tmp_path / "planar.png"
    code.draw_patch(str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_toric_draw_patch_saves_png(tmp_path):
    code = UnrotatedToricCode(
        horizontal_distance=2,
        vertical_distance=2,
        use_ancilla_qubits=True,
    )
    out = tmp_path / "toric.png"
    code.draw_patch(str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_draw_surface_code_patch_artist_counts():
    code = RotatedPlanarCode(width=3, height=3, use_ancilla_qubits=True)
    fig, ax = draw_surface_code_patch(code, margin=1, sort_stabilisers=True)

    circles = [
        artist
        for artist in ax.get_children()
        if isinstance(artist, Circle)
    ]
    polys = [
        artist
        for artist in ax.get_children()
        if isinstance(artist, Polygon)
    ]

    assert len(circles) == len(code.data_qubits) + len(code.ancilla_qubits)
    assert len(polys) == sum(len(layer) for layer in code.stabilisers)
    plt.close(fig)
