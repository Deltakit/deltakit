# (c) Copyright Riverlane 2020-2025.
from collections.abc import Collection, Iterable

import numpy as np
import plotly.graph_objects as go
from deltakit_core.decoding_graphs import (
    DecodingEdge,
    NXDecodingGraph,
    OrderedDecodingEdges,
    OrderedSyndrome,
)


class VisDecodingGraph3D:
    """Class to render 2+1D DecodingGraphs using plotly.

    Parameters
    ----------
    decoding_graph : NXDecodingGraph
        Decoding graph to base visualisations on.
    """

    def __init__(self, decoding_graph: NXDecodingGraph):
        self.decoding_graph = decoding_graph

        self.palette = ["#ff7500", "#dc4405", "#00968f",
                        "#3ccbda", "#cf6f7f", "#1d3c34", "#006f62"]
        self.logical_palette = ["#780116", "#F7B538", "#DB7C26", "#D8572A", "#C32F27"]

    def _categorise_nodes(self, nodes: Iterable[int]
                          ) -> tuple[set[int], set[int], set[int]]:
        """Categorise nodes to three types: boundary, boundary-adjacent or bulk"""
        boundary_nodes: set[int] = set()
        boundary_adj_nodes: set[int] = set()
        bulk_nodes: set[int] = set()
        for node in nodes:
            if self.decoding_graph.detector_is_boundary(node):
                boundary_nodes.add(node)
            elif any(
                self.decoding_graph.detector_is_boundary(n)
                for n in self.decoding_graph.neighbors(node)
            ):
                boundary_adj_nodes.add(node)
            else:
                bulk_nodes.add(node)
        return (boundary_nodes, boundary_adj_nodes, bulk_nodes)

    def _separate_boundary_edges(self, edges: Iterable[DecodingEdge]
                                 ) -> tuple[set[DecodingEdge], set[DecodingEdge]]:
        """Categorise edges to four types: to-boundary, spacelike, timelike or hook"""
        boundary_edges: set[DecodingEdge] = set()
        normal_edges: set[DecodingEdge] = set()
        for edge in edges:
            u, v = edge
            if (self.decoding_graph.detector_is_boundary(u)
                    or self.decoding_graph.detector_is_boundary(v)):
                boundary_edges.add(edge)
            else:
                normal_edges.add(edge)
        return (boundary_edges, normal_edges)

    def _categorise_edges(self, edges: Iterable[DecodingEdge]
                          ) -> tuple[set[DecodingEdge], set[DecodingEdge],
                                     set[DecodingEdge], set[DecodingEdge]]:
        """Categorise edges to four types: to-boundary, spacelike, timelike or hook"""
        boundary_edges: set[DecodingEdge] = set()
        spacelike_edges: set[DecodingEdge] = set()
        timelike_edges: set[DecodingEdge] = set()
        hook_edges: set[DecodingEdge] = set()
        for edge in edges:
            u, v = edge
            if (self.decoding_graph.detector_is_boundary(u)
                    or self.decoding_graph.detector_is_boundary(v)):
                boundary_edges.add(edge)
            elif edge.is_timelike(self.decoding_graph.detector_records):
                timelike_edges.add(edge)
            elif edge.is_spacelike(self.decoding_graph.detector_records):
                spacelike_edges.add(edge)
            elif edge.is_hooklike(self.decoding_graph.detector_records):
                hook_edges.add(edge)
            else:
                msg = f"Unknown edge type: {edge}"
                raise ValueError(msg)
        return (boundary_edges, spacelike_edges, timelike_edges, hook_edges)

    def get_plot_3d_traces(
        self,
        syndrome: OrderedSyndrome | None = None,
        correction_edges: OrderedDecodingEdges | None = None,
        error_edges: OrderedDecodingEdges | None = None,
        logicals: list[set[DecodingEdge]] | None = None,
        categorise_edges: bool = False,
    ) -> list[go.Trace]:
        """Plots the DecodingGraph in 3D based on the coordinates (h, v, t)
        of the nodes of the graph.

        Parameters
        ----------
        syndrome : Optional[OrderedSyndrome], optional
            Optional syndrome to visualise by highlighting nodes on the graph.
        correction_edges : Optional[OrderedDecodingEdges], optional
            Optional set of edges that represent a correction from a decoder,
            to be highlighted in the graph.
        error_edges : Optional[OrderedDecodingEdges], optional
            Optional set of edges that represent an error mechanism, to be
            highlighted in the graph.
        logicals : Optional[List[Set[DecodingEdge]]], optional
            Optional list of sets of edges that define the logicals observables
            on the decoding graph, to be highlighted.
        categorise_edges : bool
            Whether to split the edges in timelike, spacelike and hooks.
            By default, False.

        Returns
        -------
        List[go.Trace]
            List of traces representing the plot_3D.
        """
        traces: list[go.Trace] = []
        traces += self.get_node_traces()
        if categorise_edges:
            traces += self.get_categorised_edges_traces()
        else:
            traces += self.get_edges_traces()
        if syndrome is not None:
            traces += self.get_syndrome_traces(syndrome)
        if correction_edges is not None:
            traces += self.get_corrections_traces(correction_edges)
        if error_edges is not None:
            traces += self.get_error_edges_traces(error_edges)
        if logicals is not None:
            traces += self.get_logical_edges_traces(logicals)
        return traces

    def plot_3d(
        self,
        syndrome: OrderedSyndrome | None = None,
        correction_edges: OrderedDecodingEdges | None = None,
        error_edges: OrderedDecodingEdges | None = None,
        logicals: list[set[DecodingEdge]] | None = None,
        categorise_edges: bool = False,
        show: bool = True
    ) -> go.Figure:
        """Plots the DecodingGraph in 3D based on the coordinates (h, v, t)
        of the nodes of the graph.

        Parameters
        ----------
        syndrome : Optional[OrderedSyndrome], optional
            Optional syndrome to visualise by highlighting nodes on the graph.
        correction_edges : Optional[OrderedDecodingEdges], optional
            Optional set of edges that represent a correction from a decoder,
            to be highlighted in the graph.
        error_edges : Optional[OrderedDecodingEdges], optional
            Optional set of edges that represent an error mechanism, to be
            highlighted in the graph.
        logicals : Optional[List[Set[DecodingEdge]]], optional
            Optional list of sets of edges that define the logicals observables
            on the decoding graph, to be highlighted.
        categorise_edges : bool
            Whether to split the edges in timelike, spacelike and hooks.
            By default, False.
        show : bool, optional
            If the figure is shown or just returned. Default True.
        """
        traces = self.get_plot_3d_traces(
            syndrome, correction_edges, error_edges, logicals,
            categorise_edges=categorise_edges
        )
        fig = go.Figure(data=traces, layout=get_default_layout())
        if show:
            fig.show()
        return fig

    def get_node_traces(self) -> list[go.Trace]:
        """Add nodes of the base graph to traces."""
        _, boundary_adj_nodes, bulk_nodes = self._categorise_nodes(
            self.decoding_graph.nodes)
        all_nodes = [boundary_adj_nodes, bulk_nodes]
        node_names = ["Boundary-adjacent node", "Bulk node"]
        colors = ["rgb(65, 212, 228)", "rgb(24, 110, 98)"]
        traces = []
        for node_set, name, color in zip(all_nodes, node_names, colors):
            scatter = get_scatter_for_node(
                self.decoding_graph, node_set, name, color, 8, "circle")
            if scatter is not None:
                traces.append(scatter)
        return traces

    def get_edges_traces(self) -> list[go.Trace]:
        """Add edges of the base graph as lines to traces"""
        _, normal_edges = self._separate_boundary_edges(self.decoding_graph.edges)
        edge_color = "rgba(100, 100, 100, 1)"
        label = "Decoding edge"
        traces = []
        if len(normal_edges) != 0:
            edge_trace = get_line_for_edge(
                self.decoding_graph, normal_edges, label, edge_color, 2.5, "solid"
            )
            traces.append(edge_trace)
        return traces

    def get_categorised_edges_traces(self) -> list[go.Trace]:
        """Add edges of the base graph as lines to traces"""
        (
            _,
            spacelike_edges,
            timelike_edges,
            hook_edges,
        ) = self._categorise_edges(self.decoding_graph.edges)
        all_edges = [spacelike_edges, timelike_edges, hook_edges]
        edge_names = ["Spacelike edge", "Timelike edge", "Hook edge"]
        edge_colors = [
            "rgba(50, 50, 50, 1)",
            "rgba(130, 130, 130, 0.9)",
            "rgba(100, 100, 100, 0.9)",
        ]
        traces = []
        for edge_set, name, edge_color in zip(all_edges, edge_names, edge_colors):
            if len(edge_set) != 0:
                edge_trace = get_line_for_edge(
                    self.decoding_graph, edge_set, name, edge_color, 2.5, "solid"
                )
                traces.append(edge_trace)
        return traces

    def get_syndrome_traces(
        self, syndrome: OrderedSyndrome
    ) -> list[go.Trace]:
        """Given a lit up syndrome, add its nodes to traces"""
        traces = []
        if any(s not in self.decoding_graph.nodes for s in syndrome):
            msg = (
                f"SyndromeBits must belong to DecodingGraph. \
                            Invalid syndrome: {syndrome}"
            )
            raise ValueError(msg)
        if len(syndrome) != 0:
            scatter = get_scatter_for_node(
                self.decoding_graph,
                syndrome,
                "Syndrome",
                "rgba(255, 140, 0, 0.9)",
                13,
                "circle"
            )
            traces.append(scatter)
        return traces

    def highlight_edges_traces(self,
                               edges: OrderedDecodingEdges,
                               edge_type: str,
                               color: str,
                               linestyle: str
                               ) -> go.Trace:
        """Highlight given set of edges in set colour with the edge type as label."""
        boundary_edges, normal_edges = self._separate_boundary_edges(edges)

        # highlight edges to the boundary as a scatter plot with cross markers
        traces = []
        if boundary_edges:
            boundary_nodes = np.array(
                [
                    v if self.decoding_graph.detector_is_boundary(u) else u
                    for u, v in boundary_edges
                ]
            )
            if len(boundary_nodes) != 0:
                scatter = get_scatter_for_node(
                    self.decoding_graph,
                    nodes=boundary_nodes,
                    name=f"{edge_type} to boundary",
                    color=color,
                    markersize=15,
                    symbol="cross",
                )
                traces.append(scatter)

        # add all other corrections as lines
        if len(normal_edges) != 0:
            edge_trace = get_line_for_edge(
                self.decoding_graph, normal_edges, edge_type, color, 10, linestyle
            )
            traces.append(edge_trace)
        return traces

    def get_corrections_traces(
        self, correction_edges: OrderedDecodingEdges
    ) -> list[go.Trace]:
        """Add correction from decoder to traces."""
        return self.highlight_edges_traces(
            edges=correction_edges,
            edge_type="Correction",
            color="rgba(255, 187, 0, 0.8)",
            linestyle="solid"
        )

    def get_error_edges_traces(
        self, error_edges: OrderedDecodingEdges
    ) -> list[go.Trace]:
        """Add error to traces."""
        return self.highlight_edges_traces(
            edges=error_edges,
            edge_type="Error",
            color="rgba(220, 68, 5, 0.8)",
            linestyle="dash"
        )

    def get_logical_edges_traces(
        self, logicals: list[set[DecodingEdge]]
    ) -> list[go.Trace]:
        """Add logical edges to traces."""
        traces = []
        for i, edges in enumerate(logicals):
            color_idx = i % len(self.logical_palette)   # cycle colors if needed
            traces += self.highlight_edges_traces(
                edges=OrderedDecodingEdges(edges),
                edge_type=f"Logical L{i}",
                color=self.logical_palette[color_idx],
                linestyle="solid"
            )
        return traces


def get_default_layout() -> go.Layout:
    """Return Layout object for formatting the figure
    """
    return go.Layout(
        showlegend=True,
        scene={
            "xaxis": {
                "backgroundcolor": "rgb(242, 240, 239)",
                "gridcolor": "white",
                "showbackground": True,
                "zerolinecolor": "white",
                "dtick": 1,
            },
            "yaxis": {
                "backgroundcolor": "rgb(242, 240, 239)",
                "gridcolor": "white",
                "showbackground": True,
                "zerolinecolor": "white",
                "dtick": 1,
            },
            "zaxis": {
                "backgroundcolor": "rgb(230, 230, 230)",
                "gridcolor": "white",
                "showbackground": True,
                "zerolinecolor": "white",
                "dtick": 1,
            },
            "xaxis_title": "h",
            "yaxis_title": "v",
            "zaxis_title": "t",
            # fix aspect if desired
            # aspectratio=dict(x=1, y=1, z=1),
            "camera": {
                "eye": {"x": 0.2, "y": 0.9, "z": 0.2},
                "projection": {"type": "orthographic"}
                # use type="perspective" for a more immersive 3D experience
            },
        },
        margin={"t": 0, "r": 0, "l": 0, "b": 0},
        legend={"x": 0.9, "y": 0.5},
    )


def get_scatter_for_node(
    graph: NXDecodingGraph,
    nodes: Collection[int] | None = None,
    name: str | None = None,
    color: str | None = None,
    markersize: float | None = None,
    symbol: str | None = None,
    avoid_boundaries: bool = False
) -> go.Scatter3d:
    """Given a set of syndromes, plot them in 3D with specified style parameters"""
    if nodes is None:
        nodes = graph.nodes
    if avoid_boundaries:
        nodes = [n for n in nodes if not graph.detector_is_boundary(n)]

    xyz = np.array(
        [
            graph.detector_records[node].full_coord
            for node in nodes
            if not graph.detector_is_boundary(node)
        ]
    )
    if len(xyz) > 0:
        return go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            customdata=list(zip(nodes, xyz)),
            hovertemplate="Detector: %{customdata[0]}<br>"
            + "Full coord: (%{customdata[1]})",
            marker={"size": markersize, "color": color, "symbol": symbol},
            mode="markers",
            name=name,
        )
    return go.Scatter3d()


def get_line_for_edge(
    graph: NXDecodingGraph,
    edges: Collection[DecodingEdge] | None = None,
    name: str | None = None,
    color: str | None = None,
    linewidth: float | None = None,
    linestyle: str | None = None,
    avoid_boundaries: bool = False
) -> go.Scatter3d:
    """Given a set of edges, plot them in 3D with specified style parameters"""
    if edges is None:
        edges = graph.edges
    if avoid_boundaries:
        edges = [e for e in edges if not (graph.detector_is_boundary(
            e.first) or graph.detector_is_boundary(e.second))]
    x_edges, y_edges, z_edges = [], [], []
    edge_syndromes = []
    edge_weights = []
    for edge in edges:
        first_coord = graph.detector_records[
            edge.first
        ].full_coord
        second_coord = graph.detector_records[
            edge.second
        ].full_coord
        x_edges += [first_coord[0], second_coord[0], None]
        y_edges += [first_coord[1], second_coord[1], None]
        z_edges += [first_coord[2], second_coord[2], None]
        edge_syndromes += 3 * [str((edge.first, edge.second))]
        edge_weight = graph.edge_records[edge].weight
        edge_weights += 3 * [str(np.round(edge_weight, 4))]
    return go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        line={"color": color, "width": linewidth, "dash": linestyle},
        mode="lines",
        customdata=list(zip(edge_syndromes, edge_weights)),
        hovertemplate="edge: %{customdata[0]}<br>" + "weight: %{customdata[1]}",
        name=name,
    )
