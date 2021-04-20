import sys
import genotypes_to_visualize
from graphviz import Digraph


def plot(genotype, filename, output_format='pdf', return_type="render", view_render= True):
    """
    Args:
        return_type (str): Defines how the created graph should be returned. Can be one of the following:
            'render': Renders the graph    
            'source': Returns the source
            'graph': Returns the graph object
        view_render (bool): When the graph should be rendered (see option above), should it also be displayed?

    """
    g = Digraph(
        format=output_format,
        edge_attr=dict(fontsize="20", fontname="times"),
        node_attr=dict(
            style="filled",
            shape="rect",
            align="center",
            fontsize="20",
            height="0.5",
            width="0.5",
            penwidth="2",
            fontname="times",
        ),
        engine="dot",
    )
    g.body.extend(["rankdir=LR"])

    g.node("c_{k-2}", fillcolor="darkseagreen2")
    g.node("c_{k-1}", fillcolor="darkseagreen2")
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor="lightblue")

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor="palegoldenrod")
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    return_type = return_type.lower()
    if return_type == 'source':
        return g.source
    elif return_type == 'graph':
        return g
    else: #if return_type == 'render'
        #default: render
        g.render(filename, view=view_render)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval(f"genotypes_to_visualize.{genotype_name}")
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")
