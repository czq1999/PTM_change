import networkx as nx
import matplotlib.pyplot as plt

# G = nx.DiGraph()  # 有向图和无向图
# G.add_nodes_from(['1', '3'])
# G.add_edge('2', '3')
# e = ('4', '5')
# G.add_edge(*e)
# G.add_edges_from([(1, 2), (1, 3)])
# print(G.nodes())
# print(nx.adjacency_matrix(G).todense())

code_tree = {-1: {'indent': -1, 'child': [0, 6], 'parent': None, 'code': None},
             0: {'code': 'def  render_body ( context ,  options ) ', 'indent': 0, 'parent': -1, 'child': [1, 3, 5]},
             1: {'code': '       if  options . key? ( :partial ) ', 'indent': 1, 'parent': 0, 'child': [2]},
             2: {'code': '         [ render_partial ( context ,  options ) ] ', 'indent': 2, 'parent': 1, 'child': []},
             3: {'code': '       else ', 'indent': 1, 'parent': 0, 'child': [4]},
             4: {
                 'code': '         StreamingTemplateRenderer . new ( @lookup_context ) . render ( context ,  options ) ',
                 'indent': 2, 'parent': 3, 'child': []},
             5: {'code': '       end ', 'indent': 1, 'parent': 0, 'child': []},
             6: {'code': '     end', 'indent': 0, 'parent': -1, 'child': []}}


def adj_from_tree_use_networkx(code_tree, save_path=None):
    G = nx.DiGraph()
    G.add_nodes_from(code_tree.items())
    for k, v in code_tree.items():
        child = v['child']
        if len(child):
            for each in v['child']:
                G.add_edge(k, each)
                G.add_edge(each, k)

    print(list(nx.adjacency_matrix(G).todense()))

    # pos = nx.spring_layout(G)
    # # node_color = nx.get_node_attributes(G, 'indent').values()
    #
    # nx.draw(G, pos)
    # # edge_labels = nx.get_edge_attributes(G, 'weigh')
    # # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # nx.draw_networkx_labels(G, pos, alpha=1)
    #
    # plt.show()
    # plt.savefig(save_path + 'tree.jpg')
    #
    # code = ''
    # for k, v in code_tree.items():
    #     if k == -1:
    #         code += '-1\troot\n'
    #     else:
    #         code += f"{k}\t{v['code']}\n"
    # print(code)
adj_from_tree_use_networkx(code_tree)

from PIL import Image, ImageFont, ImageDraw


def draw_code(code, save_path):
    image = Image.new('RGB', (250, 250), (255, 255, 255))  # 设置画布大小及背景色
    iwidth, iheight = image.size  # 获取画布高宽
    font = ImageFont.truetype('consola.ttf', 110)  # 设置字体及字号
    draw = ImageDraw.Draw(image)

    fwidth, fheight = draw.textsize('22', font)  # 获取文字高宽

    fontx = (iwidth - fwidth - font.getoffset('22')[0]) / 2
    fonty = (iheight - fheight - font.getoffset('22')[1]) / 2

    draw.text((fontx, fonty), code, 'black', font)
    image.save(save_path + 'code.jpg')  # 保存图片


code = ''
for k, v in code_tree.items():
    if k == -1:
        code += '-1\troot\n'
    else:
        code += f"{k}\t{v['code']}\n"
print(code)
draw_code('code\n 1234\n 1267\n', './')
