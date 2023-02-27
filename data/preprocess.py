import json
from tqdm import tqdm
from datasets import load_dataset
from data_utils import parse_formula, jieba_retokenize_for_dep, convert_nodes


def make_parse():
    '''parse formulas and replace them with [MATH]'''
    # we store our data in json line format, such as
    # "{"id": "xxx", "content": "xxx"}\n{"id": "xxx", "content": "xxx"}"
    f = open('text_data.json')
    w = open('parsed_data.json', "w")
    for data in tqdm(map(json.loads, f)):
        content = data.pop('content').split(' ')
        
        content, content_parse = parse_formula(content)
        content_jieba = jieba_retokenize_for_dep(content)
        
        # in this project, we use \\fb and \\fe to identify the location of 
        # mathematical formulas (formula begin and formula end)
        content = [w for w in content if w not in ['\\fb', '\\fe']]

        data['content'] = ' '.join(content)
        data['content_jieba'] = ' '.join(content_jieba)
        data['content_parse'] = content_parse
        w.write(json.dumps(data, ensure_ascii=False) + "\n")
    f.close()
    w.close()


def construct_ms_graph():
    '''dependency parsing and construct math syntax graph (combine operator tree and dependency parsing tree)'''
    import stanza
    nlp = stanza.Pipeline(
        lang='zh', 
        processors='tokenize,pos,lemma,depparse', 
        model_dir='../stanza_resources', 
        tokenize_pretokenized=True,
    )

    with open("node_vocab.json") as f:
        node_vocab = json.load(f)
    node_vocab = {n: i for i, n in enumerate(node_vocab)}
    unk_id = node_vocab['[UNK]']

    f = open('parsed_data.json')
    w = open('dep_data.json', "w")
    for data in tqdm(map(json.loads, f)):
        content_jieba = data.pop('content_jieba')
        content_parse = data.pop('content_parse')

        sentence = nlp(content_jieba).sentences[0]

        nodes = ['ROOT'] + [word.text for word in sentence.words]
        edges = [(word.id, word.head) for word in sentence.words]
        # allign formula parse
        math_id = [i for i, word in enumerate(nodes) if word == '[MATH]']

        for mid in math_id:
            if (mid, 0) not in edges:
                edges.append((mid, 0))
        for n, e in content_parse:
            if len(n) > 0:
                edges.append((len(nodes), 0))
                edges.append((0, len(nodes)))
            for x, y in e:
                edges.append((x + len(nodes), y + len(nodes)))
                edges.append((y + len(nodes), x + len(nodes)))
            nodes.extend([convert_nodes(sn) for sn in n])
        src = [edge[0] for edge in edges]
        dst = [edge[1] for edge in edges]
        assert max(src + dst) <= len(nodes)

        data['nodes_id'] = [node_vocab.get(node, unk_id) for node in nodes]
        data['src'] = src
        data['dst'] = dst
        w.write(json.dumps(data, ensure_ascii=False) + '\n')
    f.close()
    w.close()


def make_datasets():
    dataset = load_dataset('json', data_files='dep_data.json')
    dataset.save_to_disk('graph_dataset')