import os
import json
from tqdm import tqdm
from datasets import load_dataset
from data_utils import parse_formula, jieba_retokenize_for_dep, convert_nodes


def get_operator_tree():
    '''parse formulas and replace them with [MATH]'''
    # we store our data in json line format, such as
    # "{"id": "xxx", "content": "xxx"}\n{"id": "xxx", "content": "xxx"}"
    with open('example_data.json') as f, open('opt_data.json', "w") as w:
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


def get_math_syntax_graph():
    '''dependency parsing and construct math syntax graph (combine operator tree and dependency parsing tree)'''
    import stanza
    
    if not os.path.exists('~/stanza_resources/zh-hans'):
        stanza.download('zh', processors='tokenize,pos,lemma,depparse')
        
    nlp = stanza.Pipeline(
        lang='zh', 
        processors='tokenize,pos,lemma,depparse', 
        tokenize_pretokenized=True,
    )

    with open('opt_data.json') as f, open('msg_data.json', "w") as w:
        for data in tqdm(map(json.loads, f)):
            content_jieba = data.pop('content_jieba')
            content_parse = data.pop('content_parse')

            sentence = nlp(content_jieba).sentences[0]

            nodes = ['ROOT'] + [word.text for word in sentence.words]
            edges = [(word.id, word.head, word.deprel) for word in sentence.words]
            
            # allign formula parse
            math_id = [i for i, word in enumerate(nodes) if word == '[MATH]']
            for mid in math_id:
                if (mid, 0, 'root') not in edges:
                    edges.append((mid, 0, 'root'))
                    edges.append((0, mid, 'root'))
            for n, e in content_parse:
                for x, y in e:
                    edges.append((x + len(nodes), y + len(nodes), 'formula'))
                    edges.append((y + len(nodes), x + len(nodes), 'formula'))
                nodes.extend([convert_nodes(sn) for sn in n])
            src = [edge[0] for edge in edges]
            dst = [edge[1] for edge in edges]
            rel = [edge[2] for edge in edges]
            assert max(src + dst) <= len(nodes)

            data['nodes'] = nodes
            data['src'] = src
            data['dst'] = dst
            data['rel'] = rel
            w.write(json.dumps(data, ensure_ascii=False) + '\n')


def get_dataset():
    graph_vocab = set(['[UNK]'])
    rel_vocab = set()
    with open('msg_data.json') as f:
        for data in tqdm(map(json.loads, f)):
            graph_vocab.update(data['nodes'])
            rel_vocab.update(data['rel'])

    with open('graph_vocab.json', 'w') as w:
        json.dump(list(graph_vocab), w, ensure_ascii=False)
    with open('rel_vocab.json', 'w') as w:
        json.dump(list(rel_vocab), w, ensure_ascii=False)
    
        
    # convert node to node_id
    graph_vocab = {node: i for i, node in enumerate(graph_vocab)}
    rel_vocab = {relation: i for i, relation in enumerate(rel_vocab)}
    unk_id = graph_vocab['[UNK]']
    with open('msg_data.json') as f, open('final_data.json', 'w') as w:
        for data in tqdm(map(json.loads, f)):
            nodes = data.pop('nodes')
            rel = data.pop('rel')
            data['node_ids'] = [graph_vocab.get(node, unk_id) for node in nodes]
            data['rel_ids'] = [rel_vocab[r] for r in rel]
            w.write(json.dumps(data, ensure_ascii=False) + '\n')

    dataset = load_dataset('json', data_files='final_data.json')
    print(dataset)
    dataset.save_to_disk('graph_dataset')


if __name__=='__main__':
    get_operator_tree()
    get_math_syntax_graph()
    get_dataset()
