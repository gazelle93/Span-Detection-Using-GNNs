import stanza
import spacy

def get_nlp_pipeline(_nlp_pipeline):
    if _nlp_pipeline == "stanza":
        return stanza.Pipeline('en')
    elif _nlp_pipeline == "spacy":
        return spacy.load("en_core_web_sm")
    

def word_tokenization(_input_text, nlp, _nlp_pipeline=None, _lower=False):
    if _lower == True:
        _input_text = _input_text.lower()
    
    input_tk_list = ["ROOT"]
    input_dep_list = []
    
    if _nlp_pipeline == None:
        return _input_text.split()
    
    elif _nlp_pipeline == "stanza":
        text = nlp(_input_text)

        for sen in text.sentences:
            for tk in sen.tokens:
                tk_infor_dict = tk.to_dict()[0]
                cur_tk = tk_infor_dict["text"]

                cur_id = tk_infor_dict['id']
                cur_head = tk_infor_dict['head']
                cur_dep = tk_infor_dict["deprel"]

                cur_dep_triple = (cur_id, cur_dep, cur_head)

                input_tk_list.append(cur_tk)
                input_dep_list.append(cur_dep_triple)

        return input_tk_list, input_dep_list

    elif _nlp_pipeline == "spacy":
        text = nlp(_input_text)

        for tk_idx, tk in enumerate(text):
            cur_tk = tk.text

            cur_id = tk_idx+1
            cur_dep = tk.dep_

            if cur_dep == "ROOT":
                cur_head = 0
            else:
                cur_head = tk.head.i+1

            cur_dep_triple = (cur_id, cur_dep, cur_head)
            input_tk_list.append(cur_tk)
            input_dep_list.append(cur_dep_triple)

        return input_tk_list, input_dep_list


def get_span_of_target(input_text, target_text):
    for i in range(len(input_text)):
        if input_text[i:i+len(target_text)] == target_text:
            return [x for x in range(i,i+len(target_text))]


def get_output_label(input_text, ant_span, con_span):
    label_list = ["O", "A", "C"]
    label_dict = {}
    reverse_label_dict = {}
    for idx, l in enumerate(label_list):
        label_dict[idx] = l
        reverse_label_dict[l] = idx

    output_label = [reverse_label_dict["O"] for x in range(len(input_text))]

    for idx, _ in enumerate(input_text):
        if idx in ant_span:
            output_label[idx] = reverse_label_dict["A"]
        elif idx in con_span:
            output_label[idx] = reverse_label_dict["C"]
    return output_label, label_dict, reverse_label_dict


def get_target(input_text, ant_span, con_span, nlp_pipeline):
    selected_nlp_pipeline = get_nlp_pipeline(nlp_pipeline)

    processed_cur_text, _ = word_tokenization(input_text, selected_nlp_pipeline, nlp_pipeline)
    processed_ant_span_text, _ = word_tokenization(ant_span, selected_nlp_pipeline, nlp_pipeline)
    processed_con_span_text, _ = word_tokenization(con_span, selected_nlp_pipeline, nlp_pipeline)

    ant_target_span = get_span_of_target(processed_cur_text, processed_ant_span_text[1:])
    con_target_span = get_span_of_target(processed_cur_text, processed_con_span_text[1:])

    return get_output_label(processed_cur_text, ant_target_span, con_target_span)
